from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import toolscope

from .utils import extract_tools


@dataclass
class FastMCPWrapperConfig:
    """
    Configuration for ToolScopeFastMCPClient.

    Provide exactly one of:
      - embedder (EmbeddingProvider with embed_texts)
      - embedding (EmbeddingConfig, e.g., HTTP provider)

    backend defaults to MemoryBackend.
    """
    embedder: Optional[Any] = None
    embedding: Optional[toolscope.EmbeddingConfig] = None
    backend: Optional[Any] = None
    normalizer: Optional[Any] = None  # default AutoToolNormalizer (should cover MCP tool shapes once MCP normalizer lands)
    namespace: str = "fastmcp"

    tool_text_config: Optional[toolscope.ToolTextConfig] = None
    reranking: Optional[toolscope.RerankingConfig] = None
    session_cfg: Optional[Any] = None

    # If True, we rebuild the index when the tool list changes (fingerprint diff)
    # even without notifications/tools/list_changed.
    auto_refresh: bool = True

    # Optional hook to attach tags without changing upstream tools:
    # tagger(tool) -> ["tag1", "tag2"]
    tagger: Optional[Callable[[Any], Sequence[str]]] = None


class ToolScopeFastMCPClient:
    """
    Wrap a FastMCP client and filter its tools per turn using ToolScope.

    Typical usage:
      upstream = fastmcp.Client(...)
      wrapper = ToolScopeFastMCPClient(upstream, config=FastMCPWrapperConfig(embedding=...))

      tools = await wrapper.list_tools(messages, k=12, deny_tags=["dangerous"])
      result = await wrapper.call_tool(tool_name, args)
    """

    def __init__(self, client: Any, *, config: FastMCPWrapperConfig) -> None:
        self._client = client
        self._cfg = config

        if self._cfg.embedder is not None and self._cfg.embedding is not None:
            raise ValueError("Specify either embedder or embedding, not both.")
        if self._cfg.embedder is None and self._cfg.embedding is None:
            raise ValueError("You must provide either embedder or embedding.")

        if self._cfg.backend is None:
            self._cfg.backend = toolscope.MemoryBackend()
        if self._cfg.normalizer is None:
            self._cfg.normalizer = toolscope.AutoToolNormalizer()

        self._index: Optional[Any] = None
        self._dirty: bool = True
        self._last_fps: Optional[set[str]] = None

    @property
    def client(self) -> Any:
        return self._client

    def mark_dirty(self) -> None:
        """Call this when you receive notifications/tools/list_changed."""
        self._dirty = True

    async def _fetch_tools(self) -> List[Any]:
        res = await self._client.list_tools()
        tools = extract_tools(res)

        if self._cfg.tagger is not None:
            tools = self._apply_tagger(tools, self._cfg.tagger)

        return tools

    def _apply_tagger(self, tools: List[Any], tagger: Callable[[Any], Sequence[str]]) -> List[Any]:
        """
        Attach tags without requiring upstream schema changes.
        For dict tools: merge into "toolscope_tags".
        For objects: set/extend ".tags" if possible.
        """
        out: List[Any] = []
        for t in tools:
            tags = [str(x).strip() for x in (tagger(t) or []) if str(x).strip()]
            if not tags:
                out.append(t)
                continue

            if isinstance(t, dict):
                t2 = dict(t)
                existing = t2.get("toolscope_tags")
                merged: List[str] = []
                if isinstance(existing, (list, tuple)):
                    merged.extend([str(x) for x in existing if isinstance(x, str)])
                merged.extend(tags)

                seen = set()
                merged2 = []
                for x in merged:
                    if x not in seen:
                        seen.add(x)
                        merged2.append(x)
                t2["toolscope_tags"] = merged2
                out.append(t2)
            else:
                try:
                    existing = getattr(t, "tags", None)
                    merged = list(existing) if isinstance(existing, (list, tuple)) else []
                    merged.extend(tags)
                    seen = set()
                    merged2 = []
                    for x in merged:
                        if x not in seen:
                            seen.add(x)
                            merged2.append(x)
                    setattr(t, "tags", merged2)
                except Exception:
                    pass
                out.append(t)
        return out

    def _compute_fps(self, tools: List[Any]) -> Optional[set[str]]:
        try:
            canon = self._cfg.normalizer.normalize(tools)
            return {ct.fingerprint.value for ct in canon}
        except Exception:
            return None

    async def _ensure_index(self) -> None:
        if self._index is None:
            tools = await self._fetch_tools()
            self._index = toolscope.index(
                tools,
                normalizer=self._cfg.normalizer,
                backend=self._cfg.backend,
                embedder=self._cfg.embedder,
                embedding=self._cfg.embedding,
                namespace=self._cfg.namespace,
                tool_text_config=self._cfg.tool_text_config,
                reranking=self._cfg.reranking,
                session_cfg=self._cfg.session_cfg,
            )
            self._last_fps = self._compute_fps(tools)
            self._dirty = False

    async def _maybe_refresh(self) -> None:
        await self._ensure_index()
        assert self._index is not None

        if not self._cfg.auto_refresh and not self._dirty:
            return

        tools = await self._fetch_tools()
        fps = self._compute_fps(tools)

        should_refresh = self._dirty
        if self._cfg.auto_refresh:
            # If we can diff fingerprints, only refresh when toolset changed.
            if fps is None or self._last_fps is None:
                should_refresh = True
            else:
                should_refresh = should_refresh or (fps != self._last_fps)

        if should_refresh:
            self._index.upsert_tools(tools)
            self._last_fps = fps
            self._dirty = False

    async def list_tools(
        self,
        messages: Any,
        *,
        k: int = 12,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Filtered tools for this turn (same tool objects/dicts as upstream).
        """
        await self._maybe_refresh()
        assert self._index is not None
        return self._index.filter(
            messages,
            k=k,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            session_id=session_id,
        )

    async def list_tools_with_trace(
        self,
        messages: Any,
        *,
        k: int = 12,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        trace_sink: Optional[Any] = None,
    ) -> Tuple[List[Any], Any]:
        await self._maybe_refresh()
        assert self._index is not None
        return self._index.filter_with_trace(
            messages,
            k=k,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            session_id=session_id,
            turn_id=turn_id,
            trace_sink=trace_sink,
        )

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Passthrough tool call to upstream FastMCP client.
        """
        # Try common signature variants for robustness.
        fn = getattr(self._client, "call_tool", None)
        if fn is None:
            raise AttributeError("Underlying FastMCP client has no call_tool()")

        try:
            return await fn(name=name, arguments=arguments)
        except TypeError:
            pass
        try:
            return await fn(name, arguments=arguments)
        except TypeError:
            pass
        try:
            return await fn(name, args=arguments)
        except TypeError:
            pass
        return await fn(name, arguments)
