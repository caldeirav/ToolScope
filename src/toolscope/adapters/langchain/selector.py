from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import toolscope

from .normalizer import LangChainToolNormalizer


@dataclass
class ToolSelector:
    """
    Reusable selector for LangChain tool lists.

    Typical usage:
      selector = ToolSelector(embedder=..., session_cfg=..., reranking=...)
      filtered = selector.select(messages, tools)

    The returned list contains the same tool objects (BaseTool instances), just fewer.
    """
    embedder: Optional[Any] = None
    embedding: Optional[toolscope.EmbeddingConfig] = None
    backend: Optional[Any] = None
    namespace: str = "langchain"
    tool_text_config: Optional[toolscope.ToolTextConfig] = None
    reranking: Optional[toolscope.RerankingConfig] = None
    session_cfg: Optional[Any] = None
    normalizer: Any = None

    _index: Optional[Any] = None
    _last_tools_id: int = -1

    def __post_init__(self) -> None:
        if self.embedder is not None and self.embedding is not None:
            raise ValueError("Specify either embedder or embedding, not both.")
        if self.embedder is None and self.embedding is None:
            raise ValueError("Provide embedder or embedding.")
        if self.normalizer is None:
            self.normalizer = LangChainToolNormalizer()
        if self.backend is None:
            self.backend = toolscope.MemoryBackend()

    def _ensure_index(self, tools: Sequence[Any]) -> None:
        # If caller passes a different list object, rebuild (simple & safe).
        tools_id = id(tools)
        if self._index is None or tools_id != self._last_tools_id:
            self._index = toolscope.index(
                tools,
                normalizer=self.normalizer,
                backend=self.backend,
                embedder=self.embedder,
                embedding=self.embedding,
                tool_text_config=self.tool_text_config,
                namespace=self.namespace,
                reranking=self.reranking,
                session_cfg=self.session_cfg,
            )
            self._last_tools_id = tools_id
        else:
            # Same list object; still allow incremental updates if user mutated it.
            # We can upsert_tools again (cheap for MemoryBackend; may be heavier for DB).
            self._index.upsert_tools(tools)

    def select(
        self,
        messages: Any,
        tools: Sequence[Any],
        *,
        k: int = 12,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Any]:
        self._ensure_index(tools)
        assert self._index is not None
        return self._index.filter(
            messages,
            k=k,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            session_id=session_id,
        )

    def select_with_trace(
        self,
        messages: Any,
        tools: Sequence[Any],
        *,
        k: int = 12,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        trace_sink: Optional[Any] = None,
    ) -> Tuple[List[Any], Any]:
        self._ensure_index(tools)
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


def filter_tools(
    messages: Any,
    tools: Sequence[Any],
    *,
    k: int = 12,
    embedder: Optional[Any] = None,
    embedding: Optional[toolscope.EmbeddingConfig] = None,
    backend: Optional[Any] = None,
    tool_text_config: Optional[toolscope.ToolTextConfig] = None,
    reranking: Optional[toolscope.RerankingConfig] = None,
    session_cfg: Optional[Any] = None,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
    session_id: Optional[str] = None,
) -> List[Any]:
    selector = ToolSelector(
        embedder=embedder,
        embedding=embedding,
        backend=backend,
        tool_text_config=tool_text_config,
        reranking=reranking,
        session_cfg=session_cfg,
    )
    return selector.select(
        messages,
        tools,
        k=k,
        allow_tags=allow_tags,
        deny_tags=deny_tags,
        session_id=session_id,
    )


def filter_tools_with_trace(
    messages: Any,
    tools: Sequence[Any],
    *,
    k: int = 12,
    embedder: Optional[Any] = None,
    embedding: Optional[toolscope.EmbeddingConfig] = None,
    backend: Optional[Any] = None,
    tool_text_config: Optional[toolscope.ToolTextConfig] = None,
    reranking: Optional[toolscope.RerankingConfig] = None,
    session_cfg: Optional[Any] = None,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    trace_sink: Optional[Any] = None,
) -> Tuple[List[Any], Any]:
    selector = ToolSelector(
        embedder=embedder,
        embedding=embedding,
        backend=backend,
        tool_text_config=tool_text_config,
        reranking=reranking,
        session_cfg=session_cfg,
    )
    return selector.select_with_trace(
        messages,
        tools,
        k=k,
        allow_tags=allow_tags,
        deny_tags=deny_tags,
        session_id=session_id,
        turn_id=turn_id,
        trace_sink=trace_sink,
    )
