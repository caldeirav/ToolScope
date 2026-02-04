from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import (
    CanonicalTool,
    EmbeddingProvider,
    ToolIndexBackend,
    ToolNormalizer,
    ToolFingerprint,
)
from .cache import EmbeddingCache
from .embedding_config import EmbeddingConfig
from .embeddings import make_embedder
from .normalize import AutoToolNormalizer
from .backends.memory import MemoryBackend
from .text import ToolTextConfig
from .observability import ToolScopeTrace, ToolScore, TraceSink, Stopwatch
from .rerankers import Reranker, make_reranker
from .reranking_config import RerankingConfig
from .session import StickySessionConfig, SessionCache, cosine


def messages_to_query_text(messages: Any) -> str:
    """
    Convert messages into a single query string.
    Supports:
      - str
      - OpenAI-style list[{"role":..., "content":...}]
      - anything else via str(...)
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                parts.append(f"{m.get('role','')}: {m.get('content','')}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(messages)


def _tag_set(tags: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not tags:
        return None
    s = {t.strip() for t in tags if isinstance(t, str) and t.strip()}
    return s or None


def _passes_tag_filters(tool: CanonicalTool, allow: Optional[set[str]], deny: Optional[set[str]]) -> bool:
    tset = set(tool.tags or ())
    if deny is not None and (tset & deny):
        return False
    if allow is not None and not (tset & allow):
        return False
    return True


@dataclass
class ToolIndex:
    """
    Long-lived tool index: register tools once, filter per prompt/messages.

    This object:
      - normalizes tool specs into CanonicalTool records
      - embeds tool texts (cached by fingerprint)
      - upserts vectors into the backend (Memory/MilvusLite/etc.)
      - searches vectors for relevant tools at query time
      - returns the original tool specs unchanged (round-trip)
    """
    backend: ToolIndexBackend
    embedder: EmbeddingProvider
    normalizer: ToolNormalizer
    cache: EmbeddingCache

    namespace: Optional[str] = None

    tool_text_config: ToolTextConfig = ToolTextConfig()

    reranker: Optional[Reranker] = None
    reranking_config: Optional[RerankingConfig] = None

    session_cfg: StickySessionConfig = field(default_factory=lambda: StickySessionConfig(enabled=False))
    _session_cache: Optional[SessionCache] = None

    def _ensure_session_cache(self) -> None:
        if self._session_cache is None and self.session_cfg.enabled:
            self._session_cache = SessionCache(self.session_cfg)

    def upsert_tools(self, tools: Sequence[Any]) -> None:
        """
        Normalize tools, embed them (cached), and upsert into backend.
        Safe to call multiple times; tools with same fingerprint reuse cached vectors.
        """
        canonical = self.normalizer.normalize(tools)

        # Embed only what we don't already have in cache
        to_embed: List[CanonicalTool] = []
        embed_texts: List[str] = []

        for t in canonical:
            fp = t.fingerprint
            if self.cache.get(fp) is None:
                to_embed.append(t)
                embed_texts.append(self.tool_text_config.render(t))

        if to_embed:
            vecs = self.embedder.embed_texts(embed_texts)
            for t, v in zip(to_embed, vecs, strict=True):
                self.cache.put(t.fingerprint, v)  # type: ignore[arg-type]

        # Upsert ALL tools into backend using cached vectors
        # (This keeps backend consistent even if it was empty/restarted.)
        ids: List[str] = []
        vectors: List[List[float]] = []
        payloads: List[CanonicalTool] = []

        for t in canonical:
            fp = t.fingerprint
            vec = self.cache.get(fp)
            if vec is None:
                # Should not happen
                raise RuntimeError("Missing vector in cache after embedding step.")
            ids.append(fp.value)
            vectors.append(vec)
            payloads.append(t)

        dim = len(vectors[0]) if vectors else 0
        if dim > 0:
            self.backend.ensure_namespace(namespace=self.namespace, dim=dim)
            self.backend.upsert(ids, vectors, payloads, namespace=self.namespace)

    def delete_tools(self, fingerprints: Sequence[ToolFingerprint]) -> None:
        """
        Remove tools by fingerprint from the backend (hard/soft depends on backend).
        """
        ids = [fp.value for fp in fingerprints]
        if ids:
            self.backend.delete(ids, namespace=self.namespace)

    def filter_with_trace(
        self,
        messages: Any,
        *,
        k: int = 12,
        filter: Optional[Dict[str, Any]] = None,
        reranking: Optional[RerankingConfig] = None,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        trace_sink: Optional[TraceSink] = None,
    ) -> Tuple[List[Any], ToolScopeTrace]:
        trace = ToolScopeTrace(session_id=session_id, turn_id=str(turn_id) if turn_id is not None else None)
        total_sw = Stopwatch()

        qtext_full = messages_to_query_text(messages)
        q_sw = Stopwatch()
        qvec = self.embedder.embed_texts([qtext_full])[0]
        trace.ms_embed_query = q_sw.ms()

        # --- tag filters ---
        allow = _tag_set(allow_tags)
        deny = _tag_set(deny_tags)
        if allow is not None:
            trace.allow_tags = tuple(sorted(allow))
        if deny is not None:
            trace.deny_tags = tuple(sorted(deny))

        # --- session-aware state lookup ---
        state = None
        refresh_mode = False
        if session_id and self.session_cfg.enabled:
            self._ensure_session_cache()
            assert self._session_cache is not None
            state = self._session_cache.get(session_id)
            if state is not None:
                sim = cosine(qvec, state.last_query_vec)
                trace.query_similarity_to_prev = sim

                # 1) REUSE PATH: highly similar query => try reuse last selected tools
                if sim >= self.session_cfg.similarity_threshold_reuse:
                    trace.mode = "reuse"
                    prev_tools = self.backend.get(state.last_selected_ids, namespace=self.namespace)

                    # Apply allow/deny tag filters
                    f_sw = Stopwatch()
                    if allow is not None or deny is not None:
                        prev_tools = [t for t in prev_tools if _passes_tag_filters(t, allow, deny)]
                    trace.ms_filter += f_sw.ms()

                    # Return if we still have enough to be useful
                    if prev_tools:
                        prev_tools = prev_tools[:k]
                        trace.returned_tools = len(prev_tools)
                        trace.candidates_after_filters = len(prev_tools)

                        # Update session state with what we actually returned (post-filtered)
                        selected_ids = [t.fingerprint.value for t in prev_tools]
                        self._session_cache.put(session_id, qvec, selected_ids)

                        # produce output
                        tools_out = self.normalizer.denormalize(prev_tools)
                        trace.approx_chars_after = sum(len(str(t)) for t in tools_out)

                        trace.ms_total = total_sw.ms()
                        if trace_sink:
                            trace_sink.emit(trace)
                        return tools_out, trace

                # 2) REFRESH MODE: moderately similar => refresh but keep stickiness
                refresh_mode = sim >= self.session_cfg.similarity_threshold_refresh
                if refresh_mode:
                    trace.mode = "refresh"
                else:
                    trace.mode = "fresh"

        # --- determine initial retrieval pool ---
        active_cfg = reranking or self.reranking_config
        active_reranker = self.reranker
        if reranking is not None:
            active_reranker = make_reranker(reranking=reranking)

        pool_size = k
        if active_cfg is not None and active_reranker is not None:
            pool_size = max(pool_size, int(active_cfg.pool_size))

        # If tag filtering is active, pull a bigger pool so we still get k after filtering.
        if allow is not None or deny is not None:
            pool_size = max(pool_size, min(200, k * 8))

        # If refresh_mode is active, also pull a bigger pool (sticky merge needs room)
        if refresh_mode:
            pool_size = max(pool_size, min(200, k * 8))

        trace.retrieved_candidates = pool_size

        # --- backend search (optionally pass filter hints) ---
        backend_filter = dict(filter or {})
        if allow is not None:
            backend_filter["allow_tags"] = sorted(allow)
        if deny is not None:
            backend_filter["deny_tags"] = sorted(deny)

        s_sw = Stopwatch()
        hits = self.backend.search(
            qvec,
            k=pool_size,
            namespace=self.namespace,
            filter=backend_filter or None,
        )
        trace.ms_search = s_sw.ms()
        if not hits:
            trace.ms_total = total_sw.ms()

            # still update session state to avoid thrashing, but only if session is enabled
            if session_id and self.session_cfg.enabled:
                self._ensure_session_cache()
                assert self._session_cache is not None
                self._session_cache.put(session_id, qvec, [])

            if trace_sink:
                trace_sink.emit(trace)
            return [], trace

        hit_ids = [hid for (hid, _score) in hits]
        hit_score = {hid: float(sc) for (hid, sc) in hits}

        # ---- fetch ----
        f_sw = Stopwatch()
        fetched = self.backend.get(hit_ids, namespace=self.namespace)
        trace.ms_fetch = f_sw.ms()

        # ---- order by vector rank ----
        by_id: Dict[str, CanonicalTool] = {t.fingerprint.value: t for t in fetched}
        candidates: List[CanonicalTool] = [by_id[hid] for hid in hit_ids if hid in by_id]
        if not candidates and fetched:
            candidates = fetched[:]  # best-effort fallback

        # record candidate scores (pre-filter)
        trace.candidates = [
            ToolScore(
                tool_id=t.fingerprint.value,
                tool_name=t.name,
                vector_score=hit_score.get(t.fingerprint.value),
            )
            for t in candidates[: min(len(candidates), 50)]  # cap trace size
        ]

        # --- apply allow/deny tag filters ---
        filt_sw = Stopwatch()
        if allow is not None or deny is not None:
            kept: List[CanonicalTool] = []
            kept_ids = set()
            for t in candidates:
                ok = _passes_tag_filters(t, allow, deny)
                if ok:
                    kept.append(t)
                    kept_ids.add(t.fingerprint.value)

            # mark in trace which candidates were filtered out
            kept_set = kept_ids
            for cs in trace.candidates:
                if cs.tool_id not in kept_set:
                    cs.passed_filters = False
                    reasons: List[str] = []
                    if deny is not None:
                        reasons.append("deny_tags")
                    if allow is not None:
                        reasons.append("allow_tags")
                    cs.filter_reasons = tuple(reasons)

            candidates = kept

        trace.ms_filter = filt_sw.ms()
        trace.candidates_after_filters = len(candidates)

        # --- stickiness (refresh mode) ---
        if state is not None and refresh_mode and candidates:
            prev_set = set(state.last_selected_ids)

            # 1) Boost candidates that were previously selected
            def boost_key(t: CanonicalTool) -> float:
                return self.session_cfg.sticky_boost if t.fingerprint.value in prev_set else 0.0

            # stable sort: boosted tools bubble up but original relative order is preserved
            candidates = sorted(candidates, key=boost_key, reverse=True)

            # 2) Ensure up to sticky_keep previous tools are included, but MUST respect allow/deny
            if self.session_cfg.sticky_keep > 0 and state.last_selected_ids:
                prev_tools = self.backend.get(state.last_selected_ids, namespace=self.namespace)

                # Apply allow/deny filters to prev_tools too (IMPORTANT)
                if allow is not None or deny is not None:
                    prev_tools = [t for t in prev_tools if _passes_tag_filters(t, allow, deny)]

                existing = {t.fingerprint.value for t in candidates if t.fingerprint is not None}
                added = 0
                for pt in prev_tools:
                    if pt.fingerprint is None:
                        continue
                    if pt.fingerprint.value not in existing:
                        candidates.insert(0, pt)  # insert to favor stickiness
                        existing.add(pt.fingerprint.value)
                        added += 1
                    if added >= self.session_cfg.sticky_keep:
                        break

        # --- optional reranking ---
        if active_cfg is not None and active_reranker is not None and candidates:
            r_sw = Stopwatch()

            qtext = qtext_full[: active_cfg.max_query_chars]
            docs = [self.tool_text_config.render(t)[: active_cfg.max_doc_chars] for t in candidates]
            scores = active_reranker.score(qtext, docs)

            trace.ms_rerank = r_sw.ms()
            trace.candidates_reranked = len(candidates)

            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [t for (t, _s) in scored[:k]]

            # attach rerank scores into trace
            score_map = {t.fingerprint.value: float(s) for (t, s) in scored}
            for cs in trace.candidates:
                rs = score_map.get(cs.tool_id)
                if rs is not None:
                    cs.rerank_score = rs
        else:
            candidates = candidates[:k]

        trace.returned_tools = len(candidates)

        # --- persist session state (post-filter, post-rerank) ---
        if session_id and self.session_cfg.enabled:
            self._ensure_session_cache()
            assert self._session_cache is not None
            selected_ids = [t.fingerprint.value for t in candidates if t.fingerprint is not None]
            self._session_cache.put(session_id, qvec, selected_ids)

        tools_out = self.normalizer.denormalize(candidates)
        trace.approx_chars_after = sum(len(str(t)) for t in tools_out)

        trace.ms_total = total_sw.ms()
        if trace_sink:
            trace_sink.emit(trace)

        return tools_out, trace

    def filter(
        self,
        messages: Any,
        *,
        k: int = 12,
        filter: Optional[Dict[str, Any]] = None,
        reranking: Optional[RerankingConfig] = None,
        allow_tags: Optional[Sequence[str]] = None,
        deny_tags: Optional[Sequence[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Any]:
        tools, _trace = self.filter_with_trace(
            messages,
            k=k,
            filter=filter,
            reranking=reranking,
            allow_tags=allow_tags,
            deny_tags=deny_tags,
            session_id=session_id,
            turn_id=None,
            trace_sink=None,
        )
        return tools


def make_index(
    tools: Sequence[Any],
    *,
    backend: Optional[ToolIndexBackend] = None,
    embedder: Optional[EmbeddingProvider] = None,
    embedding: Optional[EmbeddingConfig] = None,
    normalizer: Optional[ToolNormalizer] = None,
    namespace: Optional[str] = None,
    cache: Optional[EmbeddingCache] = None,
    tool_text_config: Optional["ToolTextConfig"] = None,
    reranking: Optional[RerankingConfig] = None,
    session_cfg: Optional[StickySessionConfig] = None,
) -> ToolIndex:
    """
    Convenience constructor: creates a ToolIndex and immediately upserts tools.

    If `embedder` is not provided, ToolScope resolves one from `embedding`.
    """
    from .text import ToolTextConfig  # avoid cycles

    if embedder is not None and embedding is not None:
        raise ValueError("Specify either 'embedder' or 'embedding', not both.")

    resolved_embedder = embedder or make_embedder(embedding=embedding)

    resolved_reranker = make_reranker(reranking=reranking)

    idx = ToolIndex(
        backend=backend or MemoryBackend(),
        embedder=resolved_embedder,
        normalizer=normalizer or AutoToolNormalizer(),
        cache=cache or EmbeddingCache(vectors={}),
        namespace=namespace,
        tool_text_config=tool_text_config or ToolTextConfig(),
        reranker=resolved_reranker,
        reranking_config=reranking,
        session_cfg=session_cfg or StickySessionConfig(enabled=False),
    )
    idx.upsert_tools(tools)
    return idx
