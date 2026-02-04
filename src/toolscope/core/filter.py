from typing import Any, Optional, Sequence, Tuple

from .normalize import AutoToolNormalizer
from .index import make_index
from .types import EmbeddingProvider, ToolIndexBackend, ToolNormalizer
from .text import ToolTextConfig
from .embedding_config import EmbeddingConfig
from .reranking_config import RerankingConfig
from .observability import ToolScopeTrace, TraceSink


def filter_tools_with_trace(
    messages: Any,
    tools: Sequence[Any],
    *,
    k: int = 12,
    normalizer: Optional[ToolNormalizer] = None,
    backend: Optional[ToolIndexBackend] = None,
    namespace: Optional[str] = None,
    embedder: Optional[EmbeddingProvider] = None,
    embedding: Optional[EmbeddingConfig] = None,
    tool_text_config: Optional[ToolTextConfig] = None,
    reranking: Optional[RerankingConfig] = None,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
    trace_sink: Optional[TraceSink] = None,
) -> Tuple[list[Any], ToolScopeTrace]:
    if embedder is not None and embedding is not None:
        raise ValueError("Specify either 'embedder' or 'embedding', not both.")

    norm = normalizer or AutoToolNormalizer()
    idx = make_index(
        tools,
        normalizer=norm,
        backend=backend,
        embedder=embedder,
        embedding=embedding,
        namespace=namespace,
        tool_text_config=tool_text_config,
        reranking=reranking,
    )
    return idx.filter_with_trace(
        messages,
        k=k,
        reranking=reranking,
        allow_tags=allow_tags,
        deny_tags=deny_tags,
        trace_sink=trace_sink,
    )


def filter_tools(
    messages: Any,
    tools: Sequence[Any],
    *,
    k: int = 12,
    normalizer: Optional[ToolNormalizer] = None,
    backend: Optional[ToolIndexBackend] = None,
    namespace: Optional[str] = None,
    embedder: Optional[EmbeddingProvider] = None,
    embedding: Optional[EmbeddingConfig] = None,
    tool_text_config: Optional[ToolTextConfig] = None,
    reranking: Optional[RerankingConfig] = None,
    allow_tags: Optional[Sequence[str]] = None,
    deny_tags: Optional[Sequence[str]] = None,
) -> list[Any]:
    norm = normalizer or AutoToolNormalizer()
    idx = make_index(
        tools,
        normalizer=norm,
        backend=backend,
        embedder=embedder,
        embedding=embedding,
        namespace=namespace,
        tool_text_config=tool_text_config,
        reranking=reranking,
    )
    return idx.filter(messages, k=k, allow_tags=allow_tags, deny_tags=deny_tags)
