# Public entrypoints
from .core.filter import filter_tools as filter
from .core.filter import filter_tools_with_trace as filter_with_trace
from .core.index import make_index as index

# Public configs
from .core.normalize import AutoToolNormalizer, McpToolNormalizer, OpenAIToolDictNormalizer
from .core.embedding_config import EmbeddingConfig
from .core.reranking_config import RerankingConfig
from .core.session import StickySessionConfig
from .core.text import (
    ToolTextConfig,
    lowercase,
    collapse_whitespace,
    normalize_unicode,
    strip_control_chars,
)

# Exceptions
from .core.embeddings import EmbeddingNotConfiguredError

# Backends
from .core.backends.memory import MemoryBackend
try:
    from .core.backends.milvus_lite import MilvusLiteBackend
except Exception:
    MilvusLiteBackend = None  # type: ignore

from .core.observability import ToolScopeTrace, TraceSink

from importlib.metadata import version as _version

__version__ = _version("toolscope")


__all__ = [
    # main API
    "filter",
    "filter_with_trace",
    "index",
    # configs
    "EmbeddingConfig",
    "RerankingConfig",
    "ToolTextConfig",
    "StickySessionConfig",
    "lowercase",
    "collapse_whitespace",
    "normalize_unicode",
    "strip_control_chars",
    "AutoToolNormalizer",
    "McpToolNormalizer",
    "OpenAIToolDictNormalizer",
    # exceptions
    "EmbeddingNotConfiguredError",
    # backends
    "MemoryBackend",
    "MilvusLiteBackend",
]
