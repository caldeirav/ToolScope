from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RerankingConfig:
    """
    Provider-agnostic reranking configuration.

    provider:
      - "sentence-transformers" (cross-encoder)
      - (future) "http", "openai", etc.

    model:
      Cross-encoder model name/id.

    pool_size:
      Retrieve top-N candidates from the vector DB, then rerank those and return top-k.
      Typical values: 20..200 depending on latency budget and tool count.

    allow_download:
      For local model providers, whether missing models may be downloaded.
      Default False.

    max_query_chars / max_doc_chars:
      Truncation applied before reranker scoring, to stabilize latency and avoid irrelevant tail text.

    extra:
      Provider-specific extra settings.
    """
    provider: str
    model: str

    pool_size: int = 50
    allow_download: bool = False

    max_query_chars: int = 512
    max_doc_chars: int = 256

    extra: Optional[Dict[str, Any]] = None
