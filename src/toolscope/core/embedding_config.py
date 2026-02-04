from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Provider-agnostic embedding configuration.

    provider:
      - "sentence-transformers"
      - "http"
      - (future) "openai", "azure-openai", etc.

    model:
      Provider-specific model name (e.g. sentence-transformers model id, OpenAI model id).

    allow_download:
      Only relevant for local providers that may download artifacts (e.g. sentence-transformers).
      Default False to avoid surprise downloads and token/permission issues.

    normalize:
      If True, providers should return normalized vectors when supported.
      (Useful for cosine/IP similarity.)

    endpoint / headers / timeout_s:
      Used by the "http" provider.
    """
    provider: str
    model: Optional[str] = None

    allow_download: bool = False
    normalize: bool = True

    endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout_s: float = 30.0

    extra: Optional[Dict[str, Any]] = None
