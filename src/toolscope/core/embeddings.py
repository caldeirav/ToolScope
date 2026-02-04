from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .types import EmbeddingProvider
from .embedding_config import EmbeddingConfig


class EmbeddingNotConfiguredError(RuntimeError):
    pass


# ---------------- Providers ----------------

@dataclass
class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using `sentence-transformers`.

    Requires optional dependency:
      pip install toolscope[st]
    """
    model_name: str
    allow_download: bool = False
    normalize_embeddings: bool = True

    _model: object | None = None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise EmbeddingNotConfiguredError(
                "sentence-transformers is not installed. Install with: pip install toolscope[st]"
            ) from e

        kwargs: Dict[str, Any] = {}
        if not self.allow_download:
            # Avoid surprise downloads by default
            kwargs["local_files_only"] = True

        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as e:
            if not self.allow_download:
                raise EmbeddingNotConfiguredError(
                    f"Embedding model '{self.model_name}' not found locally. "
                    f"Either pre-download it (e.g. in your image/CI) or set allow_download=True."
                ) from e
            raise

        return self._model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._load_model()

        # mypy-friendly import
        from sentence_transformers import SentenceTransformer  # type: ignore
        assert isinstance(model, SentenceTransformer)

        vecs = model.encode(
            list(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()


@dataclass
class HttpEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider that calls an HTTP embedding service.

    Requires optional dependency:
      pip install toolscope[http]

    Contract:
      POST {endpoint}
      Body: {"texts": ["...", "..."], "model": "..."}   (model optional)
      Response: {"vectors": [[...], [...]]}

    You can adapt the request/response format later via EmbeddingConfig.extra if needed.
    """
    endpoint: str
    model: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout_s: float = 30.0

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            import httpx
        except Exception as e:
            raise EmbeddingNotConfiguredError(
                "httpx is not installed. Install with: pip install toolscope[http]"
            ) from e

        payload: Dict[str, Any] = {"texts": list(texts)}
        if self.model:
            payload["model"] = self.model

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self.endpoint, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

        vectors = data.get("vectors")
        if not isinstance(vectors, list) or not vectors:
            raise RuntimeError("HTTP embedder returned no vectors (expected JSON key 'vectors').")
        return vectors


# ---------------- Resolver ----------------

def make_embedder(*, embedding: Optional[EmbeddingConfig]) -> EmbeddingProvider:
    """
    Resolve an EmbeddingProvider from EmbeddingConfig.

    Policy:
      - If embedding is None: raise with clear setup instructions.
      - Only imports provider deps when that provider is selected.
    """
    if embedding is None:
        raise EmbeddingNotConfiguredError(
            "No embedding configured. Provide either:\n"
            "  - embedder=... (custom EmbeddingProvider), or\n"
            "  - embedding=EmbeddingConfig(provider=..., model/endpoint=...)\n\n"
            "Examples:\n"
            "  EmbeddingConfig(provider='sentence-transformers', model='sentence-transformers/all-MiniLM-L6-v2')\n"
            "  EmbeddingConfig(provider='http', endpoint='http://localhost:8080/embed')"
        )

    prov = embedding.provider.strip().lower()

    if prov in ("sentence-transformers", "sentence_transformers", "st"):
        if not embedding.model:
            raise EmbeddingNotConfiguredError("sentence-transformers provider requires embedding.model")
        return SentenceTransformersEmbeddingProvider(
            model_name=embedding.model,
            allow_download=embedding.allow_download,
            normalize_embeddings=embedding.normalize,
        )

    if prov in ("http", "rest"):
        if not embedding.endpoint:
            raise EmbeddingNotConfiguredError("http provider requires embedding.endpoint")
        return HttpEmbeddingProvider(
            endpoint=embedding.endpoint,
            model=embedding.model,
            headers=embedding.headers,
            timeout_s=embedding.timeout_s,
        )

    raise EmbeddingNotConfiguredError(f"Unknown embedding provider: {embedding.provider!r}")
