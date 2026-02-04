from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .reranking_config import RerankingConfig


class RerankingNotConfiguredError(RuntimeError):
    pass


class Reranker(Protocol):
    """
    Scores (query, doc) pairs. Higher score = better.
    """
    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        ...


def _truncate(s: str, n: int) -> str:
    if n <= 0:
        return ""
    return s[:n]


@dataclass
class SentenceTransformersCrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.

    Requires optional dependency:
      pip install toolscope[st]
    """
    model_name: str
    allow_download: bool = False

    _model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RerankingNotConfiguredError(
                "sentence-transformers is not installed. Install with: pip install toolscope[st]"
            ) from e

        kwargs = {}
        if not self.allow_download:
            kwargs["local_files_only"] = True

        try:
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as e:
            if not self.allow_download:
                raise RerankingNotConfiguredError(
                    f"Reranker model '{self.model_name}' not found locally. "
                    f"Either pre-download it or set allow_download=True."
                ) from e
            raise
        return self._model

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        model = self._load()
        from sentence_transformers import CrossEncoder  # type: ignore
        assert isinstance(model, CrossEncoder)

        pairs = [(query, d) for d in docs]
        scores = model.predict(pairs)
        # predict may return numpy array
        return [float(x) for x in scores]


def make_reranker(*, reranking: Optional[RerankingConfig]) -> Optional[Reranker]:
    """
    Resolve a reranker from config. If reranking is None, returns None.
    """
    if reranking is None:
        return None

    prov = reranking.provider.strip().lower()
    if prov in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformersCrossEncoderReranker(
            model_name=reranking.model,
            allow_download=reranking.allow_download,
        )

    raise RerankingNotConfiguredError(f"Unknown reranking provider: {reranking.provider!r}")
