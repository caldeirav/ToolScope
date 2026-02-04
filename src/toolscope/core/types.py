from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


Json = Dict[str, Any]


@dataclass(frozen=True)
class ToolFingerprint:
    """
    Stable identifier for a specific version of a tool definition.

    Fingerprints are used as primary keys in vector backends.
    They MUST change when the tool meaningfully changes
    (name, description, schema, tags, etc.).
    """
    value: str


@dataclass(frozen=True)
class CanonicalTool:
    """
    Canonical internal representation of a tool.

    payload:
      The original tool specification provided by the user
      (e.g. OpenAI tool dict, MCP tool descriptor).
      Returned verbatim to the model after filtering.
    """
    name: str
    description: str
    input_schema: Json
    fingerprint: ToolFingerprint
    tags: Tuple[str, ...] = ()
    payload: Optional[Any] = None


class EmbeddingProvider(Protocol):
    """
    Produces dense vectors from text.
    """
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class ToolNormalizer(Protocol):
    """
    Converts tool specs to/from CanonicalTool.
    """
    def normalize(self, tools: Sequence[Any]) -> List[CanonicalTool]:
        ...

    def denormalize(self, canonical_tools: Sequence[CanonicalTool]) -> List[Any]:
        ...


class ToolIndexBackend(Protocol):
    """
    Vector index backend interface.

    This interface mirrors common vector DB APIs (Milvus, Qdrant, Pinecone):

      - ensure_namespace: create collection/index if needed
      - upsert: insert or update vectors by id
      - delete: remove vectors by id
      - search: similarity search
      - get: fetch stored records by id

    All backends MUST treat ToolFingerprint.value as the primary key.
    """

    def ensure_namespace(self, *, namespace: Optional[str], dim: int) -> None:
        """
        Ensure the collection/namespace exists with the given vector dimension.
        May be a no-op for simple backends.
        """

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[List[float]],
        payloads: Sequence[CanonicalTool],
        *,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Insert or update records.

        ids       : primary keys (fingerprint strings)
        vectors   : embedding vectors
        payloads  : CanonicalTool metadata
        """

    def delete(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> None:
        """
        Remove records by id (hard or soft delete).
        """

    def search(
        self,
        query_vector: List[float],
        *,
        k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k (id, score) pairs.
        Score must be higher-is-better.
        """

    def get(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> List[CanonicalTool]:
        """
        Fetch CanonicalTool records by id.
        """
