from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

from ..types import CanonicalTool, ToolIndexBackend


class MemoryBackend(ToolIndexBackend):
    """
    Simple in-memory vector backend.

    - Zero dependencies
    - Namespaces supported
    - Naive cosine similarity search
    - Intended for defaults, tests, and small catalogs
    """

    def __init__(self) -> None:
        self._vectors: Dict[str, Dict[str, List[float]]] = {}
        self._payloads: Dict[str, Dict[str, CanonicalTool]] = {}
        self._dims: Dict[str, int] = {}

    def ensure_namespace(self, *, namespace: Optional[str], dim: int) -> None:
        ns = namespace or "__default__"
        if ns not in self._vectors:
            self._vectors[ns] = {}
            self._payloads[ns] = {}
            self._dims[ns] = dim
        elif self._dims[ns] != dim:
            raise ValueError(
                f"Namespace '{ns}' already exists with dim {self._dims[ns]}, got {dim}"
            )

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[List[float]],
        payloads: Sequence[CanonicalTool],
        *,
        namespace: Optional[str] = None,
    ) -> None:
        ns = namespace or "__default__"
        if not vectors:
            return

        self.ensure_namespace(namespace=ns, dim=len(vectors[0]))

        for _id, vec, payload in zip(ids, vectors, payloads, strict=True):
            if len(vec) != self._dims[ns]:
                raise ValueError("Vector dimension mismatch")
            self._vectors[ns][_id] = vec
            self._payloads[ns][_id] = payload

    def delete(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> None:
        ns = namespace or "__default__"
        if ns not in self._vectors:
            return
        for _id in ids:
            self._vectors[ns].pop(_id, None)
            self._payloads[ns].pop(_id, None)

    def search(
        self,
        query_vector: List[float],
        *,
        k: int,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        ns = namespace or "__default__"
        if ns not in self._vectors:
            return []

        allow = None
        deny = None
        if filter:
            at = filter.get("allow_tags")
            dt = filter.get("deny_tags")
            if isinstance(at, list):
                allow = set(at)
            if isinstance(dt, list):
                deny = set(dt)

        def ok(payload: CanonicalTool) -> bool:
            tset = set(payload.tags or ())
            if deny is not None and (tset & deny):
                return False
            if allow is not None and not (tset & allow):
                return False
            return True

        qn = _norm(query_vector)
        scored: List[Tuple[str, float]] = []

        for _id, vec in self._vectors[ns].items():
            payload = self._payloads[ns][_id]
            if filter and not ok(payload):
                continue

            score = _cosine(query_vector, qn, vec)
            scored.append((_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def get(self, ids: Sequence[str], *, namespace: Optional[str] = None) -> List[CanonicalTool]:
        ns = namespace or "__default__"
        if ns not in self._payloads:
            return []
        return [self._payloads[ns][_id] for _id in ids if _id in self._payloads[ns]]


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _cosine(q: List[float], qn: float, v: List[float]) -> float:
    return sum(a * b for a, b in zip(q, v)) / (qn * _norm(v))
