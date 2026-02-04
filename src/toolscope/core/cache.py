from dataclasses import dataclass
from typing import Dict, List, Optional
from .types import ToolFingerprint


@dataclass
class EmbeddingCache:
    """
    Cache embeddings by tool fingerprint. Keeps stateless API fast.
    """
    vectors: Dict[str, List[float]]

    def get(self, fp: ToolFingerprint) -> Optional[List[float]]:
        return self.vectors.get(fp.value)

    def put(self, fp: ToolFingerprint, vec: List[float]) -> None:
        self.vectors[fp.value] = vec
