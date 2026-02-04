from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
import time


class TraceSink(Protocol):
    """
    Receives completed traces. You can implement this to:
      - log JSON
      - send to metrics/OTel
      - write to a file
    """
    def emit(self, trace: "ToolScopeTrace") -> None:
        ...


@dataclass
class ToolScore:
    tool_id: str
    tool_name: str
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    passed_filters: bool = True
    filter_reasons: Tuple[str, ...] = ()


@dataclass
class ToolScopeTrace:
    # Identity
    session_id: Optional[str] = None
    turn_id: Optional[str] = None

    # High-level mode
    mode: str = "fresh"  # "fresh" | "reuse" | "refresh"
    query_similarity_to_prev: Optional[float] = None

    # Counts
    input_tool_count: Optional[int] = None
    indexed_tool_count: Optional[int] = None  # how many exist in the backend for this namespace (optional)
    retrieved_candidates: int = 0  # k used in vector search
    candidates_after_filters: int = 0
    candidates_reranked: int = 0
    returned_tools: int = 0

    # Tag filters
    allow_tags: Tuple[str, ...] = ()
    deny_tags: Tuple[str, ...] = ()

    # Token-ish hints (very rough, but useful)
    approx_chars_before: Optional[int] = None
    approx_chars_after: Optional[int] = None

    # Timing (ms)
    ms_embed_query: float = 0.0
    ms_search: float = 0.0
    ms_fetch: float = 0.0
    ms_filter: float = 0.0
    ms_rerank: float = 0.0
    ms_total: float = 0.0

    # Why selected (top candidates + returned)
    candidates: List[ToolScore] = field(default_factory=list)

    # Free-form metadata for callers
    extra: Dict[str, Any] = field(default_factory=dict)


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
