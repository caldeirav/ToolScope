from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import math


def cosine(a: List[float], b: List[float]) -> float:
    # assume non-empty
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class StickySessionConfig:
    """
    Controls session-aware caching and toolset stickiness.

    similarity_threshold_reuse:
      If cosine(query, last_query) >= this, reuse last selected tool ids.

    similarity_threshold_refresh:
      If cosine(query, last_query) >= this but below reuse threshold, refresh via vector search
      but bias toward previous tools.

    sticky_keep:
      When refreshing, ensure up to this many previously-selected tools are kept if available.

    sticky_boost:
      Score bonus applied to tools that were selected last time when combining results.
      (Used only in refresh mode.)

    ttl_seconds:
      Expire session state after inactivity.

    max_sessions:
      Bound memory usage.
    """
    enabled: bool = True
    similarity_threshold_reuse: float = 0.92
    similarity_threshold_refresh: float = 0.80
    sticky_keep: int = 4
    sticky_boost: float = 0.03
    ttl_seconds: float = 3600.0
    max_sessions: int = 1000


@dataclass
class SessionState:
    last_seen_ts: float
    last_query_vec: List[float]
    last_selected_ids: List[str]  # fingerprint strings


class SessionCache:
    def __init__(self, cfg: StickySessionConfig) -> None:
        self.cfg = cfg
        self._states: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        st = self._states.get(session_id)
        if st is None:
            return None
        if (time.time() - st.last_seen_ts) > self.cfg.ttl_seconds:
            self._states.pop(session_id, None)
            return None
        return st

    def put(self, session_id: str, query_vec: List[float], selected_ids: List[str]) -> None:
        # Simple max_sessions eviction: drop oldest by last_seen_ts when over limit.
        now = time.time()
        self._states[session_id] = SessionState(
            last_seen_ts=now,
            last_query_vec=query_vec,
            last_selected_ids=list(selected_ids),
        )
        if len(self._states) > self.cfg.max_sessions:
            # evict oldest ~10% to amortize cost
            victims = sorted(self._states.items(), key=lambda kv: kv[1].last_seen_ts)
            n = max(1, len(self._states) // 10)
            for sid, _ in victims[:n]:
                self._states.pop(sid, None)
