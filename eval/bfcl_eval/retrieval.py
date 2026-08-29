"""
Tool retrieval baselines for comparison against ToolScope.

All retrievers implement .filter(messages, k) -> List[Dict], returning
OpenAI-format tool dicts ranked by relevance to the query.

Usage in run_eval.py:
    retrievers = {
        "Random":    RandomRetriever(tools, seed=seed),
        "BM25":      BM25Retriever(tools),
        "TF-IDF":    TFIDFRetriever(tools),
        "Oracle*":   OracleRetriever(tools, seed=seed),   # cheats – uses GT
        "ToolScope": ts_index,                            # ToolScope ToolIndex
    }
"""

import random
import re
from typing import Any, Dict, List, Optional


# ── Text helpers ────────────────────────────────────────────────────────────


def _tool_text(tool: Dict) -> str:
    """Concatenate all human-readable text from an OpenAI-format tool dict."""
    fn = tool.get("function", {})
    parts = [
        fn.get("name", "").replace("_", " "),
        fn.get("description", ""),
    ]
    for pname, pdef in fn.get("parameters", {}).get("properties", {}).items():
        parts.append(pname.replace("_", " "))
        desc = pdef.get("description", "")
        if desc:
            parts.append(desc)
    return " ".join(p for p in parts if p)


def _query_text(messages: List[Dict]) -> str:
    return " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"\W+", text.lower()) if tok]


# ── Retrievers ──────────────────────────────────────────────────────────────


class RandomRetriever:
    """Uniformly random tool selection. Retrieval lower bound.

    Each query gets a reproducible (but query-dependent) random sample so
    repeated runs give the same numbers without identical results per query.
    """

    def __init__(self, tools: List[Dict], seed: int = 42):
        self._tools = list(tools)
        self._seed = seed

    def filter(self, messages: List[Dict], k: int) -> List[Dict]:
        query = _query_text(messages)
        rng = random.Random(self._seed ^ (hash(query) & 0xFFFFFFFF))
        pool = list(self._tools)
        rng.shuffle(pool)
        return pool[:k]


class BM25Retriever:
    """BM25 sparse retrieval (Okapi BM25 via rank-bm25).

    Strong keyword-based baseline; length-normalised term frequency.
    Indexes: tool name + description + parameter names/descriptions.
    Requires: pip install rank-bm25
    """

    def __init__(self, tools: List[Dict]):
        from rank_bm25 import BM25Okapi  # lazy import so missing dep → clear error

        self._tools = list(tools)
        corpus = [_tokenize(_tool_text(t)) for t in self._tools]
        self._bm25 = BM25Okapi(corpus)

    def filter(self, messages: List[Dict], k: int) -> List[Dict]:
        query_tokens = _tokenize(_query_text(messages))
        scores = self._bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._tools[i] for i in top_idx]


class TFIDFRetriever:
    """TF-IDF cosine similarity retrieval (scikit-learn).

    Simpler than BM25: no length normalization, pure term weighting.
    Useful to isolate how much the IDF discount in BM25 contributes.
    Requires: pip install scikit-learn
    """

    def __init__(self, tools: List[Dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._tools = list(tools)
        self._cosine_similarity = cosine_similarity

        docs = [_tool_text(t) for t in self._tools]
        self._vec = TfidfVectorizer(
            token_pattern=r"[a-z0-9]+",   # consistent with _tokenize()
            sublinear_tf=True,
        )
        self._matrix = self._vec.fit_transform(docs)  # (n_tools, vocab)

    def filter(self, messages: List[Dict], k: int) -> List[Dict]:
        import numpy as np

        query = _query_text(messages)
        q_vec = self._vec.transform([query])
        sims = self._cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [self._tools[i] for i in top_idx]


class ToolScopeRetriever:
    """
    Adapts ``toolscope.adapters.langchain.ToolSelector`` to ``.filter(messages, k)``.

    Operates on the same LangChain tool list the agent will ``bind_tools`` on.
    """

    def __init__(self, selector: Any, tools: List[Any]):
        self._selector = selector
        self._tools = list(tools)

    def filter(self, messages: List[Dict], k: int) -> List[Any]:
        return self._selector.select(messages, self._tools, k=k)


class OracleRetriever:
    """Always returns the ground-truth tools. Retrieval upper bound.

    NOT a realistic retriever — uses ground truth at query time.
    Tells you: "if retrieval were perfect, what accuracy would the model achieve?"
    This isolates model errors from retrieval errors.

    Stateful: call set_ground_truth(gt_names) before each filter() call.
    evaluate_instance() handles this automatically via duck-typing.
    """

    def __init__(self, tools: List[Dict], seed: int = 42):
        self._tools = list(tools)
        self._seed = seed
        self._gt_names: List[str] = []

    def set_ground_truth(self, gt_names: List[str]) -> None:
        self._gt_names = list(gt_names)

    def filter(self, messages: List[Dict], k: int) -> List[Dict]:
        from .tools import tool_name

        gt_set = set(self._gt_names)
        gt_tools = [t for t in self._tools if tool_name(t) in gt_set]
        others = [t for t in self._tools if tool_name(t) not in gt_set]
        query = _query_text(messages)
        rng = random.Random(self._seed ^ (hash(query) & 0xFFFFFFFF))
        rng.shuffle(others)
        combined = gt_tools + others[: max(0, k - len(gt_tools))]
        return combined[:k]
