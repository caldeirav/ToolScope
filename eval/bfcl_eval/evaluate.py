"""
Per-instance evaluation and metric aggregation.

InstanceResult stores a baseline result (model sees the full per-instance pool)
plus one RetrieverResult per retriever (model sees only the retriever's top-k).
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model import ParsedToolCall


# ── Per-instance result structures ─────────────────────────────────────────


@dataclass
class RetrieverResult:
    """All metrics for one retriever on one evaluation instance."""
    name_acc: bool
    exact_match: bool
    recall: float           # fraction of GT tools present in top-k
    dcg: float              # discounted cumulative gain at k
    ndcg: float             # normalized DCG (0–1); equals DCG for single-GT entries
    gt_rank: Optional[int]  # 1-indexed rank of first GT tool; None if not retrieved
    tool_names: List[str]   # names of tools the retriever returned
    tokens: int             # approx prompt tokens (json_chars // 4)
    compression_rate: float # 1 - tokens / baseline_tokens
    raw: str                # raw model output for debugging
    predicted: Optional[ParsedToolCall]


@dataclass
class InstanceResult:
    id: str
    query: str                    # first user message, truncated for display
    ground_truth_names: List[str]

    # Baseline: model sees the full per-instance pool (pool_size tools)
    baseline_name_acc: bool
    baseline_exact_match: bool
    baseline_tokens: int
    baseline_raw: str
    baseline_pred: Optional[ParsedToolCall]

    # One entry per configured retriever
    retrievers: Dict[str, RetrieverResult]


# ── Aggregate metric structures ─────────────────────────────────────────────


@dataclass
class RetrieverMetrics:
    """Aggregate metrics for one retriever across all evaluated instances."""
    name_acc: float
    exact_match: float
    recall: float
    dcg: float
    ndcg: float
    mean_tokens: float
    mean_compression_rate: float
    delta_name_acc: float    # retriever - baseline (positive = retriever helps)
    delta_exact_match: float


@dataclass
class AggregateMetrics:
    n: int
    n_skipped: int
    baseline_name_acc: float
    baseline_exact_match: float
    mean_baseline_tokens: float
    retrievers: Dict[str, RetrieverMetrics]


# ── Metric computation ──────────────────────────────────────────────────────


def _gt_names(ground_truth: List[Dict]) -> List[str]:
    names = []
    for call in ground_truth:
        if isinstance(call, dict):
            names.extend(call.keys())
    return names


def _normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
        lower = v.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return v


def _args_match(pred_args: Dict, gt_args: Dict) -> bool:
    for key, expected in gt_args.items():
        if key not in pred_args:
            return False
        if _normalize_value(pred_args[key]) != _normalize_value(expected):
            return False
    return True


def compute_name_acc(pred: Optional[ParsedToolCall], gt_names: List[str]) -> bool:
    return pred is not None and pred.name in gt_names


def compute_exact_match(
    pred: Optional[ParsedToolCall],
    ground_truth: List[Dict],
    gt_names: List[str],
) -> bool:
    if pred is None or pred.name not in gt_names:
        return False
    for call in ground_truth:
        if isinstance(call, dict) and pred.name in call:
            gt_args = call[pred.name]
            if isinstance(gt_args, dict):
                return _args_match(pred.arguments, gt_args)
            return True
    return False


def compute_recall(gt_names: List[str], retrieved_names: List[str]) -> float:
    if not gt_names:
        return 0.0
    ts_set = set(retrieved_names)
    return sum(1 for n in gt_names if n in ts_set) / len(gt_names)


def compute_dcg(
    gt_names: List[str], retrieved_names: List[str]
) -> Tuple[float, float, Optional[int]]:
    """
    Returns (dcg, ndcg, first_rank).

    DCG@k  = sum_i  relevance_i / log2(rank_i + 1)   for each GT tool found
    IDCG@k = ideal DCG (all n_gt GT tools at ranks 1..n_gt)
    NDCG@k = DCG / IDCG  → [0, 1]

    For simple/multiple categories (single GT call), IDCG = 1.0 so NDCG = DCG.
    """
    if not gt_names:
        return 0.0, 0.0, None

    gt_set = set(gt_names)
    first_rank: Optional[int] = None
    dcg = 0.0

    for rank_0, name in enumerate(retrieved_names):
        if name in gt_set:
            rank = rank_0 + 1
            dcg += 1.0 / math.log2(rank + 1)
            if first_rank is None:
                first_rank = rank

    n_gt = min(len(gt_names), len(retrieved_names))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_gt + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return dcg, ndcg, first_rank


def _approx_tokens(tools: List[Dict]) -> int:
    """Approximate token count: JSON character length / 4."""
    return max(1, len(json.dumps(tools, ensure_ascii=False)) // 4)


# ── Core evaluation ─────────────────────────────────────────────────────────


def _run_retriever(
    retriever: Any,
    messages: List[Dict],
    ground_truth: List[Dict],
    gt_names: List[str],
    baseline_tokens: int,
    model: Any,
    k: int,
) -> RetrieverResult:
    """Run one retriever against one instance and compute all metrics."""
    # OracleRetriever (and any future GT-aware retriever) needs the GT up front
    if hasattr(retriever, "set_ground_truth"):
        retriever.set_ground_truth(gt_names)

    r_tools = retriever.filter(messages, k=k)
    r_names = [t["function"]["name"] for t in r_tools]
    r_tokens = _approx_tokens(r_tools) if r_tools else 0
    r_raw = model.predict(messages, r_tools) if r_tools else ""
    r_pred = model.parse_tool_call(r_raw) if r_tools else None
    r_dcg, r_ndcg, r_gt_rank = compute_dcg(gt_names, r_names)
    compression = 1.0 - (r_tokens / baseline_tokens) if baseline_tokens > 0 else 0.0

    return RetrieverResult(
        name_acc=compute_name_acc(r_pred, gt_names),
        exact_match=compute_exact_match(r_pred, ground_truth, gt_names),
        recall=compute_recall(gt_names, r_names),
        dcg=r_dcg,
        ndcg=r_ndcg,
        gt_rank=r_gt_rank,
        tool_names=r_names,
        tokens=r_tokens,
        compression_rate=compression,
        raw=r_raw,
        predicted=r_pred,
    )


def evaluate_instance(
    entry_id: str,
    messages: List[Dict],
    ground_truth: List[Dict],
    tool_pool: List[Dict],
    model: Any,
    retrievers: Dict[str, Any],
    k: int,
) -> Optional[InstanceResult]:
    """
    Run one evaluation instance.

    Baseline: model sees all pool_size tools (per-instance pool).
    Each retriever: model sees only its top-k tools from the global index.
    Returns None if the entry has no parseable ground truth names.
    """
    gt_names = _gt_names(ground_truth)
    if not gt_names:
        return None

    query = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"), ""
    )[:200]

    baseline_tokens = _approx_tokens(tool_pool)
    baseline_raw = model.predict(messages, tool_pool)
    baseline_pred = model.parse_tool_call(baseline_raw)

    retriever_results: Dict[str, RetrieverResult] = {
        rname: _run_retriever(
            retriever, messages, ground_truth, gt_names,
            baseline_tokens, model, k
        )
        for rname, retriever in retrievers.items()
    }

    return InstanceResult(
        id=entry_id,
        query=query,
        ground_truth_names=gt_names,
        baseline_name_acc=compute_name_acc(baseline_pred, gt_names),
        baseline_exact_match=compute_exact_match(baseline_pred, ground_truth, gt_names),
        baseline_tokens=baseline_tokens,
        baseline_raw=baseline_raw,
        baseline_pred=baseline_pred,
        retrievers=retriever_results,
    )


def aggregate(results: List[InstanceResult], n_skipped: int) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        return AggregateMetrics(
            n=0, n_skipped=n_skipped,
            baseline_name_acc=0.0, baseline_exact_match=0.0,
            mean_baseline_tokens=0.0, retrievers={},
        )

    b_name   = sum(r.baseline_name_acc   for r in results) / n
    b_exact  = sum(r.baseline_exact_match for r in results) / n
    b_tokens = sum(r.baseline_tokens      for r in results) / n

    rnames = list(results[0].retrievers.keys())
    retriever_metrics: Dict[str, RetrieverMetrics] = {}

    for rname in rnames:
        rr = [r.retrievers[rname] for r in results if rname in r.retrievers]
        nr = len(rr)
        if nr == 0:
            continue
        r_name_acc = sum(r.name_acc    for r in rr) / nr
        r_exact    = sum(r.exact_match for r in rr) / nr
        retriever_metrics[rname] = RetrieverMetrics(
            name_acc=r_name_acc,
            exact_match=r_exact,
            recall=sum(r.recall            for r in rr) / nr,
            dcg=sum(r.dcg                  for r in rr) / nr,
            ndcg=sum(r.ndcg                for r in rr) / nr,
            mean_tokens=sum(r.tokens       for r in rr) / nr,
            mean_compression_rate=sum(r.compression_rate for r in rr) / nr,
            delta_name_acc=r_name_acc - b_name,
            delta_exact_match=r_exact - b_exact,
        )

    return AggregateMetrics(
        n=n,
        n_skipped=n_skipped,
        baseline_name_acc=b_name,
        baseline_exact_match=b_exact,
        mean_baseline_tokens=b_tokens,
        retrievers=retriever_metrics,
    )
