#!/usr/bin/env python3
"""Re-run BM25 / ToolScope retriever conditions only; baseline is preserved."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAPER_CONFIG = REPO / "eval/paper/bfcl_multiple.yaml"
RESULTS_DIR = REPO / "eval/results/paper/local"

RETRIEVER_MAX_ATTEMPTS = 2


def _best_json(model_id: str) -> Path | None:
    slug = model_id.split("/")[-1]
    candidates = sorted(
        RESULTS_DIR.glob(f"bfcl_eval_{slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best: Path | None = None
    best_n = -1
    for path in candidates:
        if "inprogress" in path.name or "retriever_inprogress" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload.get("metrics", {}).get("n", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if n > best_n:
            best_n = n
            best = path
    return best


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _staging_path(model_id: str) -> Path:
    return RESULTS_DIR / f".retriever_rerun_{_model_slug(model_id)}.json"


def _inprogress_path(model_id: str) -> Path:
    return RESULTS_DIR / f"bfcl_eval_{_model_slug(model_id)}_retriever_inprogress.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _retriever_row(rr) -> dict:
    def _pred(pred):
        return asdict(pred) if pred is not None else None

    return {
        "name_acc": rr.name_acc,
        "exact_match": rr.exact_match,
        "ast_acc": rr.ast_acc,
        "recall": rr.recall,
        "dcg": rr.dcg,
        "ndcg": rr.ndcg,
        "gt_rank": rr.gt_rank,
        "tokens": rr.tokens,
        "compression_rate": rr.compression_rate,
        "tool_names": rr.tool_names,
        "predicted": _pred(rr.predicted),
        "error": rr.error,
        "latency_ms": rr.latency_ms,
        "raw": rr.raw,
        "prompt_tokens": rr.prompt_tokens,
    }


def _instance_result_from_saved_row(saved: dict):
    from eval.bfcl_eval.checkpoint import _instance_result_from_dict

    b = saved.get("baseline") or {}
    flat = {
        "id": saved["id"],
        "query": saved.get("query", ""),
        "ground_truth_names": saved.get("ground_truth_names", []),
        "baseline_name_acc": b.get("name_acc", False),
        "baseline_exact_match": b.get("exact_match", False),
        "baseline_ast_acc": b.get("ast_acc", b.get("exact_match", False)),
        "baseline_tokens": b.get("tokens", 0),
        "baseline_raw": b.get("raw", saved.get("baseline_raw", "")),
        "baseline_pred": b.get("predicted", saved.get("baseline_pred")),
        "baseline_error": b.get("error", saved.get("baseline_error")),
        "baseline_latency_ms": b.get("latency_ms", saved.get("baseline_latency_ms", 0.0)),
        "baseline_prompt_tokens": b.get(
            "prompt_tokens", saved.get("baseline_prompt_tokens")
        ),
        "retrievers": saved.get("retrievers", {}),
    }
    return _instance_result_from_dict(flat)


def _instance_row_from_result(result) -> dict:
    b = {
        "name_acc": result.baseline_name_acc,
        "exact_match": result.baseline_exact_match,
        "ast_acc": result.baseline_ast_acc,
        "tokens": result.baseline_tokens,
        "predicted": asdict(result.baseline_pred) if result.baseline_pred else None,
        "error": result.baseline_error,
        "latency_ms": result.baseline_latency_ms,
        "raw": result.baseline_raw,
        "prompt_tokens": result.baseline_prompt_tokens,
    }
    return {
        "id": result.id,
        "query": result.query,
        "ground_truth_names": result.ground_truth_names,
        "baseline": b,
        "retrievers": {rname: _retriever_row(rr) for rname, rr in result.retrievers.items()},
    }


def _apply_staging(saved_instances: list[dict], staging: dict) -> None:
    by_id = staging.get("instances_by_id") or {}
    for inst in saved_instances:
        staged = by_id.get(inst["id"])
        if staged:
            inst.clear()
            inst.update(staged)


def _has_retriever_api_fail(saved: dict) -> bool:
    for rr in (saved.get("retrievers") or {}).values():
        if rr.get("error") == "api_fail":
            return True
    return False


def _should_rerun_instance(
    saved: dict,
    *,
    only_failed: bool,
    instance_ids: set[str] | None,
) -> bool:
    if instance_ids is not None and saved["id"] not in instance_ids:
        return False
    if only_failed:
        return _has_retriever_api_fail(saved)
    return True


def _build_retrievers(tool_list: list, cfg: dict, seed: int) -> dict:
    import toolscope
    from eval.bfcl_eval.retrieval import BM25Retriever, ToolScopeRetriever

    ts_cfg = cfg.get("toolscope") or {}
    embed = ts_cfg.get("embedding") or {}
    embedding_config = toolscope.EmbeddingConfig(
        provider=embed.get("provider", "sentence-transformers"),
        model=embed.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        allow_download=bool(embed.get("allow_download", True)),
        normalize=True,
    )
    from toolscope.adapters.langchain import ToolSelector
    from eval.bfcl_eval.tools import catalog_to_langchain

    lc_catalog = catalog_to_langchain(tool_list)
    ts_selector = ToolSelector(embedding=embedding_config)
    ts_retriever = ToolScopeRetriever(ts_selector, lc_catalog)

    wanted = set(cfg.get("retrievers") or ["BM25", "ToolScope"])
    all_r = {
        "BM25": BM25Retriever(tool_list),
        "ToolScope": ts_retriever,
    }
    return {n: r for n, r in all_r.items() if n in wanted}


def _predict_with_retry(model, messages, tools, *, label: str):
    from eval.bfcl_eval.evaluate import _normalize_predict

    last = _normalize_predict(model, messages, tools)
    for attempt in range(2, RETRIEVER_MAX_ATTEMPTS + 1):
        if last.error != "api_fail":
            break
        print(
            f"  retry: {label} api_fail "
            f"(attempt {attempt - 1}/{RETRIEVER_MAX_ATTEMPTS}, "
            f"latency_ms={last.latency_ms:.0f})",
            file=sys.stderr,
        )
        last = _normalize_predict(model, messages, tools)
    return last


def _run_retrievers_only(
    model,
    inst,
    tool_pool: list,
    retrievers: dict,
    k_values: list[int],
    baseline_tokens: int,
):
    from eval.bfcl_eval.evaluate import (
        _approx_tokens,
        _gt_names,
        classify_error,
        compute_ast_valid,
        compute_dcg,
        compute_exact_match,
        compute_name_acc,
        compute_recall,
        RetrieverResult,
    )
    from eval.bfcl_eval.tools import tool_name

    gt_names = _gt_names(inst.ground_truth)
    k_max = max(k_values)
    out: dict = {}

    for rname, retriever in retrievers.items():
        ranked = retriever.filter(inst.messages, k=k_max)
        for kv in k_values:
            key = f"{rname}@{kv}"
            r_tools = ranked[:kv]
            r_names = [tool_name(t) for t in r_tools]
            if not r_tools:
                pred_res = None
            else:
                pred_res = _predict_with_retry(
                    model, inst.messages, r_tools, label=f"{inst.id}/{key}"
                )
            r_pred = pred_res.predicted if pred_res else None
            r_raw = pred_res.raw if pred_res else ""
            name_acc = compute_name_acc(r_pred, gt_names)
            exact = compute_exact_match(r_pred, inst.ground_truth, gt_names)
            ast_ok = (
                compute_ast_valid(
                    r_pred, inst.possible_answer, func_descriptions=inst.functions_bfcl
                )
                if inst.possible_answer is not None
                else exact
            )
            error = (
                pred_res.error
                if pred_res and pred_res.error
                else classify_error(
                    pred=r_pred,
                    raw=r_raw,
                    name_acc=name_acc,
                    ast_acc=ast_ok,
                    gt_names=gt_names,
                    retrieved_names=r_names,
                )
            )
            r_tokens = _approx_tokens(r_tools) if r_tools else 0
            r_dcg, r_ndcg, r_gt_rank = compute_dcg(gt_names, r_names)
            compression = (
                1.0 - (r_tokens / baseline_tokens) if baseline_tokens > 0 else 0.0
            )
            out[key] = RetrieverResult(
                name_acc=name_acc,
                exact_match=exact,
                ast_acc=ast_ok,
                recall=compute_recall(gt_names, r_names),
                dcg=r_dcg,
                ndcg=r_ndcg,
                gt_rank=r_gt_rank,
                tool_names=r_names,
                tokens=r_tokens,
                compression_rate=compression,
                raw=r_raw,
                predicted=r_pred,
                error=error,
                latency_ms=pred_res.latency_ms if pred_res else 0.0,
                prompt_tokens=pred_res.prompt_tokens if pred_res else None,
            )
    return out


def probe_retriever(model, messages, retriever, k: int = 10) -> None:
    tools = retriever.filter(messages, k=k)
    if not tools:
        raise RuntimeError("retriever probe returned no tools")
    res = _predict_with_retry(model, messages, tools, label="probe")
    if res.error == "api_fail" or (not res.raw and res.predicted is None):
        raise RuntimeError(f"retriever probe failed: error={res.error!r}")
    print(f"  probe: OK — {len(tools)} tools, latency_ms={res.latency_ms:.0f}")


def _persist_inprogress(
    model_id: str,
    source_json: str,
    payload: dict,
    saved_instances: list[dict],
) -> Path:
    from eval.bfcl_eval.evaluate import aggregate

    merged = [_instance_result_from_saved_row(s) for s in saved_instances]
    metrics = aggregate(merged, n_skipped=0)
    out_cfg = dict(payload.get("config") or {})
    out_cfg["retriever_rerun"] = {
        "source_json": source_json,
        "pending_api_fail": sum(1 for s in saved_instances if _has_retriever_api_fail(s)),
        "n_instances": len(saved_instances),
    }
    path = _inprogress_path(model_id)
    _atomic_write_json(
        path,
        {
            "config": out_cfg,
            "metrics": dataclasses.asdict(metrics),
            "instances": saved_instances,
        },
    )
    return path


def _update_checkpoint_retrievers(model_id: str, results, *, quiet: bool = False) -> None:
    ckpt_dir = RESULTS_DIR / "checkpoints"
    slug = model_id.split("/")[-1]
    matches = sorted(ckpt_dir.glob(f"{slug}_*.jsonl"))
    if not matches:
        print("warning: no checkpoint jsonl to update", file=sys.stderr)
        return
    ckpt_path = matches[-1]
    by_id = {r.id: r for r in results}

    def _rr_dict(rr):
        return {
            "name_acc": rr.name_acc,
            "exact_match": rr.exact_match,
            "ast_acc": rr.ast_acc,
            "recall": rr.recall,
            "dcg": rr.dcg,
            "ndcg": rr.ndcg,
            "gt_rank": rr.gt_rank,
            "tool_names": rr.tool_names,
            "tokens": rr.tokens,
            "compression_rate": rr.compression_rate,
            "predicted": asdict(rr.predicted) if rr.predicted else None,
            "error": rr.error,
            "latency_ms": rr.latency_ms,
            "prompt_tokens": rr.prompt_tokens,
            "raw": rr.raw,
        }

    lines = []
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row.get("id")
        if rid in by_id:
            r = by_id[rid]
            row["retrievers"] = {k: _rr_dict(v) for k, v in r.retrievers.items()}
        lines.append(json.dumps(row, ensure_ascii=False))
    ckpt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not quiet:
        print(f"checkpoint updated → {ckpt_path.relative_to(REPO)}")


def rerun_retrievers(
    model_id: str,
    *,
    source: Path | None,
    probe_only: bool,
    samples: int | None,
    only_failed: bool = True,
    resume: bool = True,
    instance_ids: set[str] | None = None,
) -> int:
    from eval.bfcl_eval.agent import LangGraphAgent
    from eval.bfcl_eval.dataset import collect_catalog, load_entries
    from eval.bfcl_eval.evaluate import _approx_tokens, _gt_names, aggregate
    from eval.bfcl_eval.report import save_results

    cfg = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8")) or {}
    entry = next(
        (e for e in cfg.get("model", {}).get("entries", []) if e["name"] == model_id),
        None,
    )
    if entry is None:
        raise SystemExit(f"unknown model id: {model_id}")

    original_src = source or _best_json(model_id)
    if original_src is None or not original_src.exists():
        raise SystemExit(f"no existing result JSON for {model_id}")

    inprogress = _inprogress_path(model_id)
    src = original_src
    if resume and inprogress.exists():
        src = inprogress
        print(f"  resuming in-progress results → {inprogress.relative_to(REPO)}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    saved_instances = payload.get("instances") or []
    if not saved_instances:
        raise SystemExit(f"no instances in {src}")

    staging_path = _staging_path(model_id)
    if resume and staging_path.exists():
        staging = json.loads(staging_path.read_text(encoding="utf-8"))
        if staging.get("source_json") in {original_src.name, src.name}:
            _apply_staging(saved_instances, staging)
            print(f"  resumed staging → {staging_path.relative_to(REPO)}")

    pending = [
        s
        for s in saved_instances
        if _should_rerun_instance(
            s, only_failed=only_failed, instance_ids=instance_ids
        )
    ]
    print(
        f"  retriever rerun: {len(pending)}/{len(saved_instances)} instances "
        f"({'only-failed' if only_failed else 'all'})"
    )

    ds = cfg.get("dataset") or {}
    seed = int(ds.get("seed", 42))
    entries = load_entries(
        categories=ds.get("categories") or ["multiple"],
        cache_dir=Path(ds.get("cache_dir", "eval/.bfcl_cache")),
        samples=samples,
        seed=seed,
    )
    by_id = {e.id: e for e in entries}
    all_tools, _ = collect_catalog(entries)
    tool_pool = list(all_tools.values())
    retrievers = _build_retrievers(tool_pool, cfg, seed)
    ts_cfg = cfg.get("toolscope") or {}
    k_values = sorted(
        {int(x) for x in (ts_cfg.get("k_values") or [ts_cfg.get("k", 10)])}
    )

    out_cfg_base = dict(payload.get("config") or {})

    if not pending and not probe_only:
        merged = [_instance_result_from_saved_row(s) for s in saved_instances]
        metrics = aggregate(merged, n_skipped=0)
        n_fail = sum(1 for s in saved_instances if _has_retriever_api_fail(s))
        if n_fail:
            print(f"  still {n_fail} instances with retriever api_fail", file=sys.stderr)
            return 1
        out_cfg = dict(out_cfg_base)
        out_cfg.pop("retriever_rerun", None)
        path = save_results(
            results=merged,
            metrics=metrics,
            config=out_cfg,
            output_dir=RESULTS_DIR,
            model_name=model_id,
        )
        _update_checkpoint_retrievers(model_id, merged)
        if staging_path.exists():
            staging_path.unlink()
        if inprogress.exists():
            inprogress.unlink()
        print(f"  promoted → {path.name}")
        return 0

    agent = LangGraphAgent(entry)
    agent.load()

    probe_inst = next(e for e in entries if e.id == pending[0]["id"])
    probe_retriever(agent, probe_inst.messages, retrievers["ToolScope"], k=10)

    if probe_only:
        print("probe-only: success")
        return 0

    staging_rows: dict[str, dict] = {}
    if staging_path.exists():
        staging_rows = json.loads(staging_path.read_text(encoding="utf-8")).get(
            "instances_by_id", {}
        )

    pending_ids = {s["id"] for s in pending}
    for saved in tqdm(saved_instances, desc=f"{model_id} retrievers", unit="inst"):
        inst = by_id.get(saved["id"])
        if inst is None:
            raise SystemExit(f"missing BFCL entry for {saved['id']}")
        if not _gt_names(inst.ground_truth):
            continue
        if saved["id"] not in pending_ids:
            continue

        baseline_tokens = int((saved.get("baseline") or {}).get("tokens") or 0)
        if baseline_tokens <= 0:
            baseline_tokens = _approx_tokens(tool_pool)

        base = _instance_result_from_saved_row(saved)
        new_retrievers = _run_retrievers_only(
            agent,
            inst,
            tool_pool,
            retrievers,
            k_values,
            baseline_tokens,
        )
        merged_rets = dict(base.retrievers)
        merged_rets.update(new_retrievers)

        result = base.__class__(
            id=base.id,
            query=base.query,
            ground_truth_names=base.ground_truth_names,
            baseline_name_acc=base.baseline_name_acc,
            baseline_exact_match=base.baseline_exact_match,
            baseline_ast_acc=base.baseline_ast_acc,
            baseline_tokens=base.baseline_tokens,
            baseline_raw=base.baseline_raw,
            baseline_pred=base.baseline_pred,
            baseline_error=base.baseline_error,
            baseline_latency_ms=base.baseline_latency_ms,
            baseline_prompt_tokens=base.baseline_prompt_tokens,
            retrievers=merged_rets,
        )

        row = _instance_row_from_result(result)
        staging_rows[result.id] = row
        saved.clear()
        saved.update(row)
        _atomic_write_json(
            staging_path,
            {
                "model_id": model_id,
                "source_json": original_src.name,
                "instances_by_id": staging_rows,
            },
        )
        _update_checkpoint_retrievers(model_id, [result], quiet=True)
        snap = _persist_inprogress(
            model_id, original_src.name, payload, saved_instances
        )
        n_clean = sum(1 for s in saved_instances if not _has_retriever_api_fail(s))
        tqdm.write(f"  saved {result.id} → {snap.name} ({n_clean}/{len(saved_instances)} clean)")

    merged = [_instance_result_from_saved_row(s) for s in saved_instances]
    metrics = aggregate(merged, n_skipped=0)
    n_fail = sum(1 for s in saved_instances if _has_retriever_api_fail(s))
    _persist_inprogress(model_id, original_src.name, payload, saved_instances)

    if n_fail:
        print(
            f"retriever rerun incomplete: {n_fail} instances still have api_fail — "
            f"progress in {_inprogress_path(model_id).name}; retry with --only-failed",
            file=sys.stderr,
        )
        return 1

    out_cfg = dict(out_cfg_base)
    out_cfg.pop("retriever_rerun", None)
    path = save_results(
        results=merged,
        metrics=metrics,
        config=out_cfg,
        output_dir=RESULTS_DIR,
        model_name=model_id,
    )
    _update_checkpoint_retrievers(model_id, merged)
    if staging_path.exists():
        staging_path.unlink()
    if inprogress.exists():
        inprogress.unlink()
    print(f"retriever name_acc (ToolScope@10) saved → {path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model id from models.yaml")
    parser.add_argument("--source", type=Path, help="Existing bfcl_eval_*.json to merge from")
    parser.add_argument("--probe-only", action="store_true", help="Run one retriever probe and exit")
    parser.add_argument("--samples", type=int, help="Limit instances (debug)")
    parser.add_argument(
        "--all-instances",
        action="store_true",
        help="Re-run retriever conditions on every instance (default: api_fail only)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore incremental staging checkpoint",
    )
    parser.add_argument(
        "--instance",
        action="append",
        dest="instances",
        metavar="ID",
        help="Re-run specific instance id(s); may be repeated",
    )
    args = parser.parse_args()
    instance_ids = set(args.instances) if args.instances else None
    return rerun_retrievers(
        args.model,
        source=args.source,
        probe_only=args.probe_only,
        samples=args.samples,
        only_failed=not args.all_instances,
        resume=not args.no_resume,
        instance_ids=instance_ids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
