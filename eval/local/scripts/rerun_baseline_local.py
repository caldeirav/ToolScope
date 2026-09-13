#!/usr/bin/env python3
"""Re-run baseline (full-catalog) only for a completed local BFCL model run."""

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
        if "inprogress" in path.name:
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


def _retriever_from_saved(d: dict):
    from eval.bfcl_eval.checkpoint import _retriever_result_from_dict

    return _retriever_result_from_dict(d)


def _instance_from_saved(d: dict):
    from eval.bfcl_eval.checkpoint import _parsed_tool_call_from_dict
    from eval.bfcl_eval.evaluate import InstanceResult

    return InstanceResult(
        id=d["id"],
        query=d.get("query", ""),
        ground_truth_names=d.get("ground_truth_names", []),
        baseline_name_acc=False,
        baseline_exact_match=False,
        baseline_ast_acc=False,
        baseline_tokens=0,
        baseline_raw="",
        baseline_pred=None,
        baseline_error=None,
        baseline_latency_ms=0.0,
        baseline_prompt_tokens=None,
        retrievers={
            name: _retriever_from_saved(rr)
            for name, rr in (d.get("retrievers") or {}).items()
        },
    )


BASELINE_MAX_ATTEMPTS = 2  # per-run retries on api_fail
BASELINE_MAX_INSTANCE_ATTEMPTS = 3  # cumulative api_fail attempts, then abandon instance


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _staging_path(model_id: str) -> Path:
    return RESULTS_DIR / f".baseline_rerun_{_model_slug(model_id)}.json"


def _inprogress_path(model_id: str) -> Path:
    return RESULTS_DIR / f"bfcl_eval_{_model_slug(model_id)}_baseline_inprogress.json"


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


def _saved_baseline_ok(saved: dict) -> bool:
    b = saved.get("baseline") or {}
    if b.get("error") == "api_fail":
        return False
    pred = b.get("predicted") or saved.get("baseline_pred")
    raw = (b.get("raw") or saved.get("baseline_raw") or "").strip()
    if pred or raw:
        return True
    lat = float(b.get("latency_ms") or saved.get("baseline_latency_ms") or 0)
    return lat >= 5000.0 and b.get("error") is None


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
    def _pred(pred):
        return asdict(pred) if pred is not None else None

    return {
        "id": result.id,
        "query": result.query,
        "ground_truth_names": result.ground_truth_names,
        "baseline": {
            "name_acc": result.baseline_name_acc,
            "exact_match": result.baseline_exact_match,
            "ast_acc": result.baseline_ast_acc,
            "tokens": result.baseline_tokens,
            "predicted": _pred(result.baseline_pred),
            "error": result.baseline_error,
            "latency_ms": result.baseline_latency_ms,
            "raw": result.baseline_raw,
            "prompt_tokens": result.baseline_prompt_tokens,
        },
        "retrievers": {
            rname: {
                "name_acc": rr.name_acc,
                "exact_match": rr.exact_match,
                "ast_acc": getattr(rr, "ast_acc", rr.exact_match),
                "recall": rr.recall,
                "dcg": rr.dcg,
                "ndcg": rr.ndcg,
                "gt_rank": rr.gt_rank,
                "tokens": rr.tokens,
                "compression_rate": rr.compression_rate,
                "tool_names": rr.tool_names,
                "predicted": _pred(rr.predicted),
                "error": getattr(rr, "error", None),
                "latency_ms": getattr(rr, "latency_ms", 0.0),
            }
            for rname, rr in result.retrievers.items()
        },
    }


def _apply_staging(saved_instances: list[dict], staging: dict) -> None:
    by_id = staging.get("instances_by_id") or {}
    for inst in saved_instances:
        staged = by_id.get(inst["id"])
        if staged:
            inst.clear()
            inst.update(staged)


def _write_staging(
    model_id: str,
    source_json: str,
    rows_by_id: dict[str, dict],
    *,
    fail_attempts: dict[str, int],
) -> None:
    _atomic_write_json(
        _staging_path(model_id),
        {
            "model_id": model_id,
            "source_json": source_json,
            "instances_by_id": rows_by_id,
            "fail_attempts": fail_attempts,
        },
    )


def _load_fail_attempts(staging: dict | None, payload: dict) -> dict[str, int]:
    attempts: dict[str, int] = {}
    if staging:
        raw = staging.get("fail_attempts") or {}
        attempts.update({str(k): int(v) for k, v in raw.items()})
    rerun = (payload.get("config") or {}).get("baseline_rerun") or {}
    raw = rerun.get("fail_attempts") or {}
    for k, v in raw.items():
        attempts[str(k)] = max(attempts.get(str(k), 0), int(v))
    return attempts


def _instance_settled(saved: dict, fail_attempts: dict[str, int]) -> bool:
    if _saved_baseline_ok(saved):
        return True
    b = saved.get("baseline") or {}
    if b.get("error") != "api_fail":
        # Model ran (parse_fail, wrong_tool, etc.) — do not block completion.
        return True
    return fail_attempts.get(saved["id"], 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS


def _abandoned_ids(saved_instances: list[dict], fail_attempts: dict[str, int]) -> list[str]:
    return [
        s["id"]
        for s in saved_instances
        if not _saved_baseline_ok(s)
        and (s.get("baseline") or {}).get("error") == "api_fail"
        and fail_attempts.get(s["id"], 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS
    ]


def _persist_snapshot(
    model_id: str,
    original_source: str,
    payload: dict,
    saved_instances: list[dict],
    *,
    api_fail_ids: list[str],
    fail_attempts: dict[str, int],
) -> Path:
    from eval.bfcl_eval.evaluate import aggregate

    merged = [_instance_result_from_saved_row(s) for s in saved_instances]
    metrics = aggregate(merged, n_skipped=0)
    out_cfg = dict(payload.get("config") or {})
    out_cfg["baseline_rerun"] = {
        "source_json": original_source,
        "context_size_note": "models.yaml context_size>=65536 for large models",
        "in_progress": not all(
            _instance_settled(s, fail_attempts) for s in saved_instances
        ),
        "api_fail_ids": api_fail_ids,
        "fail_attempts": fail_attempts,
        "abandoned_ids": _abandoned_ids(saved_instances, fail_attempts),
        "completed_baselines": sum(1 for s in saved_instances if _saved_baseline_ok(s)),
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


def _clear_staging(model_id: str) -> None:
    path = _staging_path(model_id)
    if path.exists():
        path.unlink()


def _should_rerun_instance(
    saved: dict,
    *,
    only_failed: bool,
    instance_ids: set[str] | None,
    fail_attempts: dict[str, int],
) -> bool:
    if instance_ids is not None and saved["id"] not in instance_ids:
        return False
    iid = saved["id"]
    b = saved.get("baseline") or {}
    if only_failed:
        if _saved_baseline_ok(saved):
            return False
        if b.get("error") != "api_fail":
            return False
        if fail_attempts.get(iid, 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS:
            return False
        return True
    if fail_attempts.get(iid, 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS:
        return False
    return True


def _response_ok(result) -> bool:
    if getattr(result, "error", None) == "api_fail":
        return False
    raw = (getattr(result, "raw", None) or "").strip()
    pred = getattr(result, "predicted", None)
    if pred is not None:
        return True
    if raw:
        return True
    # Allow slow empty responses only when the server clearly ran inference.
    return getattr(result, "latency_ms", 0.0) >= 5000


def probe_baseline(model, messages, tool_pool, *, min_latency_ms: float = 1000.0) -> None:
    from eval.bfcl_eval.evaluate import _approx_tokens, _normalize_predict

    approx = _approx_tokens(tool_pool)
    print(f"  probe: catalog tools={len(tool_pool)} approx_tokens={approx}")
    from eval.bfcl_eval.agent import _as_lc_tools, _to_lc_messages, make_llm
    from eval.bfcl_eval.tools import tool_name as _tn

    if hasattr(model, "entry"):
        llm = make_llm(model.entry)
        lc_tools = _as_lc_tools(tool_pool)
        lc_msgs = _to_lc_messages(messages)
        t0 = __import__("time").monotonic()
        try:
            ai = llm.bind_tools(lc_tools).invoke(lc_msgs)
            latency = (__import__("time").monotonic() - t0) * 1000
            raw = getattr(ai, "content", "") or ""
            tcs = getattr(ai, "tool_calls", None) or []
            pred_name = None
            if tcs:
                tc = tcs[0]
                pred_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            print(
                f"  probe: latency_ms={latency:.0f} pred={pred_name!r} "
                f"raw_len={len(raw)} tool_calls={len(tcs)}"
            )
            if latency < min_latency_ms:
                raise RuntimeError("baseline probe returned too quickly")
            if not tcs and not str(raw).strip():
                raise RuntimeError("baseline probe: empty model response")
            print("  probe: OK — model returned a response for the full catalog")
            return
        except Exception as exc:
            print(f"  probe: exception {type(exc).__name__}: {exc}")
            raise RuntimeError(f"baseline probe failed: {exc}") from exc

    res = _normalize_predict(model, messages, tool_pool)
    print(
        f"  probe: latency_ms={res.latency_ms:.0f} error={res.error!r} "
        f"pred={getattr(res.predicted, 'name', None)} raw_len={len(res.raw or '')}"
    )
    if res.latency_ms < min_latency_ms and res.error == "api_fail":
        raise RuntimeError(
            "baseline probe failed instantly (likely exceed_context_size / api_fail)"
        )
    if not _response_ok(res):
        raise RuntimeError(
            "baseline probe did not get a usable model response "
            f"(error={res.error!r}, latency_ms={res.latency_ms:.0f})"
        )
    print("  probe: OK — model returned a response for the full catalog")


def _predict_baseline_with_retry(model, messages, tool_pool, *, instance_id: str):
    from eval.bfcl_eval.evaluate import _normalize_predict

    last = _normalize_predict(model, messages, tool_pool)
    for attempt in range(2, BASELINE_MAX_ATTEMPTS + 1):
        if last.error != "api_fail":
            break
        print(
            f"  retry: {instance_id} api_fail "
            f"(attempt {attempt - 1}/{BASELINE_MAX_ATTEMPTS}, "
            f"latency_ms={last.latency_ms:.0f})",
            file=sys.stderr,
        )
        last = _normalize_predict(model, messages, tool_pool)
    return last


def rerun_baseline(
    model_id: str,
    *,
    source: Path | None,
    probe_only: bool,
    samples: int | None,
    only_failed: bool = False,
    resume: bool = True,
    instance_ids: set[str] | None = None,
) -> int:
    from eval.bfcl_eval.agent import LangGraphAgent
    from eval.bfcl_eval.dataset import collect_catalog, load_entries
    from eval.bfcl_eval.evaluate import (
        _approx_tokens,
        _gt_names,
        aggregate,
        classify_error,
        compute_ast_valid,
        compute_exact_match,
        compute_name_acc,
    )
    from eval.bfcl_eval.tools import tool_name
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
    staging: dict | None = None
    if resume and staging_path.exists():
        staging = json.loads(staging_path.read_text(encoding="utf-8"))
        expected_source = original_src.name
        if staging.get("source_json") in {expected_source, src.name}:
            _apply_staging(saved_instances, staging)
            print(f"  resumed staging → {staging_path.relative_to(REPO)}")
        else:
            print(
                f"warning: ignoring stale staging (source {staging.get('source_json')!r})",
                file=sys.stderr,
            )
            staging = None

    fail_attempts = _load_fail_attempts(staging, payload)
    rerun_meta = (payload.get("config") or {}).get("baseline_rerun") or {}
    for iid in rerun_meta.get("api_fail_ids") or []:
        fail_attempts[str(iid)] = max(fail_attempts.get(str(iid), 0), 1)

    # Prior runs did not always persist fail_attempts; abandon lone slow api_fail when
    # the rest of the matrix is already complete (e.g. 199/200 ok).
    n_ok = sum(1 for s in saved_instances if _saved_baseline_ok(s))
    for saved in saved_instances:
        iid = saved["id"]
        b = saved.get("baseline") or {}
        if (
            b.get("error") == "api_fail"
            and float(b.get("latency_ms") or 0) >= 2000.0
            and n_ok >= len(saved_instances) - 1
            and fail_attempts.get(iid, 0) < BASELINE_MAX_INSTANCE_ATTEMPTS
        ):
            fail_attempts[iid] = BASELINE_MAX_INSTANCE_ATTEMPTS

    pending = [
        s
        for s in saved_instances
        if _should_rerun_instance(
            s,
            only_failed=only_failed,
            instance_ids=instance_ids,
            fail_attempts=fail_attempts,
        )
    ]
    print(
        f"  baseline rerun: {len(pending)}/{len(saved_instances)} instances "
        f"({'only-failed' if only_failed else 'all'})"
    )

    ds = cfg.get("dataset") or {}
    entries = load_entries(
        categories=ds.get("categories") or ["multiple"],
        cache_dir=Path(ds.get("cache_dir", "eval/.bfcl_cache")),
        samples=samples,
        seed=int(ds.get("seed", 42)),
    )
    by_id = {e.id: e for e in entries}
    all_tools, _ = collect_catalog(entries)
    tool_pool = list(all_tools.values())

    out_cfg_base = dict(payload.get("config") or {})

    def _retryable_api_fail_ids() -> list[str]:
        return [
            s["id"]
            for s in saved_instances
            if not _saved_baseline_ok(s)
            and (s.get("baseline") or {}).get("error") == "api_fail"
            and fail_attempts.get(s["id"], 0) < BASELINE_MAX_INSTANCE_ATTEMPTS
        ]

    def _all_settled() -> bool:
        return all(_instance_settled(s, fail_attempts) for s in saved_instances)

    def _promote_if_settled() -> int | None:
        if not _all_settled():
            return None
        merged_local = [_instance_result_from_saved_row(s) for s in saved_instances]
        metrics_local = aggregate(merged_local, n_skipped=0)
        abandoned = _abandoned_ids(saved_instances, fail_attempts)
        out_cfg = dict(out_cfg_base)
        out_cfg["baseline_rerun"] = {
            "source_json": original_src.name,
            "context_size_note": "models.yaml context_size>=65536 for large models",
            "fail_attempts": fail_attempts,
            "abandoned_ids": abandoned,
        }
        path = save_results(
            results=merged_local,
            metrics=metrics_local,
            config=out_cfg,
            output_dir=RESULTS_DIR,
            model_name=model_id,
        )
        _update_checkpoint(model_id, merged_local)
        _clear_staging(model_id)
        if inprogress.exists():
            inprogress.unlink()
        if abandoned:
            print(
                f"  promoted with {len(abandoned)} abandoned api_fail "
                f"({', '.join(abandoned)}) → {path.name}",
                file=sys.stderr,
            )
        else:
            print(f"  promoted → {path.name}")
        return 0

    if not pending and not probe_only:
        merged = [_instance_result_from_saved_row(s) for s in saved_instances]
        metrics = aggregate(merged, n_skipped=0)
        api_fail_ids = _retryable_api_fail_ids()
        snap = _persist_snapshot(
            model_id,
            original_src.name,
            payload,
            saved_instances,
            api_fail_ids=api_fail_ids,
            fail_attempts=fail_attempts,
        )
        print(
            f"  nothing to rerun; baseline name_acc={metrics.baseline_name_acc:.1%} "
            f"({snap.name})"
        )
        promoted = _promote_if_settled()
        if promoted is not None:
            return promoted
        return 1

    agent = LangGraphAgent(entry)
    agent.load()

    probe_id = pending[0]["id"] if pending else saved_instances[0]["id"]
    first = next(e for e in entries if e.id == probe_id)
    probe_baseline(agent, first.messages, tool_pool)

    if probe_only:
        print("probe-only: success")
        return 0

    staging_rows: dict[str, dict] = {}
    if staging is not None:
        staging_rows = dict(staging.get("instances_by_id") or {})
    elif resume and staging_path.exists():
        staging_rows = json.loads(staging_path.read_text(encoding="utf-8")).get(
            "instances_by_id", {}
        )

    merged: list = []
    pending_ids = {s["id"] for s in pending}
    for saved in tqdm(saved_instances, desc=f"{model_id} baseline", unit="inst"):
        inst = by_id.get(saved["id"])
        if inst is None:
            raise SystemExit(f"missing BFCL entry for {saved['id']}")
        gt_names = _gt_names(inst.ground_truth)
        if not gt_names:
            continue

        if saved["id"] not in pending_ids:
            merged.append(_instance_result_from_saved_row(saved))
            continue

        query = next(
            (m.get("content", "") for m in inst.messages if m.get("role") == "user"),
            "",
        )[:200]
        baseline_tokens = _approx_tokens(tool_pool)
        base_res = _predict_baseline_with_retry(
            agent, inst.messages, tool_pool, instance_id=inst.id
        )
        if base_res.error == "api_fail":
            fail_attempts[inst.id] = fail_attempts.get(inst.id, 0) + 1

        baseline_pred = base_res.predicted
        name_acc = compute_name_acc(baseline_pred, gt_names)
        exact = compute_exact_match(baseline_pred, inst.ground_truth, gt_names)
        ast_ok = (
            compute_ast_valid(
                baseline_pred,
                inst.possible_answer,
                func_descriptions=inst.functions_bfcl,
            )
            if inst.possible_answer is not None
            else exact
        )
        baseline_error = (
            base_res.error
            if base_res.error
            else classify_error(
                pred=baseline_pred,
                raw=base_res.raw,
                name_acc=name_acc,
                ast_acc=ast_ok,
                gt_names=gt_names,
                retrieved_names=[tool_name(t) for t in tool_pool],
            )
        )

        base = _instance_from_saved(saved)
        result = base.__class__(
            id=inst.id,
            query=query,
            ground_truth_names=gt_names,
            baseline_name_acc=name_acc,
            baseline_exact_match=exact,
            baseline_ast_acc=ast_ok,
            baseline_tokens=baseline_tokens,
            baseline_raw=base_res.raw,
            baseline_pred=baseline_pred,
            baseline_error=baseline_error,
            baseline_latency_ms=base_res.latency_ms,
            baseline_prompt_tokens=base_res.prompt_tokens,
            retrievers=base.retrievers,
        )
        merged.append(result)

        row = _instance_row_from_result(result)
        staging_rows[result.id] = row
        saved.clear()
        saved.update(row)
        _write_staging(
            model_id, original_src.name, staging_rows, fail_attempts=fail_attempts
        )
        _update_checkpoint(model_id, [result], quiet=True)
        snap = _persist_snapshot(
            model_id,
            original_src.name,
            payload,
            saved_instances,
            api_fail_ids=_retryable_api_fail_ids(),
            fail_attempts=fail_attempts,
        )
        n_ok = sum(1 for s in saved_instances if _saved_baseline_ok(s))
        tqdm.write(f"  saved {result.id} → {snap.name} ({n_ok}/{len(saved_instances)} ok)")

    merged = [_instance_result_from_saved_row(s) for s in saved_instances]
    metrics = aggregate(merged, n_skipped=0)
    retryable = _retryable_api_fail_ids()
    snap = _persist_snapshot(
        model_id,
        original_src.name,
        payload,
        saved_instances,
        api_fail_ids=retryable,
        fail_attempts=fail_attempts,
    )

    promoted = _promote_if_settled()
    if promoted is not None:
        print(f"baseline name_acc={metrics.baseline_name_acc:.1%} saved")
        return promoted

    if retryable:
        print(
            f"baseline rerun incomplete: {len(retryable)} retryable api_fail "
            f"({', '.join(retryable[:5])}"
            f"{', ...' if len(retryable) > 5 else ''}) — "
            f"all progress saved → {snap.name}; retry with --only-failed",
            file=sys.stderr,
        )
        return 1

    return 1


def _update_checkpoint(model_id: str, results, *, quiet: bool = False) -> None:
    ckpt_dir = RESULTS_DIR / "checkpoints"
    slug = model_id.split("/")[-1]
    matches = sorted(ckpt_dir.glob(f"{slug}_*.jsonl"))
    if not matches:
        print("warning: no checkpoint jsonl to update", file=sys.stderr)
        return
    ckpt_path = matches[-1]
    by_id = {r.id: r for r in results}
    lines = []
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row.get("id")
        if rid in by_id:
            r = by_id[rid]
            row.update({
                "baseline_name_acc": r.baseline_name_acc,
                "baseline_exact_match": r.baseline_exact_match,
                "baseline_ast_acc": r.baseline_ast_acc,
                "baseline_tokens": r.baseline_tokens,
                "baseline_raw": r.baseline_raw,
                "baseline_pred": asdict(r.baseline_pred) if r.baseline_pred else None,
                "baseline_error": r.baseline_error,
                "baseline_latency_ms": r.baseline_latency_ms,
                "baseline_prompt_tokens": r.baseline_prompt_tokens,
            })
        lines.append(json.dumps(row, ensure_ascii=False))
    ckpt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not quiet:
        print(f"checkpoint updated → {ckpt_path.relative_to(REPO)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model id from models.yaml")
    parser.add_argument("--source", type=Path, help="Existing bfcl_eval_*.json to merge from")
    parser.add_argument("--probe-only", action="store_true", help="Run one baseline probe and exit")
    parser.add_argument("--samples", type=int, help="Limit instances (debug)")
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Re-run only api_fail instances with fewer than 3 cumulative attempts",
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
    return rerun_baseline(
        args.model,
        source=args.source,
        probe_only=args.probe_only,
        samples=args.samples,
        only_failed=args.only_failed,
        resume=not args.no_resume,
        instance_ids=instance_ids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
