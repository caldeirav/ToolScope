#!/usr/bin/env python3
"""Report baseline rerun completion for local BFCL models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "eval/results/paper/local"
BASELINE_MAX_INSTANCE_ATTEMPTS = 3


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _saved_baseline_ok(inst: dict) -> bool:
    b = inst.get("baseline") or {}
    if b.get("error") == "api_fail":
        return False
    pred = b.get("predicted") or inst.get("baseline_pred")
    raw = (b.get("raw") or inst.get("baseline_raw") or "").strip()
    if pred or raw:
        return True
    lat = float(b.get("latency_ms") or inst.get("baseline_latency_ms") or 0)
    return lat >= 5000.0 and b.get("error") is None


def _instance_settled(inst: dict, fail_attempts: dict[str, int]) -> bool:
    if _saved_baseline_ok(inst):
        return True
    b = inst.get("baseline") or {}
    if b.get("error") != "api_fail":
        return True
    return fail_attempts.get(inst["id"], 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS


def _load_payload(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def best_result_path(model_id: str) -> Path | None:
    slug = _slug(model_id)
    best: Path | None = None
    best_n = -1
    for path in sorted(
        RESULTS.glob(f"bfcl_eval_{slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if "inprogress" in path.name:
            continue
        payload = _load_payload(path)
        if not payload:
            continue
        n = int(payload.get("metrics", {}).get("n", 0))
        if n > best_n:
            best_n = n
            best = path
    return best


def inprogress_path(model_id: str) -> Path:
    return RESULTS / f"bfcl_eval_{_slug(model_id)}_baseline_inprogress.json"


def staging_path(model_id: str) -> Path:
    return RESULTS / f".baseline_rerun_{_slug(model_id)}.json"


def _load_fail_attempts_for_model(model_id: str, payload: dict) -> dict[str, int]:
    attempts: dict[str, int] = {}
    rerun = (payload.get("config") or {}).get("baseline_rerun") or {}
    for k, v in (rerun.get("fail_attempts") or {}).items():
        attempts[str(k)] = int(v)
    sp = staging_path(model_id)
    if sp.exists():
        try:
            staging = json.loads(sp.read_text(encoding="utf-8"))
            for k, v in (staging.get("fail_attempts") or {}).items():
                attempts[str(k)] = max(attempts.get(str(k), 0), int(v))
        except (json.JSONDecodeError, OSError):
            pass
    return attempts


def _apply_fail_attempt_migration(insts: list[dict], fail_attempts: dict[str, int]) -> None:
    n_ok = sum(1 for i in insts if _saved_baseline_ok(i))
    for inst in insts:
        iid = inst["id"]
        b = inst.get("baseline") or {}
        if (
            b.get("error") == "api_fail"
            and float(b.get("latency_ms") or 0) >= 2000.0
            and n_ok >= len(insts) - 1
            and fail_attempts.get(iid, 0) < BASELINE_MAX_INSTANCE_ATTEMPTS
        ):
            fail_attempts[iid] = BASELINE_MAX_INSTANCE_ATTEMPTS


def model_status(model_id: str, *, min_n: int = 200) -> dict:
    inprog = inprogress_path(model_id)
    final = best_result_path(model_id)
    path = inprog if inprog.exists() else final
    if path is None or not path.exists():
        return {
            "model_id": model_id,
            "complete": False,
            "n": 0,
            "ok": 0,
            "api_fail": 0,
            "abandoned": 0,
            "path": None,
            "in_progress_file": inprog.exists(),
        }
    payload = _load_payload(path) or {}
    insts = payload.get("instances") or []
    fail_attempts = _load_fail_attempts_for_model(model_id, payload)
    _apply_fail_attempt_migration(insts, fail_attempts)
    ok = sum(1 for i in insts if _saved_baseline_ok(i))
    api_fail = sum(
        1
        for i in insts
        if (i.get("baseline") or {}).get("error") == "api_fail"
        or i.get("baseline_error") == "api_fail"
    )
    abandoned = sum(
        1
        for i in insts
        if not _saved_baseline_ok(i)
        and (i.get("baseline") or {}).get("error") == "api_fail"
        and fail_attempts.get(i["id"], 0) >= BASELINE_MAX_INSTANCE_ATTEMPTS
    )
    settled = all(_instance_settled(i, fail_attempts) for i in insts)
    n = int(payload.get("metrics", {}).get("n", len(insts)))
    complete = (
        n >= min_n
        and settled
        and "inprogress" not in path.name
    )
    return {
        "model_id": model_id,
        "complete": complete,
        "n": n,
        "ok": ok,
        "api_fail": api_fail,
        "abandoned": abandoned,
        "path": str(path.relative_to(REPO)),
        "in_progress_file": inprog.exists(),
    }


def main() -> int:
    models = sys.argv[1:] or ["qwen3-32b", "llama-3.3-70b-instruct"]
    for mid in models:
        st = model_status(mid)
        flag = "COMPLETE" if st["complete"] else "pending"
        extra = ""
        if st.get("abandoned"):
            extra = f"  abandoned={st['abandoned']}"
        print(
            f"{mid}: {flag}  ok={st['ok']}/{st['n']}  api_fail={st['api_fail']}"
            f"{extra}  path={st['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
