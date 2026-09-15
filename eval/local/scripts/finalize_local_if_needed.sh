#!/usr/bin/env bash
# Regenerate eval/paper/artifacts when new model result JSONs are complete.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_FILE="${TOOLSCOPE_FINALIZE_STATE:-/tmp/toolscope-artifacts-finalized.json}"
MIN_N="${TOOLSCOPE_FINALIZE_MIN_N:-200}"
DO_COMMIT="${TOOLSCOPE_FINALIZE_COMMIT:-1}"
DO_PUSH="${TOOLSCOPE_FINALIZE_PUSH:-1}"

COMMIT_ARGS=()
[[ "${DO_COMMIT}" == "1" ]] && COMMIT_ARGS+=(--commit)
[[ "${DO_PUSH}" == "1" ]] && COMMIT_ARGS+=(--push)

_complete_models() {
  python3 - "${REPO_ROOT}" "${MIN_N}" <<'PY'
import json, sys
from pathlib import Path
import yaml

repo = Path(sys.argv[1])
min_n = int(sys.argv[2])
cfg = yaml.safe_load((repo / "eval/paper/bfcl_multiple.yaml").read_text()) or {}
results = repo / "eval/results/paper/local"
out = {}
for entry in cfg.get("model", {}).get("entries", []):
    mid = entry["name"]
    slug = mid.split("/")[-1]
    best_n = -1
    for path in sorted(results.glob(f"bfcl_eval_{slug}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if "inprogress" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload.get("metrics", {}).get("n", 0))
            insts = payload.get("instances") or []
            api_fail = sum(
                1 for i in insts
                if (i.get("baseline") or {}).get("error") == "api_fail"
            )
            rerun = (payload.get("config") or {}).get("baseline_rerun") or {}
            abandoned = set(rerun.get("abandoned_ids") or [])
            fail_attempts = {
                str(k): int(v) for k, v in (rerun.get("fail_attempts") or {}).items()
            }
            unsettled_api_fail = sum(
                1 for i in insts
                if (i.get("baseline") or {}).get("error") == "api_fail"
                and i.get("id") not in abandoned
                and fail_attempts.get(i.get("id", ""), 0) < 3
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if n > best_n and unsettled_api_fail == 0:
            best_n = n
    if best_n >= min_n:
        out[mid] = best_n
print(json.dumps(out, sort_keys=True))
PY
}

_load_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}"
  else
    echo '{}'
  fi
}

CURRENT="$(_complete_models)"
PREVIOUS="$(_load_state)"

if [[ "${CURRENT}" == "${PREVIOUS}" ]]; then
  exit 0
fi

echo "New complete model result(s) detected; updating local harness artifacts ..."
echo "  before: ${PREVIOUS}"
echo "  after:  ${CURRENT}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python3 eval/local/scripts/finalize_local_artifacts.py \
  --min-n "${MIN_N}" \
  "${COMMIT_ARGS[@]}"

echo "${CURRENT}" > "${STATE_FILE}"
