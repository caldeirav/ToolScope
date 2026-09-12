#!/usr/bin/env bash
# Autonomous baseline rerun for qwen3-32b + llama-3.3-70b-instruct:
# wait for any in-flight run, retry --only-failed until clean, then finalize artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LOG="${TOOLSCOPE_BASELINE_COMPLETE_LOG:-/tmp/toolscope-baseline-complete.log}"
RERUN_LOG="${TOOLSCOPE_BASELINE_LOG:-/tmp/toolscope-baseline-rerun.log}"
POLL="${TOOLSCOPE_BASELINE_POLL_SECS:-120}"
MODELS=(qwen3-32b llama-3.3-70b-instruct)

_log() { echo "$(date -Is) $*" | tee -a "${LOG}"; }

_model_complete() {
  python3 "${SCRIPT_DIR}/baseline_run_status.py" "$1" | grep -q ': COMPLETE'
}

_wait_for_rerun_idle() {
  while pgrep -f 'rerun_baseline_local\.(sh|py)' >/dev/null 2>&1; do
    _log "waiting for in-flight baseline rerun ..."
    if [[ -f "${RERUN_LOG}" ]]; then
      tail -1 "${RERUN_LOG}" 2>/dev/null | sed 's/^/  /' | tee -a "${LOG}" || true
    fi
    python3 "${SCRIPT_DIR}/baseline_run_status.py" "${MODELS[@]}" 2>/dev/null | tee -a "${LOG}" || true
    sleep "${POLL}"
  done
}

_run_model_until_complete() {
  local model_id="$1"
  local attempt=0
  while ! _model_complete "${model_id}"; do
    attempt=$((attempt + 1))
    _log "========== ${model_id} baseline attempt ${attempt} (--only-failed) =========="
    set +e
    TOOLSCOPE_BASELINE_LOG="${RERUN_LOG}" \
      bash "${SCRIPT_DIR}/rerun_baseline_local.sh" \
      --model "${model_id}" --only-failed --no-finalize \
      2>&1 | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    set -e
    python3 "${SCRIPT_DIR}/baseline_run_status.py" "${model_id}" | tee -a "${LOG}"
    if _model_complete "${model_id}"; then
      _log "done: ${model_id}"
      return 0
    fi
    slug="${model_id##*/}"
    inprog="eval/results/paper/local/bfcl_eval_${slug}_baseline_inprogress.json"
    if [[ ! -f "${inprog}" ]]; then
      _log "error: ${model_id} failed with no in-progress snapshot (rc=${rc})"
      return 1
    fi
    _log "retry scheduled: ${model_id} incomplete (rc=${rc}); sleeping ${POLL}s"
    sleep "${POLL}"
  done
}

_log "baseline matrix supervisor started (poll=${POLL}s)"
touch "${LOG}"
touch "${RERUN_LOG}"

# If start_baseline_rerun.sh / watcher already launched a run, let it finish first.
_wait_for_rerun_idle

for mid in "${MODELS[@]}"; do
  if _model_complete "${mid}"; then
    _log "skip (already complete): ${mid}"
    continue
  fi
  _run_model_until_complete "${mid}" || exit 1
done

_log "all baseline reruns complete; updating harness artifacts ..."
TOOLSCOPE_FINALIZE_COMMIT=1 TOOLSCOPE_FINALIZE_PUSH=1 \
  bash "${SCRIPT_DIR}/finalize_local_if_needed.sh" 2>&1 | tee -a "${LOG}"

_log "baseline matrix supervisor finished"
python3 "${SCRIPT_DIR}/baseline_run_status.py" "${MODELS[@]}" | tee -a "${LOG}"
