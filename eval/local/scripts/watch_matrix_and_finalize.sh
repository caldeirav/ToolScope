#!/usr/bin/env bash
# Wait for the local matrix to finish, then freeze artifacts and git push.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG="${TOOLSCOPE_WATCH_LOG:-/tmp/toolscope-watch.log}"
MATRIX_LOG="${TOOLSCOPE_MATRIX_LOG:-/tmp/toolscope-matrix-resume.log}"
POLL_SECS="${TOOLSCOPE_WATCH_POLL_SECS:-300}"
MIN_MODELS="${TOOLSCOPE_FINALIZE_MIN_MODELS:-5}"
MIN_N="${TOOLSCOPE_FINALIZE_MIN_N:-200}"

_log() {
  echo "$(date -Is) $*" | tee -a "${LOG}"
}

_matrix_running() {
  pgrep -f 'eval/local/scripts/run_local_matrix.sh' >/dev/null 2>&1
}

_checkpoint_summary() {
  local dir="${REPO_ROOT}/eval/results/paper/local/checkpoints"
  if [[ ! -d "${dir}" ]]; then
    echo "no checkpoints"
    return
  fi
  wc -l "${dir}"/*.jsonl 2>/dev/null | tail -1 || echo "0 total"
}

_log "watcher started (poll=${POLL_SECS}s, min_models=${MIN_MODELS}, min_n=${MIN_N})"

while _matrix_running; do
  _log "matrix running; checkpoints: $(_checkpoint_summary)"
  sleep "${POLL_SECS}"
done

_log "matrix process exited; waiting 30s for flush"
sleep 30

if [[ -f "${MATRIX_LOG}" ]] && ! grep -q '^Done\.$' "${MATRIX_LOG}"; then
  _log "warning: matrix log missing 'Done.' — finalizing available artifacts anyway"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if python3 eval/local/scripts/finalize_local_artifacts.py --commit --push; then
  _log "finalize complete"
else
  _log "finalize failed (exit $?)"
  exit 1
fi

_log "watcher finished"
