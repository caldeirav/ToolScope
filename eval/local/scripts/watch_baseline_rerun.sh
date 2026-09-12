#!/usr/bin/env bash
# Poll baseline rerun log until completion, then finalize artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG="${TOOLSCOPE_BASELINE_LOG:-/tmp/toolscope-baseline-rerun.log}"
POLL="${TOOLSCOPE_BASELINE_POLL_SECS:-120}"

_log() { echo "$(date -Is) $*" | tee -a "${LOG}"; }

_log "baseline watcher started (poll=${POLL}s)"

while pgrep -f 'rerun_baseline_local.sh' >/dev/null 2>&1; do
  if [[ -f "${LOG}" ]]; then
    last="$(tail -1 "${LOG}" 2>/dev/null || true)"
    _log "running ... ${last}"
  else
    _log "running ..."
  fi
  sleep "${POLL}"
done

_log "baseline rerun process exited; finalizing artifacts"
sleep 10
cd "${REPO_ROOT}"
TOOLSCOPE_FINALIZE_COMMIT=1 TOOLSCOPE_FINALIZE_PUSH=1 \
  "${SCRIPT_DIR}/finalize_local_if_needed.sh" | tee -a "${LOG}"
_log "baseline watcher finished"
