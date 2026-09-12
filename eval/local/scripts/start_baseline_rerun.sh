#!/usr/bin/env bash
# Start autonomous baseline matrix supervisor (qwen3-32b + llama-3.3-70b-instruct).
# Waits for any in-flight run, retries failed instances, finalizes harness artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LOG="${TOOLSCOPE_BASELINE_COMPLETE_LOG:-/tmp/toolscope-baseline-complete.log}"
RERUN_LOG="${TOOLSCOPE_BASELINE_LOG:-/tmp/toolscope-baseline-rerun.log}"

pkill -f 'complete_baseline_matrix.sh' 2>/dev/null || true
pkill -f 'watch_baseline_rerun.sh' 2>/dev/null || true
# Do not kill an in-flight rerun_baseline_local — supervisor waits for it.

touch "${LOG}"
touch "${RERUN_LOG}"

export TOOLSCOPE_BASELINE_LOG="${RERUN_LOG}"
export TOOLSCOPE_BASELINE_COMPLETE_LOG="${LOG}"
nohup bash "${SCRIPT_DIR}/complete_baseline_matrix.sh" >>"${LOG}" 2>&1 &
echo $! > /tmp/toolscope-baseline-supervisor.pid

echo "supervisor_pid=$(cat /tmp/toolscope-baseline-supervisor.pid)"
echo "supervisor_log: ${LOG}"
echo "rerun_log: ${RERUN_LOG}"
