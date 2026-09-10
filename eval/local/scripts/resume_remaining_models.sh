#!/usr/bin/env bash
# Stop duplicate matrix runs and stale HF locks, then resume qwen3 + 70b once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LOCK="/tmp/toolscope-matrix.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another matrix resume is already running (lock: ${LOCK})" >&2
  exit 1
fi

echo "Stopping stale matrix / llama-server / watcher processes ..."
pkill -f 'run_local_matrix.sh' 2>/dev/null || true
pkill -f 'run_in_devcontainer.sh.*run_local_matrix' 2>/dev/null || true
pkill -f 'watch_matrix_and_finalize' 2>/dev/null || true
pkill -f 'resume3_orchestrate|matrix_resume3_waiter' 2>/dev/null || true
pkill -f 'llama-server' 2>/dev/null || true
pkill -f 'hf download.*Llama-3.3-70B' 2>/dev/null || true
pkill -f 'docker build.*toolscope-dev' 2>/dev/null || true
sleep 3

find "${REPO_ROOT}/eval/local/models" -name '*.lock' -delete 2>/dev/null || true

export LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b6985}"
"${SCRIPT_DIR}/vendor_llama_cpp.sh"
MATRIX_LOG="${TOOLSCOPE_MATRIX_LOG:-/tmp/toolscope-matrix-resume4.log}"
WATCH_LOG="${TOOLSCOPE_WATCH_LOG:-/tmp/toolscope-watch4.log}"

if [[ "${REBUILD_IMAGE:-0}" == "1" ]]; then
  echo "Rebuilding toolscope-dev with llama.cpp ${LLAMA_CPP_TAG} (--no-cache) ..."
  eval/local/scripts/run_in_devcontainer.sh --rebuild bash -lc 'llama-server --version 2>/dev/null | head -3 || true'
else
  echo "Using existing toolscope-dev image (set REBUILD_IMAGE=1 to force rebuild)."
fi

# Release startup lock before backgrounding (child cannot flock while we hold fd 9).
exec 9>&-

nohup bash -lc "
  cd \"${REPO_ROOT}\"
  eval/local/scripts/run_in_devcontainer.sh \
    eval/local/scripts/run_local_matrix.sh \
    --skip-build --purge-after-eval \
    --model qwen3-32b --model llama-3.3-70b-instruct \
    2>&1 | tee \"${MATRIX_LOG}\"
" >>/tmp/toolscope-matrix.nohup.log 2>&1 &
echo "matrix_pid=$!"

nohup env TOOLSCOPE_MATRIX_LOG="${MATRIX_LOG}" TOOLSCOPE_WATCH_LOG="${WATCH_LOG}" \
  eval/local/scripts/watch_matrix_and_finalize.sh >>"${WATCH_LOG}" 2>&1 &
echo "watcher_pid=$!"
echo "logs: ${MATRIX_LOG}  ${WATCH_LOG}"
