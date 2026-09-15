#!/usr/bin/env bash
# Download one or more models sequentially (background-friendly prefetch queue).
# Safe to run while another model is being evaluated — one HF download at a time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODEL_FILTER=()

usage() {
  cat <<EOF
Usage: $0 [--model <id>] ...

Download models one at a time into TOOLSCOPE_MODEL_CACHE. Intended as a
background prefetch while run_local_matrix.sh evaluates the current model.

Log: /tmp/toolscope-prefetch.log (override with TOOLSCOPE_PREFETCH_LOG)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_FILTER+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#MODEL_FILTER[@]} -eq 0 ]]; then
  echo "error: specify at least one --model <id>" >&2
  exit 1
fi

STATE_DIR="$(llama_state_dir)"
mkdir -p "${STATE_DIR}" 2>/dev/null || true
LOG="${TOOLSCOPE_PREFETCH_LOG:-/tmp/toolscope-prefetch.log}"

_log() {
  echo "$1" | tee -a "${LOG}" 2>/dev/null || echo "$1"
}

for MODEL_ID in "${MODEL_FILTER[@]}"; do
  if find_gguf_path "${MODEL_ID}" >/dev/null 2>&1; then
    _log "$(date -Is) [prefetch] skip ${MODEL_ID} (already on disk)"
    continue
  fi
  _log "$(date -Is) [prefetch] downloading ${MODEL_ID} ..."
  if "${SCRIPT_DIR}/download_models.sh" --model "${MODEL_ID}" >>"${LOG}" 2>&1; then
    _log "$(date -Is) [prefetch] done ${MODEL_ID}"
  else
    _log "$(date -Is) [prefetch] FAILED ${MODEL_ID}"
  fi
done

_log "$(date -Is) [prefetch] queue complete"
