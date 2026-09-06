#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_ALIAS:?MODEL_ALIAS is required}"

HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8000}"
CTX="${LLAMA_CONTEXT_SIZE:-131072}"
NGL="${LLAMA_N_GPU_LAYERS:-99}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "error: model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

ARGS=(
  -m "${MODEL_PATH}"
  -a "${MODEL_ALIAS}"
  --host "${HOST}"
  --port "${PORT}"
  -c "${CTX}"
  -ngl "${NGL}"
)

# Always enable jinja when EXTRA_ARGS does not already set it.
if [[ " ${EXTRA_ARGS:-} " != *" --jinja "* ]]; then
  ARGS+=(--jinja)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARR[@]}")
fi

echo "Starting llama-server: ${MODEL_ALIAS} (${MODEL_PATH})"
exec llama-server "${ARGS[@]}"
