#!/usr/bin/env bash
# Start llama-server (native in devcontainer, or container on host).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-id>" >&2
  echo "Models: $(list_model_ids | tr '\n' ' ')" >&2
  exit 1
fi

MODEL_ID="$1"
MODE="$(inference_mode)"

GGUF="$(find_gguf_path "${MODEL_ID}")"
ALIAS="$(resolve_model_field "${MODEL_ID}" alias)"
CTX="$(resolve_model_field "${MODEL_ID}" context_size)"
NGL="$(yaml_get defaults.n_gpu_layers 2>/dev/null || echo 99)"
EXTRA="$(resolve_model_field "${MODEL_ID}" server_extra_args 2>/dev/null || true)"
PORT="$(server_port)"

"${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true

if [[ "${MODE}" == "native" ]]; then
  require_cmd llama-server
  LOG="$(llama_log_file)"
  PIDFILE="$(llama_pid_file)"

  ARGS=(
    -m "${GGUF}"
    -a "${ALIAS}"
    --host 0.0.0.0
    --port "${PORT}"
    -c "${CTX}"
    -ngl "${NGL}"
    --jinja
  )
  if [[ -n "${EXTRA}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${EXTRA})
    ARGS+=("${EXTRA_ARR[@]}")
  fi

  echo "Starting native llama-server (${ALIAS}) on port ${PORT}"
  echo "  GGUF: ${GGUF}"
  nohup llama-server "${ARGS[@]}" >"${LOG}" 2>&1 &
  echo $! >"${PIDFILE}"
  echo "  PID: $(cat "${PIDFILE}")"
  echo "  Log: ${LOG}"
  exit 0
fi

# ── Container mode (podman/docker) ───────────────────────────────────────────
require_cmd python3
IMG="$(image_name)"
CNAME="$(container_name)"
RT="$(container_runtime)"

GGUF_BASENAME="$(basename "${GGUF}")"
MODEL_DIR="$(dirname "${GGUF}")"

GPU_ARGS=()
if [[ "${RT}" == "docker" ]]; then
  GPU_ARGS=(--gpus all)
else
  GPU_ARGS=(--device nvidia.com/gpu=all)
fi

ctr run -d \
  --name "${CNAME}" \
  "${GPU_ARGS[@]}" \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/models:ro" \
  -e "MODEL_PATH=/models/${GGUF_BASENAME}" \
  -e "MODEL_ALIAS=${ALIAS}" \
  -e "LLAMA_PORT=${PORT}" \
  -e "LLAMA_CONTEXT_SIZE=${CTX}" \
  -e "LLAMA_N_GPU_LAYERS=${NGL}" \
  -e "EXTRA_ARGS=${EXTRA}" \
  "${IMG}"

echo "Started ${CNAME} (${ALIAS}) on port ${PORT}"
echo "  GGUF: ${GGUF}"
echo "  Wait for readiness: ${SCRIPT_DIR}/healthcheck.sh"
