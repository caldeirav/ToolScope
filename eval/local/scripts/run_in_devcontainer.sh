#!/usr/bin/env bash
# Run a command inside the ToolScope devcontainer (native llama.cpp inference).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

IMAGE="${TOOLSCOPE_DEV_IMAGE:-toolscope-dev:latest}"
MODEL_CACHE_HOST="${TOOLSCOPE_MODEL_CACHE:-${REPO_ROOT}/eval/local/models}"
mkdir -p "${MODEL_CACHE_HOST}"

REBUILD=false
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=true; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]}"

if ! image_exists "${IMAGE}" || [[ "${REBUILD}" == "true" ]]; then
  if [[ "${REBUILD}" == "true" ]] && image_exists "${IMAGE}"; then
    echo "Rebuilding ${IMAGE} ..."
  else
    echo "Building ${IMAGE} (first run; compiles llama.cpp + installs eval deps — may take 20+ min) ..."
  fi
  ctr build \
    -f "${REPO_ROOT}/.devcontainer/Dockerfile" \
    --build-arg LLAMA_CPP_TAG=b4897 \
    -t "${IMAGE}" \
    "${REPO_ROOT}"
fi

GPU_ARGS=()
RT="$(container_runtime)"
if [[ "${RT}" == "docker" ]]; then
  GPU_ARGS=(--gpus all)
else
  GPU_ARGS=(--device nvidia.com/gpu=all)
fi

TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

REMOTE_CMD=""
if [[ $# -gt 0 ]]; then
  REMOTE_CMD="$(printf '%q ' "$@")"
else
  REMOTE_CMD="bash"
fi

ctr run --rm \
  "${TTY_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  --shm-size=32g \
  -v "${REPO_ROOT}:/workspace" \
  -v "${MODEL_CACHE_HOST}:/workspace/eval/local/models" \
  -e TOOLSCOPE_DEVCONTAINER=1 \
  -e TOOLSCOPE_INFERENCE_MODE=native \
  -e OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_API_KEY=local \
  -e TOOLSCOPE_MODEL_CACHE=/workspace/eval/local/models \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "${REMOTE_CMD}"
