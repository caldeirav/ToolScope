#!/usr/bin/env bash
# Build llama-server from vendored sources (host or devcontainer).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

"${SCRIPT_DIR}/vendor_llama_cpp.sh"

SRC="${LOCAL_DIR}/vendor/llama.cpp"
BUILD="${SRC}/build"
BIN="${BUILD}/bin/llama-server"
PATCH="${LOCAL_DIR}/patches/llama_server_extended_ctx.patch"

if [[ -f "${PATCH}" ]]; then
  if ! grep -q "using extended context (no cap)" "${SRC}/tools/server/server.cpp" 2>/dev/null; then
    echo "Applying llama-server extended-context patch ..."
    patch -p1 -d "${SRC}" < "${PATCH}"
  fi
fi

if [[ -x "${BIN}" ]] && [[ "${REBUILD_LLAMA:-0}" != "1" ]]; then
  echo "llama-server already built: ${BIN}"
  exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "Installing cmake ..."
  apt-get update -qq && apt-get install -y -qq cmake build-essential
fi
require_cmd cmake
require_cmd g++

if [[ -f /usr/local/bin/toolscope-cuda-runtime-env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/bin/toolscope-cuda-runtime-env.sh
  cuda_runtime_env
fi

export LD_LIBRARY_PATH="/usr/local/cuda-13/compat:${LD_LIBRARY_PATH:-}"

cmake -B "${BUILD}" -S "${SRC}" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"

cmake --build "${BUILD}" --target llama-server -j "$(nproc)"
echo "Built ${BIN}"
