#!/usr/bin/env bash
# Fetch llama.cpp sources on the host (avoids flaky git clone inside docker build).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

TAG="${LLAMA_CPP_TAG:-b6985}"
VENDOR_ROOT="${LOCAL_DIR}/vendor"
SRC="${VENDOR_ROOT}/llama.cpp"
REPO_URL="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"

mkdir -p "${VENDOR_ROOT}"

if [[ -f "${SRC}/CMakeLists.txt" ]]; then
  if [[ -x "${SRC}/build/bin/llama-server" ]] \
    && grep -q "using extended context (no cap)" "${SRC}/tools/server/server.cpp" 2>/dev/null; then
    echo "llama.cpp vendor OK @ ${TAG} (patched build present)"
    exit 0
  fi
  current="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
  want="$(git ls-remote "${REPO_URL}" "refs/tags/${TAG}" 2>/dev/null | awk '{print $1}')"
  if [[ -n "${want}" && "${current}" == "${want}" ]]; then
    echo "llama.cpp vendor OK @ ${TAG} (${current:0:12})"
    exit 0
  fi
  echo "Refreshing llama.cpp vendor (${TAG}) ..."
  rm -rf "${SRC}"
fi

echo "Cloning llama.cpp ${TAG} from ${REPO_URL} ..."
git clone --depth 1 --branch "${TAG}" "${REPO_URL}" "${SRC}"
echo "Vendored → ${SRC}"
