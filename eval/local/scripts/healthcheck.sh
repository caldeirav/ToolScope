#!/usr/bin/env bash
# Poll GET /v1/models until llama-server is ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

PORT="$(server_port)"
BASE_URL="${OPENAI_BASE_URL:-$(openai_base_url)}"
TIMEOUT="${HEALTHCHECK_TIMEOUT:-600}"
INTERVAL="${HEALTHCHECK_INTERVAL:-5}"

# Strip trailing /v1 if duplicated
BASE_URL="${BASE_URL%/}"
if [[ "${BASE_URL}" != */v1 ]]; then
  BASE_URL="${BASE_URL}/v1"
fi

URL="${BASE_URL}/models"
echo "Waiting for ${URL} (timeout ${TIMEOUT}s) ..."

deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
  if curl -sf "${URL}" >/dev/null 2>&1; then
    echo "Server ready."
    curl -s "${URL}" | python3 -m json.tool 2>/dev/null || true
    exit 0
  fi
  if (( $(date +%s) >= deadline )); then
    echo "error: server not ready after ${TIMEOUT}s" >&2
    if [[ "$(inference_mode)" == "native" ]]; then
      LOG="$(llama_log_file)"
      if [[ -f "${LOG}" ]]; then
        echo "--- llama-server log (tail) ---" >&2
        tail -40 "${LOG}" >&2 || true
      fi
    elif command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then
      ctr logs "$(container_name)" 2>&1 | tail -40 || true
    fi
    exit 1
  fi
  sleep "${INTERVAL}"
done
