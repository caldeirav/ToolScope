#!/usr/bin/env bash
# Build the llama.cpp podman image for local GGUF serving.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REBUILD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--rebuild]"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

require_cmd python3

RT="$(container_runtime)"
IMG="$(image_name)"

if image_exists "${IMG}" && [[ "${REBUILD}" != "true" ]]; then
  echo "Image ${IMG} already exists (use --rebuild to force)."
  exit 0
fi

echo "Building ${IMG} with ${RT} from ${CONTAINER_DIR}/Containerfile ..."
ctr build \
  -f "${CONTAINER_DIR}/Containerfile" \
  -t "${IMG}" \
  "${CONTAINER_DIR}"

echo "Built ${IMG}"
