#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODE="$(inference_mode)"

if [[ "${MODE}" == "native" ]]; then
  PIDFILE="$(llama_pid_file)"
  if [[ -f "${PIDFILE}" ]]; then
    PID="$(cat "${PIDFILE}")"
    if kill -0 "${PID}" 2>/dev/null; then
      kill "${PID}" 2>/dev/null || true
      sleep 1
      kill -9 "${PID}" 2>/dev/null || true
    fi
    rm -f "${PIDFILE}"
    echo "Stopped native llama-server (pid ${PID})"
  fi
  pkill -f "llama-server -m " 2>/dev/null || true
  exit 0
fi

CNAME="$(container_name)"
if container_exists "${CNAME}"; then
  ctr rm -f "${CNAME}" >/dev/null 2>&1 || true
  echo "Stopped ${CNAME}"
fi
