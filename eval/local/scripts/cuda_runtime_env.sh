#!/usr/bin/env bash
# CUDA runtime env for llama-server inside nvidia/cuda containers.
#
# Build-time needs /usr/local/cuda-*/compat (no GPU in buildkit). At runtime on
# hosts with a current driver (e.g. GB10 @ 580+), prepending compat libs causes
# error 803 (driver/CUDA mismatch) and CPU-only fallback. Use the host driver
# injected by nvidia-container-toolkit instead.
#
# Source before starting llama-server (serve_model.sh, devcontainer runs).

cuda_runtime_env() {
  local path="/opt/llama.cpp/bin"
  local entry stripped=""
  local old="${LD_LIBRARY_PATH:-}"

  for entry in ${old//:/ }; do
    [[ -z "${entry}" ]] && continue
    [[ "${entry}" == *"/compat"* ]] && continue
    path="${path}:${entry}"
  done

  # Deduplicate while preserving order.
  stripped=""
  for entry in ${path//:/ }; do
    [[ -z "${entry}" ]] && continue
    [[ ":${stripped}:" == *":${entry}:"* ]] && continue
    stripped="${stripped:+${stripped}:}${entry}"
  done
  export LD_LIBRARY_PATH="${stripped}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cuda_runtime_env
  exec "${@}"
fi
