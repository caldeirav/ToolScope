#!/usr/bin/env bash
# Re-run BM25 / ToolScope retriever conditions only; baseline is preserved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODELS=()
PROBE_ONLY=false
FINALIZE=true
ALL_INSTANCES=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    --probe-only) PROBE_ONLY=true; shift ;;
    --all-instances) ALL_INSTANCES=true; shift ;;
    --no-finalize) FINALIZE=false; shift ;;
    -h|--help)
      echo "Usage: $0 --model <id> [--model <id> ...] [--probe-only] [--all-instances] [--no-finalize]"
      echo "Default: re-run retriever conditions for instances with api_fail only."
      echo "Models: qwen3-32b llama-3.3-70b-instruct"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(qwen3-32b llama-3.3-70b-instruct)
fi

LOG="${TOOLSCOPE_RETRIEVER_LOG:-/tmp/toolscope-retriever-rerun.log}"
touch "${LOG}"

_run_model() {
  local model_id="$1"
  local alias hc_timeout
  alias="$(resolve_model_field "${model_id}" alias)"
  hc_timeout="$(resolve_model_field "${model_id}" healthcheck_timeout_seconds 2>/dev/null || true)"
  hc_timeout="${hc_timeout:-2400}"

  echo ""
  echo "========== retriever rerun: ${model_id} ==========" | tee -a "${LOG}"

  if ! find_gguf_path "${model_id}" >/dev/null 2>&1; then
    echo "error: weights missing for ${model_id}" | tee -a "${LOG}" >&2
    return 1
  fi

  "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
  "${SCRIPT_DIR}/serve_model.sh" "${model_id}" | tee -a "${LOG}"
  if ! HEALTHCHECK_TIMEOUT="${hc_timeout}" "${SCRIPT_DIR}/healthcheck.sh" | tee -a "${LOG}"; then
    echo "error: healthcheck failed for ${model_id}" | tee -a "${LOG}" >&2
    "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
    return 1
  fi

  echo "Waiting for tool-call smoke (model may still be loading) ..." | tee -a "${LOG}"
  smoke_ok=false
  for _ in $(seq 1 120); do
    if python3 "${LOCAL_DIR}/smoke/tool_call_probe.py" --model "${alias}" >>"${LOG}" 2>&1; then
      smoke_ok=true
      break
    fi
    sleep 15
  done
  if [[ "${smoke_ok}" != "true" ]]; then
    echo "error: smoke test never passed for ${model_id}" | tee -a "${LOG}" >&2
    "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
    return 1
  fi

  local py_args=(python3 "${SCRIPT_DIR}/rerun_retrievers_local.py" --model "${model_id}")
  [[ "${PROBE_ONLY}" == "true" ]] && py_args+=(--probe-only)
  [[ "${ALL_INSTANCES}" == "true" ]] && py_args+=(--all-instances)

  set +e
  "${py_args[@]}" 2>&1 | tee -a "${LOG}"
  py_status=${PIPESTATUS[0]}
  set -e
  if [[ "${py_status}" -ne 0 ]]; then
    slug="${model_id##*/}"
    inprogress="eval/results/paper/local/bfcl_eval_${slug}_retriever_inprogress.json"
    if [[ -f "${inprogress}" ]]; then
      echo "warning: ${model_id} incomplete (exit ${py_status}); progress in ${inprogress}" | tee -a "${LOG}" >&2
      "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
      return "${py_status}"
    fi
    echo "error: retriever probe/rerun failed for ${model_id}" | tee -a "${LOG}" >&2
    "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
    return 1
  fi

  "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
  echo "done: ${model_id}" | tee -a "${LOG}"
}

if [[ -f /etc/toolscope/devcontainer ]]; then
  REBUILD_LLAMA="${REBUILD_LLAMA:-0}" "${SCRIPT_DIR}/build_llama_server.sh" | tee -a "${LOG}"
  for mid in "${MODELS[@]}"; do
    _run_model "${mid}" || exit 1
  done
else
  DC_ARGS=()
  for mid in "${MODELS[@]}"; do
    DC_ARGS+=(--model "${mid}")
  done
  [[ "${PROBE_ONLY}" == "true" ]] && DC_ARGS+=(--probe-only)
  [[ "${ALL_INSTANCES}" == "true" ]] && DC_ARGS+=(--all-instances)
  [[ "${FINALIZE}" == "false" ]] && DC_ARGS+=(--no-finalize)
  "${SCRIPT_DIR}/run_in_devcontainer.sh" \
    env TOOLSCOPE_INFERENCE_MODE=native TOOLSCOPE_DEVCONTAINER=1 \
    TOOLSCOPE_RETRIEVER_LOG="${LOG}" \
    eval/local/scripts/rerun_retrievers_local.sh "${DC_ARGS[@]}"
fi

if [[ "${PROBE_ONLY}" != "true" && "${FINALIZE}" == "true" ]]; then
  echo "Updating frozen artifacts ..."
  TOOLSCOPE_FINALIZE_COMMIT=1 TOOLSCOPE_FINALIZE_PUSH=1 \
    "${SCRIPT_DIR}/finalize_local_if_needed.sh" | tee -a "${LOG}"
fi

echo "Log: ${LOG}"
