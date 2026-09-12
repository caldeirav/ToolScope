#!/usr/bin/env bash
# Re-run baseline (full 443-tool catalog) for models whose prior baseline api_fail'd.
# Serves each model at models.yaml context_size (65536 for qwen3 / 70B), probes one
# full-catalog request, then re-runs baseline only and merges into existing results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODELS=()
PROBE_ONLY=false
FINALIZE=true
ONLY_FAILED="${TOOLSCOPE_BASELINE_ONLY_FAILED:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("$2"); shift 2 ;;
    --probe-only) PROBE_ONLY=true; shift ;;
    --only-failed) ONLY_FAILED=1; shift ;;
    --no-finalize) FINALIZE=false; shift ;;
    -h|--help)
      echo "Usage: $0 --model <id> [--model <id> ...] [--probe-only] [--only-failed] [--no-finalize]"
      echo "Models: qwen3-32b llama-3.3-70b-instruct"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(qwen3-32b llama-3.3-70b-instruct)
fi

LOG="${TOOLSCOPE_BASELINE_LOG:-/tmp/toolscope-baseline-rerun.log}"
# Append so a relaunch does not erase prior progress logs.
touch "${LOG}"

_run_model() {
  local model_id="$1"
  local alias ctx hc_timeout
  alias="$(resolve_model_field "${model_id}" alias)"
  ctx="$(resolve_model_field "${model_id}" context_size)"
  hc_timeout="$(resolve_model_field "${model_id}" healthcheck_timeout_seconds 2>/dev/null || true)"
  hc_timeout="${hc_timeout:-2400}"

  echo ""
  echo "========== baseline rerun: ${model_id} (ctx=${ctx}) ==========" | tee -a "${LOG}"

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
    tail -30 "$(llama_log_file)" >>"${LOG}" 2>/dev/null || true
    "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
    return 1
  fi

  local py_args=(python3 "${SCRIPT_DIR}/rerun_baseline_local.py" --model "${model_id}")
  [[ "${PROBE_ONLY}" == "true" ]] && py_args+=(--probe-only)
  [[ "${ONLY_FAILED}" == "1" ]] && py_args+=(--only-failed)

  set +e
  "${py_args[@]}" 2>&1 | tee -a "${LOG}"
  py_status=${PIPESTATUS[0]}
  set -e
  if [[ "${py_status}" -ne 0 ]]; then
    slug="${model_id##*/}"
    inprogress="eval/results/paper/local/bfcl_eval_${slug}_baseline_inprogress.json"
    if [[ -f "${inprogress}" ]]; then
      echo "warning: ${model_id} incomplete (exit ${py_status}); progress in ${inprogress}" | tee -a "${LOG}" >&2
      "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
      return "${py_status}"
    fi
    echo "error: baseline probe/rerun failed for ${model_id}" | tee -a "${LOG}" >&2
    "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
    return 1
  fi

  "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
  echo "done: ${model_id}" | tee -a "${LOG}"
}

if [[ -f /etc/toolscope/devcontainer ]]; then
  echo "Building patched llama-server (extended slot context) ..." | tee -a "${LOG}"
  REBUILD_LLAMA="${REBUILD_LLAMA:-1}" "${SCRIPT_DIR}/build_llama_server.sh" | tee -a "${LOG}"
  for mid in "${MODELS[@]}"; do
    _run_model "${mid}" || exit 1
  done
else
  echo "Running in devcontainer (native llama.cpp, ctx>=65536) ..."
  DC_ARGS=()
  for mid in "${MODELS[@]}"; do
    DC_ARGS+=(--model "${mid}")
  done
  [[ "${PROBE_ONLY}" == "true" ]] && DC_ARGS+=(--probe-only)
  [[ "${ONLY_FAILED}" == "1" ]] && DC_ARGS+=(--only-failed)
  [[ "${FINALIZE}" == "false" ]] && DC_ARGS+=(--no-finalize)
  "${SCRIPT_DIR}/run_in_devcontainer.sh" \
    env TOOLSCOPE_INFERENCE_MODE=native TOOLSCOPE_DEVCONTAINER=1 \
    TOOLSCOPE_BASELINE_LOG="${LOG}" \
    eval/local/scripts/rerun_baseline_local.sh "${DC_ARGS[@]}"
fi

if [[ "${PROBE_ONLY}" != "true" && "${FINALIZE}" == "true" ]]; then
  echo "Updating frozen artifacts ..."
  TOOLSCOPE_FINALIZE_COMMIT=1 TOOLSCOPE_FINALIZE_PUSH=1 \
    "${SCRIPT_DIR}/finalize_local_if_needed.sh" | tee -a "${LOG}"
fi

echo "Log: ${LOG}"
