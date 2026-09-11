#!/usr/bin/env bash
# Full local BFCL matrix: build image → download weights → serve → smoke → eval → stop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REBUILD=false
SAMPLES=""
DRY_RUN=false
NO_RESUME=false
MODEL_FILTER=()
SKIP_BUILD=false
SKIP_DOWNLOAD=false
SKIP_EVAL=false
USE_DEVCONTAINER=false
TIER_FILTER=""
KEEP_WEIGHTS=false
# Keep weights by default; opt in to purge when disk is tight (e.g. first full matrix).
if [[ -n "${TOOLSCOPE_PURGE_AFTER_EVAL:-}" ]]; then
  PURGE_AFTER_EVAL="${TOOLSCOPE_PURGE_AFTER_EVAL}"
else
  PURGE_AFTER_EVAL=false
fi

usage() {
  cat <<EOF
Usage: $0 [options]

  --rebuild           Force podman image rebuild
  --skip-build        Skip image build
  --skip-download     Skip HF weight download
  --skip-eval         Only build/download/serve smoke (no BFCL)
  --devcontainer      Run eval inside toolscope-dev image (native llama.cpp)
  --tier <name>       Run a tier: high_sensitivity | mid_production | control_ceiling
  --model <id>        Run one or more models (repeatable)
  --samples <n>       Limit BFCL instances (pilot runs)
  --dry-run           Use dummy model in eval (no live LLM calls)
  --no-resume         Disable checkpoint resume per model
  --keep-weights      Keep GGUF weights on disk after each model (default)
  --purge-after-eval  Delete weights after each successful eval (saves disk)
  -h, --help          Show this help

Tiers:
  high_sensitivity   SLM testbed (llama-3.2-3b, qwen2.5-7b) — largest ToolScope delta
  mid_production     Edge agents (llama-3.1-8b-instruct, qwen3-32b)
  control_ceiling    70B ceiling (llama-3.3-70b-instruct)

Models: $(list_model_ids | tr '\n' ' ')
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild) REBUILD=true; shift ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --skip-eval) SKIP_EVAL=true; shift ;;
    --devcontainer) USE_DEVCONTAINER=true; shift ;;
    --tier) TIER_FILTER="$2"; shift 2 ;;
    --model) MODEL_FILTER+=("$2"); shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --no-resume) NO_RESUME=true; shift ;;
    --keep-weights) KEEP_WEIGHTS=true; PURGE_AFTER_EVAL=false; shift ;;
    --purge-after-eval) PURGE_AFTER_EVAL=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

IDS=()
if [[ ${#MODEL_FILTER[@]} -gt 0 ]]; then
  IDS=("${MODEL_FILTER[@]}")
elif [[ -n "${TIER_FILTER}" ]]; then
  mapfile -t IDS < <(list_tier_ids "${TIER_FILTER}")
  if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "error: unknown or empty tier: ${TIER_FILTER}" >&2
    exit 1
  fi
else
  mapfile -t IDS < <(default_matrix_ids)
fi

# On host, live inference runs inside the devcontainer (native llama-server).
if [[ "${SKIP_EVAL}" != "true" && "${DRY_RUN}" != "true" && "$(inference_mode)" == "container" ]]; then
  if [[ "${USE_DEVCONTAINER}" == "true" ]] || ! command -v llama-server >/dev/null 2>&1; then
    USE_DEVCONTAINER=true
  fi
fi

if [[ "${USE_DEVCONTAINER}" == "true" && "${SKIP_EVAL}" != "true" ]]; then
  echo "Delegating matrix to devcontainer (native llama.cpp) ..."
  DC_ARGS=(--skip-build)
  [[ "${SKIP_DOWNLOAD}" == "true" ]] && DC_ARGS+=(--skip-download)
  [[ -n "${SAMPLES}" ]] && DC_ARGS+=(--samples "${SAMPLES}")
  [[ "${DRY_RUN}" == "true" ]] && DC_ARGS+=(--dry-run)
  [[ "${NO_RESUME}" == "true" ]] && DC_ARGS+=(--no-resume)
  [[ "${KEEP_WEIGHTS}" == "true" ]] && DC_ARGS+=(--keep-weights)
  for mid in "${IDS[@]}"; do
    DC_ARGS+=(--model "$mid")
  done
  "${SCRIPT_DIR}/run_in_devcontainer.sh" \
    env TOOLSCOPE_INFERENCE_MODE=native TOOLSCOPE_DEVCONTAINER=1 \
    eval/local/scripts/run_local_matrix.sh "${DC_ARGS[@]}"
  exit 0
fi

if [[ "${SKIP_BUILD}" != "true" ]] && [[ "$(inference_mode)" == "container" ]]; then
  if [[ "${REBUILD}" == "true" ]]; then
    "${SCRIPT_DIR}/build_image.sh" --rebuild
  else
    "${SCRIPT_DIR}/build_image.sh"
  fi
fi

EVAL_ARGS=(--config "${LOCAL_DIR}/bfcl_multiple_local.yaml")
if [[ -n "${SAMPLES}" ]]; then
  EVAL_ARGS+=(--samples "${SAMPLES}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  EVAL_ARGS+=(--dry-run)
fi
if [[ "${NO_RESUME}" == "true" ]]; then
  EVAL_ARGS+=(--no-resume)
fi

log_cache_disk

PREFETCH_PID=""
start_prefetch_queue() {
  local -a remaining=("$@")
  if [[ "${SKIP_DOWNLOAD}" == "true" || "${DRY_RUN}" == "true" || ${#remaining[@]} -eq 0 ]]; then
    return
  fi
  if [[ -n "${PREFETCH_PID}" ]] && kill -0 "${PREFETCH_PID}" 2>/dev/null; then
    return
  fi
  local -a args=()
  for mid in "${remaining[@]}"; do
    args+=(--model "$mid")
  done
  echo "Prefetch queue (sequential, background): ${remaining[*]}"
  "${SCRIPT_DIR}/prefetch_models.sh" "${args[@]}" &
  PREFETCH_PID=$!
}

wait_prefetch() {
  if [[ -n "${PREFETCH_PID}" ]] && kill -0 "${PREFETCH_PID}" 2>/dev/null; then
    echo "Waiting for background prefetch (PID ${PREFETCH_PID}) ..."
    wait "${PREFETCH_PID}" || echo "warning: prefetch exited non-zero" >&2
  fi
  PREFETCH_PID=""
}

for i in "${!IDS[@]}"; do
  MODEL_ID="${IDS[$i]}"
  REMAINING=()
  if [[ $i -lt $((${#IDS[@]} - 1)) ]]; then
    REMAINING=("${IDS[@]:$((i + 1))}")
  fi
  ALIAS="$(resolve_model_field "${MODEL_ID}" alias)"
  echo ""
  echo "========== ${MODEL_ID} (${ALIAS}) =========="
  log_cache_disk

  if [[ "${SKIP_DOWNLOAD}" != "true" ]]; then
    wait_prefetch
    if find_gguf_path "${MODEL_ID}" >/dev/null 2>&1; then
      echo "Weights already present for ${MODEL_ID}; skipping download"
    elif ! "${SCRIPT_DIR}/download_models.sh" --model "${MODEL_ID}"; then
      echo "warning: download failed for ${MODEL_ID}; skipping model" >&2
      continue
    fi
  fi

  if [[ "${DRY_RUN}" != "true" ]]; then
    if ! "${SCRIPT_DIR}/serve_model.sh" "${MODEL_ID}"; then
      echo "warning: serve failed for ${MODEL_ID}; skipping model" >&2
      continue
    fi
    HC_TIMEOUT="$(resolve_model_field "${MODEL_ID}" healthcheck_timeout_seconds 2>/dev/null || true)"
    HC_TIMEOUT="${HC_TIMEOUT:-1200}"
    if ! HEALTHCHECK_TIMEOUT="${HC_TIMEOUT}" "${SCRIPT_DIR}/healthcheck.sh"; then
      echo "warning: healthcheck failed for ${MODEL_ID}; skipping model" >&2
      "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
      continue
    fi
    if ! python3 "${LOCAL_DIR}/smoke/tool_call_probe.py" --model "${ALIAS}"; then
      echo "warning: smoke test failed for ${MODEL_ID}; skipping model" >&2
      "${SCRIPT_DIR}/stop_server.sh" 2>/dev/null || true
      continue
    fi
  fi

  EVAL_OK=true
  if [[ "${SKIP_EVAL}" != "true" ]]; then
    start_prefetch_queue "${REMAINING[@]}"
    EVAL_OK=false
    if (
      cd "${REPO_ROOT}"
      python3 eval/run_eval.py "${EVAL_ARGS[@]}" --model "${ALIAS}"
    ); then
      EVAL_OK=true
    else
      echo "warning: eval failed for ${MODEL_ID}; continuing matrix" >&2
    fi
  fi

  if [[ "${DRY_RUN}" != "true" ]]; then
    "${SCRIPT_DIR}/stop_server.sh"
  fi

  if [[ "${PURGE_AFTER_EVAL}" == "true" && "${EVAL_OK}" == "true" ]]; then
    echo "Purging weights for ${MODEL_ID} to reclaim disk ..."
    if "${SCRIPT_DIR}/purge_model.sh" --model "${MODEL_ID}"; then
      log_cache_disk
    else
      echo "warning: purge failed for ${MODEL_ID}" >&2
    fi
  fi

  if [[ "${EVAL_OK}" == "true" && "${SKIP_EVAL}" != "true" ]]; then
    echo "Updating local harness artifacts after ${MODEL_ID} ..."
    if ! TOOLSCOPE_FINALIZE_COMMIT=1 TOOLSCOPE_FINALIZE_PUSH=1 \
      "${SCRIPT_DIR}/finalize_local_if_needed.sh"; then
      echo "warning: artifact finalize failed for ${MODEL_ID}" >&2
    fi
  fi
done

wait_prefetch

echo ""
echo "Done."
echo "  Runtime results: ${REPO_ROOT}/eval/results/paper/local/"
echo "  Frozen artifacts: ${REPO_ROOT}/eval/paper/artifacts/local/"
