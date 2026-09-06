#!/usr/bin/env bash
# Shared helpers for eval/local/scripts/*.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LOCAL_DIR}/../.." && pwd)"
MODELS_YAML="${LOCAL_DIR}/models.yaml"
CONTAINER_DIR="${LOCAL_DIR}/container"

# Default model cache (gitignored). Override in .env or the environment.
: "${TOOLSCOPE_MODEL_CACHE:=${REPO_ROOT}/eval/local/models}"

# Inside the devcontainer, docker -e sets container paths; do not let .env
# overwrite them with host-absolute TOOLSCOPE_MODEL_CACHE (breaks GGUF lookup).
_PRESERVE_MODEL_CACHE=""
if [[ "${TOOLSCOPE_DEVCONTAINER:-}" == "1" ]]; then
  _PRESERVE_MODEL_CACHE="${TOOLSCOPE_MODEL_CACHE}"
fi

# Load repo-root .env when present (OPENAI_*, TOOLSCOPE_MODEL_CACHE, HF_TOKEN).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${_PRESERVE_MODEL_CACHE}" ]]; then
  TOOLSCOPE_MODEL_CACHE="${_PRESERVE_MODEL_CACHE}"
fi

export TOOLSCOPE_MODEL_CACHE
export REPO_ROOT LOCAL_DIR MODELS_YAML CONTAINER_DIR

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: ${cmd}" >&2
    exit 1
  fi
}

container_runtime() {
  if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    echo "${CONTAINER_RUNTIME}"
    return
  fi
  if command -v podman >/dev/null 2>&1; then
    echo podman
  elif command -v docker >/dev/null 2>&1; then
    echo docker
  else
    echo "error: install podman or docker" >&2
    exit 1
  fi
}

ctr() {
  "$(container_runtime)" "$@"
}

image_exists() {
  local img="$1"
  ctr image inspect "${img}" >/dev/null 2>&1
}

container_exists() {
  local name="$1"
  ctr container inspect "${name}" >/dev/null 2>&1
}

container_gpu_args() {
  local rt
  rt="$(container_runtime)"
  if [[ "${rt}" == "docker" ]]; then
    echo --gpus all
  else
    echo --device nvidia.com/gpu=all
  fi
}

# Resolve a dotted key from models.yaml via python (stdlib yaml not guaranteed).
yaml_get() {
  local key="$1"
  python3 - "${MODELS_YAML}" "${key}" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("error: PyYAML required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

path = Path(sys.argv[1])
key = sys.argv[2]
data = yaml.safe_load(path.read_text()) or {}

def get(d, parts):
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

parts = key.split(".")
val = get(data, parts)
if val is None:
    sys.exit(2)
if isinstance(val, (dict, list)):
    import json
    print(json.dumps(val))
else:
    print(val)
PY
}

list_model_ids() {
  python3 - "${MODELS_YAML}" <<'PY'
import sys
from pathlib import Path
import yaml

data = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
for mid in (data.get("models") or {}):
    print(mid)
PY
}

list_tier_ids() {
  local tier="$1"
  python3 - "${MODELS_YAML}" "${tier}" <<'PY'
import sys
from pathlib import Path
import yaml

data = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
tier = sys.argv[2]
tiers = data.get("tiers") or {}
if tier in tiers:
    for mid in tiers[tier].get("models", []):
        print(mid)
else:
    # Fallback: filter by per-model tier field
    for mid, spec in (data.get("models") or {}).items():
        if spec.get("tier") == tier:
            print(mid)
PY
}

default_matrix_ids() {
  # Smallest-first within each tier; ceiling last.
  list_tier_ids high_sensitivity
  list_tier_ids mid_production
  list_tier_ids control_ceiling
}

resolve_model_field() {
  local model_id="$1"
  local field="$2"
  python3 - "${MODELS_YAML}" "${model_id}" "${field}" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
model_id = sys.argv[2]
field = sys.argv[3]
data = yaml.safe_load(path.read_text()) or {}
defaults = data.get("defaults") or {}
models = data.get("models") or {}
entry = models.get(model_id)
if entry is None:
    print(f"error: unknown model id: {model_id}", file=sys.stderr)
    sys.exit(1)
val = entry.get(field, defaults.get(field))
if val is None:
    sys.exit(2)
if isinstance(val, list):
    print(" ".join(str(x) for x in val))
else:
    print(val)
PY
}

find_gguf_path() {
  local model_id="$1"
  local manifest="${TOOLSCOPE_MODEL_CACHE}/manifest.json"
  if [[ -f "${manifest}" ]]; then
    local resolved
    resolved="$(python3 - "${manifest}" "${model_id}" "${TOOLSCOPE_MODEL_CACHE}" <<'PY'
import json, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
mid = sys.argv[2]
cache = Path(sys.argv[3])
for m in manifest.get("models", []):
    if m.get("id") != mid:
        continue
    raw = m.get("path", "")
    if not raw:
        break
    p = Path(raw)
    candidates = []
    if p.is_file():
        candidates.append(p)
    else:
        candidates.append(cache / p)
        repo = (m.get("hf_repo") or "").replace("/", "__")
        if repo:
            candidates.append(cache / repo / p.name)
    for c in candidates:
        if c.is_file():
            print(c)
            break
    break
PY
)" || true
    if [[ -n "${resolved}" && -f "${resolved}" ]]; then
      echo "${resolved}"
      return 0
    fi
  fi
  local glob_pattern
  glob_pattern="$(resolve_model_field "${model_id}" file_glob)"
  local picked
  picked="$(python3 - "${TOOLSCOPE_MODEL_CACHE}" "${glob_pattern}" <<'PY'
import fnmatch, re, sys
from pathlib import Path

cache = Path(sys.argv[1])
glob_pat = sys.argv[2]
matches = [
    p for p in cache.rglob("*.gguf")
    if fnmatch.fnmatch(p.name, glob_pat)
]
if not matches:
    sys.exit(1)
shard_first = [p for p in matches if re.search(r"-00001-of-\d+\.gguf$", p.name)]
if shard_first:
    print(shard_first[0])
elif len(matches) == 1:
    print(matches[0])
else:
    print(max(matches, key=lambda p: p.stat().st_size))
PY
)" || true
  if [[ -n "${picked}" && -f "${picked}" ]]; then
    echo "${picked}"
    return 0
  fi
  echo "error: no GGUF for ${model_id} under ${TOOLSCOPE_MODEL_CACHE}" >&2
  return 1
}

image_name() {
  yaml_get defaults.image 2>/dev/null || echo "toolscope-llamacpp:latest"
}

container_name() {
  yaml_get defaults.container_name 2>/dev/null || echo "toolscope-llama"
}

server_port() {
  yaml_get defaults.port 2>/dev/null || echo "8000"
}

inference_mode() {
  if [[ -n "${TOOLSCOPE_INFERENCE_MODE:-}" ]]; then
    echo "${TOOLSCOPE_INFERENCE_MODE}"
    return
  fi
  if [[ -f /etc/toolscope/devcontainer ]] || [[ "${TOOLSCOPE_DEVCONTAINER:-}" == "1" ]]; then
    echo native
    return
  fi
  echo container
}

llama_state_dir() {
  mkdir -p "${REPO_ROOT}/eval/local/.podman"
  echo "${REPO_ROOT}/eval/local/.podman"
}

llama_pid_file() {
  echo "$(llama_state_dir)/llama-server.pid"
}

llama_log_file() {
  echo "$(llama_state_dir)/llama-server.log"
}

openai_base_url() {
  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    echo "${OPENAI_BASE_URL}"
    return
  fi
  echo "http://127.0.0.1:$(server_port)/v1"
}

gguf_path_in_models_mount() {
  local model_id="$1"
  local gguf
  gguf="$(find_gguf_path "${model_id}")"
  local rel="${gguf#"${TOOLSCOPE_MODEL_CACHE}"/}"
  echo "/models/${rel}"
}
