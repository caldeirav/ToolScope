#!/usr/bin/env bash
# Remove cached GGUF weights for one or more model ids (frees disk between matrix runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODEL_FILTER=()
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [--model <id>] ... [--dry-run]

Remove downloaded weights for the given model id(s) from TOOLSCOPE_MODEL_CACHE
and drop their entries from manifest.json. Safe to run after a model's eval
checkpoint is complete when disk is tight.

Models: $(list_model_ids | tr '\n' ' ')
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_FILTER+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

IDS=()
if [[ ${#MODEL_FILTER[@]} -gt 0 ]]; then
  IDS=("${MODEL_FILTER[@]}")
else
  echo "error: specify at least one --model <id>" >&2
  usage
  exit 1
fi

python3 - "${MODELS_YAML}" "${TOOLSCOPE_MODEL_CACHE}/manifest.json" "${TOOLSCOPE_MODEL_CACHE}" \
  "${DRY_RUN}" "${IDS[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import yaml

models_yaml = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
dry_run = sys.argv[4].lower() == "true"
ids = sys.argv[5:]

data = yaml.safe_load(models_yaml.read_text()) or {}
registry = data.get("models") or {}

manifest = {"models": []}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        pass

by_id = {m["id"]: m for m in manifest.get("models", []) if "id" in m}
freed = 0

for model_id in ids:
    if model_id not in registry:
        raise SystemExit(f"unknown model id: {model_id}")
    repo = registry[model_id]["hf_repo"]
    repo_dir = cache_root / repo.replace("/", "__")
    if repo_dir.exists():
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        freed += size
        label = f"{model_id} ({repo_dir.name}, {size / (1024**3):.1f} GiB)"
        if dry_run:
            print(f"would purge {label}")
        else:
            shutil.rmtree(repo_dir)
            print(f"purged {label}")
    else:
        print(f"skip {model_id}: no cache dir {repo_dir}")

    by_id.pop(model_id, None)

if dry_run:
    print(f"would free ~{freed / (1024**3):.1f} GiB total")
    sys.exit(0)

manifest["models"] = list(by_id.values())
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"updated manifest → {manifest_path} ({len(manifest['models'])} model(s) remain)")
print(f"freed ~{freed / (1024**3):.1f} GiB")
PY
