#!/usr/bin/env bash
# Download GGUF weights listed in eval/local/models.yaml into TOOLSCOPE_MODEL_CACHE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

MODEL_FILTER=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_FILTER+=("$2")
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model <id>] ..."
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

require_cmd python3
mkdir -p "${TOOLSCOPE_MODEL_CACHE}"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
else
  echo "error: install huggingface_hub (pip install huggingface_hub) for hf/huggingface-cli" >&2
  exit 1
fi

IDS=()
if [[ ${#MODEL_FILTER[@]} -gt 0 ]]; then
  IDS=("${MODEL_FILTER[@]}")
else
  mapfile -t IDS < <(list_model_ids)
fi

MANIFEST="${TOOLSCOPE_MODEL_CACHE}/manifest.json"
python3 - "${MODELS_YAML}" "${MANIFEST}" "${TOOLSCOPE_MODEL_CACHE}" "${IDS[@]}" <<'PY'
import fnmatch
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

models_yaml = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
filter_ids = sys.argv[4:]

data = yaml.safe_load(models_yaml.read_text()) or {}
registry = data.get("models") or {}

if filter_ids:
    registry = {k: v for k, v in registry.items() if k in filter_ids}
    missing = set(filter_ids) - set(registry)
    if missing:
        raise SystemExit(f"unknown model id(s): {', '.join(sorted(missing))}")

manifest = {"updated_at": datetime.now(timezone.utc).isoformat(), "models": []}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        pass

existing = {m["id"]: m for m in manifest.get("models", []) if "id" in m}
updated_models = []

hf_cli = "hf" if __import__("shutil").which("hf") else "huggingface-cli"

def _valid_token(token: str | None) -> str | None:
    if not token:
        return None
    t = token.strip()
    if not t or t in {"hf_your_token_here", "your_token_here"}:
        return None
    return t

for model_id, spec in registry.items():
    repo = spec["hf_repo"]
    glob_pat = spec["file_glob"]
    dest_dir = cache_root / repo.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)

    include = spec.get("hf_include") or (
        glob_pat if glob_pat.startswith("*") else f"*{glob_pat}*"
    )
    cmd = [
        hf_cli, "download", repo,
        "--include", include,
        "--local-dir", str(dest_dir),
    ]
    token = _valid_token(
        __import__("os").environ.get("HF_TOKEN")
        or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if token:
        cmd.extend(["--token", token])

    print(f"Downloading {model_id} from {repo} ({include}) via {hf_cli} ...")
    subprocess.run(cmd, check=True)

    matches = sorted(dest_dir.rglob("*.gguf"))
    matches = [p for p in matches if fnmatch.fnmatch(p.name, glob_pat)]
    if not matches:
        raise SystemExit(f"No GGUF matching {glob_pat!r} under {dest_dir}")

    import re

    def _pick_gguf(cands: list[Path]) -> Path:
        if len(cands) == 1:
            return cands[0]
        shard_first = [
            p for p in cands if re.search(r"-00001-of-\d+\.gguf$", p.name)
        ]
        if shard_first:
            return shard_first[0]
        return max(cands, key=lambda p: p.stat().st_size)

    gguf = _pick_gguf(matches)
    h = hashlib.sha256()
    with open(gguf, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    entry = {
        "id": model_id,
        "alias": spec.get("alias", model_id),
        "hf_repo": repo,
        "file_glob": glob_pat,
        "path": str(gguf.relative_to(cache_root)),
        "bytes": gguf.stat().st_size,
        "sha256": h.hexdigest(),
    }
    updated_models.append(entry)
    existing[model_id] = entry
    print(f"  → {gguf} ({entry['bytes']:,} bytes, sha256={entry['sha256'][:12]}…)")

manifest["models"] = list(existing.values())
manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote manifest → {manifest_path}")
PY
