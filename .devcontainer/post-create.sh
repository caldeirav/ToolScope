#!/usr/bin/env bash
set -euo pipefail
cd /workspace

mkdir -p eval/local/models eval/local/.podman eval/results/paper/local

if python3 -c "import langchain_core, toolscope" 2>/dev/null; then
  echo "ToolScope + eval dependencies already installed."
else
  echo "Installing ToolScope + eval dependencies ..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -e ".[st,dev]"
  python3 -m pip install -r eval/requirements.txt
fi

if command -v llama-server >/dev/null 2>&1; then
  echo "llama-server: $(command -v llama-server) (tag ${LLAMA_CPP_TAG:-unknown})"
else
  echo "warning: llama-server not found in PATH" >&2
fi

echo "ToolScope devcontainer ready."
echo "  Inference: eval/local/scripts/serve_model.sh <model-id>"
echo "  Full matrix: eval/local/scripts/run_local_matrix.sh --samples 5"
