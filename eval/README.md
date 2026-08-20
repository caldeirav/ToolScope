# BFCL Evaluation

Measures LLM tool-calling accuracy on the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) dataset, comparing two conditions:

- **Baseline** — the model receives all tools in the pool and must pick the right one.
- **ToolScope** — ToolScope filters the pool to the top-k most query-relevant tools; the model sees only those.

The evaluation uses BFCL's AST-based methodology: no real API calls are made. Tools and ground-truth answers are static definitions.

---

## Setup

From the project root:

```bash
# 1. Install ToolScope (with sentence-transformers support)
pip install -e ".[st]"

# 2. Install eval dependencies
pip install -r eval/requirements.txt
```

BFCL data is downloaded automatically on first run and cached in `eval/.bfcl_cache/`.

---

## Running

All commands run from the **project root**:

```bash
# Full run with default config (requires a GPU and ~14 GB VRAM for the default model)
python eval/run_eval.py

# Test the pipeline without a model or GPU
python eval/run_eval.py --dry-run --samples 20

# Override model and key parameters
python eval/run_eval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --samples 200 \
  --pool-size 150 \
  --k 10

# Add more categories (simple has 1 tool/entry; multiple has 2-5)
python eval/run_eval.py --category simple multiple

# Print per-instance details
python eval/run_eval.py --verbose --samples 20 --dry-run

# 4-bit quantization for low-VRAM setups (requires bitsandbytes)
# Edit config.yaml: load_in_4bit: true   (currently not a CLI flag)
```

---

## Configuration

`eval/config.yaml` controls all defaults. CLI flags override individual values.

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `name` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `model` | `device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `model` | `dtype` | `auto` | `auto` \| `float16` \| `bfloat16` \| `float32` |
| `model` | `max_new_tokens` | `256` | Max tokens for tool-call generation |
| `dataset` | `categories` | `[multiple]` | BFCL categories to include |
| `dataset` | `samples` | `100` | Instances to evaluate (`null` = all) |
| `dataset` | `pool_size` | `100` | Total tools in the distractor pool |
| `dataset` | `seed` | `42` | Random seed for pool shuffling and sampling |
| `toolscope` | `k` | `5` | Tools ToolScope selects per query |
| `toolscope.embedding` | `model` | `all-MiniLM-L6-v2` | Sentence-Transformers model |

---

## How It Works

1. BFCL entries are loaded and parsed; each entry has a user query, a set of function definitions, and a ground-truth function call.
2. A **global tool pool** is built by collecting unique function definitions across all loaded entries. Functions referenced in any ground truth are always included; the remaining slots are filled with distractor functions from other entries.
3. The **ToolScope index** is built once over this pool using sentence-transformers embeddings.
4. For each instance:
   - **Baseline**: model receives all `pool_size` tools and generates a tool call.
   - **ToolScope**: `ts_index.filter(query, k=k)` returns the top-k tools; model receives only those.
5. Predictions are parsed from the model output and compared against ground truth.

---

## Metrics

| Metric | Description |
|---|---|
| **Tool name accuracy** | % of instances where the model named the correct function |
| **Exact match** | % where name is correct AND all required arguments match |
| **ToolScope recall@k** | % of instances where the correct tool appeared in ToolScope's top-k (a retrieval quality metric — values below ~90% indicate k or the embedding model should be improved) |

Deltas show ToolScope − Baseline in percentage points (pp).

---

## Results

Results are saved as JSON to `eval/results/bfcl_eval_{timestamp}.json`. Each file contains:
- `config`: the full configuration used
- `metrics`: aggregate numbers
- `instances`: per-instance breakdown including raw model output, predictions, and per-instance metrics

---

## Notes

- The BFCL **multiple** category is the most informative: each entry has 2–5 function candidates, giving the model genuine choice. After pooling across entries, the model must pick from ~100 tools — a much harder task.
- `--dry-run` uses a dummy model that always picks the first tool in its list. It validates the full pipeline (data loading, indexing, evaluation, reporting) without any GPU or model download.
- Models that support native tool calling (Qwen2.5, Llama-3.1, Mistral-v0.3) via `apply_chat_template(tools=...)` are used in that mode automatically; others fall back to a JSON system-prompt approach.
