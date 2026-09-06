# High-cardinality paper eval (BFCL V4 Multiple)

This protocol asks:

> Can **locally-runnable models** — from high-sensitivity SLMs (3B–8B) through mid-sized
> production models (32B) to control-ceiling agents (70B) — handle a 443-tool catalog,
> and does **selecting k=10 with ToolScope** beat binding the full catalog?
>
> **Headline claim:** ToolScope filters 50+ tools to the 3–5 most relevant before the
> LLM sees them, often lifting selection accuracy from ~30% to 85%+ on smaller models —
> enabling an 8B model to match 70B tool-calling reliability.

Scores are **BFCL-derived**. They are not official Gorilla leaderboard numbers: the catalog, the agent, and the grader all live in this repo.

The shared runner is documented in [`eval/README.md`](../README.md). **Local GGUF serving** on DGX Spark is documented in [`eval/local/README.md`](../local/README.md).

---

## Design

Every query uses the **same catalog C**: the unique function definitions in the Multiple split (first-seen schema wins if a name appears with two signatures). Baseline, BM25, and ToolScope all draw from C. That is the difference from the default eval, which builds a smaller per-instance distractor pool.

| Condition | What the model can call |
|---|---|
| **Baseline** | All of C (high-cardinality prompt) |
| **BM25** | Top-k from C (sparse lexical retrieval) |
| **ToolScope** | Top-k from C via `ToolSelector` (LangChain adapter, MiniLM by default) |

`k` defaults to **10**. Retrieval metrics (Recall@k, NDCG@k) are identical across models for a given retriever: the embedder and BM25 index do not depend on the LLM.

**Agent.** One LangGraph turn: retrieve → `bind_tools` → read `AIMessage.tool_calls`. Tools are **never executed**. Native tool calling is required; there is no JSON-prompt fallback.

**Grading.**

- **Name accuracy** — was a ground-truth function named? This is the selection-quality headline.
- **AST accuracy** — do arguments match BFCL `possible_answer` (any-of lists; optional args may be omitted)?
- **Compression** — how much tool JSON disappeared relative to baseline.

If the provider rejects a call for one condition, that condition is scored fail-closed (`api_fail` / no call). The other conditions on the same query still run.

---

## Catalog quirks (worth knowing)

BFCL names are entry-local. Across the split you will see:

- **Same name, different schema** — recorded in `tool_name_collisions.json`. The first definition is kept in C.
- **Dotted vs underscored aliases** (for example `car.rental` and `car_rental`) — unique in C, but OpenAI/Gemini tool names must match `^[A-Za-z0-9_-]{1,64}$`. Before `bind_tools`, the harness keeps the first tool per **sanitized** name and stores `original_name` in metadata so the grader still uses BFCL names.

---

## Setup

From the ToolScope repository root:

```bash
pip install -e ".[st]"
pip install -r eval/requirements.txt
pip install -r eval/paper/requirements.txt
```

Create a gitignored `.env` in the repo root. Copy [`.env.example`](../../.env.example):

```
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
TOOLSCOPE_MODEL_CACHE=eval/local/models
```

For **local GGUF models** (primary path), use the automated matrix:

```bash
eval/local/scripts/run_local_matrix.sh
```

For **hosted API models** (legacy reference run), set `OPENAI_BASE_URL` / `OPENAI_API_KEY` to your `/v1` host and use [`bfcl_multiple_hc.yaml`](bfcl_multiple_hc.yaml).

---

## Pointing at your models

The paper path uses `backend: langchain`.

- **Local llama.cpp** (primary) — serve GGUF via [`eval/local/`](../local/). `ChatOpenAI` talks to `http://127.0.0.1:8000/v1`. Config: [`eval/local/bfcl_multiple_local.yaml`](../local/bfcl_multiple_local.yaml).
- **OpenAI-compatible** (`provider: openai`, the default) — `ChatOpenAI` against `OPENAI_BASE_URL` (vLLM, cloud gateway, llama.cpp, …).
- **Google** (`provider: google`) — `ChatGoogleGenerativeAI`. Legacy API run only.

The active local config lists five GGUF models in two tiers:

```yaml
# eval/local/bfcl_multiple_local.yaml
model:
  defaults:
    backend: langchain
    provider: openai
    base_url: http://127.0.0.1:8000/v1
  entries:
    # SLM tier
    - name: llama-3.2-3b-instruct
    - name: qwen2.5-7b-instruct
    # Local agent tier
    - name: glm-4.7-32b
    - name: qwen3-32b
    - name: llama-3.3-70b-instruct
```

The legacy [`bfcl_multiple_hc.yaml`](bfcl_multiple_hc.yaml) (DeepSeek / Qwen397B / Gemini API run) remains for historical comparison. Frozen API results: [`artifacts/`](artifacts/). New local results: [`artifacts/local/`](artifacts/local/).

Other knobs in that file:

- `dataset.protocol: shared_catalog` and `pool_size: null` — use all of C
- `dataset.samples: null` — all Multiple items (200)
- `retrievers: [BM25, ToolScope]`
- `toolscope.k: 10` and `sentence-transformers/all-MiniLM-L6-v2`

---

## Run

```bash
# Full local matrix (llama.cpp on DGX Spark)
eval/local/scripts/run_local_matrix.sh

# Pilot: five queries per model
eval/local/scripts/run_local_matrix.sh --samples 5

# Config wiring only (no GPU / no server)
eval/local/scripts/run_local_matrix.sh --dry-run --samples 20

# Single local model (server must already be running)
python eval/run_eval.py --config eval/local/bfcl_multiple_local.yaml --model qwen3-32b

# Legacy API YAML (historical)
python eval/run_eval.py --config eval/paper/bfcl_multiple_hc.yaml --dry-run --samples 20
```

Checkpoints resume automatically (the key includes protocol and `|C|`). `--no-resume` starts that model from scratch. After each model the process execs a fresh interpreter so the next model starts with a clean memory budget.

---

## Outputs

Runtime files go to gitignored `eval/results/paper/`:

| File | Contents |
|---|---|
| `bfcl_eval_{model}_{timestamp}.json` | Per-query traces and aggregates |
| `summary.csv` | One row per (model × condition) |
| `table.md` | Name accuracy, AST accuracy, compression |
| `harness_results.md` | Full analysis (name/AST, McNemar, error taxonomy, flips, catalog hazards) |
| `tool_name_collisions.json` | Duplicate BFCL names with differing schemas |
| `checkpoints/*.jsonl` | Resume log |
| `logs/*_errors.log` | Instances that threw before fail-close |

A single-model rerun still **merges** sibling result JSONs into `summary.csv` / `table.md` / `harness_results.md`, so you can finish Gemini without re-running the OpenAI-compatible models.

A **full** paper run (not `--dry-run`, not `--samples`) also copies `table.md`, `summary.csv`, and `harness_results.md` to the configured `output.versioned_dir`:

- Local GGUF runs → [`artifacts/local/`](artifacts/local/)
- Legacy API runs → [`artifacts/`](artifacts/)

That is the git-tracked snapshot. Per-model JSON traces stay gitignored under `eval/results/` because they can contain host URLs.

---

## Reading the table

- **Name accuracy** is the claim about *selection*. If ToolScope beats baseline here, shrinking the prompt helped the model pick the right tool.
- **AST accuracy** is *arguments*. Retrieval cannot invent a missing field; a large name–AST gap means the model still fumbles parameters.
- **Recall@k ≈ 1** with **name accuracy ≪ 1** means the right tool was bound and the model chose a near-duplicate instead.
- Compression near `1 − k/|C|` is expected (k=10 over a few hundred tools).

Do not mix these numbers with the official BFCL generate/eval pipeline or with the default distractor-pool protocol in `eval/config.yaml`.
