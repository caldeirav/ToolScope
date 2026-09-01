# High-cardinality paper eval (BFCL V4 Multiple)

This protocol asks a simple question:

> If the model is allowed to see **every** unique tool in BFCL V4 Non-Live **Multiple** (~400+ functions), does **selecting k of them** with Tool RAG beat binding the full catalog?

Scores are **BFCL-derived**. They are not official Gorilla leaderboard numbers: the catalog, the agent, and the grader all live in this repo.

The shared runner is documented in [`eval/README.md`](../README.md). This page is the experiment design.

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
OPENAI_BASE_URL=https://your-host/v1
OPENAI_API_KEY=...
GOOGLE_API_KEY=...          # only if you run the Gemini entry
```

Switching hosts is those two `OPENAI_*` lines. The runner loads `.env` on start.

---

## Pointing at your models

The paper path uses `backend: langchain`.

- **OpenAI-compatible** (`provider: openai`, the default) — `ChatOpenAI` against whatever `OPENAI_BASE_URL` is (or a per-entry `base_url`). Bearer token is `OPENAI_API_KEY`.
- **Google** (`provider: google`) — `ChatGoogleGenerativeAI`. `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).

A YAML that defers URL and key to `.env`:

```yaml
model:
  defaults:
    backend: langchain
    provider: openai
    api_key_env: OPENAI_API_KEY
    max_new_tokens: 512
  entries:
    - name: your-model-id
```

The checked-in [`bfcl_multiple_hc.yaml`](bfcl_multiple_hc.yaml) is that pattern: two OpenAI-compatible chat models plus Gemini. Point `.env` at your `/v1` server and change `entries` as needed.

Other knobs in that file:

- `dataset.protocol: shared_catalog` and `pool_size: null` — use all of C
- `dataset.samples: null` — all Multiple items (200)
- `retrievers: [BM25, ToolScope]`
- `toolscope.k: 10` and `sentence-transformers/all-MiniLM-L6-v2`

---

## Run

```bash
# Full matrix from the paper YAML
python eval/run_eval.py --config eval/paper/bfcl_multiple_hc.yaml

# Dummy model, no keys
python eval/run_eval.py --config eval/paper/bfcl_multiple_hc.yaml --dry-run --samples 20

# Live APIs, five queries
python eval/run_eval.py --config eval/paper/bfcl_multiple_hc.yaml --samples 5

# One model from the YAML
python eval/run_eval.py --config eval/paper/bfcl_multiple_hc.yaml --model gemini-3.7-flash
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

A **full** paper run (not `--dry-run`, not `--samples`) also copies `table.md`, `summary.csv`, and `harness_results.md` to [`artifacts/`](artifacts/) (`output.versioned_dir`). That is the git-tracked snapshot. Per-model JSON traces stay gitignored under `eval/results/paper/` because they can contain host URLs.

---

## Reading the table

- **Name accuracy** is the claim about *selection*. If ToolScope beats baseline here, shrinking the prompt helped the model pick the right tool.
- **AST accuracy** is *arguments*. Retrieval cannot invent a missing field; a large name–AST gap means the model still fumbles parameters.
- **Recall@k ≈ 1** with **name accuracy ≪ 1** means the right tool was bound and the model chose a near-duplicate instead.
- Compression near `1 − k/|C|` is expected (k=10 over a few hundred tools).

Do not mix these numbers with the official BFCL generate/eval pipeline or with the default distractor-pool protocol in `eval/config.yaml`.
