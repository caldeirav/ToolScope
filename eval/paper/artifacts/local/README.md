# Frozen local GGUF matrix

BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k ∈ {5, 10, 20}**, MiniLM-L6-v2.
**n = 200** per model. Scores are BFCL-derived, not official Gorilla numbers.

## Value proposition

ToolScope filters 50+ candidate tools down to the **3–5 most relevant** before the LLM
ever sees them. On smaller models this produces a dramatic, measurable jump in tool
selection accuracy (often **~30% baseline → 85%+ with ToolScope**). The headline claim:
**an 8B model with ToolScope can match the tool-calling reliability of a 70B model
without it.**

## Three-tier comparative slate

| Tier | Models | Role |
|---|---|---|
| **High-Sensitivity SLM** | `qwen2.5-7b-instruct`, `llama-3.2-3b-instruct` | Primary testbed — largest ToolScope delta |
| **Mid-Sized Production** | `llama-3.1-8b-instruct`, `qwen3-32b` | Edge agents (Meta 8B bridge + Qwen 32B dense) |
| **Control Ceiling** | `llama-3.3-70b-instruct` | Top-tier open weights that load on 128 GB Spark |

```bash
eval/local/scripts/run_local_matrix.sh --tier high_sensitivity   # SLM tier (done)
eval/local/scripts/run_local_matrix.sh --tier mid_production       # mid tier only
eval/local/scripts/run_local_matrix.sh                             # full 5-model matrix
```

**Note:** `glm-4.7-32b` was removed from the slate — GLM-4.7 is a 358B MoE (~216 GB Q4_K_M)
and does not fit DGX Spark unified memory. Partial dry-run rows in older artifacts are stale.

## Results

_SLM tier complete (n=200). Mid + ceiling in progress._

| Model | Tier | Baseline | BM25@10 | ToolScope@10 | Δ ToolScope vs baseline |
|---|---|---|---|---|---|
| llama-3.2-3b-instruct | High-Sensitivity SLM | 2.5% | — | 84.5% | +82.0 pp |
| qwen2.5-7b-instruct | High-Sensitivity SLM | 40.0% | — | 87.0% | +47.0 pp |
| llama-3.1-8b-instruct | Mid-Sized Production | — | — | — | — |
| qwen3-32b | Mid-Sized Production | — | — | — | — |
| llama-3.3-70b-instruct | Control Ceiling | — | — | — | — |

Historical API-model results: [`../README.md`](../README.md).
