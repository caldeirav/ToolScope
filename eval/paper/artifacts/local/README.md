# Frozen k=10 local GGUF matrix

BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k = 10**, MiniLM-L6-v2.
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
| **Mid-Sized Production** | `glm-4.7-32b`, `qwen3-32b` | Fast deployment sweet spot — does ToolScope lift 32B to frontier? |
| **Control Ceiling** | `llama-3.3-70b-instruct` | Top-tier open weights — speed, context savings, precision at ceiling |

```bash
eval/local/scripts/run_local_matrix.sh --tier high_sensitivity   # pilot SLM tier
eval/local/scripts/run_local_matrix.sh                         # full 5-model matrix
```

## Results

_Fill in after the first full matrix run._

| Model | Tier | Baseline | BM25 | ToolScope | Δ ToolScope vs baseline |
|---|---|---|---|---|---|
| llama-3.2-3b-instruct | High-Sensitivity SLM | — | — | — | — |
| qwen2.5-7b-instruct | High-Sensitivity SLM | — | — | — | — |
| glm-4.7-32b | Mid-Sized Production | — | — | — | — |
| qwen3-32b | Mid-Sized Production | — | — | — | — |
| llama-3.3-70b-instruct | Control Ceiling | — | — | — | — |

Historical API-model results: [`../README.md`](../README.md).
