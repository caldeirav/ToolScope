# Frozen local GGUF k=10 matrix



BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k = 10** (anchor; k-ablation
{k ∈ 5, 10, 20} in [`harness_results.md`](harness_results.md)), MiniLM-L6-v2.
**n = 200** per completed model. Scores are BFCL-derived, not official Gorilla numbers.

Runtime writes go to gitignored `eval/results/paper/local/`. This directory is the
checked-in snapshot of the run of record (same layout as the historical API matrix in
[`../README.md`](../README.md)).

| Model | Source file (gitignored) | SHA-256 |
|---|---|---|
| llama-3.2-3b-instruct | `bfcl_eval_llama-3.2-3b-instruct_1788642468.json` | `8ca48c218af824ee6369a1c55471621e7b8492b65dff3121a5374fd20ecce2d6` |
| qwen2.5-7b-instruct | `bfcl_eval_qwen2.5-7b-instruct_1788656165.json` | `1ce620d6bacc9bd0d77ad2d97fd1403c16bd5339bf22025bbefd51b726bdb45d` |
| llama-3.1-8b-instruct | `bfcl_eval_llama-3.1-8b-instruct_1788679598.json` | `24cd33f6ae875b71df8e22fbb4d3838660b447a99558a12cc5340e2bd411fff1` |
| qwen3-32b | `bfcl_eval_qwen3-32b_1789275906.json` | `5e608768f2aa862c4f85ea643cb9afc203a47a99acc50fe209df945f19a56f6b` |
| llama-3.3-70b-instruct | `bfcl_eval_llama-3.3-70b-instruct_1789279888.json` | `36d17d93f2985951bbb4d4f7a8611c87680f4edc9b9e4abf34099d31e651ae76` |

## Tool name accuracy

| Model | Baseline | BM25 | ToolScope | Δ ToolScope vs baseline |
|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.5% | **84.5%** | **+82.0 pp** (McNemar exact p = < 0.001; +165 / −1) |
| llama-3.1-8b-instruct | 6.0% | 91.5% | **92.0%** | **+86.0 pp** (McNemar exact p = < 0.001; +173 / −1) |
| qwen2.5-7b-instruct | 40.0% | 86.0% | **87.0%** | **+47.0 pp** (McNemar exact p = < 0.001; +104 / −10) |
| qwen3-32b | 72.5% | 79.5% | **78.0%** | **+5.5 pp** (McNemar exact p = 0.20; +36 / −25) |
| llama-3.3-70b-instruct | 79.0% | 38.5% | **38.0%** | **-41.0 pp** (McNemar exact p = < 0.001; +9 / −91) |

BM25 / ToolScope columns are **@k=10** (`BM25@10`, `ToolScope@10` in the full matrix).

## AST accuracy

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.0% | 46.5% |
| llama-3.1-8b-instruct | 3.5% | 50.5% | 52.0% |
| qwen2.5-7b-instruct | 23.5% | 53.5% | 53.5% |
| qwen3-32b | 45.0% | 50.0% | 49.0% |
| llama-3.3-70b-instruct | 46.5% | 27.0% | 27.5% |

Retrieval (identical across models): BM25 Recall@10 **97.0%** / NDCG **0.881**; ToolScope Recall@10 **98.5%** / NDCG **0.885**. Compression **97.7%** (~60,051 → ~1,362 prompt tokens at k=10).

See [harness_results.md](harness_results.md) for the analysis (name/AST, McNemar, error taxonomy, flips). [table.md](table.md) and [summary.csv](summary.csv) are the compact matrix (includes k-ablation columns). Historical API-model results: [`../README.md`](../README.md). Follow-up experiments: [../../next-experiments.md](../../next-experiments.md).
