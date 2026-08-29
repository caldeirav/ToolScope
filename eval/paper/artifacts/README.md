# Frozen k=10 paper matrix

BFCL V4 Non-Live Multiple, shared catalog **C = 443**, **k = 10**, MiniLM-L6-v2.
**n = 200** per model, **0 skipped**. Scores are BFCL-derived, not official Gorilla numbers.

Runtime writes go to gitignored `eval/results/paper/`. This directory is the
checked-in snapshot of the run of record.

| Model | Source file (gitignored) | SHA-256 |
|---|---|---|
| deepseek-v4-flash-0731 | `bfcl_eval_deepseek-v4-flash-0731_1787876679.json` | `c2573b7a02054299f12a694265ab151a09de65c87c2cfa2645818cd6e3414d47` |
| qwen3.5-397b-a17b | `bfcl_eval_qwen3.5-397b-a17b_1787879041.json` | `dd9d842f737a2646ba6f7556ff67abfa851dcba39b0f77b8c5d478cf6d341e19` |
| gemini-3.7-flash | `bfcl_eval_gemini-3.7-flash_1787966586.json` | `3e61a5b7dfd4e72a2fbfa37dfbcb7e8894a0d362897373da531dc90821efe3ad` |

Committed JSON copies have host URLs and API-key fields stripped. Do not treat
the Aug 27 n=2 smokes or the Gemini n=0 skip run as paper cells.

## Tool name accuracy

| Model | Baseline | BM25 | ToolScope | Δ ToolScope vs baseline |
|---|---|---|---|---|
| qwen3.5-397b-a17b | 85.5% | 88.5% | **91.0%** | **+5.5 pp** (McNemar exact p = 0.013; +14 / −3) |
| deepseek-v4-flash-0731 | 87.0% | 87.0% | 88.0% | +1.0 pp (p = 0.80; +9 / −7) |
| gemini-3.7-flash | **90.0%** | 88.5% | 88.5% | −1.5 pp (p = 0.55; +4 / −7) |

## AST accuracy

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| qwen3.5-397b-a17b | 64.5% | 67.5% | 67.0% |
| deepseek-v4-flash-0731 | 60.5% | 63.0% | 62.5% |
| gemini-3.7-flash | 65.0% | 62.5% | 61.5% |

Retrieval (identical across models): BM25 Recall@10 **97.0%** / NDCG **0.881**;
ToolScope Recall@10 **98.5%** / NDCG **0.885**. Compression **97.7%**
(~60,051 → ~1,362–1,401 prompt tokens).

See [harness_results.md](harness_results.md) for the analysis (name/AST, McNemar, error taxonomy, flips). [table.md](table.md) and [summary.csv](summary.csv) are the compact matrix. Follow-up experiments: [../next-experiments.md](../next-experiments.md).
