# BFCL V4 Multiple — high-cardinality Tool RAG

Protocol: `shared_catalog` (BFCL-derived; **not** an official Gorilla leaderboard score).
Catalog size: 443 tools. k=10.

## Tool name accuracy

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| deepseek-v4-flash-0731 | 87.0% | 87.0% | 88.0% |
| qwen3.5-397b-a17b | 85.5% | 88.5% | 91.0% |
| gemini-3.7-flash | 90.0% | 88.5% | 88.5% |

## AST accuracy

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| deepseek-v4-flash-0731 | 60.5% | 63.0% | 62.5% |
| qwen3.5-397b-a17b | 64.5% | 67.5% | 67.0% |
| gemini-3.7-flash | 65.0% | 62.5% | 61.5% |

## Context compression

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| deepseek-v4-flash-0731 | 0.0% | 97.7% | 97.7% |
| qwen3.5-397b-a17b | 0.0% | 97.7% | 97.7% |
| gemini-3.7-flash | 0.0% | 97.7% | 97.7% |
