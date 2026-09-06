# BFCL V4 Multiple — high-cardinality Tool RAG

Protocol: `shared_catalog` (BFCL-derived; **not** an official Gorilla leaderboard score).
Catalog size: 443 tools. k=10.

## Tool name accuracy

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.0% | 85.5% | 82.5% | 84.5% | 84.5% | 83.5% |
| qwen2.5-7b-instruct | 40.0% | 83.5% | 86.0% | 88.5% | 84.5% | 87.0% | 87.5% |
| glm-4.7-32b | 0.0% | — | — | — | — | — | — |
| qwen3-32b | 0.0% | — | — | — | — | — | — |
| llama-3.3-70b-instruct | 0.0% | — | — | — | — | — | — |

## AST accuracy

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.5% | 47.0% | 44.5% | 47.5% | 46.5% | 44.0% |
| qwen2.5-7b-instruct | 23.5% | 52.0% | 53.5% | 55.0% | 51.5% | 53.5% | 56.0% |
| glm-4.7-32b | 0.0% | — | — | — | — | — | — |
| qwen3-32b | 0.0% | — | — | — | — | — | — |
| llama-3.3-70b-instruct | 0.0% | — | — | — | — | — | — |

## Context compression

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 0.0% | 98.8% | 97.7% | 95.4% | 98.9% | 97.7% | 95.5% |
| qwen2.5-7b-instruct | 0.0% | 98.8% | 97.7% | 95.4% | 98.9% | 97.7% | 95.5% |
| glm-4.7-32b | 0.0% | — | — | — | — | — | — |
| qwen3-32b | 0.0% | — | — | — | — | — | — |
| llama-3.3-70b-instruct | 0.0% | — | — | — | — | — | — |
