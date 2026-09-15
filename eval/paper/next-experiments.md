# Next experiments after paper v1.0

The frozen v1.0 matrix lives in [`artifacts/`](artifacts/). See
[`harness_results.md`](artifacts/harness_results.md) for conclusions.

## Immediate follow-up (data quality)

**Re-run BM25 / ToolScope conditions** for `qwen3-32b` and `llama-3.3-70b-instruct`.
Their baselines were fixed after extending context windows, but retrieval columns still carry
`api_fail` from the first matrix pass. Until re-run, cite 70B **baseline only** (79.0% name acc).

## Paper claims to lock (writing)

- **Selection over injection is an interaction**, not “ToolScope always wins.” Largest deltas on
  the SLM tier (3B–8B); diminishing returns by 32B; 70B already handles the full catalog.
- Report **name accuracy** as selection and **AST** as calling. Retrieval does not fix `bad_args`.
- Add **McNemar / Wilson CIs** on the v1.0 table (appendix).

## Experiment 0 — stats from frozen artifacts (no GPU)

Recompute paired tests from `eval/results/paper/local/bfcl_eval_*.json`:

1. Exact McNemar on name-acc flips (ToolScope vs baseline, BM25 vs baseline).
2. Wilson intervals on name acc and AST acc.
3. Collision sensitivity (33 colliding names / 25 affected queries).
4. Error taxonomy: `wrong_tool` given recall=1 vs `retrieval_miss` vs `bad_args`.

Suggested entry point: `eval/paper/stats_from_artifacts.py` (not yet implemented).

## Experiment 1 — k-ablation extension

**Question.** Does ToolScope's gain hold at k ∈ {40}, and does a weaker model recover as k grows?

| Knob | Value |
|---|---|
| k | 5, 10, 20 (done); optionally 40 |
| Models | Best SLM cell + qwen2.5-7b-instruct |
| Baseline | Reuse frozen v1.0 baseline from `artifacts/` |

Nested shortlists: retrieve once at `k_max`, score prefixes. Checkpoints under
`eval/results/paper/local/k{k}/`.

## Experiment 2 — optional

- Official `bfcl-eval` AST as sensitivity footnote.
- Stronger embedder / cross-encoder reranker for sibling confusion.
- Additional quantizations (Q5_K_M on 70B).

## Out of scope

Official Gorilla generate/eval pipeline, Live or Multi-Turn BFCL.
