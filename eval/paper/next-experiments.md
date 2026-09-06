# Next experiments after the local k=10 matrix

The active run of record will live in [`artifacts/local/`](artifacts/local/) once
the DGX Spark matrix completes. Historical API-model results remain in
[`artifacts/`](artifacts/) for reference only.

## Paper claim to lock (writing, not a run)

- **Selection over injection is an interaction**, not “ToolScope always wins.”
  Expect the largest deltas on the **SLM tier** (3B–7B), where full-catalog binding
  is weakest; the **local agent tier** (32–70B) fills out the interaction curve.
- Report **name accuracy** as the selection metric and **AST** as calling.
  Retrieval does not fill arguments (`bad_args` remains high even given a
  correct name).
- Put **McNemar / CIs** on the k=10 local table. Compare against the legacy
  API interaction curve only as background — do not mix numbers across backends.

## Experiment 0 — stats from frozen artifacts (no GPU)

Recompute paired tests from `eval/paper/artifacts/local/bfcl_eval_*.json` once
the matrix lands. Output appendix tables next to the artifacts.

1. **Paired tests.** Exact McNemar on name-acc flips (ToolScope vs baseline,
   BM25 vs baseline, ToolScope vs BM25). Wilson intervals on name acc and AST acc.
2. **Collision sensitivity.** Same 33 colliding names / 25 affected queries as the
   legacy run (`tool_name_collisions.json`). Report n=175 vs n=200 slice.
3. **Error taxonomy.** `wrong_tool` given recall=1 vs `retrieval_miss` vs `bad_args`.

Suggested entry point: `eval/paper/stats_from_artifacts.py` (stdlib + checked-in JSONs).

## Experiment 1 — k-ablation (local priority)

**Question.** On the best local cell, does ToolScope’s gain hold at k ∈ {5, 20, 40},
and does a weaker local model recover as k grows?

| Knob | Value |
|---|---|
| k | **5, 10, 20, 40** |
| Models | Best local cell from the k=10 matrix + one SLM (e.g. `qwen2.5-7b-instruct`) |
| Catalog / queries / embedder | Same C=443, same 200 Multiple items, MiniLM-L6-v2 |
| Conditions | BM25 and ToolScope at each k. **Reuse frozen k=10 baseline from `artifacts/local/`.** |

### Nested shortlists (required)

Retrieve once at `k_max=40`, score prefixes k ∈ {5, 20, 40}. BM25 and ToolScope
keep separate ranked lists.

### Cost control

- `--skip-baseline` / `--conditions` in `eval/run_eval.py` (not yet implemented).
- One `run_local_matrix.sh` pass per model with nested k scoring preferred over
  three full server restarts per k.
- Checkpoints per k under `eval/results/paper/local/k{k}/`.

## Experiment 2 — only if k-ablation is in

- Official `bfcl-eval` AST as a sensitivity footnote.
- Stronger embedder / reranker if sibling confusion at recall=1 still dominates.
- Additional local quantizations for footprint sensitivity (e.g. Q5_K_M on 70B).

## Implementation order

1. Complete the local k=10 matrix via `eval/local/scripts/run_local_matrix.sh`.
2. Freeze `table.md`, `summary.csv`, `harness_results.md` into `artifacts/local/`.
3. **Flip weight purge to opt-in.** Current matrix uses purge-after-eval by default
   (`--keep-weights` to disable). After the full run lands, change the default to
   keep weights and add `--purge-after-eval` for disk-tight hosts. Full slate is
   ~80 GiB at Q4_K_M (dense models only; no GLM-4.7 MoE).
4. `--skip-baseline` / nested k driver for local k-ablation.
5. `stats_from_artifacts.py` on local JSONs.

## Out of scope until local k=10 lands

Official Gorilla generate/eval pipeline, Live or Multi-Turn BFCL, and reviving
the legacy API-model k-ablation plan.
