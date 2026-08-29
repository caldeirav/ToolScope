# Next experiments after the k=10 matrix

The frozen k=10 run of record lives in [`artifacts/`](artifacts/). This plan
is what to implement next. Do not start extra embedders, official Gorilla AST,
more models, or Live/multi-turn until the k-ablation lands.

## Paper claim to lock (writing, not a run)

- **Selection over injection is an interaction**, not “ToolScope always wins.”
  Qwen is the headline cell; DeepSeek is a modest/noisy middle; Gemini is the
  ceiling (already 90% name acc on 443 tools).
- Report **name accuracy** as the selection metric and **AST** as calling.
  Retrieval does not fill arguments (`bad_args` remains ~24–31% even given a
  correct name).
- Put **McNemar / CIs** on the k=10 table. Only Qwen’s +5.5 pp name-acc delta
  is significant. Do not write Gemini’s −1.5 pp as “Tool RAG hurts Gemini.”

## Experiment 0 — stats from frozen artifacts (no API)

Recompute everything below from `eval/paper/artifacts/bfcl_eval_*.json`.
Output a small appendix table (markdown + CSV) next to the artifacts.

1. **Paired tests.** Exact McNemar on name-acc flips (ToolScope vs baseline,
   BM25 vs baseline, ToolScope vs BM25). Wilson (or Clopper–Pearson) intervals
   on name acc and AST acc. Same script must reproduce the numbers already in
   [`artifacts/README.md`](artifacts/README.md).
2. **Collision sensitivity.** `tool_name_collisions.json` has 33 colliding
   names; 25 queries have a colliding GT name. Report name/AST acc on n=175
   (drop those 25) vs n=200. First-seen-wins is a catalog defect; this is a
   footnote, not a new run of record.
3. **Error taxonomy.** `wrong_tool` given recall=1 (sibling confusion) vs
   `retrieval_miss` vs `bad_args`. That is the leftover-error story.

Suggested entry point: `eval/paper/stats_from_artifacts.py` (stdlib + the
checked-in JSONs). Unit-test McNemar on a 3-row fixture.

## Experiment 1 — k-ablation (priority API run)

**Question.** Does Gemini’s wash come from an overly aggressive shortlist, and
does Qwen’s gain hold when more siblings leak back in?

| Knob | Value |
|---|---|
| k | **5, 10, 20, 40** |
| Models | **qwen3.5-397b-a17b** and **gemini-3.7-flash** required; DeepSeek optional (cheap, fills the interaction curve) |
| Catalog / queries / embedder | Same C=443, same 200 Multiple items, MiniLM-L6-v2 |
| Conditions | BM25 and ToolScope at each k. **Baseline is k-invariant — reuse the frozen k=10 baseline; do not re-bind 443 tools.** |
| k=10 cells | **Reuse** [`artifacts/`](artifacts/). Do not re-pay that matrix. |

### Nested shortlists (required)

For each query, retrieve **once** at `k_max=40`, then bind prefixes
`k ∈ {5, 20, 40}` (and skip 10). Then ToolScope@5 ⊂ @20 ⊂ @40. Independent
top-k draws would make the ablation harder to interpret.

BM25 and ToolScope still produce their own ranked lists; do not mix them.

### Cost control

- New flag `--skip-baseline` (or `--conditions BM25,ToolScope`) so
  `evaluate_instance` does not call the 443-tool prompt.
- Checkpoint signature already includes `k` (`eval/bfcl_eval/checkpoint.py`).
  Write each k to `eval/results/paper/k{k}/` via `--output-dir` so a crash at
  k=20 cannot resume into k=5.
- One process per (model, k) is acceptable if nested retrieve is too invasive;
  prefer one pass per model that scores three k values per instance (one
  retrieve, three `bind_tools`).
- Gemini mean latency on the frozen run is ~9 s/condition with a long tail.
  Budget roughly 2 models × 2 retrievers × 3 new k × 200 ≈ **2,400** LLM calls
  if nested; more if k is a separate full pass. Resume from checkpoints.

### Reporting

A table (and a plot in the paper): name acc vs k, one series per
(model × retriever), baseline as a horizontal line. At each k also report
Recall@k, NDCG@k, mean tokens, and McNemar vs baseline.

**Success.** You can answer: (a) Qwen’s ToolScope > baseline at k=5 and k=20,
or it is k=10-specific; (b) Gemini remains ≤ baseline at every k, or it
recovers as k grows.

## Experiment 2 — only if k-ablation is in

Not blockers for the current split:

- Official `bfcl-eval` AST as a sensitivity footnote (local checker is the
  run of record).
- A stronger embedder / reranker, only if sibling confusion at recall=1
  still dominates after k=20/40.
- Extra models, only to thicken the “Δ vs baseline strength” interaction.

## Implementation order

1. McNemar and the n=175 collision slice are generated into `harness_results.md` on every paper eval. Keep that file in sync by running the full protocol (it copies into `eval/paper/artifacts/`). Add Wilson CIs later if a reviewer asks.
2. `--skip-baseline` / `--conditions` in `eval/run_eval.py` and
   `evaluate_instance`.
3. Nested top-`k_max` retrieve with prefix scoring, or a thin driver
   `eval/paper/run_k_ablation.py` that loops `--k` and `--output-dir`.
4. YAML or driver defaults: Qwen + Gemini, k ∈ {5, 20, 40}, output under
   `eval/results/paper/k{k}/`.
5. Merge script: frozen k=10 + new k dirs → `eval/paper/artifacts/k-ablation.md`.
6. Run Qwen, then Gemini. Copy sanitized JSONs into
   `eval/paper/artifacts/k{k}/` the same way as k=10.

## Files likely to change

| File | Why |
|---|---|
| `eval/run_eval.py` | `--skip-baseline`, `--output-dir` already exists, k loop / exec chain |
| `eval/bfcl_eval/evaluate.py` | Skip baseline generate; optional multi-k prefix scoring |
| `eval/bfcl_eval/retrieval.py` | Retrieve `k_max`, slice to k |
| `eval/bfcl_eval/report.py` | k column; McNemar on the paper table |
| `eval/paper/bfcl_multiple_hc.yaml` | Leave k=10 as the run of record |
| `eval/paper/run_k_ablation.py` (new) | Driver so the ablation is one command |
| `tests/test_bfcl_paper.py` | skip-baseline, nested prefixes, McNemar fixture |

Keep `eval/config.yaml` (distractor-pool / local HF) working.

## Out of scope until this lands

Official Gorilla generate/eval pipeline, Live or Multi-Turn BFCL, additional
embedding models, and adding a fourth chat model.
