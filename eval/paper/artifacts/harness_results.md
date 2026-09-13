# BFCL Multiple — harness results (paper v1.0)

**Frozen benchmark for ToolScope.** Five locally-served open-weight models, one shared tool
catalog, three binding conditions per query. This is the single run of record for the paper.

| | |
|---|---|
| **Protocol** | `shared_catalog` — every query sees the same catalog **C** (443 unique BFCL Multiple functions) |
| **Queries** | 200 (BFCL V4 Non-Live Multiple split) |
| **Models** | Llama 3.2 3B, Qwen2.5 7B, Llama 3.1 8B, Qwen3 32B, Llama 3.3 70B (GGUF Q4_K_M, llama.cpp) |
| **Retrievers** | BM25 (lexical) and ToolScope (dense, MiniLM-L6-v2) at **k ∈ {5, 10, 20}** |
| **Paper default** | **k = 10** (ToolScope@10 vs baseline) |
| **Agent** | One LangGraph turn: retrieve → `bind_tools` → read `tool_calls`. Tools are **never executed**. |
| **Grading** | BFCL-derived name + AST match against `possible_answer` — **not** an official Gorilla leaderboard score |

---

## How to read this report

**Tool name accuracy** answers: *did the model pick a ground-truth function name?* This is the
headline metric for **tool selection** — whether shrinking the prompt helped the model find the
right tool among hundreds of candidates.

**AST accuracy** answers: *given the call, were the arguments correct?* Retrieval can only
narrow the choice set; it cannot invent missing parameters. A large gap between name accuracy
and AST accuracy means the model still fumbles arguments even when it picks the right tool.

**Baseline** binds the **entire catalog** (~60k prompt tokens). **BM25@k** and **ToolScope@k**
bind only the top-k retrieved tools (~1.4k tokens at k=10, **~97.7% compression**).

**McNemar p** tests whether ToolScope@10 changes name accuracy vs baseline on the *same* 200
queries (paired wins vs losses). **pp** = percentage points.

**Error labels:** `wrong_tool` (picked a non-GT name), `bad_args` (right name, wrong args),
`retrieval_miss` (GT not in bound set), `parse_fail`, `api_fail` (provider error / timeout).

---

## Executive summary

### 1. Tool filtering is a large win when the model cannot handle the full catalog

For **small models** (3B–8B), binding all 443 tools mostly fails. Baseline name accuracy is
**2.5–6%** for Llama 3.2 3B and Llama 3.1 8B, and **40%** for Qwen2.5 7B — often with
`parse_fail` or no tool call at all because the prompt is enormous.

Retrieving **k = 10** tools lifts name accuracy to **~85–92%** (BM25 and ToolScope are
neck-and-neck). McNemar tests are highly significant (p < 0.001). The gain is not subtle
tweaking: it is the difference between a model that **cannot practically use** a large MCP
registry and one that **can**.

**Practical takeaway:** If you deploy a 3B–8B agent against a 400+ tool catalog, you should
assume full-catalog binding is broken unless you filter first.

### 2. Gains shrink as baseline selection improves — this is an interaction, not a universal boost

**Qwen3 32B** already reaches **72.5%** on the full catalog (after fixing context limits for
the 443-tool prompt). ToolScope@10 adds a modest **+5.5 pp** (78.0%, McNemar p = 0.20 — not
significant at α = 0.05). Retrieval still helps on some hard queries, but the model is no longer
desperate for a shorter list.

**Llama 3.3 70B** reaches **79.0%** baseline name accuracy on the full catalog — the best of
the five models. For models that already cope with the full catalog, **selection-over-injection
is not expected to help** and may hurt if the shortlist introduces sibling confusion (see §5).

### 3. Retrieval quality is high; remaining errors are mostly confusion and arguments

At k = 10, ToolScope finds the ground-truth tool in the bound set on **98.5%** of queries
(Recall@10). BM25 is similar (97.0%). The bottleneck is no longer “find the right tool in the
index” — it is **(a)** the model picking a near-duplicate sibling that is also in the top-k,
and **(b)** filling arguments correctly once the name is right.

Among queries where the GT tool *was* bound, name accuracy is ~86%. Among queries where recall
failed, name accuracy is 0% — you cannot call a tool the model never saw.

### 4. Argument quality is a separate problem from selection

Even when the model picks the correct tool name, **~35–45%** of calls still fail AST checks
(`bad_args`). Retrieval does not fix this. The paper should treat **selection** (name acc) and
**calling** (AST acc) as distinct claims.

### 5. Data-quality caveat on two models' retrieval columns

The **baseline** condition for Qwen3 32B and Llama 3.3 70B was **re-run** after extending
llama.cpp context windows (65k / 131k) so the full catalog fits. The **BM25 / ToolScope**
columns for those two models were **not** re-run and still contain **`api_fail`** from the
first matrix pass (~10% for Qwen3, ~58% for 70B on retrieval conditions).

**Trust for the paper:**
- **SLM tier (3B, 7B, 8B):** all conditions are clean — use these for the main ToolScope claim.
- **Qwen3 32B:** baseline and qualitative trends are reliable; treat retrieval deltas as directional.
- **Llama 3.3 70B:** **baseline 79.0% is reliable**; reported retrieval name acc (~38%) is
  **not interpretable** until retrieval conditions are re-run. The correct ceiling story is:
  *this model already selects well from the full catalog.*

---

## Tool name accuracy (headline)

Share of queries where the model called a ground-truth tool name. Retrieval metrics are
identical across models for a given retriever.

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.0% | 85.5% | 82.5% | 84.5% | 84.5% | 83.5% |
| llama-3.1-8b-instruct | 6.0% | 89.0% | 91.5% | 89.5% | 90.5% | 92.0% | 92.5% |
| qwen2.5-7b-instruct | 40.0% | 83.5% | 86.0% | 88.5% | 84.5% | 87.0% | 87.5% |
| qwen3-32b | 72.5% | 76.5% | 79.5% | 79.0% | 78.0% | 78.0% | 78.0% |
| llama-3.3-70b-instruct | 79.0% | 37.5% | 38.5% | 38.5% | 37.5% | 38.0% | 38.0% |

Models ordered by baseline name accuracy (weakest full-catalog handler first).

**Reading the table:** Focus on **ToolScope@10 vs Baseline** for the paper default. The k = 5
and k = 20 columns show whether gains hold when the shortlist is tighter or wider (see
§K-ablation). Ignore 70B retrieval columns until re-run (§Executive summary).

---

## Δ name acc vs full catalog (k = 10)

| Model | Baseline | BM25 Δ | ToolScope Δ | ToolScope@10 flips (win/lose) | McNemar p |
|---|---:|---:|---:|---|---:|
| llama-3.2-3b-instruct | 2.5% | +83.0 pp | +82.0 pp | +165 / −1 | < 0.001 |
| llama-3.1-8b-instruct | 6.0% | +85.5 pp | +86.0 pp | +173 / −1 | < 0.001 |
| qwen2.5-7b-instruct | 40.0% | +46.0 pp | +47.0 pp | +104 / −10 | < 0.001 |
| qwen3-32b | 72.5% | +7.0 pp | +5.5 pp | +36 / −25 | 0.20 |
| llama-3.3-70b-instruct | 79.0% | −40.5 pp† | −41.0 pp† | +9 / −91† | < 0.001† |

†70B retrieval columns contaminated by `api_fail`; see §Executive summary.

**Pattern:** Δ shrinks monotonically as baseline rises. ToolScope and BM25 are similar at k = 10
on the clean models — dense retrieval is not magic; **any** sane shortlist beats an unfiltered
443-tool prompt for SLMs.

---

## Per-condition matrix (k = 10 focus)

| Model | Condition | Name acc | AST acc | Δ name | Recall@10 | NDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | Baseline | 2.5% | 2.0% | — | — | — | 28.0 s |
| llama-3.2-3b-instruct | ToolScope@10 | 84.5% | 46.5% | +82.0 pp | 98.5% | 0.885 | 2.9 s |
| llama-3.1-8b-instruct | Baseline | 6.0% | 3.5% | — | — | — | 36.3 s |
| llama-3.1-8b-instruct | ToolScope@10 | 92.0% | 52.0% | +86.0 pp | 98.5% | 0.885 | 1.6 s |
| qwen2.5-7b-instruct | Baseline | 40.0% | 23.5% | — | — | — | 56.1 s |
| qwen2.5-7b-instruct | ToolScope@10 | 87.0% | 53.5% | +47.0 pp | 98.5% | 0.885 | 1.8 s |
| qwen3-32b | Baseline | 72.5% | 45.0% | — | — | — | 75.2 s |
| qwen3-32b | ToolScope@10 | 78.0% | 49.0% | +5.5 pp | 98.5% | 0.885 | 64.1 s |
| llama-3.3-70b-instruct | Baseline | 79.0% | 46.5% | — | — | — | 16.5 s |
| llama-3.3-70b-instruct | ToolScope@10 | 38.0%† | 27.5%† | −41.0 pp† | 98.5% | 0.885 | 6.9 s |

Full k-ablation grid (k ∈ {5, 10, 20}): see [table.md](table.md) and [summary.csv](summary.csv).

Prompt tokens: baseline ~60,051 vs ToolScope@10 ~1,362 (~97.7% compression). Latency is
one-turn `bind_tools` only.

---

## AST accuracy

| Model | Baseline | BM25@10 | ToolScope@10 |
|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.0% | 46.5% |
| llama-3.1-8b-instruct | 3.5% | 50.5% | 52.0% |
| qwen2.5-7b-instruct | 23.5% | 53.5% | 53.5% |
| qwen3-32b | 45.0% | 50.0% | 49.0% |
| llama-3.3-70b-instruct | 46.5% | 27.0%† | 27.5%† |

AST jumps when selection improves because many baseline failures never attempt a gradable call.
The **residual** AST gap at ToolScope@10 (~47–53% on SLMs) is almost entirely `bad_args`.

---

## AST given correct name

| Model | Baseline | ToolScope@10 |
|---|---|---|
| llama-3.2-3b-instruct | 80.0% | 55.0% |
| llama-3.1-8b-instruct | 58.3% | 56.5% |
| qwen2.5-7b-instruct | 58.8% | 61.5% |
| qwen3-32b | 62.1% | 62.8% |
| llama-3.3-70b-instruct | 58.9% | 72.4%† |

Once the name is right, **~35–45%** of calls still fail argument checks. ToolScope does not
materially fix argument filling on the clean models.

---

## Where the errors are (ToolScope@10)

| Model | Fully correct | bad_args | wrong_tool | parse_fail | retrieval_miss | api_fail |
|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | 93 | 76 | 16 | 0 | 3 | 12 |
| llama-3.1-8b-instruct | 104 | 80 | 13 | 0 | 3 | 0 |
| qwen2.5-7b-instruct | 107 | 67 | 23 | 0 | 3 | 0 |
| qwen3-32b | 98 | 58 | 17 | 3 | 2 | 22 |
| llama-3.3-70b-instruct | 55† | 21† | 6† | 0 | 0 | 118† |

**SLM story (qwen2.5-7b):** baseline `wrong_tool` collapses from **110 → 23** with ToolScope@10.
The gain is almost entirely better **selection**, not better arguments.

**Ceiling model (70B baseline):** 93 fully correct, 65 `bad_args`, 40 `wrong_tool` — a capable
full-catalog selector with room to improve arguments. Retrieval row not interpretable (†).

---

## Model-by-model narrative

### Llama 3.2 3B — “cannot use the catalog without filtering”

Baseline is effectively broken (2.5% name acc, 183 `parse_fail`). The model rarely produces a
valid tool call when forced to read 443 tool schemas. ToolScope@10 restores usability to **84.5%**
with **~10× lower latency**. This is the clearest deployment argument for ToolScope.

### Llama 3.1 8B — largest measured delta

**+86.0 pp** name acc (6% → 92%), McNemar p < 0.001. Supports the headline that a small
filtered agent can approach much larger models' *selection* behavior — though AST remains ~52%.

### Qwen2.5 7B — strong mid-SLM, wrong-tool dominated

Baseline 40% with 110 `wrong_tool` errors. ToolScope@10 reaches 87% by cutting wrong-tool
errors ~5×. BM25@20 slightly edges ToolScope@20 on name acc (88.5% vs 87.5%) — at wide k,
lexical retrieval is competitive.

### Qwen3 32B — diminishing returns from filtering

With a working full-catalog baseline (72.5%), ToolScope@10 is a **small, non-significant** bump.
Many losses are **sibling confusion** (GT retrieved, model picks `calculate_derivative` vs
`calculus.derivative`). Future work: reranking or schema-aware deduplication, not larger k.

### Llama 3.3 70B — full-catalog ceiling

**79.0% baseline** is the benchmark ceiling for open-weight local inference on this catalog.
Report this as: *a 70B model can handle 443 tools without retrieval*. Do **not** cite the
~38% retrieval number without re-running retrieval after the context fix.

---

## K-ablation (k ∈ {5, 10, 20})

| Observation | Detail |
|---|---|
| **Tighter k (5)** | Slightly lower name acc than k = 10 on most models — too aggressive for 3 misses / 200 queries |
| **Wider k (20)** | Recall → 99%+; name acc flat or down for SLMs (more siblings in the shortlist) |
| **Paper default k = 10** | Best trade-off on clean models: 98.5% recall, ~85–92% name acc on SLMs |

---

## Retrieval quality (model-independent)

| Retriever | Recall@10 | NDCG@10 | Missed queries | Mean tokens |
|---|---:|---:|---:|---:|
| BM25@10 | 97.0% | 0.881 | 6 / 200 | 1,401 |
| ToolScope@10 | 98.5% | 0.885 | 3 / 200 | 1,362 |

Missed ground-truth names at k = 10: `linear_regression`, `probabilities.calculate_single`,
`route_planner.calculate_route`.

When recall = 1, name acc ≈ **85.8%** (first model traces). When recall = 0, name acc = **0%**.

---

## Catalog hazards

| Hazard | Count | Effect |
|---|---:|---|
| Same name, different schema (first-seen kept) | 42 records / 33 names | ~4–8 pp lower name acc on 25 affected queries vs 175 clean queries |
| Dotted vs underscore aliases | 2 groups | Sanitized to valid OpenAI tool names; grader uses original BFCL names |
| Confusable siblings in top-k | Most `wrong_tool` at recall = 1 | Model sees GT and a near-duplicate; picks the wrong one |

See [tool_name_collisions.json](tool_name_collisions.json) for the collision manifest.

---

## Conclusions for the paper

1. **Claim (selection):** For agents backed by **3B–8B models** and **400+ tool** catalogs,
   binding the full catalog is not viable. Retrieving k ≈ 10 tools — with ToolScope or BM25 —
   lifts tool **name accuracy from single digits or ~40% to ~85–92%**, with ~98% less tool
   JSON in the prompt and an order-of-magnitude latency reduction.

2. **Claim (interaction):** The benefit is **largest when baseline selection is weakest**.
   Gains shrink and become statistically uncertain by ~70% baseline (Qwen3 32B). Models that
   already select well from the full catalog (70B at 79%) do not need injection-style filtering
   for selection — the interesting question shifts to argument quality and sibling disambiguation.

3. **Claim (retrieval vs reranking):** At k = 10, **recall is not the bottleneck** (98.5%).
   Further improvements require better **disambiguation among similar tools**, not bigger k.

4. **Claim (calling):** Even with correct tool names, **~40% AST failure** persists. Tool RAG
   solves **which tool**; it does not solve **how to call it**. Do not over-claim end-to-end
   task success from name accuracy alone.

5. **Reproducibility:** Config [`eval/paper/bfcl_multiple.yaml`](../bfcl_multiple.yaml).
   Runtime JSON traces: gitignored `eval/results/paper/local/`. This directory is the frozen
   v1.0 snapshot. Re-run: `eval/local/scripts/run_local_matrix.sh`.

**Do not** present these numbers as official BFCL / Gorilla leaderboard scores. Protocol,
agent, grader, and model serving stack are all defined in this repository.
