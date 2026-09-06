# BFCL Multiple — harness results

Shared catalog C = 443 tools, 200 queries, k = 10, sentence-transformers/all-MiniLM-L6-v2.
Protocol: `shared_catalog`. BFCL-derived; **not** an official Gorilla leaderboard score.

| | |
|---|---|
| Queries scored | 200 per model |
| Models | 3 |
| Catalog C | 443 tools |
| Context compression at k=10 | 97.7% |
| Largest name-acc Δ vs baseline | +86.0 pp (llama-3.1-8b-instruct, ToolScope@10) |
| Instances skipped | 0 |

---

Skipped instances: **0**. `api_fail` on at least one condition: **22** queries across the matrix.

## Tool name accuracy (headline)

Share of queries where the model called a ground-truth tool name. Retrieval metrics are identical across models for a given retriever.

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.0% | 85.5% | 82.5% | 84.5% | 84.5% | 83.5% |
| llama-3.1-8b-instruct | 6.0% | 89.0% | 91.5% | 89.5% | 90.5% | 92.0% | 92.5% |
| qwen2.5-7b-instruct | 40.0% | 83.5% | 86.0% | 88.5% | 84.5% | 87.0% | 87.5% |

Models ordered by baseline name accuracy (weakest catalog handler first).

## Δ name acc vs full catalog

Selection gain shrinks as baseline name accuracy rises. McNemar is exact two-sided on paired name-acc flips (ToolScope@10 vs baseline).

| Model | Baseline name acc | BM25 Δ | ToolScope Δ | ToolScope@10 flips (win/lose) | McNemar p |
|---|---:|---:|---:|---|---:|
| llama-3.2-3b-instruct | 2.5% | +83.0 pp | +82.0 pp | +165 / −1 | < 0.001 |
| llama-3.1-8b-instruct | 6.0% | +85.5 pp | +86.0 pp | +173 / −1 | < 0.001 |
| qwen2.5-7b-instruct | 40.0% | +46.0 pp | +47.0 pp | +104 / −10 | < 0.001 |

## Per-condition matrix

| Model | Condition | Name acc | AST acc | Δ name | Recall@10 | NDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | Baseline | 2.5% | 2.0% | — | — | — | 28.0 s |
| llama-3.2-3b-instruct | BM25@5 | 85.0% | 47.5% | +82.5 pp | 95.0% | 0.874 | 1.8 s |
| llama-3.2-3b-instruct | BM25@10 | 85.5% | 47.0% | +83.0 pp | 97.0% | 0.881 | 2.7 s |
| llama-3.2-3b-instruct | BM25@20 | 82.5% | 44.5% | +80.0 pp | 99.0% | 0.886 | 3.6 s |
| llama-3.2-3b-instruct | ToolScope@5 | 84.5% | 47.5% | +82.0 pp | 96.0% | 0.877 | 1.7 s |
| llama-3.2-3b-instruct | ToolScope@10 | 84.5% | 46.5% | +82.0 pp | 98.5% | 0.885 | 2.9 s |
| llama-3.2-3b-instruct | ToolScope@20 | 83.5% | 44.0% | +81.0 pp | 99.5% | 0.888 | 3.4 s |
| llama-3.1-8b-instruct | Baseline | 6.0% | 3.5% | — | — | — | 36.3 s |
| llama-3.1-8b-instruct | BM25@5 | 89.0% | 49.5% | +83.0 pp | 95.0% | 0.874 | 1.5 s |
| llama-3.1-8b-instruct | BM25@10 | 91.5% | 50.5% | +85.5 pp | 97.0% | 0.881 | 1.6 s |
| llama-3.1-8b-instruct | BM25@20 | 89.5% | 49.0% | +83.5 pp | 99.0% | 0.886 | 2.5 s |
| llama-3.1-8b-instruct | ToolScope@5 | 90.5% | 49.0% | +84.5 pp | 96.0% | 0.877 | 1.4 s |
| llama-3.1-8b-instruct | ToolScope@10 | 92.0% | 52.0% | +86.0 pp | 98.5% | 0.885 | 1.6 s |
| llama-3.1-8b-instruct | ToolScope@20 | 92.5% | 52.0% | +86.5 pp | 99.5% | 0.888 | 2.4 s |
| qwen2.5-7b-instruct | Baseline | 40.0% | 23.5% | — | — | — | 56.1 s |
| qwen2.5-7b-instruct | BM25@5 | 83.5% | 52.0% | +43.5 pp | 95.0% | 0.874 | 1.8 s |
| qwen2.5-7b-instruct | BM25@10 | 86.0% | 53.5% | +46.0 pp | 97.0% | 0.881 | 1.9 s |
| qwen2.5-7b-instruct | BM25@20 | 88.5% | 55.0% | +48.5 pp | 99.0% | 0.886 | 2.4 s |
| qwen2.5-7b-instruct | ToolScope@5 | 84.5% | 51.5% | +44.5 pp | 96.0% | 0.877 | 1.7 s |
| qwen2.5-7b-instruct | ToolScope@10 | 87.0% | 53.5% | +47.0 pp | 98.5% | 0.885 | 1.8 s |
| qwen2.5-7b-instruct | ToolScope@20 | 87.5% | 56.0% | +47.5 pp | 99.5% | 0.888 | 2.2 s |

Prompt tokens: baseline ~60,051 vs BM25@5 ~699, BM25@10 ~1,401, BM25@20 ~2,789, ToolScope@5 ~683, ToolScope@10 ~1,362, ToolScope@20 ~2,700 (~97.7% compression). Latency is one-turn `bind_tools` only; tools are never executed.

## AST accuracy

Name selection does not close the AST gap. Leftover error after a correct name is almost entirely `bad_args`.

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.5% | 47.0% | 44.5% | 47.5% | 46.5% | 44.0% |
| llama-3.1-8b-instruct | 3.5% | 49.5% | 50.5% | 49.0% | 49.0% | 52.0% | 52.0% |
| qwen2.5-7b-instruct | 23.5% | 52.0% | 53.5% | 55.0% | 51.5% | 53.5% | 56.0% |

## AST given correct name

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 80.0% | 55.9% | 55.0% | 53.9% | 56.2% | 55.0% | 52.7% |
| llama-3.1-8b-instruct | 58.3% | 55.6% | 55.2% | 54.7% | 54.1% | 56.5% | 56.2% |
| qwen2.5-7b-instruct | 58.8% | 62.3% | 62.2% | 62.1% | 60.9% | 61.5% | 64.0% |

Once the name is right, ~20–47% of calls still fail AST (`bad_args`). Retrieval does not fix argument quality.

## Where the remaining errors are

Counts. Fully correct (name + AST) is listed first; the rest are the error taxonomy.

| Model | Condition | Fully correct | bad_args | wrong_tool | parse_fail | no_call | retrieval_miss | api_fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | Baseline | 4 | 1 | 11 | 183 | 0 | 0 | 1 |
| llama-3.2-3b-instruct | BM25@5 | 95 | 75 | 14 | 0 | 0 | 10 | 6 |
| llama-3.2-3b-instruct | BM25@10 | 94 | 77 | 12 | 0 | 0 | 6 | 11 |
| llama-3.2-3b-instruct | BM25@20 | 89 | 76 | 21 | 0 | 0 | 2 | 12 |
| llama-3.2-3b-instruct | ToolScope@5 | 95 | 74 | 17 | 0 | 0 | 8 | 6 |
| llama-3.2-3b-instruct | ToolScope@10 | 93 | 76 | 16 | 0 | 0 | 3 | 12 |
| llama-3.2-3b-instruct | ToolScope@20 | 88 | 79 | 20 | 0 | 0 | 1 | 12 |
| llama-3.1-8b-instruct | Baseline | 7 | 5 | 27 | 161 | 0 | 0 | 0 |
| llama-3.1-8b-instruct | BM25@5 | 99 | 79 | 12 | 0 | 0 | 10 | 0 |
| llama-3.1-8b-instruct | BM25@10 | 101 | 82 | 11 | 0 | 0 | 6 | 0 |
| llama-3.1-8b-instruct | BM25@20 | 98 | 81 | 19 | 0 | 0 | 2 | 0 |
| llama-3.1-8b-instruct | ToolScope@5 | 98 | 83 | 11 | 0 | 0 | 8 | 0 |
| llama-3.1-8b-instruct | ToolScope@10 | 104 | 80 | 13 | 0 | 0 | 3 | 0 |
| llama-3.1-8b-instruct | ToolScope@20 | 104 | 81 | 14 | 0 | 0 | 1 | 0 |
| qwen2.5-7b-instruct | Baseline | 47 | 33 | 110 | 10 | 0 | 0 | 0 |
| qwen2.5-7b-instruct | BM25@5 | 104 | 63 | 22 | 3 | 0 | 8 | 0 |
| qwen2.5-7b-instruct | BM25@10 | 107 | 65 | 22 | 2 | 0 | 4 | 0 |
| qwen2.5-7b-instruct | BM25@20 | 110 | 67 | 21 | 2 | 0 | 0 | 0 |
| qwen2.5-7b-instruct | ToolScope@5 | 103 | 66 | 22 | 2 | 0 | 7 | 0 |
| qwen2.5-7b-instruct | ToolScope@10 | 107 | 67 | 23 | 0 | 0 | 3 | 0 |
| qwen2.5-7b-instruct | ToolScope@20 | 112 | 63 | 24 | 1 | 0 | 0 | 0 |

qwen2.5-7b-instruct's ToolScope@10 name-acc gain is almost entirely fewer `wrong_tool` (110 → 23), not better arguments.

## ToolScope@10 vs baseline name-acc flips

### llama-3.2-3b-instruct

Name acc 2.5% → 84.5% (+82.0 pp). Flips +165 / −1, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_187` GT `whole_foods.check_price`: baseline `—` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_101` GT `math.gcd`: baseline `—` → ToolScope@10 `math.gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_193` GT `maps.get_distance_duration`: baseline `—` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...
- `multiple_13` GT `corporate_finance.revenue_forecast`: baseline `—` → ToolScope@10 `corporate_finance.revenue_forecast` (recall=1). How much revenue would company XYZ generate if we increase the sales units of product A...

Losses (baseline right, retriever wrong):
- `multiple_39` GT `ride_hailing.get_rides`: baseline `ride_hailing.get_rides` → ToolScope@10 `—` (recall=1). Find a ride from New York to Philadelphia with maximum cost of $50

1 of 1 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### llama-3.1-8b-instruct

Name acc 6.0% → 92.0% (+86.0 pp). Flips +173 / −1, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_66` GT `traffic_estimate`: baseline `—` → ToolScope@10 `traffic_estimate` (recall=1). How much traffic should I expect from Las Vegas to Los Angeles this weekend?
- `multiple_187` GT `whole_foods.check_price`: baseline `—` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_101` GT `math.gcd`: baseline `—` → ToolScope@10 `math.gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_193` GT `maps.get_distance_duration`: baseline `geo_distance.calculate` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...

Losses (baseline right, retriever wrong):
- `multiple_24` GT `route_planner.calculate_route`: baseline `route_planner.calculate_route` → ToolScope@10 `maps.shortest_path` (recall=0). What is the fastest route from London to Edinburgh for playing a chess championship? Al...

### qwen2.5-7b-instruct

Name acc 40.0% → 87.0% (+47.0 pp). Flips +104 / −10, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_187` GT `whole_foods.check_price`: baseline `wholefoods.vegan_products` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_193` GT `maps.get_distance_duration`: baseline `route_planner.calculate_route` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...
- `multiple_13` GT `corporate_finance.revenue_forecast`: baseline `calculate_return_on_investment` → ToolScope@10 `corporate_finance.revenue_forecast` (recall=1). How much revenue would company XYZ generate if we increase the sales units of product A...
- `multiple_2` GT `country_info.capital`: baseline `get_highest_scoring_player` → ToolScope@10 `country_info.capital` (recall=1). What is the capital of Brazil?

Losses (baseline right, retriever wrong):
- `multiple_99` GT `calculus.derivative`: baseline `calculus.derivative` → ToolScope@10 `calculate_derivative` (recall=1). Calculate the derivative of the function 2x^2 at x = 1.
- `multiple_21` GT `generate_sound_wave`: baseline `generate_sound_wave` → ToolScope@10 `audio.generate` (recall=1). I want to generate a sound of 440Hz frequency for 5 seconds. What is the function and h...
- `multiple_119` GT `database.query`: baseline `database.query` → ToolScope@10 `db_fetch_records` (recall=1). Find records in database in user table where age is greater than 25 and job is 'engineer'.
- `multiple_96` GT `solve_quadratic_equation`: baseline `solve_quadratic_equation` → ToolScope@10 `solve_quadratic` (recall=1). Solve a quadratic equation where a=2, b=6, and c=5
- `multiple_11` GT `math_roots.quadratic`: baseline `math_roots.quadratic` → ToolScope@10 `solve_quadratic` (recall=1). Calculate the roots of a quadratic equation with coefficients 5, 20, and -25

9 of 10 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

## Retrieval quality (model-independent)

| Retriever | Recall@10 | NDCG@10 | Missed queries | Mean tokens |
|---|---:|---:|---:|---:|
| BM25@5 | 95.0% | 0.874 | 10 / 200 | 699 |
| BM25@10 | 97.0% | 0.881 | 6 / 200 | 1,401 |
| BM25@20 | 99.0% | 0.886 | 2 / 200 | 2,789 |
| ToolScope@5 | 96.0% | 0.877 | 8 / 200 | 683 |
| ToolScope@10 | 98.5% | 0.885 | 3 / 200 | 1,362 |
| ToolScope@20 | 99.5% | 0.888 | 1 / 200 | 2,700 |

When ToolScope@10 recall is 1, name acc is 85.8% on the first model's traces. When recall is 0, name acc is 0% — the agent cannot call a tool that is not bound.
Missed ground-truth names: `linear_regression`, `probabilities.calculate_single`, `route_planner.calculate_route`.

## Catalog hazards

| Hazard | Count | Effect on scores |
|---|---:|---|
| Same name, different schema (first-seen kept) | 42 records / 33 names | llama-3.2-3b-instruct ToolScope@10 name acc 80.0% on 25 colliding-GT queries vs 85.1% on 175 others; llama-3.1-8b-instruct ToolScope@10 name acc 84.0% on 25 colliding-GT queries vs 93.1% on 175 others; qwen2.5-7b-instruct ToolScope@10 name acc 68.0% on 25 colliding-GT queries vs 89.7% on 175 others |
| Dotted vs underscore aliases after sanitizing | 2 groups | `car.rental` / `car_rental` → `car_rental`; `solve.quadratic_equation` / `solve_quadratic_equation` → `solve_quadratic_equation`. Dedupe keeps first-seen; original_name stays in metadata. |
| Confusable siblings inside top-k | Most remaining `wrong_tool` | Ground truth is retrieved (recall = 1) but the model prefers a near-duplicate still in the shortlist. |

## What this supports for the paper

Selection over injection is not a uniform lift. It helps the model that struggles with a 443-tool prompt (llama-3.2-3b-instruct, +82.0 pp name acc, ~97.7% less tool JSON) and is a wash for models that already pick the right name from the full catalog (qwen2.5-7b-instruct baseline 40.0%).

Retrieval at k=10 is nearly solved (Recall 98.5%). The leftover selection error is sibling confusion, and the leftover calling error is arguments.

Do not treat these numbers as an official BFCL / Gorilla leaderboard score. Shared-catalog protocol, local AST vs `possible_answer`, one-turn LangGraph, no tool execution. `table.md` / `summary.csv` are the compact matrix; this file is the analysis.
