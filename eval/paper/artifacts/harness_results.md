# BFCL Multiple — harness results

Shared catalog C = 443 tools, 200 queries, k = 10, sentence-transformers/all-MiniLM-L6-v2.
Protocol: `shared_catalog`. BFCL-derived; **not** an official Gorilla leaderboard score.

| | |
|---|---|
| Queries scored | 200 per model |
| Models | 3 |
| Catalog C | 443 tools |
| Context compression at k=10 | 97.7% |
| Largest name-acc Δ vs baseline | +5.5 pp (Qwen 3.5-397B, ToolScope) |
| Instances skipped | 0 |

---

No instances skipped. No `api_fail` cells in the loaded traces.

## Tool name accuracy (headline)

Share of queries where the model called a ground-truth tool name. Retrieval metrics are identical across models for a given retriever.

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| Qwen 3.5-397B | 85.5% | 88.5% | 91.0% |
| DeepSeek-V4-Flash | 87.0% | 87.0% | 88.0% |
| Gemini 3.7 Flash | 90.0% | 88.5% | 88.5% |

Models ordered by baseline name accuracy (weakest catalog handler first).

## Δ name acc vs full catalog

Selection gain shrinks as baseline name accuracy rises. McNemar is exact two-sided on paired name-acc flips (ToolScope vs baseline).

| Model | Baseline name acc | BM25 Δ | ToolScope Δ | ToolScope flips (win/lose) | McNemar p |
|---|---:|---:|---:|---|---:|
| Qwen 3.5-397B | 85.5% | +3.0 pp | +5.5 pp | +14 / −3 | 0.013 |
| DeepSeek-V4-Flash | 87.0% | +0.0 pp | +1.0 pp | +9 / −7 | 0.80 |
| Gemini 3.7 Flash | 90.0% | -1.5 pp | -1.5 pp | +4 / −7 | 0.55 |

## Per-condition matrix

| Model | Condition | Name acc | AST acc | Δ name | Recall@10 | NDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.5-397B | Baseline | 85.5% | 64.5% | — | — | — | 4.5 s |
| Qwen 3.5-397B | BM25 | 88.5% | 67.5% | +3.0 pp | 97.0% | 0.881 | 3.5 s |
| Qwen 3.5-397B | ToolScope | 91.0% | 67.0% | +5.5 pp | 98.5% | 0.885 | 3.4 s |
| DeepSeek-V4-Flash | Baseline | 87.0% | 60.5% | — | — | — | 1.3 s |
| DeepSeek-V4-Flash | BM25 | 87.0% | 63.0% | +0.0 pp | 97.0% | 0.881 | 975 ms |
| DeepSeek-V4-Flash | ToolScope | 88.0% | 62.5% | +1.0 pp | 98.5% | 0.885 | 985 ms |
| Gemini 3.7 Flash | Baseline | 90.0% | 65.0% | — | — | — | 11.0 s |
| Gemini 3.7 Flash | BM25 | 88.5% | 62.5% | -1.5 pp | 97.0% | 0.881 | 8.4 s |
| Gemini 3.7 Flash | ToolScope | 88.5% | 61.5% | -1.5 pp | 98.5% | 0.885 | 8.9 s |

Prompt tokens: baseline ~60,051 vs BM25 ~1,401, ToolScope ~1,362 (~97.7% compression). Latency is one-turn `bind_tools` only; tools are never executed.
Gemini 3.7 Flash latency tail: p50 3.0 s, p95 52.5 s (baseline).

## AST accuracy

Name selection does not close the AST gap. Leftover error after a correct name is almost entirely `bad_args`.

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| Qwen 3.5-397B | 64.5% | 67.5% | 67.0% |
| DeepSeek-V4-Flash | 60.5% | 63.0% | 62.5% |
| Gemini 3.7 Flash | 65.0% | 62.5% | 61.5% |

## AST given correct name

| Model | Baseline | BM25 | ToolScope |
|---|---|---|---|
| Qwen 3.5-397B | 75.4% | 76.3% | 73.6% |
| DeepSeek-V4-Flash | 69.5% | 72.4% | 71.0% |
| Gemini 3.7 Flash | 72.2% | 70.6% | 69.5% |

Once the name is right, ~24–31% of calls still fail AST (`bad_args`). Retrieval does not fix argument quality.

## Where the remaining errors are

Counts. Fully correct (name + AST) is listed first; the rest are the error taxonomy.

| Model | Condition | Fully correct | bad_args | wrong_tool | parse_fail | no_call | retrieval_miss | api_fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5-397B | Baseline | 129 | 42 | 20 | 5 | 4 | 0 | 0 |
| Qwen 3.5-397B | BM25 | 135 | 42 | 10 | 7 | 4 | 2 | 0 |
| Qwen 3.5-397B | ToolScope | 134 | 48 | 9 | 5 | 2 | 2 | 0 |
| DeepSeek-V4-Flash | Baseline | 121 | 53 | 23 | 3 | 0 | 0 | 0 |
| DeepSeek-V4-Flash | BM25 | 126 | 48 | 18 | 4 | 1 | 3 | 0 |
| DeepSeek-V4-Flash | ToolScope | 125 | 51 | 19 | 3 | 0 | 2 | 0 |
| Gemini 3.7 Flash | Baseline | 130 | 50 | 15 | 5 | 0 | 0 | 0 |
| Gemini 3.7 Flash | BM25 | 125 | 52 | 13 | 8 | 0 | 2 | 0 |
| Gemini 3.7 Flash | ToolScope | 123 | 54 | 16 | 5 | 0 | 2 | 0 |

Qwen 3.5-397B's ToolScope name-acc gain is almost entirely fewer `wrong_tool` (20 → 9), not better arguments.

## ToolScope vs baseline name-acc flips

### Qwen 3.5-397B

Name acc 85.5% → 91.0% (+5.5 pp). Flips +14 / −3, McNemar p = 0.013.

Wins (baseline wrong, retriever right):
- `multiple_161` GT `find_exhibition`: baseline `—` → ToolScope `find_exhibition` (recall=1). Find the top rated modern sculpture exhibition happening in New York in the upcoming mo...
- `multiple_65` GT `geodistance.find`: baseline `calculate_distance` → ToolScope `geodistance.find` (recall=1). Find the distance between New York City and Los Angeles.
- `multiple_142` GT `weather.humidity_forecast`: baseline `humidity_temperature_forecast` → ToolScope `weather.humidity_forecast` (recall=1). What is the humidity level in Miami, Florida in the upcoming 7 days?
- `multiple_41` GT `magnetic_field.calculate`: baseline `electromagnetism.ampere_law` → ToolScope `magnetic_field.calculate` (recall=1). Calculate the magnetic field at point P using Ampere’s law where current I is 10 Ampere...
- `multiple_93` GT `hotel.book`: baseline `hotel` → ToolScope `hotel.book` (recall=1). Book a deluxe room for 2 nights at the Marriott hotel in New York and add breakfast as ...

Losses (baseline right, retriever wrong):
- `multiple_121` GT `geometry.area_triangle`: baseline `geometry.area_triangle` → ToolScope `triangle.area` (recall=1). Calculate the area of a triangle with base 6 and height 10.
- `multiple_97` GT `geometry.area_circle`: baseline `geometry.area_circle` → ToolScope `circle.area` (recall=1). What's the area of a circle with a radius of 10?
- `multiple_197` GT `mutation_type.find`: baseline `mutation_type.find` → ToolScope `—` (recall=1). Find the type of gene mutation based on SNP (Single Nucleotide Polymorphism) ID rs6034464.

3 of 3 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### DeepSeek-V4-Flash

Name acc 87.0% → 88.0% (+1.0 pp). Flips +9 / −7, McNemar p = 0.80.

Wins (baseline wrong, retriever right):
- `multiple_153` GT `get_event_date`: baseline `history.get_event_date` → ToolScope `get_event_date` (recall=1). When was the signing of the Treaty of Lisbon?
- `multiple_102` GT `calculate_displacement`: baseline `kinematics.calculate_displacement` → ToolScope `calculate_displacement` (recall=1). Calculate the displacement of a car given the initial velocity of 10 and acceleeration ...
- `multiple_89` GT `recipe_search`: baseline `find_recipes` → ToolScope `recipe_search` (recall=1). Find a healthy lunch recipe under 500 calories that uses chicken and mushrooms.
- `multiple_152` GT `history.get_key_events`: baseline `european_history.get_events` → ToolScope `history.get_key_events` (recall=1). Provide key war events in German history from 1871 to 1945.
- `multiple_141` GT `lawsuit.check_case`: baseline `legal_case.fetch` → ToolScope `lawsuit.check_case` (recall=1). I need the details of the lawsuit case with case ID of 1234 and verify if it's already ...

Losses (baseline right, retriever wrong):
- `multiple_76` GT `sculpture.create_custom`: baseline `sculpture.create_custom` → ToolScope `sculpture_availability.check` (recall=1). I want to order a custom bronze sculpture of a horse. What material options are available?
- `multiple_52` GT `currency_conversion`: baseline `currency_conversion` → ToolScope `currency_conversion.convert` (recall=1). I have 100 euro. How much is it in USD?
- `multiple_135` GT `get_case_info`: baseline `get_case_info` → ToolScope `court_case.search` (recall=1). Who was the victim in the case docket numbered 2022/AL2562 in California?
- `multiple_99` GT `calculus.derivative`: baseline `calculus.derivative` → ToolScope `calculate_derivative` (recall=1). Calculate the derivative of the function 2x^2 at x = 1.
- `multiple_53` GT `linear_regression`: baseline `linear_regression` → ToolScope `—` (recall=0). Predict the house prices for next 5 years based on interest rates and unemployment rates.

6 of 7 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### Gemini 3.7 Flash

Name acc 90.0% → 88.5% (-1.5 pp). Flips +4 / −7, McNemar p = 0.55.

Wins (baseline wrong, retriever right):
- `multiple_34` GT `math.lcm`: baseline `calculate_lcm` → ToolScope `math.lcm` (recall=1). Calculate the Least Common Multiple (LCM) of 18 and 12.
- `multiple_92` GT `walmart.vegan_products`: baseline `—` → ToolScope `walmart.vegan_products` (recall=1). Get me a list of available vegetarian and gluten-free foods at the Walmart near Denver.
- `multiple_11` GT `math_roots.quadratic`: baseline `solve_quadratic` → ToolScope `math_roots.quadratic` (recall=1). Calculate the roots of a quadratic equation with coefficients 5, 20, and -25
- `multiple_55` GT `stock_forecast`: baseline `stock_market_forecast` → ToolScope `stock_forecast` (recall=1). Predict the stock price for Google for the next 3 days.

Losses (baseline right, retriever wrong):
- `multiple_101` GT `math.gcd`: baseline `math.gcd` → ToolScope `calculate_gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_95` GT `currency_exchange.convert`: baseline `currency_exchange.convert` → ToolScope `currency_conversion` (recall=1). Convert 200 euros to US dollars using current exchange rate.
- `multiple_138` GT `legal_case.fetch`: baseline `legal_case.fetch` → ToolScope `—` (recall=1). How to obtain the detailed case information of the R vs Adams legal case?
- `multiple_132` GT `finance.calculate_future_value`: baseline `finance.calculate_future_value` → ToolScope `future_value` (recall=1). Calculate the future value of an investment with an annual rate of return of 8%, an ini...
- `multiple_21` GT `generate_sound_wave`: baseline `generate_sound_wave` → ToolScope `—` (recall=1). I want to generate a sound of 440Hz frequency for 5 seconds. What is the function and h...

5 of 7 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

## Retrieval quality (model-independent)

| Retriever | Recall@10 | NDCG@10 | Missed queries | Mean tokens |
|---|---:|---:|---:|---:|
| BM25 | 97.0% | 0.881 | 6 / 200 | 1,401 |
| ToolScope | 98.5% | 0.885 | 3 / 200 | 1,362 |

When ToolScope recall is 1, name acc is 92.4% on the first model's traces. When recall is 0, name acc is 0% — the agent cannot call a tool that is not bound.
Missed ground-truth names: `linear_regression`, `probabilities.calculate_single`, `route_planner.calculate_route`.

## Catalog hazards

| Hazard | Count | Effect on scores |
|---|---:|---|
| Same name, different schema (first-seen kept) | 42 records / 33 names | Qwen 3.5-397B ToolScope name acc 72.0% on 25 colliding-GT queries vs 93.7% on 175 others; DeepSeek-V4-Flash ToolScope name acc 68.0% on 25 colliding-GT queries vs 90.9% on 175 others; Gemini 3.7 Flash ToolScope name acc 64.0% on 25 colliding-GT queries vs 92.0% on 175 others |
| Dotted vs underscore aliases after sanitizing | 2 groups | `car.rental` / `car_rental` → `car_rental`; `solve.quadratic_equation` / `solve_quadratic_equation` → `solve_quadratic_equation`. Dedupe keeps first-seen; original_name stays in metadata. |
| Confusable siblings inside top-k | Most remaining `wrong_tool` | Ground truth is retrieved (recall = 1) but the model prefers a near-duplicate still in the shortlist. |

## What this supports for the paper

Selection over injection is not a uniform lift. It helps the model that struggles with a 443-tool prompt (Qwen 3.5-397B, +5.5 pp name acc, ~97.7% less tool JSON) and is a wash for models that already pick the right name from the full catalog (Gemini 3.7 Flash baseline 90.0%).

Retrieval at k=10 is nearly solved (Recall 98.5%). The leftover selection error is sibling confusion, and the leftover calling error is arguments.

Do not treat these numbers as an official BFCL / Gorilla leaderboard score. Shared-catalog protocol, local AST vs `possible_answer`, one-turn LangGraph, no tool execution. `table.md` / `summary.csv` are the compact matrix; this file is the analysis.
