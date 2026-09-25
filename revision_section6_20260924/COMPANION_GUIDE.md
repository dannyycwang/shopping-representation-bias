# Reading the evidence companions

The CSVs retain 17 significant digits. Percentages and percentage points appear only in manuscript/plot formatting: an inclusion estimate of `0.024` is 2.4%, and a paired difference of `0.024` is 2.4 percentage points. nDCG and cNDCG remain on the 0-1 scale. Query/pair counts are integers. JSON arrays of product IDs in CSV/Parquet are exact sets, not representative examples.

| Question | Authoritative query records | Full estimates/comparisons |
|---|---|---|
| Does order change inclusion? | `data/inclusion_per_query.parquet` or `.csv` | `tables/inclusion_complete.csv` |
| Which relevant identities changed? | `data/membership_per_query.parquet` or `.csv.gz` | `tables/membership_all_contrasts.csv` |
| What is graded effectiveness? | `data/effectiveness_per_query.parquet` or `.csv.gz` | Joined into method and membership companions |
| How do controls compare? | `data/controls_per_query.parquet` or `.csv.gz` | `tables/method_complete_companion.csv` |
| Where do raw states move under canonicalization? | `data/transitions_per_query.parquet` | `tables/transitions_complete.csv` |
| What changes after harmonizing execution? | `harmonized/<model>/<dataset>_ranks.parquet`, `data/harmonized_per_query.csv`, `data/historical_to_harmonized_per_query.csv` | `tables/harmonized_summary.csv`, `tables/harmonized_increments.csv`, `tables/historical_to_harmonized.csv` |

The last row becomes available only after both harmonized runs and their summary finish. `qa/final_validation.json` is the final package's completion check; `complete.json` in each model directory is a model-specific inference completion check.

## Keys and denominators

- Inclusion query key: dataset, model, intervention, target_support, K, query_id. `n_highest` is the denominator **after** the stated support restriction. Each state is a within-query proportion; the estimate is its equally weighted mean across nonempty queries. Target-only source/mean inclusion is not coherent-catalog Recall.
- Membership key: dataset, model, method, schedule, K, query_id. Every alternative is compared with source C0. `changed` means at least one gained or lost highest-label ID. `exact_replacement` means integer gain_count = loss_count > 0. `equal_positive_share_all` and `equal_positive_share_changed` use distinct denominators. Conditional absolute nDCG quantiles exclude unchanged-membership queries; the signed `delta_ndcg` mean includes every eligible query.
- Effectiveness key: dataset, model, method, schedule, query_id. BM25 uses `model=lexical` in this authoritative rank-derived file, then repeats the same lexical reference under each encoder in the comparison companion. `rank_source` points to the saved all-judged rank table; `included_ids_K20/K100` are highest-label membership sets.
- Control key: dataset, model, method, schedule, K, query_id. For raw/raw_hybrid, `C0` and `seven_mean` select different **effectiveness** references. The persistent states and VI always describe the method's whole seven-order family; a C0 effectiveness row does not redefine VI as a single-order statistic. `fixed` methods use one order-invariant index. `vectors_per_product` counts dense vectors only; `sparse_branch` marks extra lexical indexing/scoring. BM25 needs zero neural product encodings. Set-Mean needs one encoding per attribute plus its fixed-text atom.
- The long method companion distinguishes `record_type=estimate`, `paired_comparison`, and `unavailable`. A delta is method minus the explicitly named comparator on identical query IDs. Missing encoder/method experiments are unavailable rather than zero; do not pool them with structural VI=0.
- Transition key: dataset, model, family, rule, K, query_id, reference_state, control_state. Each of six `cell_fraction` entries uses all highest-label support for that query. Dense and hybrid have different raw references. These are not conditional probabilities given a raw state.
- Harmonized key: dataset, model, support, family, K, query_id. `common_all_fitting` holds the all-32 WANDS or exhaustive-target ESCI support fixed across families. Rank records preserve actual native scores, stable catalog indices, text/input hashes, and cached-vector references. The aliases connect every query, competitor and target to one pinned profile.

## Schedules, methods, and provenance

`C0` = Source; `C1` = Reverse; `C2s1` through `C2s5` = Shuffle 1-5. Original seeds and serialized strings are unchanged. WANDS `P2026092201` through `P2026092225` extend the original seven; ESCI `E0`, `E1`, ... enumerate distinct whole-attribute orders for each target. Exhaustive never means all jointly permuted catalogs.

`lexical_ascending`, `lexical_descending`, and `field_priority_type` are the three pure canonical rules. `canonical_hybrid_<rule>` combines that dense rule with BM25; `raw_hybrid` combines each raw schedule with BM25. `centroid_M` constructs M views but stores/scores one vector; `max_M` stores/scores M vectors. All saved variants remain in the complete method file even when omitted from the compact main table.

Intervals are 95% unadjusted paired query-cluster percentile intervals with 10,000 resamples and seed 2026091701. Every compared condition and all products of a sampled query stay together. Zero-containing intervals do not establish equivalence. Identity-count ranges across six contrasts are not unions of queries.

The claim ledger audits original v11 assertions; the four expansion rows remain `NEEDS_CORRECTION` for that historical manuscript even after their `resolution` points to the completed replacement supplement. A numerical match does not repair mixed execution provenance. The final audit and scoped TeX use the new supplement only once validated.

Original scripts, array paths/hashes, serializer/model revisions and unavailable runtime fields are enumerated in `data/source_inventory.csv`, `data/historical_execution_matrix.csv`, and `section6_scope_and_provenance.md`. `output_manifest.csv` fingerprints the final package; large vector hashes are in the corresponding model's `complete.json`.
