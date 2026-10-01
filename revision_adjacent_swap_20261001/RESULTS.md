# One adjacent attribute swap: completed results

This controlled follow-up evaluates one adjacent transposition of complete attribute entries with the query and every competitor fixed at raw C0. It is a different population and permutation family from the seven-/32-schedule analyses; its percentages are not directly comparable to their VI estimates. The protocol and cohort were frozen before new swap outcomes, not externally preregistered.

Starting repository revision: `3948a7e610a2b73b7d2fd56dab47a374cf099163`. Dedicated branch: `codex/adjacent-swap-20261001`. Source data, model revisions, tokenizer details, execution settings, cache provenance and hashes are recorded in `execution_profile.json` and `freeze_manifest.json`.

## Exact population and support

Start from held-out eligible queries and WANDS Exact / ESCI E judgments. Eligibility requires at least one distinct non-no-op adjacent swap and complete C0 plus every such variant fitting the encoder context, including special tokens. C0 competitors use the established native encoding, including truncation where applicable. Sample up to 128 queries and up to 16 eligible targets per query by encoder-independent SHA-256 priorities. No selection uses previous outcomes, ranks, margins or attribution.

| Setting | Highest-label pairs / held-out queries | Eligible pairs / queries | Sampled pairs / queries | Pair-swap outcomes |
|---|---:|---:|---:|---:|
| WANDS / MiniLM | 21,299 / 308 | 917 / 117 | 712 / 117 | 12,807 |
| WANDS / BGE | 21,299 / 308 | 6,789 / 239 | 1,319 / 128 | 46,211 |
| ESCI / MiniLM | 4,434 / 499 | 2,200 / 427 | 641 / 128 | 1,089 |
| ESCI / BGE | 4,434 / 499 | 3,166 / 461 | 769 / 128 | 1,318 |

Catalogs contain 42,994 WANDS and 10,076 ESCI products. The 3,441 sampled query-product-setting pairs yield 61,425 pair-swap outcomes. Targets repeated across queries share encodings. These are setting-specific supports, not unique products pooled across settings.

| Setting | Excluded: no distinct swap | Excluded: context | Sampled unique products |
|---|---:|---:|---:|
| WANDS / MiniLM | 8 | 20,374 | 679 |
| WANDS / BGE | 8 | 14,502 | 1,301 |
| ESCI / MiniLM | 441 | 1,793 | 641 |
| ESCI / BGE | 441 | 827 | 769 |

Context exclusions are evaluated only among targets with a distinct swap, so exclusion counts are disjoint. Inputs failing C0 are rejected immediately; all variants of every retained target are tokenized. Full attribute-count and swap-count distributions are in `data/attribute_swap_distributions.csv`; they include population, eligible and sampled supports. Encoder-specific fitting populations differ, so cross-encoder differences are descriptive, not causal model comparisons. These are not prevalence estimates over all relevant products.

## Primary results, K=20

A is the query-macro share of sampled targets affected by at least one distinct adjacent swap. F first averages the flip fraction within each target, then across targets within a query, then across queries. Long records do not receive extra weight just because they have more swaps. Brackets are 95% unadjusted query-cluster percentile intervals from 10,000 resamples (seed 2026100101).

| Setting | A: any swap | F: one uniformly selected swap | Affected targets / sampled | Confirmed flip events |
|---|---:|---:|---:|---:|
| WANDS / MiniLM | 6.01% [3.55%, 8.89%] | 0.92% [0.53%, 1.36%] | 48 / 712 | 140 / 12807 |
| WANDS / BGE | 3.50% [2.36%, 4.92%] | 0.48% [0.29%, 0.70%] | 58 / 1319 | 249 / 46211 |
| ESCI / MiniLM | 2.22% [1.20%, 3.41%] | 1.44% [0.79%, 2.20%] | 21 / 641 | 23 / 1089 |
| ESCI / BGE | 1.43% [0.65%, 2.34%] | 0.95% [0.41%, 1.60%] | 14 / 769 | 16 / 1318 |

| Setting | Baseline inclusion | Loss share | Gain share |
|---|---:|---:|---:|
| WANDS / MiniLM | 23.70% [17.91%, 29.62%] | 4.08% [1.93%, 6.72%] | 1.92% [0.88%, 3.21%] |
| WANDS / BGE | 25.16% [20.10%, 30.49%] | 1.63% [0.97%, 2.40%] | 1.87% [1.14%, 2.74%] |
| ESCI / MiniLM | 69.43% [62.95%, 75.58%] | 1.71% [0.81%, 2.78%] | 0.51% [0.14%, 0.98%] |
| ESCI / BGE | 73.92% [68.05%, 79.40%] | 0.13% [0.00%, 0.33%] | 1.31% [0.55%, 2.19%] |

Loss and gain use all sampled eligible targets in each query as denominator and sum to A. They are not conditional failure rates. Baseline inclusion and the target-only interventions are not catalog-wide Recall: different interventions do not form one jointly realized index. Uniform selection refers only to the enumerated distinct-swap family, not a model of real-world edits.

The observed one-swap flip frequencies are modest: 0.48%-1.44% at K=20 in these samples. The existence of a confirmed crossing does not make most adjacent swaps consequential. ESCI/BGE has only two loss-affected targets at K=20 (versus twelve gain-affected targets); its loss-share interval reaches zero. These sparse outcomes are retained and should temper broad failure-prevalence claims.

## Secondary results, K=100

| Setting | A | F | Affected targets | Confirmed / unresolved flip events |
|---|---:|---:|---:|---:|
| WANDS / MiniLM | 6.52% [4.30%, 8.92%] | 1.41% [0.73%, 2.35%] | 57 / 712 | 213 / 0 |
| WANDS / BGE | 6.34% [4.09%, 9.01%] | 1.09% [0.47%, 2.00%] | 76 / 1319 | 338 / 0 |
| ESCI / MiniLM | 2.66% [0.78%, 5.08%] | 1.95% [0.46%, 4.07%] | 9 / 641 | 10 / 0 |
| ESCI / BGE | 0.56% [0.18%, 1.04%] | 0.32% [0.10%, 0.61%] | 9 / 769 | 10 / 0 |

Intervals describe query-level variation in this frozen sample. They do not account for all possible target samples, all relevant products or arbitrary permutations. For any zero-event cell, the observed count and support remain in the table; an empirical [0,0] bootstrap interval does not establish zero underlying event probability.

Prespecified breakdowns by distinct-swap count (1, 2-3, 4-7, 8-15, 16+) and C0 rank (1-10, 11-20, 21-40, >40) are in `data/descriptive_breakdowns.csv`, with query, target and swap supports. Within each stratum, target means are averaged within represented queries and then over represented queries. These are descriptive analyses of the same frozen cohort. Native score, margin and rank-change distributions are in the per-variant/per-target files and `data/supporting_distributions.csv`.

## Numerical validity and preservation

All competitors and query representations stay fixed. Ranks use exact FP32 scores and ascending saved catalog index for ties; the original target is removed before insertion. Margins are signed differences between native target and fixed Kth competitor scores, with boundary indices and exact-tie flags saved. Diagnostic subtraction represents the exact difference of the two saved FP32 scores in float64; membership uses the original native score comparison. No epsilon changes membership.

- MiniLM: 1,353 no-op pair replays, 1,320 distinct inputs; maximum C0 score discrepancy 0; 0 replay rank discrepancies. 386 variants crossing either cutoff received three C0 and three swapped forward checks each (shared identical inputs reused across query checks). Maximum crossing score errors: C0 0, swap 0.
- BGE: 2,088 no-op pair replays, 2,070 distinct inputs; maximum C0 score discrepancy 0; 0 replay rank discrepancies. 613 variants crossing either cutoff received three C0 and three swapped forward checks each (shared identical inputs reused across query checks). Maximum crossing score errors: C0 0, swap 0.

Every observed crossing reproduced at its cutoff in all three repeats. There are zero unresolved observed variant-cutoff events; confirmed and observed estimates coincide.

All 59,983 distinct product-setting variants passed entry-multiset (including duplicates), key/value, numeric-substring, fixed-text, adjacent-transposition, other-relative-order, identity and no-truncation checks. The repository representation audit was reused. Zero audit failures. C0 and every adjacent swap map to exactly the same ascending canonical serialization; this invariance is structural and is not claimed as a new empirical result.

The installed PyTorch/Transformers/tokenizers versions, CUDA/GPU, model snapshots, precision, pooling, padding/batching and serializer matched the earlier harmonized C0 reference. Payload hashes and fresh input aliases were verified. No mixed-profile historical score arrays were used. Queries missing from that reference were encoded under the same profile. Exact compatible swap-input aliases, especially the already enumerated ESCI orders, also reuse vectors; this does not mean that every tested serialization required a new forward pass. Full repeat records and numerical reports retain native discrepancies.

## Deterministic illustrative case

The smallest prespecified case hash among confirmed K=20 loss pairs selected WANDS/MiniLM, query 63, product 25295. Within that pair the smallest qualifying swap hash selected positions [23]. This selection is conditional on a confirmed crossing, not a representative prevalence sample.

Query: 'monthly calendar'. C0 rank 19 becomes 21; scores 0.393814832 and 0.388081193; K=20 margins +0.004287809 and -0.001445830. Tokens 251 and 251 / limit 256. All competing products remain C0. Full serializations and all adjacent outcomes are in `data/case_selection.json` and `data/case_all_swap_outcomes.csv`.

## Contribution and manuscript placement

The experiment adds a distinct sufficiency finding: a single adjacent exchange of complete attributes can change a relevant target's inclusion with fixed competitors and fully fitting inputs. Confirmed K=20 events occur in 4 of the four evaluated settings. This narrows the size of the perturbation needed compared with the earlier schedule analyses; it does not identify an attention mechanism, establish semantic responsibility for an attribute, or show typical real-world editing effects.

Recommended placement: a short main-text finding with the compact K=20 aggregate figure if space permits; put the full numerical audit, K=100, support distributions, descriptive strata and illustrative case in the appendix. Keep the severe fully-fitting selection and setting-specific supports next to the claim. Do not present this as stronger prevalence evidence over the entire catalog or as an encoder ranking.

## Artifacts

- Primary figure: `figures/aggregate_K20.pdf` and `.png`; supplementary: `figures/aggregate_K100.pdf` and `.png`.
- Illustrative case: `figures/illustrative_case.pdf` and `.png` when a confirmed case exists; source data are beside the figures.
- Outcomes: `data/per_variant.parquet`, `data/per_target.parquet` / `.csv`, `data/per_query.parquet` / `.csv`, `data/aggregate_estimates.csv`.
- Population/audits: `data/eligibility_population.parquet`, `data/population_counts.csv`, `data/representation_audit.parquet`, `REPRESENTATION_AUDIT.md`, `NUMERICAL_VERIFICATION.md`.
- Protocol and reproduction: `protocol.md`, `config.json`, `cohort_manifest.csv`, `execution_profile.json`, `freeze_manifest.json`, `REPRODUCE.md`.
- English manuscript fragments: `manuscript/captions.tex`, `manuscript/paragraph.tex`. Existing manuscripts and experiments are preserved.
