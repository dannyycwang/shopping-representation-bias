# D. Completed permutation coverage

WANDS uses the 128 queries selected by SHA256("coverage-20260922|"+decimalID), numeric tie-breaker:7,125 pairs,6,444 products, identical queries for both encoders. Exact old seven schedules are followed by seeds 2026092201–2209 then 2210–2225 through the original product-ID-seeded random-order function. Families are nested 7/16/32. Duplicates remain in schedule-weighted frequencies and distinct texts are separately counted.

ESCI uses all 499 queries/4,434 E pairs. Every distinct permutation of whole nonempty entries is enumerated, maximum 3 entries / 6 orders, preserving duplicate multiplicities, fixed blocks and separators. This exhausts per-target orders with competitors at C0, not joint catalog combinations.

The original seven schedules left missing target orders for 2,713 of 4,427 distinct E-target products. Per-target coverage is below; three distinct orders arise when a three-entry multiset contains duplicates. Counts here are products, not queries or query–product pairs. Global unique text counts in the compute table additionally deduplicate identical strings across products.

| distinct_full_family_texts | distinct_old_seven_texts | target_products |
| --- | --- | --- |
| 1 | 1 | 441 |
| 2 | 2 | 1060 |
| 3 | 2 | 1 |
| 3 | 3 | 3 |
| 6 | 2 | 15 |
| 6 | 3 | 319 |
| 6 | 4 | 1272 |
| 6 | 5 | 1106 |
| 6 | 6 | 210 |

For WANDS, 14 of 6,444 target products have fewer than 32 distinct full strings; the average is 31.958876. Complete per-product old/full counts are in data/wands_distinct_order_counts.csv and data/esci_distinct_order_counts.csv.

Only missing distinct target model inputs are encoded. Exact old query/competitor payloads are reused; no whole-catalog/GTE encoding or training. Original-seven token lengths match historical values. Each model's fixed common fitting support fits every 32 WANDS schedule or every ESCI permutation; nonempty query sets stay constant across the curve. Full supports are separate; the 128-query supplement is not the 308-query primary population.

## Frozen estimates and measured computation

| dataset | model | products | schedule_records | distinct_full_texts | distinct_model_inputs | new_model_inputs | new_capped_tokens | runtime_estimate_seconds | full_queries | full_pairs | fully_fitting_queries | fully_fitting_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wands | minilm | 6444 | 206208 | 205943 | 200480 | 156419 | 39555524 | 791.110480 | 128 | 7125 | 48 | 354 |
| wands | bge_base | 6444 | 206208 | 205943 | 205757 | 160723 | 74780523 | 8308.947000 | 128 | 7125 | 92 | 2186 |
| esci | minilm | 4427 | 51094 | 20097 | 15471 | 3199 | 500818 | 10.016360 | 499 | 4434 | 460 | 2578 |
| esci | bge_base | 4427 | 51094 | 20097 | 18667 | 4199 | 1031720 | 114.635556 | 499 | 4434 | 488 | 3587 |

Counts/estimates were frozen before new scores in COVERAGE_PLAN_FROZEN.json. Full-text hashes resolve through explicit complete-token-input aliases: identical input_ids/types/masks after truncation share one vector. The execution addendum records this before new outcomes. No strings, schedules or support pairs are dropped.

Actually encoded 324,540 new inputs. Block encoding/compression time totals 1968.6s (32.8min), excluding token planning, loading, forward audit, scoring and reporting:

The recorded total also excludes an interrupted partial block. BGE initially used native batch 48; at the 8 GiB GPU's memory limit, the run resumed after 50 complete WANDS blocks (51,200 inputs) with batches capped at 24 and unused allocator memory released between blocks. ENCODING_MEMORY_ADDENDUM.json documents this resource-driven adjustment before remaining BGE scores. Completed vectors were hash-verified and reused; no model, input, seed, support or outcome selection changed. Batch-dependent numerical differences are part of the disclosed forward audit, not assumed zero.

| model | inputs | blocks | seconds |
| --- | --- | --- | --- |
| minilm | 159618 | 157 | 390.715942 |
| bge_base | 164922 | 162 | 1577.872573 |

Each block metadata file retains input hashes, full encoder fingerprint, payload hash and elapsed time. Large payloads are local/ignored, metadata and scientific results tracked. Resume verifies hashes/profile; cache keys resolve full-text hash plus profile through the explicit model-input aliases.

## Results

Top-20 query-macro crossing mass, percentage points with uncorrected 95% intervals:

| dataset | model | support | family | K | queries | pairs | mean_pp | CI_low_pp | CI_high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wands | minilm | full | 7 | 20 | 128 | 7125 | 20.848855 | 16.848047 | 25.294852 |
| wands | minilm | full | 16 | 20 | 128 | 7125 | 24.497557 | 20.389674 | 28.967031 |
| wands | minilm | full | 32 | 20 | 128 | 7125 | 28.209606 | 23.760261 | 32.950172 |
| wands | minilm | common_all_fitting | 7 | 20 | 48 | 354 | 11.364939 | 5.846814 | 17.630884 |
| wands | minilm | common_all_fitting | 16 | 20 | 48 | 354 | 15.348553 | 8.808623 | 22.597304 |
| wands | minilm | common_all_fitting | 32 | 20 | 48 | 354 | 16.613202 | 9.899363 | 23.905158 |
| esci | minilm | full | 7 | 20 | 499 | 4434 | 2.909198 | 2.142806 | 3.805415 |
| esci | minilm | full | exhaustive_target | 20 | 499 | 4434 | 3.245325 | 2.460565 | 4.138384 |
| esci | minilm | common_all_fitting | 7 | 20 | 460 | 2578 | 3.204667 | 2.253954 | 4.254574 |
| esci | minilm | common_all_fitting | exhaustive_target | 20 | 460 | 2578 | 3.491594 | 2.536987 | 4.563748 |
| wands | bge_base | full | 7 | 20 | 128 | 7125 | 11.400071 | 8.956929 | 14.232139 |
| wands | bge_base | full | 16 | 20 | 128 | 7125 | 15.029755 | 12.066860 | 18.325313 |
| wands | bge_base | full | 32 | 20 | 128 | 7125 | 17.545299 | 14.432516 | 20.928254 |
| wands | bge_base | common_all_fitting | 7 | 20 | 92 | 2186 | 8.743601 | 5.987506 | 11.955138 |
| wands | bge_base | common_all_fitting | 16 | 20 | 92 | 2186 | 13.085566 | 9.172882 | 17.512331 |
| wands | bge_base | common_all_fitting | 32 | 20 | 92 | 2186 | 17.600469 | 13.130771 | 22.657733 |
| esci | bge_base | full | 7 | 20 | 499 | 4434 | 2.423639 | 1.758347 | 3.178709 |
| esci | bge_base | full | exhaustive_target | 20 | 499 | 4434 | 3.134741 | 2.332601 | 4.035441 |
| esci | bge_base | common_all_fitting | 7 | 20 | 488 | 3587 | 2.552728 | 1.868484 | 3.315560 |
| esci | bge_base | common_all_fitting | exhaustive_target | 20 | 488 | 3587 | 3.334196 | 2.497723 | 4.279357 |

All K100, persistent inclusion/omission, tested-order inclusion frequency and pair-micro results are in coverage_summary.csv. Frequency averages separate target counterfactuals, not coherent-catalog Recall/nDCG. WANDS weights requested schedules including duplicates. ESCI retains seven original observations plus each missing distinct text once: frequency weights that observation multiset, not uniformly unique permutations; inclusion states cover all distinct target orders.

Incremental crossing, percentage points:

| dataset | model | support | K | contrast | queries | pairs | delta_pp | CI_low_pp | CI_high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wands | minilm | full | 20 | 16 minus 7 | 128 | 7125 | 3.648701 | 2.765992 | 4.670348 |
| wands | minilm | full | 100 | 16 minus 7 | 128 | 7125 | 5.931511 | 4.262928 | 8.030201 |
| wands | minilm | full | 20 | 32 minus 16 | 128 | 7125 | 3.712049 | 2.681776 | 4.876548 |
| wands | minilm | full | 100 | 32 minus 16 | 128 | 7125 | 4.992865 | 3.789657 | 6.378602 |
| wands | minilm | common_all_fitting | 20 | 16 minus 7 | 48 | 354 | 3.983614 | 0.935921 | 8.832849 |
| wands | minilm | common_all_fitting | 100 | 16 minus 7 | 48 | 354 | 7.438115 | 2.508224 | 14.023319 |
| wands | minilm | common_all_fitting | 20 | 32 minus 16 | 48 | 354 | 1.264650 | 0.173611 | 2.705717 |
| wands | minilm | common_all_fitting | 100 | 32 minus 16 | 48 | 354 | 5.521620 | 0.841255 | 11.916600 |
| esci | minilm | full | 20 | exhaustive_target minus 7 | 499 | 4434 | 0.336127 | 0.166637 | 0.547174 |
| esci | minilm | full | 100 | exhaustive_target minus 7 | 499 | 4434 | 0.075614 | 0.007422 | 0.174423 |
| esci | minilm | common_all_fitting | 20 | exhaustive_target minus 7 | 460 | 2578 | 0.286927 | 0.116198 | 0.501757 |
| esci | minilm | common_all_fitting | 100 | exhaustive_target minus 7 | 460 | 2578 | 0.025664 | 0.000000 | 0.064915 |
| wands | bge_base | full | 20 | 16 minus 7 | 128 | 7125 | 3.629684 | 2.475209 | 5.003508 |
| wands | bge_base | full | 100 | 16 minus 7 | 128 | 7125 | 2.825200 | 2.108365 | 3.624895 |
| wands | bge_base | full | 20 | 32 minus 16 | 128 | 7125 | 2.515544 | 1.565358 | 3.715464 |
| wands | bge_base | full | 100 | 32 minus 16 | 128 | 7125 | 1.866036 | 1.372598 | 2.409629 |
| wands | bge_base | common_all_fitting | 20 | 16 minus 7 | 92 | 2186 | 4.341965 | 1.814164 | 7.790032 |
| wands | bge_base | common_all_fitting | 100 | 16 minus 7 | 92 | 2186 | 2.841101 | 1.640921 | 4.260675 |
| wands | bge_base | common_all_fitting | 20 | 32 minus 16 | 92 | 2186 | 4.514903 | 2.134808 | 7.736289 |
| wands | bge_base | common_all_fitting | 100 | 32 minus 16 | 92 | 2186 | 2.407295 | 1.274525 | 3.868612 |
| esci | bge_base | full | 20 | exhaustive_target minus 7 | 499 | 4434 | 0.711102 | 0.306607 | 1.255785 |
| esci | bge_base | full | 100 | exhaustive_target minus 7 | 499 | 4434 | 0.086656 | 0.021737 | 0.173236 |
| esci | bge_base | common_all_fitting | 20 | exhaustive_target minus 7 | 488 | 3587 | 0.781468 | 0.350840 | 1.358619 |
| esci | bge_base | common_all_fitting | 100 | exhaustive_target minus 7 | 488 | 3587 | 0.100116 | 0.025615 | 0.196993 |

Original-seven always-in/out origins are separately exported with all identities. Already-crossing pairs are not counted as new. Query-cluster bootstrap uses 10,000 draws, seed 2026091701, uncorrected percentile intervals, conditional on fixed catalogs/encoders. It does not quantify training or annotation uncertainty.

All per-pair monotonicity checks pass: VI nondecreasing, persistent inclusion/omission nonincreasing. This follows structurally from nesting. A small last increment is not convergence or all-permutation invariance. Zero additional ESCI mass is valid, without inventing extra views.

## Numerical sensitivity

New native inference uses pinned tokenizer/weights, empty prefixes, FP16 model/FP32 pooling, original encoder and cached query vectors. Current transformers 4.46.2 differs from4.55.4 in some historical metadata; original arrays/ranks are never regenerated. Original-seven ranks remain authoritative.

Each dataset/encoder audits 16 hash-fixed C0 products against cached vectors and all cached queries. Forward audit CSVs show actual vector/score errors. The sensitivity table identifies added-crossing witnesses inside the largest observed audit error; this small audit is not a certified global error bound or replacement tolerance. Ranking decisions remain exact.

The WANDS/BGE 16-product probe matches cached vectors exactly, yielding a sampled error threshold of zero. Its “beyond audit error” count therefore does not establish numerical robustness. Separate single-input BGE forwards in the frozen XAI panel differ from saved native scores by up to 8.3565712e-05; this illustrates why a zero in one batch/sample cannot certify all batch shapes or target representations. All coverage estimates remain conditional on the documented native execution profile.

| dataset | model | contrast | observed_audit_score_error | new_crossing_pairs | pairs_with_witness_beyond_audit_error | pairs_near_observed_error |
| --- | --- | --- | --- | --- | --- | --- |
| wands | minilm | 16 minus 7 | 0.000218 | 278 | 274 | 4 |
| wands | minilm | 32 minus 16 | 0.000218 | 248 | 242 | 6 |
| wands | bge_base | 16 minus 7 | 0.000000 | 174 | 174 | 0 |
| wands | bge_base | 32 minus 16 | 0.000000 | 148 | 148 | 0 |
| esci | minilm | exhaustive_target minus 7 | 0.000211 | 19 | 17 | 2 |
| esci | bge_base | exhaustive_target minus 7 | 0.000132 | 21 | 20 | 1 |

coverage_validation.json confirms original ranks, boundary identities and monotonicity. coverage_plan_validation.json verifies serializers, token lengths and whole-entry multiplicities. No resource block remains. Coverage plots/source data are supplementary.
