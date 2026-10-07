# Recorded product-option coverage diagnostic

Exploratory post-hoc analysis, 7 October 2026. No training, new query generation, inference, retrieval rerun, manuscript edit, commit or push. All planned saved ranking contrasts are available. All new outputs are isolated in `option_coverage_20261007/`.

## Finding and strength of evidence

The original evaluation anchors reproduce: **308 eligible queries and 73 exact-cancellation queries**. Recorded option identities can change while per-query Exact Recall@20 stays exactly constant. However, strict metadata observation leaves only **4 material, 6 shape and 9 color cancellation queries**, so the event percentages do not estimate prevalence across all 73 cancellations. Material is a zero result (0/4 loss); shape shows 1/6 loss; color shows 7/9 loss, alongside 7/9 gains. Product substitution also leaves field option sets unchanged in 4/4, 3/6 and 1/9 respectively.

**Recommended placement: appendix diagnostic, with one cautious RQ2 pointer after the cancellation results.** The evidence supports existence and concrete audited examples, but is too limited and selected by metadata availability for a broad main-text quantitative claim. A short RQ3 cross-reference can note that fixed canonical representations retain different recorded descriptors; it must retain the small-support and historical batch-execution qualifications. No attempt was made to maximize positive findings.

## Provenance and exact support

Repository commit `0f7a359500fad32f9f051efc4a9302a7e12c18d6`; branch `codex/teal-chair-fig1-captum-20261002`. Full paths, byte counts, SHA-256 hashes, runtime versions and configuration are in `data/input_manifest.json`; additional text-input hashes are in `data/serialization_audit.json`. Pre-existing manuscript and tracked edits were preserved and hashed in `data/preexisting_state.json`.

The full WANDS catalog contains 42,994 IDs. The frozen held-out split has 384 queries; 76 have no retained Exact judgments, leaving 308 and 21,299 Exact query-product pairs. The 96 development queries were not substituted. Raw-label cleaning exactly reproduces processed qrels: 14 conflicting pairs excluded and 1559 remaining duplicate rows collapsed. Unjudged products are unknown, never called irrelevant.

The raw inputs are the saved Phase II catalog-wide Top-50 parquet files, truncated to the original first 20 IDs. Canonical inputs are saved graded-revision Top-1000 NPZ files mapped through the validated catalog axis. All 11 conditions, 3,388 query-run records have unique complete Top-20 IDs, known catalog IDs, identical judged Top-20 ranks, and Recall/nDCG agreement with saved results. No target-only ranks are used. All seven raw serializations match the original parser and frozen shuffle seeds, byte-for-byte source hashes and catalog ordering. Both canonical text hashes also reproduce. Comparisons use C0, C1, C2s1–C2s5 (seeds 20260911–20260915) and pure full-entry `(casefold(entry), entry)` lexical ascending/descending, never historical template M2 or JSON key sorting.

BGE is `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, native CLS pooling, 512-token cap, empty prefixes, FP16 model/FP32 pooling. The raw seven share encoder implementation v1; canonical ascending/descending share v3. Each within-family comparison uses the same saved query embeddings, catalog, judgments and tie rule. Ties use descending FP32 score then original catalog index (ascending product IDs in this catalog). Dense and hybrid are separate. Canonical hybrids reuse full-catalog BM25 and dense ranks with FP32 equal-weight RRF, constant 60; no truncated-list fusion is constructed here.

**Execution limitation:** Ascending inherited configured ceiling 48; descending effective ceiling 16 under frozen addendum. Scientific settings, encoder implementation, precision, query vectors and catalog match; numerical forward execution is not identical. Strict batch-matched saved canonical pair unavailable. Comparisons describe these fixed artifacts, not isolated causal sorting effects.

Missing planned artifacts: **none**. The additional condition needed for a strictly identical-execution sorting contrast is absent: WANDS/BGE ascending and descending canonical product encodings/rankings with the same effective batch ceiling, and corresponding hybrid rankings. No invented filename or alternative experiment is substituted. The historical 48-to-16 addendum is hashed and retained in the provenance records.

## Frozen extraction and catalog audit

`extraction_spec_v1.json` was frozen and the full-catalog extraction audited before option results were computed. Only exact generic raw keys `material`, `shape`, `color` are mapped, one field per key. Frame/upholstery/base/top/primary/main/detail keys and finish are not harmonized into generic keys. The census retains every raw key so narrower mapping and unused coverage remain inspectable.

Normalization is NFC, casefold, trim and whitespace collapse. Wood versus engineered wood, gray versus dark gray, and rectangle versus rectangular remain distinct. Pipe separates source entries; no within-value list delimiter is documented, so composite scalar strings (e.g. `microfiber / polyester`, `taupe , burnt orange`) remain single literal descriptors, not decomposed options. Distinct duplicate-key values are unknown; missing/unknown values and malformed entries also exclude the product-field. The specification gives explicit field-specific handling of `none`. No attributes are inferred from titles or generated by an LLM. These are recorded descriptor identities, not a harmonized physical ontology.

| Field | Catalog products | Raw key present | Valid | Unknown | Missing key | Conflicting values | Unknown sentinel |
|---|---:|---:|---:|---:|---:|---:|---:|
| material | 42994 | 8253 | 7576 (17.6%) | 35418 | 34741 | 677 | 0 |
| shape | 42994 | 8992 | 8868 (20.6%) | 34126 | 34002 | 115 | 9 |
| color | 42994 | 15214 | 11938 (27.8%) | 31056 | 27780 | 3218 | 59 |

Reason counts can overlap. The deterministic hash sample contains 30 present-key records per field (90 total): 28/29/23 valid and 2/1/7 conflicting unknowns for material/shape/color. All 90 source-entry checks passed, with **zero extraction errors** and no mapping revisions. `data/attribute_audit.jsonl` preserves complete original attributes; `data/attribute_audit_review.json` records semantic decisions, including opaque color name `soda` retained literally. Every processed catalog attribute list also matches the raw `product_features` pipe split. This is a source-fidelity audit, not independent physical validation.

## Estimands, eligibility and uncertainty

For each query, A is Top-20 intersected with the query’s Exact judgments. Cancellation is `|A_before − A_after| = |A_after − A_before| > 0`, using integer sets before any metadata filtering. A query-field is eligible only if both A sets are nonempty and every product in their union has valid, unambiguous values. Eligibility is symmetric; zero-change observations stay in the denominator. Option loss means a descriptor is absent from every relevant after-product, not merely carried by a departing product. Option gain is defined symmetrically. Equal option counts can conceal changed identities.

Each field and subgroup resamples whole eligible queries 10,000 times, NumPy PCG64 seed 20261007, and reports unadjusted 95% percentile intervals. Event rates and query-macro counts include zeros. These intervals condition on the eligible sample, fixed artifacts and frozen extraction; they omit annotation error, selection uncertainty and model uncertainty. Supports below 20 are flagged as tiny; every cancellation field support is tiny. Degenerate intervals, especially [0,0] for material, do not establish that events cannot occur. Intervals containing zero do not establish equivalence. Machine-readable summaries include gain, unchanged-set, set-change, size and net-change intervals, and degeneracy flags.

Full-ranking nDCG@20 is secondary and uses direct WANDS gains Exact=3, Partial=1, Irrelevant=0 and all judged gains for IDCG. Unjudged candidates contribute zero gain only to this metric while retaining unknown relevance. Recall cancellation is not an nDCG equality criterion.

## Membership and effectiveness: every comparison

All contrasts use 308 queries. Recall values are percentages; signed deltas and their intervals are percentage points. Gains/losses always use the named before-to-after direction. Ascending is a directional reference, not ground truth.

| Contrast | Changed membership | Unchanged membership | Cancellation | Changed, non-cancellation | Recall before -> after (%) | Delta Recall pp [CI] | Delta nDCG pp [CI] |
|---|---:|---:|---:|---:|---|---|---|
| canonical_dense | 188/308 | 120/308 | 65/308 | 123/308 | 36.072 -> 36.276 | 0.203 [-0.676, 1.156] | 0.428 [-0.155, 1.015] |
| canonical_hybrid | 164/308 | 144/308 | 66/308 | 98/308 | 37.450 -> 38.373 | 0.923 [0.190, 1.862] | 0.327 [-0.048, 0.707] |
| raw_C0_vs_C1 | 187/308 | 121/308 | 73/308 | 114/308 | 36.105 -> 36.199 | 0.094 [-0.400, 0.560] | 0.151 [-0.367, 0.662] |
| raw_C0_vs_C2s1 | 179/308 | 129/308 | 78/308 | 101/308 | 36.105 -> 36.196 | 0.092 [-0.614, 0.751] | 0.027 [-0.437, 0.484] |
| raw_C0_vs_C2s2 | 184/308 | 124/308 | 78/308 | 106/308 | 36.105 -> 36.124 | 0.019 [-0.855, 0.693] | 0.125 [-0.383, 0.630] |
| raw_C0_vs_C2s3 | 181/308 | 127/308 | 68/308 | 113/308 | 36.105 -> 36.249 | 0.144 [-0.358, 0.646] | 0.122 [-0.361, 0.607] |
| raw_C0_vs_C2s4 | 182/308 | 126/308 | 69/308 | 113/308 | 36.105 -> 35.642 | -0.463 [-1.317, 0.169] | -0.140 [-0.583, 0.311] |
| raw_C0_vs_C2s5 | 182/308 | 126/308 | 67/308 | 115/308 | 36.105 -> 35.752 | -0.352 [-1.368, 0.613] | -0.267 [-0.743, 0.191] |

Primary C0/C1 has 34 empty before-sets and 33 empty after-sets. Those field observations are ineligible, not zero changes. Exact cancellation reproduces 73/308; 121 have unchanged membership and 114 change membership without cancellation. `data/membership.jsonl` retains all queries, full relevant IDs, integer counts and both scores.

## Primary field results

All field-eligible evaluation queries:

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| raw_C0_vs_C1 | material | 20/308 | 0/20 (0.0%) [0.0, 0.0] | 1/20 (5.0%) | 19/20 (95.0%) | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] |
| raw_C0_vs_C1 | shape | 38/308 | 3/38 (7.9%) [0.0, 15.8] | 5/38 (13.2%) | 30/38 (78.9%) | 0.08 [0.00, 0.16] | 0.13 [0.03, 0.24] |
| raw_C0_vs_C1 | color | 37/308 | 12/37 (32.4%) [18.9, 48.6] | 13/37 (35.1%) | 20/37 (54.1%) | 0.59 [0.27, 0.97] | 0.62 [0.32, 0.97] |

Primary scientific subgroup: exact Recall cancellation. The n/N eligibility fractions are 4/73, 6/73 and 9/73; they are only 5.5%, 8.2% and 12.3% of the cancellation pool. Supports differ by field and are never pooled.

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| raw_C0_vs_C1 | material | 4/73 | 0/4 (0.0%) [0.0, 0.0] | 0/4 (0.0%) | 4/4 (100.0%) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C1 | shape | 6/73 | 1/6 (16.7%) [0.0, 50.0] | 2/6 (33.3%) | 3/6 (50.0%) | 0.17 [0.00, 0.50] | 0.33 [0.00, 0.67] |
| raw_C0_vs_C1 | color | 9/73 | 7/9 (77.8%) [44.4, 100.0] | 7/9 (77.8%) | 1/9 (11.1%) | 1.44 [0.67, 2.33] | 1.44 [0.67, 2.22] |

Color has equal mean lost and gained counts (1.44 each), yet 8/9 option sets change identity. Shape has three changed sets (1 loss event, 2 gain events) and three unchanged sets. Material has no changed option sets on four cancellation queries, despite product substitutions in all four. These positive, mixed and zero results are all retained.

## Exclusions and metadata coverage

Exclusions apply to the union of original relevant candidates, with no cancellation recomputation. The following reasons overlap and must not be added. `data/exclusion_summary.csv` also provides mutually exclusive reason combinations, which do sum to excluded query counts.

| Subgroup | Field | Excluded | Missing key | Conflicting values | Unknown sentinel | Empty before | Empty after |
|---|---|---:|---:|---:|---:|---:|---:|
| all | material | 288/308 | 246 | 24 | 0 | 34 | 33 |
| cancellation | material | 69/73 | 64 | 11 | 0 | 0 | 0 |
| all | shape | 270/308 | 235 | 6 | 1 | 34 | 33 |
| cancellation | shape | 67/73 | 67 | 2 | 1 | 0 | 0 |
| all | color | 271/308 | 227 | 54 | 1 | 34 | 33 |
| cancellation | color | 64/73 | 63 | 18 | 0 | 0 | 0 |

Coverage below is restricted to excluded primary query-fields. Candidate occurrences are query-specific (a product can recur across queries); macro fractions average nonempty query sets, and empty sets are reported separately.

| Subgroup | Field | Setting | Excluded queries | With missing candidates | Known/total candidate occurrences | Known fraction micro | Known fraction macro |
|---|---|---|---:|---:|---:|---:|---:|
| all | material | before | 288 | 252 | 357/2480 | 14.4% | 9.3% |
| all | material | after | 288 | 252 | 358/2500 | 14.3% | 9.9% |
| cancellation | material | before | 69 | 68 | 184/1031 | 17.8% | 15.3% |
| cancellation | material | after | 69 | 67 | 179/1031 | 17.4% | 15.1% |
| all | shape | before | 270 | 235 | 338/2262 | 14.9% | 9.8% |
| all | shape | after | 270 | 236 | 349/2288 | 15.3% | 9.9% |
| cancellation | shape | before | 67 | 67 | 153/979 | 15.6% | 12.5% |
| cancellation | shape | after | 67 | 67 | 161/979 | 16.4% | 13.1% |
| all | color | before | 271 | 234 | 739/2276 | 32.5% | 22.1% |
| all | color | after | 271 | 237 | 765/2296 | 33.3% | 22.7% |
| cancellation | color | before | 64 | 64 | 334/927 | 36.0% | 30.1% |
| cancellation | color | after | 64 | 64 | 338/927 | 36.5% | 30.5% |

Per-query missing IDs/reasons and before/after known fractions are retained in `data/query_fields.jsonl` and `data/exclusions.jsonl`. Observed descriptor arrays in incomplete records are descriptive coverage diagnostics only; they are not verified disappearance results. No incomplete-record loss percentage is presented. No combined “any field” percentage is computed on shifting supports.

## Five-shuffle robustness

Each row has its own cancellation subgroup and eligible denominator. The overlapping query groups are not independent samples and are not summed.

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| raw_C0_vs_C2s1 | material | 7/78 | 2/7 (28.6%) [0.0, 57.1] | 0/7 (0.0%) | 5/7 (71.4%) | 0.29 [0.00, 0.57] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s1 | shape | 11/78 | 3/11 (27.3%) [0.0, 54.5] | 3/11 (27.3%) | 6/11 (54.5%) | 0.27 [0.00, 0.55] | 0.27 [0.00, 0.55] |
| raw_C0_vs_C2s1 | color | 8/78 | 8/8 (100.0%) [100.0, 100.0] | 5/8 (62.5%) | 0/8 (0.0%) | 2.00 [1.62, 2.38] | 1.00 [0.38, 1.75] |
| raw_C0_vs_C2s2 | material | 5/78 | 1/5 (20.0%) [0.0, 60.0] | 0/5 (0.0%) | 4/5 (80.0%) | 0.20 [0.00, 0.60] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s2 | shape | 6/78 | 1/6 (16.7%) [0.0, 50.0] | 1/6 (16.7%) | 4/6 (66.7%) | 0.17 [0.00, 0.50] | 0.17 [0.00, 0.50] |
| raw_C0_vs_C2s2 | color | 8/78 | 4/8 (50.0%) [12.5, 87.5] | 5/8 (62.5%) | 2/8 (25.0%) | 1.00 [0.25, 1.75] | 1.12 [0.38, 2.00] |
| raw_C0_vs_C2s3 | material | 3/68 | 1/3 (33.3%) [0.0, 100.0] | 0/3 (0.0%) | 2/3 (66.7%) | 0.33 [0.00, 1.00] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s3 | shape | 9/68 | 0/9 (0.0%) [0.0, 0.0] | 2/9 (22.2%) | 7/9 (77.8%) | 0.00 [0.00, 0.00] | 0.22 [0.00, 0.56] |
| raw_C0_vs_C2s3 | color | 8/68 | 7/8 (87.5%) [62.5, 100.0] | 5/8 (62.5%) | 1/8 (12.5%) | 1.25 [0.62, 2.12] | 1.12 [0.25, 2.38] |
| raw_C0_vs_C2s4 | material | 3/69 | 1/3 (33.3%) [0.0, 100.0] | 0/3 (0.0%) | 2/3 (66.7%) | 0.33 [0.00, 1.00] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s4 | shape | 11/69 | 1/11 (9.1%) [0.0, 27.3] | 1/11 (9.1%) | 9/11 (81.8%) | 0.09 [0.00, 0.27] | 0.09 [0.00, 0.27] |
| raw_C0_vs_C2s4 | color | 7/69 | 6/7 (85.7%) [57.1, 100.0] | 6/7 (85.7%) | 0/7 (0.0%) | 1.57 [1.00, 2.00] | 1.57 [0.86, 2.29] |
| raw_C0_vs_C2s5 | material | 3/67 | 0/3 (0.0%) [0.0, 0.0] | 0/3 (0.0%) | 3/3 (100.0%) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s5 | shape | 7/67 | 0/7 (0.0%) [0.0, 0.0] | 0/7 (0.0%) | 7/7 (100.0%) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s5 | color | 11/67 | 8/11 (72.7%) [45.5, 100.0] | 9/11 (81.8%) | 1/11 (9.1%) | 1.36 [0.73, 2.09] | 1.36 [0.73, 2.18] |

The color loss pattern occurs in every shuffle contrast (4/8 to 8/8), but the supports remain only 7–11 queries and are selected differently. Material loss ranges 0–2 events and shape loss 0–3; the reversal material zero does not generalize to every shuffle. These results support a recurring descriptive phenomenon, not stable population rates or consistent benefits/harms.

For completeness, all field-eligible evaluation-query results (including unchanged membership and non-cancellation queries) are:

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| raw_C0_vs_C2s1 | material | 21/308 | 3/21 (14.3%) [0.0, 28.6] | 1/21 (4.8%) | 17/21 (81.0%) | 0.14 [0.00, 0.29] | 0.05 [0.00, 0.14] |
| raw_C0_vs_C2s1 | shape | 39/308 | 3/39 (7.7%) [0.0, 17.9] | 3/39 (7.7%) | 34/39 (87.2%) | 0.08 [0.00, 0.18] | 0.08 [0.00, 0.18] |
| raw_C0_vs_C2s1 | color | 36/308 | 12/36 (33.3%) [19.4, 50.0] | 11/36 (30.6%) | 20/36 (55.6%) | 0.58 [0.31, 0.89] | 0.47 [0.22, 0.75] |
| raw_C0_vs_C2s2 | material | 21/308 | 3/21 (14.3%) [0.0, 28.6] | 0/21 (0.0%) | 18/21 (85.7%) | 0.14 [0.00, 0.29] | 0.00 [0.00, 0.00] |
| raw_C0_vs_C2s2 | shape | 39/308 | 3/39 (7.7%) [0.0, 17.9] | 4/39 (10.3%) | 32/39 (82.1%) | 0.08 [0.00, 0.18] | 0.10 [0.03, 0.21] |
| raw_C0_vs_C2s2 | color | 37/308 | 8/37 (21.6%) [8.1, 35.1] | 12/37 (32.4%) | 23/37 (62.2%) | 0.41 [0.16, 0.70] | 0.54 [0.27, 0.86] |
| raw_C0_vs_C2s3 | material | 21/308 | 3/21 (14.3%) [0.0, 28.6] | 2/21 (9.5%) | 16/21 (76.2%) | 0.14 [0.00, 0.29] | 0.10 [0.00, 0.24] |
| raw_C0_vs_C2s3 | shape | 38/308 | 1/38 (2.6%) [0.0, 7.9] | 3/38 (7.9%) | 34/38 (89.5%) | 0.03 [0.00, 0.08] | 0.08 [0.00, 0.18] |
| raw_C0_vs_C2s3 | color | 39/308 | 14/39 (35.9%) [20.5, 51.3] | 10/39 (25.6%) | 24/39 (61.5%) | 0.59 [0.31, 0.90] | 0.41 [0.15, 0.72] |
| raw_C0_vs_C2s4 | material | 20/308 | 2/20 (10.0%) [0.0, 25.0] | 2/20 (10.0%) | 16/20 (80.0%) | 0.10 [0.00, 0.25] | 0.10 [0.00, 0.25] |
| raw_C0_vs_C2s4 | shape | 39/308 | 2/39 (5.1%) [0.0, 12.8] | 3/39 (7.7%) | 34/39 (87.2%) | 0.05 [0.00, 0.13] | 0.08 [0.00, 0.18] |
| raw_C0_vs_C2s4 | color | 37/308 | 12/37 (32.4%) [18.9, 48.6] | 15/37 (40.5%) | 19/37 (51.4%) | 0.59 [0.30, 0.92] | 0.65 [0.35, 0.97] |
| raw_C0_vs_C2s5 | material | 20/308 | 2/20 (10.0%) [0.0, 25.0] | 1/20 (5.0%) | 17/20 (85.0%) | 0.10 [0.00, 0.25] | 0.05 [0.00, 0.15] |
| raw_C0_vs_C2s5 | shape | 39/308 | 0/39 (0.0%) [0.0, 0.0] | 4/39 (10.3%) | 35/39 (89.7%) | 0.00 [0.00, 0.00] | 0.10 [0.03, 0.21] |
| raw_C0_vs_C2s5 | color | 37/308 | 12/37 (32.4%) [18.9, 48.6] | 11/37 (29.7%) | 23/37 (62.2%) | 0.57 [0.27, 0.92] | 0.49 [0.22, 0.84] |

Cross-contrast bootstrap resamples whole query vectors on a fixed support complete and cancelling in all six raw contrasts. Common n is only 2 material, 2 shape and 4 color queries. Five-shuffle mean loss probabilities minus reversal are +0.20 [0.00, 0.40], -0.20 [-0.40, 0.00] and +0.15 [0.00, 0.45], respectively. These intervals are descriptive and too sparse to establish consistency or differences. Every individual contrast and the joint resampling support are saved in `data/joint_query_bootstrap.csv`; no query-contrast independence assumption is used.

## Canonical-rule comparison

All eligible queries (no equal-Recall restriction):

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| canonical_dense | material | 21/308 | 1/21 (4.8%) [0.0, 14.3] | 2/21 (9.5%) | 18/21 (85.7%) | 0.05 [0.00, 0.14] | 0.10 [0.00, 0.24] |
| canonical_dense | shape | 38/308 | 3/38 (7.9%) [0.0, 18.4] | 4/38 (10.5%) | 32/38 (84.2%) | 0.08 [0.00, 0.18] | 0.11 [0.03, 0.21] |
| canonical_dense | color | 34/308 | 9/34 (26.5%) [11.8, 41.2] | 9/34 (26.5%) | 22/34 (64.7%) | 0.44 [0.18, 0.74] | 0.44 [0.18, 0.76] |
| canonical_hybrid | material | 23/308 | 0/23 (0.0%) [0.0, 0.0] | 1/23 (4.3%) | 22/23 (95.7%) | 0.00 [0.00, 0.00] | 0.04 [0.00, 0.13] |
| canonical_hybrid | shape | 37/308 | 0/37 (0.0%) [0.0, 0.0] | 2/37 (5.4%) | 35/37 (94.6%) | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.22] |
| canonical_hybrid | color | 34/308 | 5/34 (14.7%) [2.9, 26.5] | 6/34 (17.6%) | 26/34 (76.5%) | 0.21 [0.06, 0.41] | 0.26 [0.06, 0.50] |

Canonical exact-cancellation subgroups, reported separately:

| Contrast | Field | Eligible / subgroup N | Lost: n/N (%) [95% CI, %] | Gained: n/N (%) | Unchanged options: n/N (%) | Mean lost [CI] | Mean gained [CI] |
|---|---|---:|---|---|---|---|---|
| canonical_dense | material | 4/65 | 0/4 (0.0%) [0.0, 0.0] | 0/4 (0.0%) | 4/4 (100.0%) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| canonical_dense | shape | 10/65 | 1/10 (10.0%) [0.0, 30.0] | 1/10 (10.0%) | 8/10 (80.0%) | 0.10 [0.00, 0.30] | 0.10 [0.00, 0.30] |
| canonical_dense | color | 5/65 | 5/5 (100.0%) [100.0, 100.0] | 3/5 (60.0%) | 0/5 (0.0%) | 1.40 [1.00, 2.20] | 1.00 [0.20, 2.00] |
| canonical_hybrid | material | 3/66 | 0/3 (0.0%) [0.0, 0.0] | 1/3 (33.3%) | 2/3 (66.7%) | 0.00 [0.00, 0.00] | 0.33 [0.00, 1.00] |
| canonical_hybrid | shape | 9/66 | 0/9 (0.0%) [0.0, 0.0] | 0/9 (0.0%) | 9/9 (100.0%) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| canonical_hybrid | color | 4/66 | 3/4 (75.0%) [25.0, 100.0] | 1/4 (25.0%) | 1/4 (25.0%) | 0.75 [0.25, 1.00] | 0.25 [0.00, 0.75] |

Dense ascending versus descending changes relevant membership in 188/308 queries (65 cancellations). Overall strict supports are 21 material, 38 shape and 34 color queries: losses occur in 1/21, 3/38 and 9/34, with gains in 2/21, 4/38 and 9/34. Descending-minus-ascending Recall@20 is +0.203 pp [-0.676, +1.156]; the interval does not establish equivalence.

Hybrid ascending versus descending changes relevant membership in 164/308 queries (66 cancellations). Overall strict supports are 23/37/34: losses occur in 0/23, 0/37 and 5/34, gains in 1/23, 2/37 and 6/34. Recall is 37.450% versus 38.373%, difference +0.923 pp [+0.190, +1.862]. This unadjusted interval is conditional on the saved artifacts and does not select a globally superior rule. The hybrid can show a mean Recall gain while some eligible queries lose recorded color descriptors.

These are differences **between** fixed canonical representations. They do not show nonzero VI within either invariant method, nor isolate sorting from the documented batch-execution difference. Full field eligibility and zero-result rows are reported above rather than selecting favorable fields.

## Verified cases and figure

`CASE_AUDIT.md` verifies query 231 (“anti fatigue mat”): 5 products leave and 5 enter, Exact Recall remains 20/307 and nDCG@20 remains 1. Departing Anette Salon ID 19170 is the only relevant before-product recording `semi-circle`; no relevant after-product supplies that shape. Foam and black remain supplied. Material is complete and unchanged; color is ineligible due to one conflicting and one missing record. The full-list audit supports a recorded-shape disappearance; the two-row product example alone could not establish it.

Additional cases are deterministically hash-selected: query 341 illustrates color loss; query 453 illustrates unchanged material options despite cancellation. They are field-specific and include every before/after relevant ID, full original attributes, ranks and extraction provenance in `data/cases.jsonl`. No category was replaced with a maximum-effect example.

`table_option_coverage.tex` is an unintegrated manuscript-candidate table; `figures/option_coverage.pdf` is the vector figure with `option_coverage.png` preview. Both derive directly from `data/option_summary.csv`; the figure repeats every support denominator and separates metadata eligibility from eligible-query event rates.

## Scope and unmeasured outcomes

Supported claim: among the observed field-complete Exact candidate sets in these frozen WANDS/BGE retrieval artifacts, relevant-product substitution can change which literal recorded attribute descriptors are available despite exact Recall@20 cancellation. This happens for both losses and gains, and some substitutions leave descriptors unchanged. Different canonical fixed indexes also retain different descriptors. This is not total catalog diversity or an estimate over all relevant real-world products.

Unmeasured: utility, user satisfaction, fairness, sales, downstream agent choices, physical truth/completeness, annotation error, unjudged relevance, combination-of-preference satisfaction, population representativeness, and model uncertainty. Sparse generic keys, conflicting entries, unharmonized synonyms and composite labels can all limit interpretation. Eligibility selection can correlate with product categories and retrieval outcomes; no missing-at-random claim is made. Do not interpret an appearing option as better or a disappearing option as harm.

## Candidate results paragraph (computed evidence only)

Suggested placement: RQ2, after cancellation results; retain the quantitative table in the appendix. The following English paragraph is 106 words:

> A post-hoc WANDS/BGE diagnostic reproduced 73 exact Recall@20 cancellations among 308 queries. Requiring complete recorded metadata for both relevant candidate sets left only 4 material, 6 shape, and 9 color cases. Recorded options disappeared in 0/4, 1/6, and 7/9 cases, respectively, while option sets remained unchanged in 4/4, 3/6, and 1/9. Color gains also occurred in 7/9 cases; these shifts are not evidence of utility gains or losses. Color disappearance recurred across five saved shuffle contrasts, but small, varying supports limit generalization. Fixed canonical rules also retained different recorded options (RQ3). These exploratory results motivate an appendix diagnostic rather than a population-level claim about preference satisfaction.

Suggested short RQ3 cross-reference: “The appendix option audit also finds descriptor differences between ascending and descending fixed canonical indexes, under limited metadata support and the recorded execution caveat; this does not contradict within-rule invariance.”

## Reproduction and deliverables

See `REPRODUCE.md` for exact commands, dependency versions, artifact paths and missing-artifact behavior. Focused tests cover loss redundancy, equal-count identity changes, deduplication, missing/conflicting exclusion, identical lists, before/after symmetry, integer cancellation, empty sets, unknown IDs and normalization. `qa/validation.json` verifies output identities, input hashes, exact table cells, counts, bootstrap behavior, and preservation of pre-existing files. `qa/visual_review.json` records PDF rendering inspection.
