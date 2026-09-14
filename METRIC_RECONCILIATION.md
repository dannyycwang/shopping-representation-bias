# Metric Reconciliation (P0 Hard Gate)

Status: **PASS after correction**  
Audit date: 2026-09-06  
Independent implementation: `phase3/scripts/audit_metrics.py`  
Machine-readable audit: `phase3/results/phase3_audit/`  
Unit tests: `phase3/tests/test_metric_reconciliation.py`

## Executive finding

The Phase II primary cNDCG and VI values are correct. The apparent WANDS discrepancy had two exact causes:

1. The former SAR implementation inner-joined all-query cNDCG with highest-label VI. It therefore averaged WANDS cNDCG over only 379 queries containing an Exact product, instead of the 479 relevance-eligible queries used in Table 3. This produced MiniLM 0.738094 rather than 0.779260. The implementation is corrected to left-join VI and use a zero highest-label instability penalty for queries with no highest-label pair. SAR(λ=0) now equals the reported relevance component to floating-point tolerance.
2. Phase I MiniLM R0 used 254-token, no-overlap, full-document chunk mean pooling. Phase II's headline C0 result uses native first-256-token mean pooling. Phase II's matched `mean_254_o0` control is 0.738328 and reproduces Phase I's 0.738332 to 0.0000044. The 0.779260 headline value is a different, explicitly named encoder profile.

No primary Phase II representation-sensitivity result changed. The correction changes only the SAR query population and its dependent SAR CSV values.

## Table A — Metric definitions

| Metric | Dataset | Gain mapping | Query count | Unjudged handling | Aggregation | Implementation |
|---|---|---|---:|---|---|---|
| cNDCG@10/20 | WANDS | Exact=3, Partial=1, Irrelevant=0 | 479 of 480 with IDCG>0 | Remove unjudged products before rank discounting | Mean of per-query cNDCG | `phase2/src/evaluation.py`; independently recomputed in `phase3/scripts/audit_metrics.py` |
| cNDCG@10/20 | ESCI union catalog | E=1, C=0.1, S=0.01, I=0 | 500 | Remove unjudged products before rank discounting | Mean of per-query cNDCG | Same as above |
| Recall@20 | WANDS/ESCI | Positive iff gain>0 | 479/500 | Complete retrieval catalog retained; denominator is known positive judgments | Mean of per-query known-positive recall | Same as above |
| HighestRecall@20 | WANDS | Highest label=Exact | 379 | Complete catalog retained | Mean over queries containing Exact | Same as above |
| HighestRecall@20 | ESCI | Highest label=E | 499 | Complete union catalog retained | Mean over queries containing E | Same as above |
| RelevantHidden@20 | WANDS/ESCI | Highest label as above | 379/499 | Complete catalog retained | `1 - HighestRecall@20` per eligible query, then mean | Same as above |
| VI@20 micro | WANDS/ESCI | Highest-label pairs only | 25,470 WANDS pairs / 4,434 ESCI pairs | Uses stored complete-catalog ranks for all seven variants | Pair-weighted mean of crossing indicators | `phase2/scripts/analyze_sensitivity.py`; independent merge in P0 audit |
| VI@20 query-macro | WANDS/ESCI | Highest-label pairs only | 379/499 eligible queries | Same ranks as micro | First mean within query, then equal-weight query mean | Same as above |
| Query-macro VI 95% CI | WANDS/ESCI | Same as query-macro VI | 379/499 | Not applicable | 10,000 query bootstrap samples | `phase2/src/evaluation.py::mean_bootstrap` |
| SAR(λ) | WANDS/ESCI | cNDCG mapping above; highest-label VI penalty | 479/500 after fix | cNDCG is condensed; VI uses complete ranks | Mean of `cNDCG@10 - λ·VI@20_query`; missing highest-label VI is 0 | `phase2/scripts/analyze_mitigations.py` |

### Candidate catalogs and ranking details

- WANDS: all 42,994 products; 480 queries; 231,859 unique non-conflicting judgments. One query has no positive gain and is excluded automatically from cNDCG means.
- ESCI Phase II: 10,076-product union of products judged for the deterministically selected 500 test queries. It is not yet the official per-query candidate-pool experiment.
- Duplicate identical judgments are collapsed; 14 WANDS query-product pairs with conflicting labels are excluded.
- Scores are sorted descending with a stable sort. Catalogs are sorted deterministically, so score ties resolve by product order/product ID.
- WANDS and ESCI grades are used as linear DCG gains in Phase II. Phase I stored WANDS grades 2/1/0 and transformed them using `2^grade-1`, yielding the same effective 3/1/0 gain schedule.

## Table B — Independently recomputed C0 native values

Every value below is reconstructed from stored judged-pair ranks and judgments, without calling the Phase II evaluation function.

| Dataset | Retriever | Metric | Existing value | Recomputed value | Difference | Explanation |
|---|---|---:|---:|---:|---:|---|
| WANDS | MiniLM | cNDCG@10 | 0.779259909262 | 0.779259909262 | <1e-15 | Exact match |
| WANDS | BGE-base | cNDCG@10 | 0.827696491629 | 0.827696491629 | <1e-15 | Exact match |
| WANDS | GTE-ModernBERT | cNDCG@10 | 0.821896465712 | 0.821896465712 | <1e-15 | Exact match |
| ESCI | MiniLM | cNDCG@10 | 0.685337076412 | 0.685337076412 | <1e-15 | Exact match |
| ESCI | BGE-base | cNDCG@10 | 0.712445470359 | 0.712445470359 | <1e-15 | Exact match |
| ESCI | GTE-ModernBERT | cNDCG@10 | 0.716782838474 | 0.716782838474 | <1e-15 | Exact match |
| WANDS | MiniLM | Micro VI@20 | 0.132626619552 | 0.132626619552 | <1e-15 | 25,470 Exact pairs, pair weighted |
| WANDS | MiniLM | Query-macro VI@20 | 0.191675741413 | 0.191675741413 | <1e-15 | 379 Exact-eligible queries, query weighted |

The full six-cell audit also reproduces cNDCG@20, Recall@20, HighestRecall@20, RelevantHidden@20, micro VI@20, and query-macro VI@20. The largest absolute difference across audited primary fields is `1.11e-16`. Complete values are in `phase3/results/phase3_audit/recomputed_metrics.csv` and `summary_reconciliation.csv`.

## Table C — Phase I, Phase II, and former SAR

| Source | WANDS MiniLM C0/R0 cNDCG@10 | Query population | Encoder profile | Exact reason for difference |
|---|---:|---:|---|---|
| Phase I R0 Dense | 0.738331940850 | 479 relevance-eligible queries | 254-token non-overlapping chunks, token-count-weighted mean across chunks | Baseline historical implementation |
| Phase II C0 `mean_254_o0` | 0.738327519140 | 479 | Matched 254/0 chunk mean profile | Reproduces Phase I; residual −0.00000442 is attributable to the revised encoder implementation, seed/batching order, and fp16 numerical ordering |
| Phase II C0 `native` | 0.779259909262 | 479 | Native MiniLM, truncated to 256 tokens, attention-mask mean pooling | Different prespecified profile; this is the value in Phase II Table 3 |
| Former SAR λ=0 | 0.738094 (old file) | 379 Exact-eligible queries | Phase II native embeddings | Bug was query-population restriction from an inner join, not an encoder or gain change |
| Corrected SAR λ=0 | 0.779259909262 | 479 | Phase II native embeddings | Now exactly equals the all-query cNDCG relevance component |

The numerical similarity between former SAR MiniLM (0.738094) and Phase I chunked MiniLM (0.738332) was coincidental. They arose from different causes: one from query selection, the other from encoding profile.

## Micro/macro confidence-interval audit

The six primary Phase II rows correctly store separate fields:

- `VI@20_micro`
- `VI@20_query_macro`
- `VI@20_macro_ci_low`
- `VI@20_macro_ci_high`

For WANDS MiniLM, the coherent report is:

- Micro VI@20: **13.26%**
- Query-macro VI@20: **19.17%**
- Query-macro 95% CI: **[17.09%, 21.27%]**

The old Phase II generated table that labeled a column only `VI@20` beside the macro CI was ambiguous even though its stored fields were correct. Final-paper tables must use the three explicit labels above. The publication Figure 1 uses the query-macro point with its query-macro CI and is coherent.

## Correction log

1. Added an independent low-level recomputation in `phase3/scripts/audit_metrics.py`.
2. Added tests enforcing native WANDS MiniLM cNDCG, distinct micro/macro estimands, and SAR λ=0 equality.
3. Changed `phase2/scripts/analyze_mitigations.py` from an inner join to a left join for SAR, with missing highest-label VI set to zero.
4. Regenerated `phase2/results/phase2_tables/stability_adjusted_relevance_native.csv`; it now includes a `queries` provenance column and uses 479 WANDS / 500 ESCI queries.
5. Confirmed that Phase II Table 3, Table 4, direct-intervention results, VI figures, and the central cross-dataset finding do not change.

## P0 exit-gate checklist

- [x] All important Phase II values reproduced from stored pair ranks and judgments.
- [x] SAR(λ=0) equals its relevance component within floating-point tolerance.
- [x] Micro and query-macro VI are separate and correctly labeled.
- [x] Phase I versus Phase II MiniLM difference traced to encoder profile.
- [x] Former SAR difference traced to query eligibility and corrected.
- [x] No unexplained metric drift remains in the audited core metrics.

**Decision: P0 passes. Phase III main experiments may begin.**
