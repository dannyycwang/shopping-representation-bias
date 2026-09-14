# Result Provenance

Every rounded number in the main paper comes from the following machine-readable sources. Percentages in the paper equal the stored proportion multiplied by 100.

| Paper location | Source | Fields / filter |
|---|---|---|
| Abstract, Table 1, Fig. 2 general retrievers | `../phase2/results/phase2_tables/table1_representation_sensitivity_native.csv` | all rows; `rank_range_median`, `VI@20_micro`, `VI@20_query_macro`, CI fields |
| Abstract, Table 1, Fig. 2 e-commerce retriever | `../phase3/results/phase3_tables/ecommerce_retriever_sensitivity.csv` | both rows; matching sensitivity and cNDCG fields |
| Official ESCI paragraph and appendix | `../phase3/results/phase3_tables/esci_official_candidate_pool.csv` | all retrievers; `NDCG@10`, `VI@10_micro`, `VI@20_micro`, macro fields |
| Reranker paragraph, Table 2, Fig. 3 | `../phase3/results/phase3_tables/wands_reranker_pipeline.csv`; `esci_reranker_pipeline.csv` | `common_query_macro_*`, `Irrecoverable@100_*`, rank ranges |
| Table 3 and Fig. 4 | `../phase3/results/phase3_tables/robustness_relevance_pareto.csv` | methods Original, M1 factual, M2 canonical, set_mean |
| Appendix late variants | same Pareto CSV | methods late_mean, late_max, late_top3 |
| Numerical invariance tolerance | `../phase3/results/progress.json` and method summaries in `../phase3/results/phase3_invariant_method/` | `p4_max_score_diff` |
| Historical whole-catalog example, retained in RQ2 | `../figure1_candidates.csv` | query `turquoise chair`, product 34536; C2s4 rank 3 and C2s1 rank 1303; these are NOT the revised Figure 1 ranks |
| Encoding-control ranges | `../phase2/results/phase2_tables/chunk_pooling_controls.csv` | WANDS, MiniLM/BGE profiles |
| Dataset and query counts in Experimental Setup | `../phase2/results/dataset_statistics.json` | WANDS and ESCI `products`, `queries`, and `judgments` |
| RQ2 relevance-stratum contrasts | `../phase2/results/phase2_tables/relevance_stratum_contrasts_native.csv` | Exact-minus-Irrelevant and E-minus-I rows; `delta`, CI, and `holm_within_dataset` |
| RQ2 single-product intervention ranges | `../phase2/results/phase2_tables/single_product_summary_native.csv` | `subset=highest`; M1/M2 `single_crossing@20` and `full_crossing@20` |
| Fact-equivalence audit statements | `../phase2/results/equivalence/wands_equivalence_summary.csv`; `esci_equivalence_summary.csv` | primary C0/C1/C2 rows and equality/audit counts |

The audit script `../phase3/scripts/audit_metrics.py` recomputes primary Phase II values from lower-level outputs. `../phase3/tests/test_metric_reconciliation.py` and `../phase3/tests/test_phase3_outputs.py` assert reconciliation and Phase III invariants.

## Final focused revision (2026-09-06)

Table numbers above describe the pre-revision manuscript. The detailed catalog-wide sensitivity table is now in Appendix B (`tab:sensitivity`), while Figure 2 remains in the main text. Pipeline and mitigation tables retain their original values. No validated Phase III result artifacts were changed.

| Revised claim | Artifact and fields |
|---|---|
| Abstract 4.4–13.3% | Original Phase II `table1_representation_sensitivity_native.csv`, three general retrievers, `VI@20_micro`: minimum 4.42%, maximum 13.26%, rounded to one decimal; catalog-wide intervention only |
| Figure 1, introduction, RQ2: ranks 5 and 1,527, range 1,522 | `figures/figure1_target_only_source.csv`; `../phase3/results/target_only_permutations/wands_minilm_pairs.parquet`, query_id=162, product_id=34536, C2s4=5, C2s1=1527, C0=486 |
| Target-only six-cell VI@20 and macro CIs in RQ2 and Appendix C | `../phase3/results/target_only_permutations/summary.csv`, `VI@20_micro`, `VI@20_query_macro`, `VI@20_ci_low`, `VI@20_ci_high`; proportions multiplied by 100 |
| All target ranks and signed C0-relative changes | `../phase3/results/target_only_permutations/*_pairs.parquet`, C0/C1/C2s1–C2s5, delta_*; positive delta means worse rank |
| Exact C0 reconciliation | `../phase3/results/target_only_permutations/provenance.json`, each source: max_score_error=0 and C0_rank_exact_match=true |

New analysis: `../phase3/scripts/analyze_target_only_permutations.py`. Figure generation: `../phase3/scripts/make_target_only_figure.py`. The intervention changes one target at a time against C0 competitors, excludes its former entry, and resolves score ties by original catalog index. It reuses frozen scores and cached embeddings; no models were retrained or re-encoded. Query bootstrap uses 10,000 resamples and seed 20260963. The new rank algorithm is checked against explicit stable reranking in `../phase3/tests/test_target_only_permutations.py`.

The target-only analysis is post hoc and distinct from both the original catalog-wide experiment and earlier M1/M2 single-product interventions. Source/output SHA-256 records are stored in `../phase3/results/target_only_permutations/file_hashes.json`.

## Expanded 10-page main text (2026-09-06)

This is a presentation expansion only. The main PDF contains ten body pages and one references page; the appendix is compiled separately as `supplementary.pdf`, with tables S1 and S2. Figure/table numbers in earlier audit entries are historical; use labels and artifact names below for the current mapping.

| Current addition | Verified existing source |
|---|---|
| RQ1 encoding-control table (`tab:encodingcontrols`) | `../phase2/results/phase2_tables/chunk_pooling_controls.csv`; WANDS rows, `VI@20_micro` × 100; all five profiles for MiniLM/BGE |
| Figure 3, target-only cutoff profiles (`fig:targetcutoffs`) | `../phase3/results/target_only_permutations/summary.csv`; `VI@10_micro`, `VI@20_micro`, `VI@50_micro` × 100; source snapshot `figures/figure5_target_cutoffs_source.csv` |
| RQ2 target-only median ranges | Same summary, `rank_range_median`, MiniLM/BGE/GTE: WANDS 210/83/105; ESCI 0/1/2 |
| Full motivating-case table (`tab:caseorders`) | `figures/figure1_target_only_source.csv` and `../phase3/results/target_only_permutations/wands_minilm_pairs.parquet`; query 162, product 34536; all seven ranks, `whole_*`, `delta_*` |
| Candidate-coverage table (`tab:coverage`) and explanation | `../phase3/results/phase3_tables/{wands,esci}_reranker_pipeline.csv`; `highest_pairs`, `ever_in_top100`, `common_top100`; 3856 and 44 computed as ever minus common; conditional rates from `Irrecoverable@100_given_ever_retrieved` × 100 |
| RQ3 C0 recall before/after reranking | Same pipeline CSVs, `all_highest_C0_dense_recall@20` and `all_highest_C0_rerank_recall@20` × 100, rounded to two decimals; no significance asserted |
| Figure 6, complete effectiveness intervals (`fig:effectivenessci`) | `../phase3/results/phase3_tables/robustness_relevance_pareto.csv`; `delta_cNDCG@10`, `cNDCG_ci_low`, `cNDCG_ci_high`; source snapshot `figures/figure6_effectiveness_intervals_source.csv` |
| RQ4 late-aggregation negative results | Same Pareto CSV, late_mean/late_max/late_top3, `cNDCG@10` and CI fields; same values previously in appendix |

No new empirical values are introduced by the conceptual explanations, estimator examples, or proposed evaluation practice. New diagrams are plots of frozen summaries, not new model runs. The complete presentation checksum record is `expanded_revision_manifest.json`.
