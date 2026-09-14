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


## Retrieval-first extension (2026-09-07): authoritative current mapping

Earlier entries describe archived manuscript versions. The current manuscript follows the official 8-main/12-total research-track limit and integrates an appendix in one PDF: 8 main pages, 11 total pages. Old result files are preserved. ESCI effectiveness below supersedes historical effectiveness **only in the revised presentation**, because the old Substitute/Complement gains were reversed relative to the benchmark paper.

| Current claim/table/figure | Artifact relative to repository root; fields and scope |
|---|---|
| Abstract / RQ1 general-encoder catalog VI range | `phase2/results/phase2_tables/table1_representation_sensitivity_native.csv`, `VI@20_micro` min/max, six cells, seven serializations |
| Cross-model visualization | Regenerated native-column `paper_www2027/figures/figure2_cross_model_vi.pdf` (same validated values); Phase II sensitivity table and Phase III `ecommerce_retriever_sensitivity.csv`, macro VI and query CI; no effectiveness values used |
| Introduction / Figure 1 / all case ranks | `phase3/results/target_only_permutations/wands_minilm_pairs.parquet`, query 162 product 34536; figure1 target-only source CSV; `phase4/results/illustrative_case.json` raw title and atoms |
| Target-only primary-family aggregate | `phase3/results/target_only_permutations/summary.csv`, VI@20 micro/macro; generated `sensitivitydetail.tex` |
| RQ1 always/sometimes/never counts and appendix | `phase4/results/{wands,esci}_visibility_states.csv`, K/state/pairs/macro/CI; all historical queries. WANDS 13,230 never=25,470−12,240; ESCI 395=4,434−4,039 |
| Rank-conditioned Figure and RQ2 differences | `phase4/results/*_rank_controlled_contrasts.csv`, `delta_macro_vi20`, CI and common_queries; query common support within original rank band |
| RQ2 full-input and truncated strata | `phase4/results/wands_diagnostic_strata.csv`, factor fully_fits, label Exact, level True/False, macro_vi20/CI/pairs/queries; underlying `*_diagnostic_pairs.parquet` |
| Other factor diagnostics | `phase4/results/*_diagnostic_strata.csv`, `additional_diagnostic_strata.csv`; margin, length, multiplicity, query length and literal overlap. Display bins are descriptive, not strategy tuning |
| Development selection / all unsuccessful screening configurations | `phase4/results/development_selection.csv`, frozen `phase4/config/selection.json`; Original/hybrid development per-query Recall@100 means |
| Main baseline recall/effectiveness table | `phase4/results/{dataset}_{encoder}_retrieval_summary.csv`, mean/CI/query eligibility; WANDS held-out only; ESCI standard gains; generated recall_main.tex |
| Transfer table and result differences | `phase4/results/*_primary_contrasts.csv`, method_a/method_b, Recall@100 delta/CI; `presentation_generalization.csv`; fixed selected rule transferred unchanged |
| Selected rule target-only net gain/crossing | `phase4/results/*_target_only_summary.csv`, K=100, net_macro_recall/CI/crossing_micro; rank-level selected_rule_target_only.parquet |
| Set mean and canonical comparisons / negative late variants | `phase4/results/repaired_legacy_mitigation.csv`; cNDCG@10 and paired delta/CI. Direct WANDS comparison has 383 positive-IDCG queries. No equivalence claim |
| Gain repair | `phase4/results/esci_gain_repair.csv` and per-query parquet, 79 historical rank artifacts; benchmark source https://arxiv.org/pdf/2206.06588 Section 3.1; pinned official code discrepancy in BRIEF_AUDIT.md |
| Recall-budget curve / Original budget needed | `phase4/results/*_retrieval_summary.csv` Recall@K, K=20/50/100/200/500/1000; `recall_equivalent_candidate_budget.csv` discrete grid match, not exact cost matching |
| Main cost table / full method timing | `phase4/results/*_exact_timing.csv` actual four-thread score/sort times and vector/posting bytes; `*_query_encoding_timing.csv`; summaries presentation_cost.csv |
| Estimated processing-cost curve | `phase4/results/recall_cost_frontier.csv`; query encoding + exact retrieval + K times amortized pair inference; first-stage recall, not final recall or online latency |
| Main multi-view cost table | `phase4/results/saturation_summary.csv` and `*_exact_timing.csv`; fixed m=2/4/7, no winner reselection |
| Shared-query numerical audit | `phase4/results/shared_query_cache_audit.csv`; historical reconstruction versus common-query extension condition |
| Query/product transfer scope and field coverage | `phase4/results/generalization_scope.json` and `selected_rule_coverage.csv`; no designed product-disjoint evaluation |
| Reranking budget contrasts | `phase4/results/reranking_budget_contrasts.csv`; paired query differences and bootstrap intervals |
| Hardware, tests, and final layout | `phase4/config/hardware.json`, `phase4/results/integrity_tests.log`, `pdf_validation.json`; 9 tests, 8 main / 11 total pages |
| Offline encoding cost | `phase4/results/encoding_costs.csv` from content-addressed embedding metadata; historical cached build times are not claimed as matched cold builds |
| Unique view counts / saturation | `phase4/results/unique_view_summary.csv`, underlying product-level parquet; `saturation_summary.csv` from complete exact ranks, no new selection |
| Canonical reranker table / appendix | `phase4/results/*_canonical_reranking_summary.csv`; per-query CSV and Top-20 parquet; paired contrasts `canonical_reranking_paired_contrasts.csv`; timing JSON records measured batched throughput |
| K=500 extension decision | `phase4/results/reranking_extension_decision.json`: frozen development threshold and measured local cost; final K set is visible in reranker summary |
| Legacy common-pool reranker VI | `phase3/results/phase3_tables/{wands,esci}_reranker_pipeline.csv`, common_query_macro fields; varying-input condition, conditional population; no new fixed-text result substituted |
| Legacy candidate reconciliation | `phase4/results/legacy_candidate_reconciliation.csv` and per-query detail; zero highest-relevant membership changes, despite numerical/order differences |
| Rescued/newly missed/net | `phase4/results/transition_summary.csv` and per-query `*_transitions.csv`, same query highest population, K specified |
| Additional baseline contrasts | `phase4/results/descriptive_baseline_contrasts.csv`; explicitly descriptive beyond frozen primary contrasts; no selection or p-value screening |
| Historical dataset counts | `phase2/results/dataset_statistics.json`; new data counts `phase4/data/catalog_manifest.json`; metric-specific eligibility `evaluation_denominators.csv` |
| Split sizes/seeds/rules | `phase4/config/splits.json`, frozen protocol, source hashes; 96 development / 384 held-out WANDS and 100 newly sampled ESCI queries |

Generation chain: `complete_evidence.py` → `build_presentation.py` → `write_manuscript.py` → `write_results.py` → `write_appendix.py`, followed by reviewed main.tex and LaTeX compilation. Table values are rounded from the linked CSVs. Narrative values are checked against the same artifacts, with historical facts retained only where the audited intervention/denominator matches. SHA-256 inventory: `phase4/results/revision_file_hashes.json` (excludes itself and large embedding arrays; each embedding metadata file contains its input-content hash and model fingerprint). Original archived paper and prior provenance remain available under `phase4/archive/`.

## Optimization-robustness extension (2026-09-08): authoritative Phase V mapping

Phase V adds a frozen target-only audit of two adapted E-GEO prompts. It does not alter any Phase III/IV numerical artifact. Percentages in the manuscript are stored proportions multiplied by 100; confidence intervals are query-cluster bootstrap intervals over the eight frozen queries per dataset.

| Current claim/table/figure | Artifact relative to repository root; fields and scope |
|---|---|
| Frozen sample: 8 queries, 310 WANDS pairs/products; 8 queries, 111 ESCI pairs/products | `phase5/config/sample.json`; exact query/product IDs and source hashes |
| Raw, Canonical, G1, C→G1, G2, C→G2 target inclusion, VI, robust coverage, always/sometimes/never | `phase5/results/optimization_robustness_summary.csv`; dataset/model/method filters; target-only counterfactual indexes, not catalog Recall |
| Figure `fig:optimization` and Appendix `tab:optimization` | Same summary CSV; `phase5/scripts/build_presentation.py`; plot/table use BGE, K=100 query-macro mean/VI/robust coverage and pair-micro Never |
| Prespecified inclusion@100 contrasts and CIs | `phase5/results/primary_contrasts.csv`; `delta`, `ci_low`, `ci_high`, `p_raw`, `p_holm`, 8 query clusters |
| Every order | `phase5/results/per_order_inclusion.csv`; C0/C1/C2s1–C2s5 query-macro and pair-micro inclusion@20/100 |
| Rescued/newly missed statements | `phase5/results/rescued_missed.csv`; same fixed query-target-order trials relative to Raw |
| Factuality, truncation, output length, and generation time | `phase5/results/factuality_cost.csv` and `fact_failure_components.csv`; strict automated checker, not human ground truth |
| Common-valid support | `phase5/results/common_valid_support.csv`, `common_valid_metrics.csv`; all seven generated orders must pass on identical support |
| Guarded pipelines | `optimization_robustness_summary.csv`; `Guard*` rows; rank inputs and fallback flags in `phase5/results/*_target_ranks.parquet` and `*_primary_text_manifest.parquet` |
| Stochastic order and same-input controls | `phase5/results/generation_noise_order_effect.csv`, `generation_noise_same_input.csv`, `generation_noise_factuality.csv`; three replicates are not pooled |
| MiniLM encoder transfer | `optimization_robustness_summary.csv`, model=minilm; `primary_contrasts.csv`; identical generated strings reused |
| Full-catalog pure canonical Recall/cNDCG contrasts | `phase5/results/canonical_catalog_contrasts.csv`; frozen Phase IV exact ranks, query-paired bootstrap |
| Raw generation records, hashes, prompts, revision, decoding, fact checks | `phase5/results/generations.jsonl`; 6,736 primary task rows plus 384 noise task rows, 5,872 primary and 342 noise unique content-addressed generations |
| Method sources and protocol | `METHOD_SOURCE_AUDIT.md`, `OPTIMIZATION_ROBUSTNESS_PROTOCOL.md`, `phase5/config/protocol.sha256`; official repositories retained under `phase5/vendor/` |
| Cache/canonical/target-insertion/failure integrity | `phase5/tests/test_extension.py`, final integrity log and manifest |

Primary generation measured approximately 10.1 summed GPU inference hours and the stochastic control approximately 1.4 hours. This exceeded the frozen eight-hour stopping target. The run continued from content-addressed checkpoints after OOM-related batch-size reductions; no sample, prompt, model, decoding condition, or method choice was adapted to ranks, which were computed only after all generation and the frozen fact recheck completed. This nonadaptive protocol deviation is reported in the paper and revision report.
