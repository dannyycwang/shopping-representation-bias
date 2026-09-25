# Paragraph-level editorial handoff

Scope: v11 pages 6–9 and directly connected setup/appendix pointers. The matching component source is preserved in `inputs/reference_section6/`; a scoped replacement is in `manuscript/section6_revised.tex`. No entire-manuscript rewrite or overwrite is made.

| Original paragraph or asset | Action | Instruction | Verified result |
|---|---|---|---|
| p6 opening, lines 623–627 | rewrite | Use one short reporting-convention sentence; move repeated bootstrap definitions to setup. | section6_claim_ledger.csv; section6_scope_and_provenance.md |
| p6 RQ1 P1, lines 631–635 | keep | Catalog-wide VI ranges; retain finite-family interpretation. | tables/inclusion_complete.csv: catalog_wide/full/K20/VI |
| p6 RQ1 P2, lines 636–647 | rewrite | Separate fixed-competitor evidence from fully-fitting evidence and keep population warning. | tables/inclusion_complete.csv: target_only/K20; Figure 3 |
| p6 RQ1 P3, lines 648–656 | appendix | At most one main sentence from harmonized results; retain historical/runtime comparison and all fitting supports in supplement. Remove saturation/causal-runtime conflation. | harmonized_supplement.md; tables/historical_to_harmonized.csv |
| p7 Figure 3 | rewrite | One GTE point labelled Full = fully fitting; color-independent markers, counts, explicit different scales, no connecting line. | figures/figure3_target_fitting.pdf; data/figure3_plotted.csv |
| p6 RQ2 P1, lines 660–670 | rewrite | Separate macro cancellation from exact within-query equality. Give 73/308 reversal count and all-six range; never a union. | tables/membership_all_contrasts.csv; data/membership_per_query.parquet |
| p7 Figure 4 | rewrite | Display Reverse uniformly in every setting; mark it as an editorial choice after outcomes. Include exact replacement counts beside bars. Keep all shuffles in companion. | figures/figure4_membership_replacement.pdf; data/figure4_plotted.csv |
| RQ2 new compact illustration | appendix | Hash-selected anti fatigue mat case: exact unchanged 20/307 with five gained/five lost IDs. All orders in CSV. No all-fitting claim. | figures/illustrative_replacement_case.pdf; data/illustrative_case_selection.json |
| p6 RQ2 P2, lines 671–680 | rewrite | Keep conditional absolute nDCG and unconditional signed means distinct. Remove the 5-of-36 significant-count sentence; use descriptive intervals without global inference. | tables/membership_all_contrasts.csv; tables/controls_paired_comparisons.csv |
| p6 RQ3 P1, lines 685–690 | keep | State structural invariance once; effectiveness and identities remain empirical. | tables/controls_complete.csv; method metadata |
| p8 Table 1 | rewrite | Keep raw dense, all three canonical rules, BM25, raw hybrid, all canonical hybrids and Set-Mean. Move centroid/max budgets to appendix. Explicit candidate-vs-ranking cutoff rationale. | tables/main_table_candidate.tex; tables/method_complete_companion.csv |
| p6 RQ3 transition P2, lines 691–696; p8 lines 848–852 | rewrite | Add 0.43% persistent omission to inclusion alongside 0.23% opposite cell; explain crossing mass prevents subtracting only these cells for net Recall. | tables/transitions_complete.csv; data/transitions_per_query.parquet |
| p8 duplicate transition paragraph, lines 853–856 | omit-from-submission | Delete duplicate beginning with a stray period. | v11 PDF; inputs/section6_v11_raw.txt |
| p9 Table 2 | keep | Exactly one main six-cell dense/hybrid table, same convention-based ascending example. | tables/transition_single.tex; data/table2_plotted.csv |
| p9 Table 3 | omit-from-submission | Delete duplicate; update main text reference to the retained transition-table label. | v11 duplicate cells checked in claim ledger |
| p8 RQ3 effectiveness P3, lines 857–862 | rewrite | Name raw-seven means; report effect sizes and paired intervals without inferring a general trade-off. | section6_audit.md named contrasts; tables/controls_paired_comparisons.csv |
| p8 RQ3 canonical hybrid P4, lines 863–909 across columns | rewrite | Replace significance counting with named comparator effects; retain ascending HYBRID Recall@20 loss against raw-hybrid seven mean. No equivalence inference. | tables/controls_paired_comparisons.csv |
| p8 RQ3 cost P5, lines 910–916 | rewrite | Centroid: M construction encodings, one stored/scored vector; max: M stored/scored vectors. Preserve warm-cache CPU/excludes-query/exact/not-ANN caveats. | revision_completion_20260921/data/strategy_costs.csv; controls metadata |
| p6 setup, lines 581–596 and 599–620 | rewrite | Primary full population plus secondary fitting restriction; canonical rules are in methods Section 4; replace tDCG, specify direct gains/IDCG/coverage and prior feedback. | manuscript/setup_corrections.tex; section6_scope_and_provenance.md |
| Appendix native score-margin example | appendix | Optional; retain existing numerical guard and historical discrepancy. No expansion of IG/attention. | revision_final_strengthening_20260922/BOUNDARY_AUDIT.md |
| Appendix C.4 attribution plus main pointers | omit-from-submission | Proposed submission fragment omits attribution analysis and its main pointers. Preserve all historical outputs and failure records; do not claim a unique mechanism. | revision_final_strengthening_20260922/XAI_AUDIT.md |
| Appendix view budgets and costs | appendix | Keep every method/budget and all named comparisons in complete outputs; do not pick best-per-column synthetic methods. | tables/method_complete_companion.csv; inherited cost tables |

Full-paper page count and float placement remain an author integration check because the full matching LaTeX is unavailable. The standalone ACM proof verifies local references and supplied asset geometry only. No citations have been invented.

Optional future work (not run): a previously uninspected confirmation cohort would answer generalization beyond inspected queries; a matched same-support input-length intervention would address truncation causation; deployment measurements would address ANN/agent latency. None is needed to claim the finite-family inclusion and membership results here.
