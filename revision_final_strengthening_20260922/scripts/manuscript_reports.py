def claims_and_map(put,cp,wline,eline):
 put('CLAIM_VERDICTS.md',f"""
# Claim verdicts

| Claim | Verdict | Evidence and limits |
|---|---|---|
| Order changes cross fixed competitor boundaries | Supported | Six boundary parquets, K20/K100, no inclusion reconstruction contradictions. Algebra is validation, not novelty. |
| Sensitivity is solely truncation | Unsupported | All-seven-fitting rows in boundary_distributions.csv and 139/1059-token historical cases. Other sources of sensitivity are not excluded. |
| Main crossing has a validated IG explanation | Inconclusive | BGE q235/p7253 maximum margin {cp.native_margin:.9g}, error {cp.diagnostic_native_error:.9g}; five-error guard fails, no reselection. |
| Stable-control primary IG is complete | Unsupported for maximum-order zero run | BGE q260/p41359 fails completeness at256; xai_forward_completeness.csv. |
| IG establishes a unique causal mechanism | Unsupported | Scalar/path/baseline-dependent diagnostic, no causal mechanism study. |
| Panel patterns estimate population prevalence | Unsupported | Stratified24-pair panel plus one post-hoc case. |
| Seven WANDS schedules exhaust sensitivity | Unsupported | Actual nested7/16/32 states and increments, coverage_summary.csv/coverage_increments.csv. |
| WANDS expansion measures additional sensitivity | Bounded | Fixed128q/7,125 pairs, two encoders, target-only; full/common-fitting and precision sensitivity separate. |
| ESCI target orders are exhaustive | Supported within this representation | <=3 whole entries/<=6 permutations,499q/4,434 pairs; coverage_plan_validation.json. No joint catalog exhaustiveness. |
| Nested VI is nondecreasing | Supported structurally | Per-pair checks; not a discovery or convergence proof. |
| Canonicalization gives input-order invariance | Supported structurally | Reused complete-input equality and deterministic ranking, expected VI=0. |
| Fixed controls preserve every raw always-in product | Unsupported | All48 identity tables; WANDS/MiniLM ascending dense lost mass .0023190235491176 atK20=.2319pp. |
| Fixed controls can gain raw always-out products | Supported, bounded | Identity table gained cells; all rules retained, not selected from results. |
| Changed membership usually has negligible graded effect | Unsupported at descriptive .01 threshold | Inherited rq2_compact_evidence.csv:64.1–79.2% WANDS,81.1–96.2% ESCI changed queries exceed .01. |
| Gain correction changes ranks/highest-label inclusion | Unsupported | Inherited graded audit; ESCI1/.1/.01/0 and WANDS3/1/0; frozen ranks/supports unchanged. |
| All historical reports or training were affected | Unsupported | PhaseIV had already repaired some graded outputs; no relabeling of frozen/pretrained retrieval or separate WANDS training. |
| Canonical–BM25 improves nDCG20 versus BM25 | Supported in tested cells | Inherited canonical_hybrid_claim_verdicts.csv:12/12 positive uncorrected intervals. |
| Hybrid improves over own dense branch | Bounded |9/12 positive intervals, three ESCI/BGE inconclusive. |
| Hybrids universally preserve/surpass raw-hybrid mean | Unsupported | Ten nDCG20 intervals span zero; WANDS/BGE ascending Recall20 −.00624=−.624pp. |
| Universal effectiveness–instability trade-off | Unsupported | Method-/metric-specific comparisons, no Pareto law. |
| Available judgments cover all relevance | Unsupported | Unjudged-zero convention, distinct condensed cNDCG and JudgedCoverage. |
| Frozen models outperform PI-FT or its VI is nonzero | Not evaluated | No matched training/setup comparison; no training requested. |
| Seller harm, behavior, fairness or ANN cost established | Not evaluated | Retrieval diagnostics and local computation only. |

Intervals are uncorrected, conditional on frozen catalogs/encoders. Zero-spanning intervals are not equivalence. Machine-readable values use0–1; prose explicitly converts to percentage points.
""")
 put('CHAPTER6_EVIDENCE_MAP.md',f"""
# Chapter6 evidence map

Retain the three-RQ structure. The following English text/captions are candidates for author review; no manuscript file was changed. Limit main additions to one case figure and one compact table. Coverage and detailed attribution QA are supplementary.

## 6.1 Attribute-Order Sensitivity

**Evidence:** original population findings in revision_evidence_20260917/data/inclusion_states_summary.csv and rq1_target_fully_fitting_plot_data.csv; new boundary records/distributions and bounded coverage. Distinguish original catalog-wide experiments from fixed-C0-competitor target replacements. Primary WANDS308q/21,299 pairs and ESCI499q/4,434 remain fixed; WANDS coverage uses128q/7,125 pairs. Full and fitting supports are separate.

**Insertion:**
“With all competing products fixed at their source-order representations, alternative serializations can move a highest-label target across a Top-K boundary. Exact competitor thresholds reproduce every audited Top-20 and Top-100 inclusion decision; a previously documented one-rank BGE discrepancy occurs far below these cutoffs. Fully fitting examples remain order-sensitive: the139-token French-molding item spans ranks15–174, and the1,059-token teal-chair item spans11–939. These selected illustrations exclude truncation for those inputs but do not estimate population prevalence.”

**Supplement coverage insertion:**
“On the full target support of a fixed 128-query WANDS supplement, Top-20 query-macro crossing mass increases across 7, 16 and 32 tested schedules ({wline}). On the full target support of all 499 ESCI queries, exhaustive whole-entry target orders give {eline}. Competitors remain fixed; separate fitting curves reuse each encoder's common all-family-fitting support. Nested monotonicity is structural; the WANDS curve does not prove convergence or universal invariance, and ESCI exhaustiveness is per target rather than over joint catalog orders.”

**XAI insertion:**
“The preselected query-conditioned attribution cases do not yield a validated explanation of the main crossing. The BGE crossing retains native inclusion decisions under the FP32 diagnostic, but one native margin fails the predefined five-error separation guard. The stable control has a primary-baseline completeness failure at256 integration nodes. We retain the cases and native rank/margin evidence, reporting attribution as inconclusive rather than selecting replacement examples.”

**Main figure placement:** after the fixed-competitor paragraph, full manuscript width. Use figures/main_query_conditioned_cases.pdf; first eight source occurrence IDs plus an exact other-content total, no largest-change selection.

**Main caption:**
“Native Top-20 margins and diagnostic query-conditioned entry attribution for a preselected BGE crossing (q235/product7253) and stable included control (q260/product41359), WANDS. Each target replaces its old entry while all42,993 competitors stay atC0; zero margin is the boundary. Points denote seven schedules, not time, and labels are authoritative native ranks. Minimum/maximum-score orders are highlighted. All seven inputs fit the512-token limit. Zero-baseline IG is aligned by attribute occurrence on one signed scale, with other content summed. The crossing fails numerical separation; the control's maximum-order zero-baseline run fails completeness. IG colors therefore do not establish an explanation of the crossing.”

**Supplement captions:**
- French molding: “Post-hoc MiniLM illustration q359/product12575,139 tokens under a256-token cap. Reordering the same whole entries spans native target ranks15–174 with all competitors atC0. Entry IG uses the cached query and zero-embedding reference; PAD sensitivity and complete QA accompany it. Historical selection does not estimate prevalence.”
- Coverage: “Query-macro target-only crossing mass atK20/K100 for WANDS128-query nested7/16/32 families and ESCI499-query seven versus exhaustive target orders. Full supports and each encoder's fixed common all-family-fitting support are separate, with denominators shown. Bars are uncorrected95% query-cluster intervals. Separate target counterfactuals fix all competitors atC0; these are not coherent-catalog Recall.”

**Remove/qualify:** all-permutation or convergence claims for WANDS; truncation as sole explanation; human-like query attention; unique causal mechanism; XAI validation of the main crossing.

## 6.2 Aggregate Effectiveness and Product Membership

**Inherited evidence, not rerun:** revision_graded_20260922/tables/effectiveness_wide.csv, paired_contrasts.csv, rq2_distribution_and_thresholds.csv, rq2_compact_evidence.csv; data/rq2_membership_ndcg_per_query.parquet. Keep sourceC0 and a query-level mean over seven schedules distinct.

**Insertion:**
“Unchanged aggregate counts need not imply unchanged product membership: equal numbers of highest-label gains and losses cancel in Recall. This limited non-identification result should not be generalized to negligible effectiveness changes. Among queries whose Top-20 highest-label membership changes, |delta nDCG20| exceeds .01 in64.1–79.2% of WANDS cases and81.1–96.2% of ESCI cases across tested raw contrasts. The threshold is descriptive. Numerical nDCG equality and integer Recall cancellation are separately audited phenomena.”

**Metric paragraph:**
“We use direct label-derived gains: WANDS Exact/Partial/Irrelevant=3/1/0 and ESCI E/S/C/I=1/.1/.01/0, without exponential transformation. The correction changes graded evaluation but preserves frozen ranks and highest-label inclusion. Full-ranking nDCG discounts at catalog positions and assigns unjudged products zero gain only as an evaluation convention. Condensed cNDCG removes unjudged products before assigning positions. JudgedCoverage is the judged fraction of original Top-K results, including zero-gain judged items. These complementary quantities do not establish the true relevance of unjudged products.”

**Placement:** retain the existing corrected effectiveness table and nDCG/cNDCG/JudgedCoverage companions; do not add another new main table here.

**Remove/qualify:** blanket “effectiveness barely changes”; equivalence from zero-spanning intervals; claims every historical report or training was affected; target-only ranks treated as one coherent query ranking.

## 6.3 Effectiveness and Consistent Inclusion under Controls

**Evidence:** new control_transitions.csv, complete identities and per-query cells; inherited canonical_hybrid_claim_verdicts.csv, canonical_hybrid_state_contrasts.csv, canonical_hybrid_costs.csv and inherited_strategy_costs.csv. Retain all three rules. Canonical VI=0 is structural; finite-schedule raw stability is empirical.

**Insertion:**
“Canonicalization makes the permitted attribute-order transformation input-invariant, so deterministic retrieval has VI=0 by construction. Identity transitions clarify which products the fixed policy consistently includes: a canonical index can omit products included under every raw reference schedule and include products always omitted. For example, WANDS/MiniLM ascending dense loses0.232 percentage points of highest-label support from the always-in row. This is a membership description, not an estimate of seller harm.”

“Corrected graded evaluation remains condition-specific. All12 canonical–BM25 settings have positive uncorrected paired nDCG20 intervals versus BM25; nine do versus their dense branch. Most contrasts with the raw-hybrid seven-schedule mean remain inconclusive. WANDS/BGE ascending lowers Recall20 by .00624, or .624 percentage points, against that mean. These findings do not establish universal equivalence, dominance or an effectiveness–instability trade-off.”

**Main table placement:** extend controls table using tables/main_control_transitions.tex/CSV. Twelve dataset/encoder/rule rows retain both dense/hybrid families atK20. Complete six-cell matrices, K100, intervals and conditional denominators are supplementary.

**Caption:**
“Identity-based destinations under fixed canonical dense and canonical–BM25 indexes relative to each family's raw catalog-wide seven-schedule reference, K20. Lost means raw always-in but fixed-control omitted; gained means raw always-out but fixed-control included. Crossing-in/out partitions the reference crossing row. Values are query-macro percentage points of Hq,308 WANDS/499 ESCI queries. Every canonical rule is retained; full six-cell counts, conditional rates, denominators and uncorrected query-cluster intervals accompany the table.”

**Cost insertion:**
“Each canonical hybrid retains one dense vector per product plus the BM25 sparse index and scores both branches before deterministic fusion. Inherited dense+sparse array sizes are118,698,552/184,737,336 bytes for WANDS MiniLM/BGE and24,049,044/39,525,780 for ESCI. These exclude model weights and vocabulary/metadata. Local batch work and historical construction costs are not online or ANN latency; centroid and multi-vector construction/storage budgets remain distinct.”

**Remove/qualify:** structural zero VI as a new algorithmic discovery; universal preservation or Pareto law; PI-FT VI/superiority without matched setup; deployment/ANN conclusions.

## Units, inference and completion

This package closes the planned bounded empirical supplement, including negative attribution diagnostics; drafting Chapter6 does not require another open-ended experiment. Cluster resampling keeps each query's compared schedules/methods together,10,000 draws, seed2026091701, uncorrected percentile intervals. Machine-readable values stay0–1; prose explicitly converts to percentage points. Retain distinctions among query-macro/pair-micro, sourceC0/schedule mean, empirical/structural stability, and per-target/joint coverage.
""")
