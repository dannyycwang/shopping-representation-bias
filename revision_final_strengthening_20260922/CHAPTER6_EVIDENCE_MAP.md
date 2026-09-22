# Chapter 6 evidence map

Retain the three-RQ structure. The following English text/captions are candidates for author review; no manuscript file was changed. Limit main additions to one case figure and one compact table. Coverage and detailed attribution QA are supplementary.

## 6.1 Attribute-Order Sensitivity

**Evidence:** original population findings in revision_evidence_20260917/data/inclusion_states_summary.csv and rq1_target_fully_fitting_plot_data.csv; new boundary records/distributions and bounded coverage. Distinguish original catalog-wide experiments from fixed C0-competitor target replacements. Primary WANDS 308q/21,299 pairs and ESCI 499q/4,434 remain fixed; WANDS coverage uses 128q/7,125 pairs. Full and fitting supports are separate.

**Insertion:**
“With all competing products fixed at their source-order representations, alternative serializations can move a highest-label target across a Top-K boundary. Exact competitor thresholds reproduce every audited Top-20 and Top-100 inclusion decision; a previously documented one-rank BGE discrepancy occurs far below these cutoffs. Fully fitting examples remain order-sensitive: the 139-token French-molding item spans ranks 15–174, and the 1,059-token teal-chair item spans 11–939. These selected illustrations exclude truncation for those inputs but do not estimate population prevalence.”

**Supplement coverage insertion:**
“On the full target support of a fixed 128-query WANDS supplement, Top-20 query-macro crossing mass increases across 7, 16 and 32 tested schedules (MiniLM: 20.85% → 24.50% → 28.21%; BGE: 11.40% → 15.03% → 17.55%). On the full target support of all 499 ESCI queries, exhaustive whole-entry target orders give MiniLM: 2.91% → 3.25%; BGE: 2.42% → 3.13%. Competitors remain fixed; separate fitting curves reuse each encoder's common all-family-fitting support. Nested monotonicity is structural; the WANDS curve does not prove convergence or universal invariance, and ESCI exhaustiveness is per target rather than over joint catalog orders.”

**XAI insertion:**
“The preselected query-conditioned attribution cases do not yield a validated explanation of the main crossing. The BGE crossing retains native inclusion decisions under the FP32 diagnostic, but one native margin fails the predefined five-error separation guard. The stable control has a primary-baseline completeness failure at 256 integration nodes. We retain the cases and native rank/margin evidence, reporting attribution as inconclusive rather than selecting replacement examples.”

**Main figure placement:** after the fixed-competitor paragraph, full manuscript width. Use figures/main_query_conditioned_cases.pdf; first eight source occurrence IDs plus an exact other-content total, no largest-change selection.

**Main caption:**
“Native Top-20 margins and diagnostic query-conditioned entry attribution for a preselected BGE crossing (q235/product7253) and stable included control (q260/product41359), WANDS. Each target replaces its old entry while all 42,993 competitors stay at C0; zero margin is the boundary. Points denote seven schedules, not time, and labels are authoritative native ranks. Minimum/maximum-score orders are highlighted. All seven inputs fit the 512-token limit. Zero-baseline IG is aligned by attribute occurrence on one signed scale, with other content summed. The crossing fails numerical separation; the control's maximum-order zero-baseline run fails completeness. IG colors therefore do not establish an explanation of the crossing.”

**Supplement captions:**
- French molding: “Post-hoc MiniLM illustration q359/product12575,139 tokens under a 256-token cap. Reordering the same whole entries spans native target ranks 15–174 with all competitors at C0. Entry IG uses the cached query and zero-embedding reference; PAD sensitivity and complete QA accompany it. Historical selection does not estimate prevalence.”
- Coverage: “Query-macro target-only crossing mass at K=20/100 for WANDS 128-query nested7/16/32 families and ESCI 499-query seven versus exhaustive target orders. Full supports and each encoder's fixed common all-family-fitting support are separate, with denominators shown. Bars are uncorrected 95% query-cluster intervals. Separate target counterfactuals fix all competitors at C0; these are not coherent-catalog Recall.”

**Remove/qualify:** all-permutation or convergence claims for WANDS; truncation as sole explanation; human-like query attention; unique causal mechanism; XAI validation of the main crossing.

## 6.2 Aggregate Effectiveness and Product Membership

**Inherited evidence, not rerun:** revision_graded_20260922/tables/effectiveness_wide.csv, paired_contrasts.csv, rq2_distribution_and_thresholds.csv, rq2_compact_evidence.csv; data/rq2_membership_ndcg_per_query.parquet. Keep source C0 and a query-level mean over seven schedules distinct.

**Insertion:**
“Unchanged aggregate counts need not imply unchanged product membership: equal numbers of highest-label gains and losses cancel in Recall. This limited non-identification result should not be generalized to negligible effectiveness changes. Among queries whose Top-20 highest-label membership changes, |delta nDCG20| exceeds .01 in 64.1–79.2% of WANDS cases and 81.1–96.2% of ESCI cases across tested raw contrasts. The threshold is descriptive. Numerical nDCG equality and integer Recall cancellation are separately audited phenomena.”

**Metric paragraph:**
“We use direct label-derived gains: WANDS Exact/Partial/Irrelevant=3/1/0 and ESCI E/S/C/I=1/.1/.01/0, without exponential transformation. The correction changes graded evaluation but preserves frozen ranks and highest-label inclusion. Full-ranking nDCG discounts at catalog positions and assigns unjudged products zero gain only as an evaluation convention. Condensed cNDCG removes unjudged products before assigning positions. JudgedCoverage is the judged fraction of original Top-K results, including zero-gain judged items. These complementary quantities do not establish the true relevance of unjudged products.”

**Placement:** retain the existing corrected effectiveness table and nDCG/cNDCG/JudgedCoverage companions; do not add another new main table here.

**Remove/qualify:** blanket “effectiveness barely changes”; equivalence from zero-spanning intervals; claims every historical report or training was affected; target-only ranks treated as one coherent query ranking.

## 6.3 Effectiveness and Consistent Inclusion under Controls

**Evidence:** new control_transitions.csv, complete identities and per-query cells; inherited canonical_hybrid_claim_verdicts.csv, canonical_hybrid_state_contrasts.csv, canonical_hybrid_costs.csv and inherited_strategy_costs.csv. Retain all three rules. Canonical VI=0 is structural; finite-schedule raw stability is empirical.

**Insertion:**
“Canonicalization makes the permitted attribute-order transformation input-invariant, so deterministic retrieval has VI=0 by construction. Identity transitions clarify which products the fixed policy consistently includes: a canonical index can omit products included under every raw reference schedule and include products always omitted. For example, WANDS/MiniLM ascending dense loses 0.232 percentage points of highest-label support from the always-in row. This is a membership description, not an estimate of seller harm.”

“Corrected graded evaluation remains condition-specific. All 12 canonical–BM25 settings have positive uncorrected paired nDCG20 intervals versus BM25; nine do versus their dense branch. Most contrasts with the raw-hybrid seven-schedule mean remain inconclusive. WANDS/BGE ascending lowers Recall20 by .00624, or .624 percentage points, against that mean. These findings do not establish universal equivalence, dominance or an effectiveness–instability trade-off.”

**Main table placement:** extend controls table using tables/main_control_transitions.tex/CSV. Twelve dataset/encoder/rule rows retain both dense/hybrid families at K=20. Complete six-cell matrices, K100, intervals and conditional denominators are supplementary.

**Caption:**
“Identity-based destinations under fixed canonical dense and canonical–BM25 indexes relative to each family's raw catalog-wide seven-schedule reference, K20. Lost means raw always-in but fixed-control omitted; gained means raw always-out but fixed-control included. Crossing-in/out partitions the reference crossing row. Values are query-macro percentage points of Hq,308 WANDS/499 ESCI queries. Every canonical rule is retained; full six-cell counts, conditional rates, denominators and uncorrected query-cluster intervals accompany the table.”

**Cost insertion:**
“Each canonical hybrid retains one dense vector per product plus the BM25 sparse index and scores both branches before deterministic fusion. Inherited dense+sparse array sizes are 118,698,552/184,737,336 bytes for WANDS MiniLM/BGE and 24,049,044/39,525,780 for ESCI. These exclude model weights and vocabulary/metadata. Local batch work and historical construction costs are not online or ANN latency; centroid and multi-vector construction/storage budgets remain distinct.”

**Remove/qualify:** structural zero VI as a new algorithmic discovery; universal preservation or Pareto law; PI-FT VI/superiority without matched setup; deployment/ANN conclusions.

## Units, inference and completion

This package closes the planned bounded empirical supplement, including negative attribution diagnostics; drafting Chapter 6 does not require another open-ended experiment. Cluster resampling keeps each query's compared schedules/methods together, 10,000 draws, seed 2026091701, uncorrected percentile intervals. Machine-readable values stay 0–1; prose explicitly converts to percentage points. Retain distinctions among query-macro/pair-micro, source C0/schedule mean, empirical/structural stability, and per-target/joint coverage.
