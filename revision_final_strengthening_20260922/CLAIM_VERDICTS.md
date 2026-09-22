# Claim verdicts

| Claim | Verdict | Evidence and limits |
|---|---|---|
| Order changes cross fixed competitor boundaries | Supported | Six boundary parquets, K=20/100, no inclusion reconstruction contradictions. Algebra is validation, not novelty. |
| Sensitivity is solely truncation | Unsupported | All-seven-fitting rows in boundary_distributions.csv and 139/1059-token historical cases. Other sources of sensitivity are not excluded. |
| Main crossing has a validated IG explanation | Inconclusive | BGE q235/p7253 maximum margin 9.74535942e-05, error 2.72989273e-05; five-error guard fails, no reselection. |
| Stable-control primary IG is complete | Unsupported for maximum-order zero run | BGE q260/p41359 fails completeness at 256; xai_forward_completeness.csv. |
| IG establishes a unique causal mechanism | Unsupported | Scalar/path/baseline-dependent diagnostic, no causal mechanism study. |
| Panel patterns estimate population prevalence | Unsupported | Stratified24-pair panel plus one post-hoc case. |
| Seven WANDS schedules exhaust sensitivity | Unsupported | Actual nested7/16/32 states and increments, coverage_summary.csv/coverage_increments.csv. |
| WANDS expansion measures additional sensitivity | Bounded | Fixed128q/7,125 pairs, two encoders, target-only; full/common-fitting and precision sensitivity separate. |
| ESCI target orders are exhaustive | Supported within this representation | <=3 whole entries/<=6 permutations,499q/4,434 pairs; coverage_plan_validation.json. No joint catalog exhaustiveness. |
| Nested VI is nondecreasing | Supported structurally | Per-pair checks; not a discovery or convergence proof. |
| Canonicalization gives input-order invariance | Supported structurally | Reused complete-input equality and deterministic ranking, expected VI=0. |
| Fixed controls preserve every raw always-in product | Unsupported | All 48 identity tables; WANDS/MiniLM ascending dense lost mass .0023190235491176 at K=20=.2319pp. |
| Fixed controls can gain raw always-out products | Supported, bounded | Identity table gained cells; all rules retained, not selected from results. |
| Changed membership usually has negligible graded effect | Unsupported at descriptive .01 threshold | Inherited rq2_compact_evidence.csv:64.1–79.2% WANDS,81.1–96.2% ESCI changed queries exceed .01. |
| Gain correction changes ranks/highest-label inclusion | Unsupported | Inherited graded audit; ESCI1/.1/.01/0 and WANDS3/1/0; frozen ranks/supports unchanged. |
| All historical reports or training were affected | Unsupported | Phase IV had already repaired some graded outputs; no relabeling of frozen/pretrained retrieval or separate WANDS training. |
| Canonical–BM25 improves nDCG20 versus BM25 | Supported in tested cells | Inherited canonical_hybrid_claim_verdicts.csv:12/12 positive uncorrected intervals. |
| Hybrid improves over own dense branch | Bounded |9/12 positive intervals, three ESCI/BGE inconclusive. |
| Hybrids universally preserve/surpass raw-hybrid mean | Unsupported | Ten nDCG20 intervals span zero; WANDS/BGE ascending Recall20 −.00624=−.624pp. |
| Universal effectiveness–instability trade-off | Unsupported | Method-/metric-specific comparisons, no Pareto law. |
| Available judgments cover all relevance | Unsupported | Unjudged-zero convention, distinct condensed cNDCG and JudgedCoverage. |
| Frozen models outperform PI-FT or its VI is nonzero | Not evaluated | No matched training/setup comparison; no training requested. |
| Seller harm, behavior, fairness or ANN cost established | Not evaluated | Retrieval diagnostics and local computation only. |

Intervals are uncorrected, conditional on frozen catalogs/encoders. Zero-spanning intervals are not equivalence. Machine-readable values use 0–1; prose explicitly converts to percentage points.
