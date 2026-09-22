# Claim verdicts

| Claim | Verdict | Evidence and scope |
| --- | --- | --- |
| Phase II stored ESCI S/C gains are reversed | Supported | Original labels, processed grade columns and 42 raw artifacts traced; paper Section 3.1 confirms E/S/C/I=1/.1/.01/0. |
| Every historical graded result still used the error | Unsupported | Phase IV already corrected label-derived cNDCG and reconciled 79 historical rank artifacts. |
| The correction alters frozen training or retrieval ranks | Unsupported | Reused frozen encoders/query embeddings/RRF do not consume graded training targets; correction is evaluation only. |
| Highest-label Hq, inclusion, Recall20/100 and VI survive the gain swap | Supported | All reused supports and query records verified; 80 prior Recall summary cells reproduced. |
| Membership changes can occur with numerically identical nDCG20 | Supported, bounded | Observed records retained; atol=1e-12, rtol=0; distinct from integer Recall cancellation. |
| Membership changes usually have negligible graded effect | Unsupported at the frozen .01 descriptive threshold | Most changed queries exceed .01 in every raw dataset/encoder/schedule cell; complete distributions retained. |
| Mean nDCG is invariant/equivalent across raw schedules | Unsupported as a blanket claim | Five uncorrected raw nDCG20 intervals establish reductions; zero-spanning intervals cannot establish equivalence. |
| All canonical–BM25 indexes are independent of incoming attribute order | Supported structurally | Complete-input equality, fixed branch inputs, full ranks and deterministic ties; VI=0. |
| Canonical–BM25 improves nDCG20 over BM25 | Supported in all 12 tested cells | Paired uncorrected intervals above zero, all rules retained. |
| Canonical–BM25 improves nDCG20 over its own dense branch | Supported in 9 cells; inconclusive in 3 ESCI/BGE cells | Full condition-specific verdict table included. |
| Canonical–BM25 is equivalent or uniformly superior to raw-seven hybrid | Unsupported as a universal claim | Mostly inconclusive nDCG contrasts; WANDS/BGE ascending Recall20 is lower. |
| There is a general effectiveness–instability trade-off | Unsupported | Fixed canonical hybrids retain lexical benefit in several settings; no universal Pareto law follows. |
| This establishes all-permutation coverage for raw models | Unsupported / not evaluated | Optional P2 expansion deferred; seven-schedule observations are finite. |
| Judged gains are complete relevance ground truth for the catalog | Unsupported | Incomplete judgments; unjudged-zero is an evaluation convention, with coverage and condensed measures reported. |

Intervals are paired query-cluster percentile intervals, 10,000 draws, seed 2026091701, without multiplicity correction. They quantify query sampling conditional on fixed catalogs/encoders, not model, training-seed or annotation uncertainty. Descriptive thresholds are not equivalence margins. See `tables/canonical_hybrid_claim_verdicts.csv` for each setting, comparator and metric.
