# Final optimization-robustness revision — 2026-09-08

- **Target-only intervention:** The validated raw-order control remains unchanged: all competitors are C0 and the turquoise-chair target ranks 5 versus 1,527 solely from its attribute order. The new optimization audit uses the same independent-target logic on 310 WANDS and 111 ESCI highest-label pairs.
- **Optimization result:** Both adapted E-GEO rewriters increase BGE VI@100. On WANDS, Raw/heuristic/technical VI is 14.8%/31.6%/23.2%; on ESCI it is 3.0%/17.2%/10.0%. ESCI inclusion falls by 12.7 and 4.9 percentage points, respectively. WANDS mean changes are inconclusive.
- **Factuality and guard:** No WANDS or technical-prompt output passes the conservative field checker; ESCI heuristic pass is 7.6%–8.1%. Guarded pipelines consequently reduce almost entirely to raw canonical text and show no benefit over canonicalization. This is an adapted 0.5B-model audit, not a reproduction or refutation of GPT-4.1 E-GEO.
- **Writing/layout:** The abstract, RQs, Results, Discussion, and Conclusion now center optimization robustness. The main paper includes the quality-versus-VI figure, exact primary summary, contrast CIs, factuality table, transition counts, encoder transfer, cost, and carefully separated stochastic controls. It compiles to 8 self-contained main pages and 11 total pages.
- **Remaining risks:** eight query clusters per dataset; almost empty common-valid support; strict automated fact checker without human validation; adapted small generator; incomplete ESCI judgments; no deployed agent/user outcomes; and a disclosed nonadaptive generation-time deviation (approximately 11.5 GPU hours versus the planned eight).

Validation: 27 isolated tests passed; all 31 final generation/rank/cache/hash/population manifest checks passed; no undefined references, overfull boxes, or fatal LaTeX errors; every PDF page was rendered and inspected.

---

# Completed retrieval-first extension — 2026-09-07

- All scheduled experiments and conditional K=500 canonical reranking are complete. The integrated PDF has 8 main pages and 11 total pages; nine integrity tests pass.
- Figure 1 retains the verified primary-family target-only example: the same Exact-relevant turquoise chair ranks 5 versus 1,527 while all competitors remain in C0 (target C0 rank 486). These are distinct from whole-catalog ranks 3 and 1,303. The new selected-rule target-only net Recall@100 is −0.10 percentage points on WANDS and +0.12 on ESCI; the latter is small and not a broad practical-benefit claim.
- Figure 2 was replotted at native column size using unchanged validated macro VI and confidence intervals. The opening retains the broad Web visibility motivation, and Figure 1 does not interrupt its first paragraph. Detailed sensitivity statistics are in the integrated appendix.
- Added recall-first baselines, fixed-selection transfer, view-count/cost checks, candidate transitions, canonical reranking, and qualified diagnostics. ESCI gain repair and the historical/shared query-cache distinction are documented; original artifacts remain archived.
- Main interpretation: representation-induced instability is supported, but the development-selected field rule does not consistently improve held-out recall. Hybrid is a strong comparator; zero order sensitivity alone does not establish adequate recall or effectiveness equivalence.
- Remaining reviewer risks: novelty relative to tabular representational stability, post-hoc historical evaluation, English offline benchmarks, limited model/domain scope, and no demonstrated commercial exposure or user outcome.

See `REVISION_REPORT.md`, `REVIEWER_RESPONSE_MAP.md`, and `paper_www2027/RESULT_PROVENANCE.md` for the final results and source map.

---

## Historical focused-revision notes (preserved)

# Final focused revision

## Target-only intervention

Each highest-relevance target was independently evaluated under C0, C1, and five C2 permutations, with every competitor fixed in C0. Product facts, query, model, and competitors are unchanged. The calculation reuses frozen scores and cached embeddings; it requires no new model inference. C0 scores and highest-relevance ranks match the frozen baseline exactly in all six cells.

| Dataset | MiniLM | BGE-base | GTE |
|---|---:|---:|---:|
| WANDS, target-only micro VI@20 | 12.72% | 8.84% | 11.39% |
| ESCI, target-only micro VI@20 | 3.11% | 3.16% | 6.68% |

VI counts pairs that are inside Top-20 under at least one tested order and outside under another; it is not the fraction that necessarily falls out relative to C0. All variant ranks, signed C0-relative changes, query summaries, and bootstrap intervals are in `phase3/results/target_only_permutations/`. This is a post-hoc causal control, not an additional preregistered experiment.

## Figure 1

Changed to the target-only `turquoise chair` case (query 162, product 34536, Exact relevance, MiniLM). Rank is **5 under C2s4 and 1,527 under C2s1**, with all competitors fixed in C0; C0 rank is 486. The original whole-catalog ranks **3 and 1,303** remain historical results and are explicitly distinguished in RQ2. The new figure, its source CSV, and regeneration script are provided. This is an illustrative extreme case, not a typical effect size.

## Writing and layout

- Abstract retains Web-level motivation and summarizes the original catalog-wide result as 4.4–13.3%; it separately identifies the new target-only control.
- Invariant encoding removes measured instability, with a small, statistically inconclusive effectiveness change. Equivalence is not claimed.
- Figure 1 follows complete introductory paragraphs on page 1 and does not interrupt the opening sentence or paragraph.
- Figure 2 stays in the main text; redundant detailed catalog-wide statistics move unchanged to Appendix B. Target-only details appear in Appendix C.
- Section 7.2 is now “Scope of the Visibility Claim.” No fairness or disparate-impact finding is claimed.
- `paper_www2027/RESULT_PROVENANCE.md` links the revised claims to their artifacts. Original Phase III numerical results remain unchanged.

## Remaining reviewer risks

- Evidence covers two English offline product benchmarks and synthetic fact-preserving orders, not deployed Web search or other domains.
- Rank visibility is not measured user exposure, clicks, sales, satisfaction, or objective product quality.
- The new control establishes target-serialization effects conditional on C0 competitors. It does not quantify all possible competitor configurations.
- The reranker experiment remains the validated catalog-wide experiment; it was not rerun for the new target-only control.
- Zero measured order instability does not establish semantic robustness to other transformations; confidence intervals spanning zero do not prove effectiveness equivalence.
- The opening example is selected for clarity and a large crossing. Aggregate rates, including smaller ESCI effects, are retained.
- Existing bibliography entries still have incomplete optional proceedings metadata (BibTeX warnings); publication metadata and the conference-header placeholders need a final submission pass.

## Validation

All six Phase III output/reconciliation and target-ranking tests pass. The target-ranking check includes tied scores, removal of the old target, unchanged C0 scores, and comparison with full stable reranking. The PDF compiles to eight pages, with no undefined citations/references or overfull boxes; all pages were rendered and visually inspected. Existing BibTeX metadata warnings remain as noted above.
