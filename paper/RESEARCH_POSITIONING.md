# Research positioning after the September 3, 2026 literature check

The safest initial title is **Same Product Information, Different Visibility: A Catalog-Wide Study of Representation Sensitivity in E-Commerce Retrieval**. Keep the proposed shopping-agent title for a later version that actually evaluates agents.

The intended distinction is a combination of treatment scope, invariant source content, bidirectional exposure measurement and human relevance evaluation. It is not simply the observation that text changes rankings.

| Prior work | What is already covered | What this pilot examines |
|---|---|---|
| [E-GEO, v2](https://arxiv.org/html/2511.20867v2) | Rewriting among fixed ten-product candidate sets; cross-engine tests; observational length/bullet analyses; widespread adoption raised as future work | All-catalog treatment before retrieval; exact-labeled exposure gains and losses; deterministic invariance controls |
| [SAGEO Arena, v2](https://arxiv.org/html/2602.12187v2) | Stage-level retrieval, reranking and generation; structural fields; shopping-domain losses under tested optimization strategies | Human relevance alignment and source-preserving catalog serialization; no end-to-end agent claim in Phase 1 |
| [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375) | Document-side expansion for retrieval is established prior art | A fidelity-constrained sensitivity audit, rather than a claim that query-independent document transformation is new |
| [Revisiting Query Variation Robustness](https://aclanthology.org/2024.findings-emnlp.248/) | Retrieval robustness to query variation has been studied | Product-side representation variation, with fixed queries and relevance labels |

E-GEO's current dataset count is 13,747 queries, rather than the older count in the supplied outline. Its length/formatting analysis is observational; it explicitly does not claim a controlled causal experiment. SAGEO v2 is marked accepted at KDD 2026. These version changes should be reflected before submission.

This search verifies relevant prior work; it does not establish exhaustive novelty or priority. A submission would need a broader review of product field weighting, attribute normalization, document robustness and competitive search.

## Claims the pilot can support

With identical source information and a fixed retrieval procedure, catalog serialization may alter which human-labeled Exact products are exposed. Gains should be shown together with harms, judged coverage and controls. Failure of Dual View to improve mean relevance would be a meaningful negative result, not a reason to hide the experiment.

## Claims that need additional experiments

* Calling a product objectively the best, or treating every WANDS attribute as verified real-world truth.
* Claiming an isolated merchant's direct effect from universal-adoption ranks; changing competitors causes interference.
* Claiming that deterministic wrappers constitute a comprehensive test of marketing prose versus factual prose.
* Claiming agent recommendations or fairness across merchants from retrieval results alone.
* Claiming length is fully controlled by R0 duplication: it matches repetition approximately, while the order-only control preserves exact length.
* Claiming that a small general-purpose MiniLM encoder represents all strong dense retrieval models.

## Method corrections to the original outline

1. Use **source-equivalent** until factual consistency is independently verified. The outline's table example does not establish equal information: the first wording does not state solid wood or seating capacity. That comparison cannot isolate presentation alone.
2. Use max-minus-min for an absolute rank span, and max/min for a rank ratio. They measure different properties. Do not call the ratio a range without clarification.
3. Keep product scores from independently rebuilt lexical indexes out of a raw score-variance claim. Corpus statistics and score scales can change.
4. Do not truncate a representation to claim a fact-preserving length control. Truncation drops source information.
5. An objective would be relevance alignment **minus** weighted penalties, not multiplication by penalties. Phase 1 does not train or optimize this objective.
6. A schema containing conflicting variant attributes must preserve provenance and ambiguity. Selecting one value or assigning missing units creates a factuality confound.
7. Evaluate incomplete relevance pools transparently. Condensed metrics remove unknowns but alter the meaning of rank; they must not be mislabeled full-catalog NDCG.

The incomplete-judgment metric choice follows the distinction studied in [Sakai and Kando, On information retrieval metrics designed for evaluation with incomplete relevance assessments](https://link.springer.com/article/10.1007/s10791-008-9059-7). It reduces one assumption but does not repair missing labels or pooling bias.
