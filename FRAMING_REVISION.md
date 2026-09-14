# Framing Revision for the WWW 2027 Draft

## 1. Old framing summary

The earlier draft opened with product catalogs, dense retrieval, attribute order, and benchmark details. Its first-page logic was accurate but narrow: it presented a product-serialization effect before explaining the more general Web research problem. The abstract led with implementation and experimental particulars, and the introduction reached the product testbed before establishing retrieval visibility as a system property. Related work mixed product retrieval, robustness, and visibility without the three-way distinction needed for a WWW audience.

## 2. New framing summary

The revised paper follows a funnel:

1. Web users encounter content after search, recommendation, retrieval-augmented, or agentic systems select it.
2. Semantically equivalent Web information should ideally receive comparable opportunities to be retrieved.
3. Structured and semi-structured Web records are often flattened into ordered sequences, creating a mismatch between set-like facts and sequential encoders.
4. E-commerce is a controlled testbed because product attributes can be permuted without changing product identity, facts, query relevance, or catalog membership.
5. The experiments measure whether this semantically irrelevant choice changes Top-K visibility, whether the effect survives a two-stage pipeline, and whether a set representation mitigates it.
6. The conclusion returns to representation robustness as a Web retrieval evaluation and design objective.

The paper retains the product-specific title **Same Product, Different Visibility: Representation Robustness in Neural E-Commerce Retrieval**. A broader “Neural Web Retrieval” title would imply empirical coverage beyond the product domain.

## 3. Major changes made

- Rewrote the abstract from the Web-level robustness expectation downward; dataset and model names no longer appear before the problem is established.
- Rebuilt the first three introduction paragraphs around Web-mediated visibility, structured Web records, and e-commerce as a controlled intervention setting.
- Kept Figure 1 on the first page and simplified its caption to the causal contrast: same product, facts, query, and model; only attribute order changes.
- Added four conceptual research questions and four contribution statements organized around problem formulation, evidence, pipeline consequence, and mitigation.
- Reorganized Related Work into neural product retrieval, content/visibility optimization, and representation/serialization robustness.
- Generalized the problem formulation to a Web item with an equivalence class of semantic representations before specializing to products and attribute permutations.
- Reframed set-mean aggregation as a proof-of-concept mitigation based on the structural fact that attributes are sets, without claiming general novelty for mean pooling.
- Reorganized Results into RQ1 visibility, RQ2 human relevance and direct interventions, RQ3 reranking and candidate loss, and RQ4 mitigation.
- Expanded the discussion to job, hotel, restaurant, event, and other catalog records as hypotheses for future replication, while explicitly withholding empirical generalization.
- Expanded the provenance map to cover the newly restored relevance-stratum and single-product results.

## 4. Claims deliberately weakened

- The paper does not claim the first discovery that neural retrieval is sensitive to serialization.
- The set-based method “eliminates measured order-induced instability”; it is not described as preserving relevance without loss. Both paired cNDCG confidence intervals include zero, which is statistically inconclusive and does not prove equivalence.
- The paper states that highly relevant products are affected in both datasets, but disproportionate Exact-versus-Irrelevant instability is established only on WANDS; ESCI contrasts are mixed and inconclusive.
- Retrieval visibility is described as an upstream opportunity rather than realized exposure, fairness, clicks, sales, purchase, satisfaction, or welfare.
- Other structured Web domains are presented as plausible replication settings, not as domains in which the effect has already been demonstrated.

## 5. Related-work additions

The revision adds and verifies the following sources:

- Bhandari et al. (2026), *Improving Robustness of Tabular Retrieval via Representational Stability* (arXiv:2604.24040), the closest prior work on equivalent serialization in neural retrieval.
- Kim et al. (2026), *SAGEO Arena*, KDD 2026, which evaluates visibility optimization across retrieval, reranking, and generation.
- Choi et al. (2020), *Semantic Product Search for Matching Structured Product Catalogs in E-Commerce*, on multi-instance structured fields and lexical/semantic matching.
- Loughnane et al. (2024), *Explicit Attribute Extraction in e-Commerce Search*, on structured attribute signals in deployed semantic retrieval and ranking.
- Goren et al. (2020), *Ranking-Incentivized Quality Preserving Content Modification*, to connect GEO with earlier competitive content modification.
- Robertson and Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*, for the lexical retrieval baseline lineage.

The text explicitly distinguishes GEO/E-GEO's rank-improvement objective from this paper's symmetric robustness question.

## 6. Unresolved writing and positioning risks

The revised framing is broad, but the empirical evidence remains product-specific. Reviewers may still ask for another structured Web domain; the paper now treats that as external-validity work rather than implying it has been completed. The synthetic permutations are fact-preserving but may differ in naturalness, so a blinded human audit remains the highest-value addition if annotation time becomes available. The paper also needs a final bibliography pass against the official WWW 2027 template and camera-ready metadata. The current manuscript uses the available ACM review format, stays within the page limits, and keeps the first eight pages self-contained.


## Phase V positioning update (2026-09-08)

The manuscript now makes optimization robustness the second half of the funnel: raw representation-induced visibility instability motivates an audit of whether public, query-blind product rewriting survives the same controlled intervention. The paper does not present rewriting as a successful mitigation. It distinguishes raw-atom equivalence from free-text semantic preservation, separates incoming-order variation from generation randomness, and treats canonical-first caching as a structural control rather than evidence that the generator is invariant.

The new results deliberately weaken the method claim. The adapted rewriters amplify VI; ESCI inclusion falls; strict fact-valid support is almost empty; and guarded pipelines reduce to canonical text. These findings are attributed to the tested 0.5B adaptation and first-stage setting, not generalized to GPT-4.1 E-GEO. The broader contribution is an evaluation framework linking relevance, representation robustness, factuality, and candidate access. Remaining positioning risks are the small eight-query optimization sample, automated factuality checks, and the absence of a strong-model reproduction.
