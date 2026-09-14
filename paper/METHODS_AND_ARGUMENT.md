# Draft methods and argument

This text is a source for the first manuscript. Numeric findings belong in the completed manuscript and are read from the recorded experiments.

## Research question

Product search systems must map a consumer's request to a ranking of available products. That mapping depends on a textual representation of each product, often assembled from titles, descriptions, taxonomies and attribute feeds. A representation can change without changing the source information it carries. This creates a measurement problem: when the same source record is serialized differently, how stable is the visibility of products judged relevant by humans?

We study this question before the recommendation stage. A failure to retrieve a relevant product constrains what a later reranker or shopping assistant can select. Conversely, a retrieval gain does not establish a downstream recommendation gain. Our study therefore treats agentic shopping as the motivation and catalog retrieval as the measured system.

Let a catalog be C={p1,...,pN}, with source records Fp and human judgments Y(q,p). A deterministic transformation r maps each record to a string. Retrieval under universal treatment is a list L(q,r(C)). The visibility indicator is V_K(q,p;r)=1[rank(q,p;r(C))<=K]. We hold queries, product identities, source atoms, labels and model parameters fixed while changing r. Corpus statistics and competitors' representations change together under this policy intervention.

The estimand is consequently a catalog-wide treatment effect, not an isolated product's direct effect. A product can move because its own text is more effective, because competitors become more effective, or because index statistics change. Isolating those mechanisms requires a later single-adopter experiment with explicit controls for corpus-statistic changes.

## Source equivalence

We use source equivalence rather than claiming that every retained statement is true in the world. A source record may contain multiple conflicting values or attributes drawn from different variants. Every serializer retains those values and their provenance instead of resolving them. Ratings are omitted from every retrieval representation, so no condition gains a factual input that is absent from the original baseline.

The serializer never receives a query or a relevance label. Original descriptions are copied intact. Feature entries retain their raw keys and values, including duplicate keys, negative values, blank keys and unknown units. Some conditions add field labels, reorder entries or repeat a complete source view. Added shopping invitations are generic framing, not product-specific factual claims.

An auditor checks retention of each source atom, novel numeric values, residual terms outside the fixed serialization scaffolds and equality to the deterministic generator. These checks provide strong guarantees for this restricted transformation family. They are not a general semantic entailment auditor for arbitrary LLM paraphrases, and they do not turn conflicting source data into reliable product specifications.

## Evaluation under incomplete judgments

Only a small subset of query-product pairs has relevance judgments. Unknown products remain unknown. We compute judged-condensed NDCG by removing unjudged products from a ranked list before applying the usual discounted-gain calculation, with gains 3, 1 and 0 for Exact, Partial and Irrelevant. We explicitly name this metric cNDCG; it is not NDCG at the first ten positions of the full catalog ranking.

Known-relevant recall and reciprocal rank retain full-catalog positions. The recall denominator includes the judged Exact and Partial products for a query. This is a useful reproducible measure but does not estimate recall over all truly relevant products when labels are incomplete. ExactHit is defined only for queries with at least one Exact label. We report judgment coverage and explicit nonrelevance separately, because a low count of known irrelevant products can merely reflect more unknown results.

For Exact pairs, RelevantHidden@K counts transitions from outside the original top K to inside the alternative top K. ReverseHidden@K counts the opposite transition. Both use the same denominator. We report raw counts and pair-micro rates, and query-macro net changes with confidence intervals. Reporting only recovered products would conceal exposure redistribution and the possibility that a method harms more relevant products than it helps.

Rank sensitivity uses the full ranking rather than a top-K-censored surrogate. The absolute span is max_r rank minus min_r rank; the rank ratio is max_r rank divided by min_r rank. Hit instability indicates whether the minimum rank is inside K and the maximum rank is outside K. Extremes selected for qualitative review are examples, not unbiased estimates of typical effects.

## Interpretation

An aggregate relevance improvement and representational stability are distinct goals. A transformation could improve average retrieval while making individual products less stable. Conversely, large movements can cancel out in the mean. A retrieval interface can therefore exhibit substantial exposure sensitivity even when a proposed standardized representation fails to improve relevance.

This is why the pilot treats the phenomenon and the mitigation as separate hypotheses. A negative result for a dual-view serializer does not prove that representation is unimportant. It does constrain a stronger claim that adding a structured duplicate universally improves machine understanding.

## Limits of Phase 1

The deterministic marketing and factual-prose conditions are intentionally conservative proxies. They retain the original description and are not comprehensive style rewrites. Attribute normalization is limited to field labeling and entry ordering; fused attribute names are not semantically repaired and absent units are not guessed. The dense encoder is a compact general-purpose MiniLM baseline with a documented custom chunk-pooling policy. Hybrid retrieval shares that dense component and is not an independent second neural model.

All queries are used for exploratory evaluation, with no learned optimization and no relevance-label tuning. The controls, effect estimates and statistical tests improve interpretability but do not constitute an externally preregistered confirmatory study. A final paper needs held-out evaluation for any method subsequently chosen using this pilot, additional neural models, and manual judgments for newly exposed unjudged products.
