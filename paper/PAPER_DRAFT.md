# Same Product Information, Different Visibility: A Catalog-Wide Study of Representation Sensitivity in E-Commerce Retrieval

**First empirical manuscript draft — exploratory Phase 1, September 4, 2026.** Authors and affiliations to be supplied. This is not a submission-ready claim of shopping-agent recommendation effects.

## Abstract

We investigate whether product visibility depends on how unchanged source information is serialized. Using the complete WANDS catalog of 42,994 products and 480 queries, we evaluate six deterministic source-preserving representations under BM25, a chunk-pooled MiniLM dense retriever and full-list reciprocal rank fusion. Two additional controls isolate source repetition and attribute order. After deduplication and exclusion of contradictory judgments, evaluation uses 231,859 unique query-product labels, including 25,470 Exact pairs. We report judged-condensed NDCG, known-relevant recall and bidirectional Top-20 exposure transitions without labeling unknown products as irrelevant. Dual View produced cNDCG@10 differences of +0.0032, +0.0004, and +0.0057 for BM25, Dense, and Hybrid, respectively; none survived Holm correction across the three designated comparisons. In a strict order-only control with identical character counts and lexical and WordPiece multisets, 1,260 of 25,470 Exact pairs (4.95%) crossed the Dense Top-20 boundary despite no detectable mean cNDCG improvement. The results distinguish representation sensitivity from successful relevance improvement and motivate provenance-preserving evaluation before making end-to-end claims about shopping agents.

## 1. Introduction

Product discovery depends on the interface between product information and retrieval systems. A title, merchant description, taxonomy and attribute feed can describe the same source record in different ways. A shopping system's candidate set may consequently depend on serialization choices as well as product relevance. We ask whether source-equivalent catalog representations change the exposure of products judged relevant by humans.

Our focus is the retrieval stage. A downstream shopping assistant cannot choose an absent candidate, but improved retrieval does not guarantee improved recommendations. We therefore distinguish the motivation of agentic commerce from the measured behavior of retrieval models. We also distinguish exposure sensitivity from a successful mitigation: substantial movement can coexist with little or negative aggregate relevance change.

The study contributes a reproducible whole-catalog pilot, explicit accounting of both recovered and newly hidden Exact products, and deterministic controls for formatting, repetition and attribute order. It does not claim a new universal representation method or exhaustive coverage of language styles.

## 2. Related work and positioning

[E-GEO v2](https://arxiv.org/abs/2511.20867v2) studies product-description rewriting across generative engines with fixed candidate sets. [SAGEO Arena v2](https://arxiv.org/abs/2602.12187v2) extends visibility evaluation across retrieval, reranking and generation. Document-side modification itself is established prior art, including [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375). Our narrower emphasis is source-preserving universal catalog treatment, human relevance labels, and both directions of exposure change. A broader novelty review remains necessary before submission.

## 3. Problem formulation

Let C be a catalog, F_p the source record for product p, Y(q,p) its fixed human relevance label, and r a query-independent serializer. Applying r to every product yields ranking L(q,r(C)). Define V_K(q,p;r)=1[rank(q,p;r(C))≤K]. A change in visibility under different source-preserving r indicates representation sensitivity at the catalog level.

Universal treatment changes competitors and corpus statistics together. These effects are not the isolated direct causal effect of rewriting only one product. We use **source-preserving** rather than assuming all catalog statements are correct: the data contain multiple values under the same attribute key, missing fields and malformed entries.

For Exact-labeled pairs, RelevantHidden@K is the fraction with original rank >K and alternative rank ≤K. ReverseHidden@K reverses those inequalities. Their shared denominator is all 25,470 Exact pairs. We also calculate query-macro net transition rates. Rank span is maximum minus minimum rank, rank ratio is maximum divided by minimum rank, and HitInstability@K records whether a pair crosses the boundary across representations.

## 4. Experimental design

### 4.1 Data

We use [WANDS](https://github.com/wayfair/WANDS), pinned to commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`. The release contains 233,448 annotation rows. We collapse identical duplicate pair judgments and exclude 14 pairs with contradictory judgments, producing 231,859 unique pairs: 25,470 Exact, 145,670 Partial and 60,719 Irrelevant. The catalog and all 480 queries are retained. Only 379 queries have Exact judgments, and 479 have at least one known relevant product. Queries average 3.38 whitespace-delimited words; findings should not be generalized directly to long conversational requests.

We inspect every product field. Descriptions are absent for 13.97% of products, product class for 6.63%, category hierarchy for 3.62%, and rating/review fields for 21.98%. There are 29,547 records with multiple distinct values under at least one nonempty normalized key, and 34,990 with malformed or blank feature keys. Multiple values may reflect product variants; the pilot does not resolve them.

### 4.2 Representations and fidelity

R0 concatenates original name, class, category, description and features. R2 adds generic shopping invitations. R3 introduces factual field framing but retains the source description. R4 replaces feature separators with bullets while preserving the BM25 token multiset. R5 labels fields, sorts raw feature entries and places the unchanged description after attributes. R8 concatenates R0 and R5. C_repeat duplicates R0. C_order reverses raw attribute entry order without changing characters or token multisets.

These are conservative serialization prototypes. R2 is not a comprehensive persuasive rewrite, R3 does not remove source marketing language, and R5 does not infer units or fully segment fused attribute keys. All conditions use identical source fields; ratings are excluded uniformly. The generator sees no queries or labels. Auditing checks source-atom retention, novel numeric values, unapproved residual words and deterministic output equality. All 343,952 generated records pass these restricted checks. This verifies the transformations, not real-world truth or semantic equivalence of arbitrary paraphrases.

### 4.3 Retrievers

BM25 uses lowercased alphanumeric tokens, k1=1.2 and b=0.75, no stemming, and a fresh catalog-specific index per representation. Unique query terms are accumulated in sorted order. Dense retrieval uses the fixed `sentence-transformers/all-MiniLM-L6-v2` revision in the configuration. It is a compact general-purpose baseline. All WordPieces are encoded in non-overlapping 254-token chunks with special tokens added; chunk means are weighted by content length, averaged and L2-normalized. The implementation uses FP16 model weights and FP32 pooling on an RTX 4060-class GPU. It does not apply the model's default first-256-token truncation. For R0, 93.36% of products exceed 254 content WordPieces.

Hybrid retrieval uses equal-weight reciprocal rank fusion over complete lists, with constant 60. Full lists permit uncensored ranks for every Exact pair. Ties are resolved by ascending product ID, and boundary ties are reported. No model, hyperparameter or representation is optimized against WANDS relevance judgments.

### 4.4 Metrics and statistics

Unknown products remain unjudged. Following the distinction in [incomplete-assessment evaluation](https://link.springer.com/article/10.1007/s10791-008-9059-7), we report **judged-condensed NDCG** (cNDCG): unjudged products are removed before rank-based discounting, using gains 3, 1 and 0. This is not ordinary full-catalog NDCG and may be optimistic under incomplete pooling. Judgment coverage accompanies it.

Known-relevant Recall@K uses Exact and Partial judgments and full-catalog ranks. MRR is the reciprocal full-catalog rank of the first known relevant item. ExactHit is the fraction of Exact-bearing queries with an Exact result in top K. ExplicitIrrelevant@10 and Unjudged@10 are kept separate. Metrics with no positive judgments are undefined and excluded, rather than assigned a misleading success or failure.

We use 10,000 query-paired bootstrap replicates and paired sign-permutation tests with seed 20260903. The designated primary comparisons are R8 versus R0 on cNDCG@10, with Holm correction across the three retrievers. Other analyses are exploratory. All queries are used to measure this pilot; any subsequently selected method requires held-out evaluation. The study was not externally preregistered.

## 5. Results

### 5.1 Catalog-wide relevance

| representation | retriever | cNDCG@10 | cNDCG@20 | Recall@10 | Recall@20 | Recall@50 | MRR | ExactHit@20 | JudgedCoverage@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_order | BM25 | 0.7896 | 0.7987 | 0.0568 | 0.1067 | 0.2284 | 0.8347 | 0.8813 | 0.7885 |
| C_order | Dense | 0.7361 | 0.7572 | 0.0508 | 0.0936 | 0.1896 | 0.7971 | 0.7441 | 0.7094 |
| C_order | Hybrid | 0.7824 | 0.7966 | 0.0597 | 0.1110 | 0.2314 | 0.8768 | 0.8760 | 0.8125 |
| C_repeat | BM25 | 0.7927 | 0.8010 | 0.0558 | 0.1048 | 0.2230 | 0.8291 | 0.8892 | 0.7688 |
| C_repeat | Dense | 0.7311 | 0.7513 | 0.0485 | 0.0884 | 0.1817 | 0.7796 | 0.7467 | 0.6790 |
| C_repeat | Hybrid | 0.7820 | 0.7956 | 0.0585 | 0.1097 | 0.2279 | 0.8543 | 0.8734 | 0.7998 |
| R0 | BM25 | 0.7896 | 0.7987 | 0.0568 | 0.1067 | 0.2284 | 0.8347 | 0.8813 | 0.7885 |
| R0 | Dense | 0.7383 | 0.7573 | 0.0498 | 0.0930 | 0.1901 | 0.7979 | 0.7625 | 0.6902 |
| R0 | Hybrid | 0.7822 | 0.7960 | 0.0597 | 0.1121 | 0.2306 | 0.8648 | 0.8734 | 0.8075 |
| R2 | BM25 | 0.7910 | 0.8000 | 0.0568 | 0.1068 | 0.2288 | 0.8363 | 0.8865 | 0.7883 |
| R2 | Dense | 0.7274 | 0.7500 | 0.0477 | 0.0871 | 0.1784 | 0.7804 | 0.7177 | 0.6677 |
| R2 | Hybrid | 0.7788 | 0.7920 | 0.0579 | 0.1085 | 0.2263 | 0.8645 | 0.8602 | 0.7983 |
| R3 | BM25 | 0.7908 | 0.7999 | 0.0567 | 0.1069 | 0.2290 | 0.8374 | 0.8865 | 0.7888 |
| R3 | Dense | 0.7484 | 0.7674 | 0.0506 | 0.0942 | 0.1951 | 0.8199 | 0.7731 | 0.7129 |
| R3 | Hybrid | 0.7912 | 0.8047 | 0.0604 | 0.1123 | 0.2341 | 0.8793 | 0.8813 | 0.8227 |
| R4 | BM25 | 0.7896 | 0.7987 | 0.0568 | 0.1067 | 0.2284 | 0.8347 | 0.8813 | 0.7885 |
| R4 | Dense | 0.7432 | 0.7621 | 0.0510 | 0.0944 | 0.1942 | 0.8089 | 0.7573 | 0.7046 |
| R4 | Hybrid | 0.7852 | 0.7988 | 0.0598 | 0.1129 | 0.2329 | 0.8673 | 0.8839 | 0.8142 |
| R5 | BM25 | 0.7908 | 0.7995 | 0.0568 | 0.1068 | 0.2287 | 0.8375 | 0.8813 | 0.7890 |
| R5 | Dense | 0.7487 | 0.7696 | 0.0527 | 0.0964 | 0.1980 | 0.8342 | 0.7467 | 0.7267 |
| R5 | Hybrid | 0.7857 | 0.8002 | 0.0597 | 0.1128 | 0.2356 | 0.8823 | 0.8602 | 0.8179 |
| R8 | BM25 | 0.7928 | 0.8012 | 0.0558 | 0.1050 | 0.2231 | 0.8303 | 0.8892 | 0.7688 |
| R8 | Dense | 0.7387 | 0.7612 | 0.0509 | 0.0942 | 0.1948 | 0.8050 | 0.7546 | 0.7100 |
| R8 | Hybrid | 0.7879 | 0.8009 | 0.0592 | 0.1120 | 0.2343 | 0.8629 | 0.8602 | 0.8117 |

Table 1. Complete results, including two controls. Recall measures recovery of known positives, not true recall over the entire incompletely judged catalog. ExactHit uses 379 eligible queries. cNDCG, Recall and MRR use 479 queries with known positives.

The designated Dual View intervention did not yield a confirmed relevance gain. Its Hybrid difference had a bootstrap interval above zero and an unadjusted permutation p value of 0.0333, but the Holm-adjusted p value was 0.0999; the BM25 and Dense comparisons were also nonsignificant. Exploratory results suggest that factual field framing (R3) increased cNDCG@10 by 0.0101 for Dense and 0.0090 for Hybrid, while labeled and sorted fields (R5) increased Dense cNDCG@10 by 0.0104. These estimates were selected from the same evaluation set and require held-out confirmation. Generic shopping invitations (R2) reduced Dense cNDCG@10 by 0.0109 (95% CI -0.0177 to -0.0042), showing that added fluent text can be harmful.

| retriever | delta | ci_low | ci_high | permutation_p | holm_primary_p |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.0032 | -0.0007 | 0.0075 | 0.1209 | 0.2418 |
| Dense | 0.0004 | -0.0063 | 0.0070 | 0.9064 | 0.9064 |
| Hybrid | 0.0057 | 0.0004 | 0.0109 | 0.0333 | 0.0999 |

Table 2. Dual View minus Original on cNDCG@10; paired 95% bootstrap confidence intervals and permutation p values. Holm-adjusted values apply to the three designated primary comparisons.

### 5.2 Exposure redistribution

| retriever | exact_pairs | rescued | harmed | RelevantHidden_micro | ReverseHidden_micro | net_macro | net_macro_ci_low | net_macro_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 25470 | 128 | 118 | 0.0050 | 0.0046 | 0.0127 | 0.0042 | 0.0230 |
| Dense | 25470 | 649 | 581 | 0.0255 | 0.0228 | -0.0057 | -0.0180 | 0.0058 |
| Hybrid | 25470 | 485 | 431 | 0.0190 | 0.0169 | -0.0015 | -0.0139 | 0.0113 |

Table 3. Exact-pair Top-20 transitions under Dual View. Rescue and harm share a denominator of 25,470 pairs. Macro net rates average within query; their confidence intervals resample queries, not independent product pairs.

Dual View redistributed rather than uniformly improved exposure. At Top-20, BM25 rescued 128 and harmed 118 Exact pairs; Dense rescued 649 and harmed 581; Hybrid rescued 485 and harmed 431. Dense therefore moved 1,230 pairs (4.83% of all Exact pairs) across the boundary, while Hybrid moved 916 (3.60%). Query-macro crossing rates were 7.20% and 6.02%, respectively. The query-macro net intervals for Dense and Hybrid included zero, and their signs need not match micro totals because queries contain unequal numbers of judgments. Reporting only rescued items would overstate benefit.

![Bidirectional Top-20 transitions](../results/figures/hidden_products.png)

### 5.3 Controls

| retriever | delta | ci_low | ci_high | permutation_p |
| --- | --- | --- | --- | --- |
| BM25 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Dense | -0.0022 | -0.0088 | 0.0045 | 0.5116 |
| Hybrid | 0.0002 | -0.0047 | 0.0050 | 0.9406 |

| retriever | rescued | harmed | RelevantHidden_micro | ReverseHidden_micro |
| --- | --- | --- | --- | --- |
| BM25 | 0 | 0 | 0.0000 | 0.0000 |
| Dense | 640 | 620 | 0.0251 | 0.0243 |
| Hybrid | 488 | 474 | 0.0192 | 0.0186 |

Table 4. Order-only control: relevance changes and exposure transitions. Character counts, lexical token multisets and WordPiece multisets are verified identical for every product. Whole-catalog BM25 score and rank equality is verified separately.

| retriever | metric | Dual_minus_repeat | ci_low | ci_high | p |
| --- | --- | --- | --- | --- | --- |
| BM25 | cNDCG@10 | 0.0001 | -0.0003 | 0.0005 | 0.5275 |
| BM25 | Recall@20 | 0.0002 | 0.0000 | 0.0004 | 0.0188 |
| Dense | cNDCG@10 | 0.0076 | 0.0018 | 0.0134 | 0.0113 |
| Dense | Recall@20 | 0.0059 | 0.0041 | 0.0077 | 0.0001 |
| Hybrid | cNDCG@10 | 0.0058 | 0.0008 | 0.0113 | 0.0288 |
| Hybrid | Recall@20 | 0.0023 | 0.0014 | 0.0033 | 0.0001 |

Table 5. Dual View minus repeated Original. Repetition approximates length but is not a strict per-product length match. C_order provides the stricter length-and-content control.

The order-only control changed no character count, lexical token multiset, or WordPiece multiset. BM25 scores and ranks were exactly invariant, as expected. Dense nevertheless moved 640 Exact pairs into and 620 out of Top-20 (4.95% total crossing), while its cNDCG@10 difference was -0.00225 (95% CI -0.00877 to 0.00449). Hybrid moved 488 in and 474 out (3.78%), with a near-zero mean difference. In an exploratory subset of 3,385 records without repeated-key multi-values or malformed keys, Dense still moved 139 of 1,811 Exact pairs (7.68%). This subset reduces source ambiguity but does not verify external product truth. Repeating the original text lowered Dense cNDCG@10 by 0.00720, so added length alone is not a benign control.

### 5.4 Qualitative evidence

We select 20 strongest improvements and 20 strongest degradations crossing the Top-20 boundary from the main representations, deduplicating query-product pairs. Original descriptions, features and ranks are retained in the qualitative appendix. The source inspection was conducted by the research assistant producing this pilot, not by an independent human annotation panel. Extreme cases are not representative samples and do not establish population-level mechanisms.

Among the 20 strongest improvements, 15 had explicit source support, four had partial support, and one query was underspecified. Large gains often involved explicit attributes that became prominent under R5 or R3, such as cow print (rank 1,562 to 9), leather-chair information (1,417 to 13), and an LED bed (791 to 13). Among the 20 strongest degradations, 16 had explicit support, three partial support, and one underspecified query. Losses clustered around short name, collection, or style terms, including Amarillo, Kisner, bohemian, and mid-century modern, which suggests that attribute-heavy serialization can dilute concise identifiers. Several cases share a query and the extreme-case appendix is illustrative rather than an independent sample; query-paired statistics carry the inferential burden.

## 6. Discussion

The pilot supports a representation-sensitivity claim: a neural retriever can alter the candidate visibility of human-labeled relevant products when source content and token inventories are held fixed but sequence order changes. It does not support the stronger claim that a product representation reliably improves recommendation quality. Mean relevance effects were small and intervention-dependent, the planned Dual View comparisons failed multiplicity correction, and gains were accompanied by harms. The contrast between explicit attribute gains and short-identifier losses offers a testable mechanism for a follow-up study, but it is not established by the selected cases alone.

The appropriate unit of interpretation is the information environment. A product's exposure gain alone does not demonstrate that users see more relevant results. The bidirectional transition analysis and the repetition control guard against that inference. Standardizing representations can also enforce invariance to a restricted transformation class without improving mean relevance; these are separate properties to test.

## 7. Limitations and next experiments

This pilot uses one catalog, short queries, incomplete judgments and one neural encoder. Hybrid shares its dense component and is not an independent cross-model replication. cNDCG changes ranking semantics by removing unjudged items. Raw attribute feeds may mix variants; a human Exact label does not verify all source statements or certify the objectively best product. Deterministic wrappers do not establish results for arbitrary marketing language, factual prose, schema markup or LLM paraphrases. Full-list fusion and chunk pooling are specific design choices, and no reranker or shopping agent is evaluated.

A next study should replicate the order-only effect with a second strong neural retriever, compare single-product and universal interventions, obtain relevance judgments for newly exposed candidates, and separate normalization effects from text length and repetition on a held-out query set. A learned factual product compiler must be evaluated on both preservation and relevance. Expensive shopping-agent experiments should follow evidence that survives these checks.

## 8. Conclusion

Catalog serialization is a consequential component of e-commerce retrieval, particularly for neural encoders, but the tested Dual View is not a validated mitigation. The strongest evidence is bidirectional visibility instability under an order-only, source-preserving control, coupled with no average relevance gain. A second retriever, alternative pooling choices, single-product interventions, new judgments for exposed candidates, and held-out evaluation are required before extending the claim to shopping agents or recommending a standardized representation method.

## Reproducibility

The repository contains deterministic serializers, sparse BM25, a configurable dense encoder, full-list fusion, per-query evaluation, Exact-pair ranks, 10,000-replicate uncertainty estimates, fidelity records, selected source examples and PNG/SVG figures. The manifest records dataset checksums, model revision, package versions, seeds and code hashes. No paid API calls or generated experimental numbers are used. See [README](../README.md), [numerical appendix](../results/NUMERICAL_APPENDIX.md) and [Chinese research report](../results/REPORT_zh.md).

## References

1. Chen et al. (2022). *WANDS: Dataset for Product Search Relevance Assessment*. ECIR. [Dataset and citation](https://github.com/wayfair/WANDS).
2. Bagga et al. (2026 version). *E-GEO: A Testbed for Generative Engine Optimization in E-Commerce*. [arXiv:2511.20867v2](https://arxiv.org/abs/2511.20867v2).
3. Kim et al. (2026). *SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization*. [arXiv:2602.12187v2](https://arxiv.org/abs/2602.12187v2).
4. Nogueira et al. (2019). *Document Expansion by Query Prediction*. [arXiv:1904.08375](https://arxiv.org/abs/1904.08375).
5. Sakai and Kando (2008). *On information retrieval metrics designed for evaluation with incomplete relevance assessments*. [Publisher record](https://link.springer.com/article/10.1007/s10791-008-9059-7).
6. Sentence Transformers. *all-MiniLM-L6-v2 model card*. [Model documentation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
