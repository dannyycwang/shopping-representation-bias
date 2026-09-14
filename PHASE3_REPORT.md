# Phase III Report: Mechanism, Pipeline Consequences, and Mitigation

## Executive assessment

Phase III is complete and changes the project from a sensitivity observation into a coherent retrieval paper. The evidence now supports a qualified **GO** decision for a Web Conference 2027 submission. The strongest contribution is the combination of a controlled intervention, cross-model and cross-dataset replication, an end-to-end candidate-generation diagnosis, and a simple invariant alternative. The main remaining risk is positioning: the paper must make the Web-search consequence clear and avoid overstating the external validity of two public English product datasets.

## P0: metric reconciliation

An independent audit reproduced every primary Phase II metric to numerical precision (maximum absolute discrepancy (1.11\times10^{-16})). It also found one bookkeeping defect in the soft attribute regularization analysis: an inner join with a visibility table restricted WANDS to the 379 queries that contained an Exact-labelled item, rather than all 479 relevance-eligible queries. The corrected analysis uses a left join, fills absent exact-item visibility with zero, and reports the query count.

The corrected zero-penalty score is exactly the baseline cNDCG: 0.779260 on WANDS and the corresponding baseline on all 500 ESCI queries. The small Phase I/II MiniLM difference (0.738332 versus 0.738328 for the matched 254-token condition) is rounding. The larger 0.779260 result uses the Phase II native 256-token representation; it is not directly comparable to Phase I's full-document chunk pooling.

Files:

- Audit code: `phase3/scripts/audit_metrics.py`
- Audit tests: `phase3/tests/test_metric_reconciliation.py`
- Audit outputs: `phase3/results/phase3_audit/`
- Corrected mitigation code: `phase2/scripts/analyze_mitigations.py`
- Detailed explanation: `METRIC_RECONCILIATION.md`

## P1: official ESCI candidate pools

The official judged-candidate formulation was added for the same 500 ESCI queries. Candidate pools contain a median of 16 products. Visibility instability remains measurable and is most pronounced at the operationally important top ten:

| Retriever | NDCG@10 | micro VI@10 | micro VI@20 | query-macro VI@20 |
|---|---:|---:|---:|---:|
| MiniLM | 0.6853 | 0.0760 | 0.0300 | 0.0130 |
| BGE-base | 0.7124 | 0.0665 | 0.0286 | 0.0130 |
| GTE-ModernBERT | 0.7168 | 0.1155 | 0.0589 | 0.0288 |

Because condensed NDCG removes unjudged documents, its value is mathematically identical to NDCG over the official judged pool for the same ranking. The value of this check is therefore the operational rank/visibility measurement under official candidate sets, not an independent effectiveness replication.

Files:

- Analysis: `phase3/scripts/analyze_official_esci.py`
- Table: `phase3/results/phase3_tables/esci_official_candidate_pool.csv`

## P2: e-commerce-specific retriever

We added the pinned model `Marqo/marqo-ecommerce-embeddings-B` at revision `856b7151a2d1698bf940bc1b3813021c47991481`. It is trained for e-commerce representations and provides a domain-specific stress test beyond the three general-purpose encoders.

| Dataset | median rank range | micro VI@20 | query-macro VI@20 (95% CI) | cNDCG@10 |
|---|---:|---:|---:|---:|
| WANDS | 149 | 0.0607 | 0.0846 [0.0699, 0.1008] | 0.7561 |
| ESCI union | 10 | 0.0722 | 0.0618 [0.0517, 0.0730] | 0.6786 |

Domain training lowers WANDS instability relative to the three generic retrievers, but does not eliminate it. On ESCI its VI@20 is comparable to, or larger than, the generic models. This rules out the simple explanation that the phenomenon is only a mismatch between generic sentence encoders and product text.

Files:

- Analysis: `phase3/scripts/run_ecommerce_retriever.py`
- Embeddings and query-level results: `phase3/results/phase3_ecommerce_retriever/`
- Summary: `phase3/results/phase3_tables/ecommerce_retriever_sensitivity.csv`

## P3: first-stage retrieval plus reranking

The pipeline experiment retrieves the top 100 with BGE-base and reranks each candidate using the pinned cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. It distinguishes two effects:

1. **within-pool instability**, where a product survives candidate generation under every serialization but moves after reranking; and
2. **irrecoverable loss**, where a relevant product is missing from at least one first-stage top-100 pool and the reranker has no opportunity to recover it.

Among products present in every top-100 pool, reranking reduces query-macro VI@20 from 25.94% to 13.03% on WANDS and from 3.95% to 1.37% on ESCI. Median cross-variant rank range falls from 16 to 8 on WANDS and from 2 to 1 on ESCI. Yet query-macro Irrecoverable@100 is 12.52% on WANDS (95% CI 10.94--14.22%) and 0.72% on ESCI (0.47--1.00%).

The practical conclusion is precise: reranking is useful, but it cannot repair a product that representation choice removed before reranking. Candidate generation is therefore part of the fairness and robustness boundary.

Files:

- Pipeline: `phase3/scripts/run_reranker.py`
- Detailed outputs: `phase3/results/phase3_reranking/`
- Tables: `phase3/results/phase3_tables/wands_reranker_pipeline.csv` and `esci_reranker_pipeline.csv`

## P4: permutation-invariant retrieval

The proposed method separates non-attribute product text from attributes. It embeds each attribute independently, averages the attribute embeddings as an unordered set, and interpolates the result with the non-attribute embedding. Because averaging is commutative, permuting attributes leaves the product vector unchanged up to floating-point noise.

The interpolation weight was selected on a deterministic 20% WANDS development split and evaluated on the remaining 384 WANDS queries plus all 500 ESCI queries. The selected set-mean method uses α=0.5.

| Dataset | Method | cNDCG@10 | ΔcNDCG@10 (95% paired CI) | query-macro VI@20 |
|---|---|---:|---:|---:|
| WANDS held-out | Original BGE | 0.8247 | reference | 0.1193 |
| WANDS held-out | Set mean | 0.8151 | -0.0096 [-0.0195, 0.0003] | 0.0000 |
| ESCI union | Original BGE | 0.7124 | reference | 0.0366 |
| ESCI union | Set mean | 0.7059 | -0.0066 [-0.0187, 0.0055] | 0.0000 |

Maximum cross-permutation score difference is (2.38\times10^{-7}), below the predeclared (10^{-6}) invariance tolerance. On both datasets the paired confidence interval includes zero, so the experiment does not establish a relevance loss at the chosen threshold. It also does not prove equivalence; the correct wording is that measured instability is eliminated while the observed relevance change is small and statistically inconclusive.

Canonical sorting (M2) also produces VI=0 and slightly improves ESCI, but causes a confirmed WANDS cNDCG decrease. Late-interaction ablations are invariant but have confirmed WANDS losses. Set mean is therefore the best aligned choice under the preregistered criterion.

Files:

- Method: `phase3/scripts/run_invariant_method.py`
- Pareto analysis: `phase3/scripts/analyze_pareto.py`
- Method outputs: `phase3/results/phase3_invariant_method/`
- Pareto table: `phase3/results/phase3_tables/robustness_relevance_pareto.csv`

## Figures and narrative

The paper uses four complementary figures. Figure 1 makes the intervention concrete using the same turquoise chair whose MiniLM rank changes from 3 to 1,303 under fact-preserving serializations. Figure 2 summarizes VI@20 across four retrievers and two datasets. Figure 3 separates reranker attenuation from irrecoverable first-stage loss. Figure 4 shows the robustness--relevance frontier and highlights set mean.

Files:

- Figure script: `phase3/scripts/make_phase3_figures.py`
- Publication figures and source CSVs: `phase3/results/phase3_figures/`
- Candidate verification: `FIGURE1_CANDIDATES.md` and `figure1_candidates.csv`

## Limitations and next decision

The current package is strong enough to write and circulate as a serious WWW/SIGIR/CIKM submission. Acceptance is not predictable from results alone. The largest substantive limitations are two English-language public product benchmarks, offline retrieval rather than logged user behavior, synthetic but fact-preserving serializations, and no human-validated query taxonomy. The latter has been removed from the main claim; see `QUERY_TAXONOMY_DECISION.md`.

If more compute or annotation budget becomes available, the highest-value extension is a small blinded human audit of whether each variant is semantically equivalent and natural, followed by a multilingual or live-catalog replication. Neither is required to make the present causal result internally valid, but both would strengthen external validity.

