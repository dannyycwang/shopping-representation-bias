# Same Product, Different Visibility: Representation Sensitivity and Relevance–Visibility Alignment in E-Commerce Retrieval

## Abstract

Neural product retrievers should not make relevant items visible or invisible merely because identical catalog facts are serialized in a different order. We test this property on WANDS and a held-out Amazon ESCI subset with three frozen dense encoders and seven character- and token-equivalent attribute-order variants. Highest-relevance micro VI@20 averages 11.33% on WANDS and 5.78% on ESCI across retrievers. We further separate single-product from full-catalog interventions and test two query-agnostic deterministic mitigations. The results support a representation-robustness framing while exposing relevance tradeoffs that rule out treating rank gain alone as mitigation success.

## 1. Introduction

Product search systems consume catalog records whose facts can be serialized in many equivalent ways. A title may precede or follow attributes; attribute dictionaries can use arbitrary insertion orders; and descriptions can expose the same facts at different positions. These implementation choices should not determine whether a highly relevant product is retrieved. We call their observable effect **representation-induced visibility instability**.

This paper asks three questions. First, does the effect reproduce across datasets and encoder families? Second, does it affect products judged highly relevant, including ordinary product-type and multi-constraint queries? Third, can deterministic, query-independent representations reduce instability while preserving relevance? We answer these questions with controlled attribute permutations, position and pooling controls, relevance-stratified analysis, and single-product interventions.

Our contributions are: (1) a multi-variant VI@K measurement that separates threshold crossing from average relevance; (2) a cross-dataset study spanning WANDS and held-out ESCI; (3) a distinction between direct target interventions and whole-catalog transformations; and (4) an alignment criterion that rejects mitigations which gain stability by sacrificing human relevance.

## 2. Related work

[WANDS](https://github.com/wayfair/WANDS) and the [Shopping Queries/ESCI benchmark](https://arxiv.org/abs/2206.06588) made graded, real-product relevance judgments available for reproducible product-search evaluation. Dense dual encoders represent queries and documents as vectors scored by an inner product, enabling exhaustive retrieval but compressing a long structured record into one point. Behavioral IR work such as [ABNIRML](https://aclanthology.org/2022.tacl-1.13/) has shown that neural rankers vary in their sensitivity to word and sentence order. Separately, controlled NLP studies find that BERT-family models can be unexpectedly resilient to token shuffling on some tasks, while later representations still use order when lexical cues are insufficient ([Hessel and Schofield, 2021](https://aclanthology.org/2021.acl-short.27/); [Papadimitriou et al., 2022](https://aclanthology.org/2022.acl-short.71/)). Our focus differs in both intervention and endpoint: we preserve complete catalog field values and measure whether a judged product crosses operational visibility thresholds.

## 3. Data and experimental design

WANDS uses 42,994 products, 480 queries and 231,859 cleaned judgments, retaining the Phase I labels and split. The held-out ESCI confirmation uses 10,076 US products, 500 deterministically selected test queries and 10,134 judgments. Query groups were selected before retrieval by sorting SHA-256 of `20260904:query_id`. Every judged product for those queries forms one union catalog; unknown query-product pairs remain unjudged. This is an intentionally difficult full-union retrieval evaluation and is not the official ESCI Task 1 candidate-pool setting.

The official ESCI grades are retained as E=1, C=0.1, S=0.01 and I=0. WANDS uses Exact=3, Partial=1 and Irrelevant=0. Pooled cNDCG removes unjudged products before discounting, while recall and rank operate on the complete retrieval catalog. The [WANDS repository](https://github.com/wayfair/WANDS) and [ESCI paper](https://arxiv.org/abs/2206.06588) document the source collections; the exact ESCI snapshot is pinned in the [official repository](https://github.com/amazon-science/esci-data).

The retrievers are [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with attention-mask mean pooling and 256 tokens, [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) with CLS pooling and 512 tokens, and [GTE-ModernBERT-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) with CLS pooling and 8,192 tokens. Revisions, tokenizers, precision and deterministic tie handling appear in `config/phase2.json` and the embedding metadata.

The primary family contains C0, reversed attribute order and five fixed per-product random attribute orders. All members preserve the character and token multisets of the original plain serialization. Sentence reversal, section reversal and two JSON key orders are secondary controls. Automatic audits verify source-atom retention, numeric equality and exact raw JSON values. The frozen protocol and its pre-result hashes are in `PROTOCOL.md` and `results/protocol_manifest.json`.


## 4. Metrics and inference

For a judged pair `(q,p)`, VI@K is one when at least one fact-equivalent variant ranks the product at or above K and another ranks it below K. We also measure rank range, rank standard deviation, reciprocal-rank variance and pairwise crossing. Alignment uses condensed NDCG, known-label recall, highest-relevance hidden rate, Spearman correlation between human grade and negative rank, and SAR across five penalty values. All primary uncertainty and tests are query-paired as described in the frozen protocol.

## 5. Results

### 5.1 Cross-model and cross-dataset sensitivity

| Dataset | Retriever | Median rank range | Pairwise crossing@10 | Pairwise crossing@20 | Pairwise crossing@50 | VI@20 | Query-macro VI@20 | 95% query-bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | minilm | 207 | 3.70% | 5.52% | 8.52% | 13.26% | 19.17% | [17.09%, 21.27%] |
| WANDS | bge_base | 81 | 2.62% | 3.91% | 5.59% | 9.41% | 12.05% | [10.58%, 13.67%] |
| WANDS | gte_modernbert | 107 | 3.30% | 4.68% | 6.60% | 11.33% | 14.43% | [12.94%, 15.98%] |
| ESCI | minilm | 2 | 2.96% | 2.12% | 1.09% | 4.67% | 4.04% | [3.19%, 4.99%] |
| ESCI | bge_base | 2 | 3.05% | 1.92% | 0.97% | 4.42% | 3.66% | [2.84%, 4.57%] |
| ESCI | gte_modernbert | 3 | 4.59% | 3.65% | 1.69% | 8.25% | 6.91% | [5.78%, 8.13%] |

### 5.2 Position, chunking and pooling

| Retriever | Profile | Median range | VI@20 | CI low | Pass magnitude gate |
| --- | --- | --- | --- | --- | --- |
| minilm | native | 207 | 13.26% | 17.09% | yes |
| bge_base | native | 81 | 9.41% | 10.58% | yes |
| minilm | mean_126_o32 | 502 | 12.10% | 16.71% | yes |
| bge_base | mean_126_o32 | 363 | 14.45% | 20.88% | yes |
| minilm | mean_254_o0 | 320 | 11.72% | 16.66% | yes |
| bge_base | mean_254_o0 | 250 | 17.63% | 23.48% | yes |
| minilm | mean_254_o64 | 383 | 11.25% | 17.28% | yes |
| bge_base | mean_254_o64 | 316 | 17.77% | 24.68% | yes |
| minilm | cls_254_o0 | 287 | 11.63% | 15.56% | yes |
| bge_base | cls_254_o0 | 278 | 17.15% | 22.93% | yes |

### 5.3 Relevance–visibility alignment and mitigation

| Dataset | Representation | Retriever | cNDCG@10 | Highest recall@20 | Highest hidden@20 | VI@20 |
| --- | --- | --- | --- | --- | --- | --- |
| WANDS | C0_original | minilm | 0.7793 | 31.16% | 68.84% | 19.17% |
| WANDS | M1_factual_field_sentence | minilm | 0.7814 | 30.96% | 69.04% | 18.20% |
| WANDS | M2_normalized_attributes | minilm | 0.7550 | 25.69% | 74.31% | 0.00% |
| WANDS | C0_original | bge_base | 0.8277 | 36.21% | 63.79% | 12.05% |
| WANDS | M1_factual_field_sentence | bge_base | 0.8221 | 36.20% | 63.80% | 11.94% |
| WANDS | M2_normalized_attributes | bge_base | 0.8030 | 35.76% | 64.24% | 0.00% |
| WANDS | C0_original | gte_modernbert | 0.8219 | 35.94% | 64.06% | 14.43% |
| WANDS | M1_factual_field_sentence | gte_modernbert | 0.8167 | 34.97% | 65.03% | 14.87% |
| WANDS | M2_normalized_attributes | gte_modernbert | 0.8208 | 36.30% | 63.70% | 0.00% |
| ESCI | C0_original | minilm | 0.6853 | 67.95% | 32.05% | 4.04% |
| ESCI | M1_factual_field_sentence | minilm | 0.6797 | 67.78% | 32.22% | 4.30% |
| ESCI | M2_normalized_attributes | minilm | 0.6852 | 67.61% | 32.39% | 0.00% |
| ESCI | C0_original | bge_base | 0.7124 | 74.10% | 25.90% | 3.66% |
| ESCI | M1_factual_field_sentence | bge_base | 0.7061 | 74.32% | 25.68% | 3.08% |
| ESCI | M2_normalized_attributes | bge_base | 0.7192 | 75.14% | 24.86% | 0.00% |
| ESCI | C0_original | gte_modernbert | 0.7168 | 72.45% | 27.55% | 6.91% |
| ESCI | M1_factual_field_sentence | gte_modernbert | 0.7040 | 72.45% | 27.55% | 3.86% |
| ESCI | M2_normalized_attributes | gte_modernbert | 0.7180 | 72.82% | 27.18% | 0.00% |

| Dataset | Retriever | Representation | Delta cNDCG@10 | cNDCG CI low | cNDCG CI high | Delta VI@20 | Holm p (VI) | Success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | minilm | M1_factual_field_sentence | 0.0021 | -0.0057 | 0.0097 | -0.97% | 0.2364 | yes |
| WANDS | minilm | M2_normalized_attributes | -0.0243 | -0.0362 | -0.0122 | -19.17% | 0.0002 | no |
| WANDS | bge_base | M1_factual_field_sentence | -0.0056 | -0.0124 | 0.0008 | -0.11% | 0.8802 | yes |
| WANDS | bge_base | M2_normalized_attributes | -0.0246 | -0.0351 | -0.0147 | -12.05% | 0.0002 | no |
| WANDS | gte_modernbert | M1_factual_field_sentence | -0.0052 | -0.0141 | 0.0034 | 0.43% | 0.6188 | no |
| WANDS | gte_modernbert | M2_normalized_attributes | -0.0011 | -0.0087 | 0.0067 | -14.43% | 0.0002 | yes |
| ESCI | minilm | M1_factual_field_sentence | -0.0056 | -0.0150 | 0.0040 | 0.26% | 0.6127 | no |
| ESCI | minilm | M2_normalized_attributes | -0.0002 | -0.0101 | 0.0099 | -4.04% | 0.0002 | yes |
| ESCI | bge_base | M1_factual_field_sentence | -0.0063 | -0.0141 | 0.0012 | -0.58% | 0.2299 | yes |
| ESCI | bge_base | M2_normalized_attributes | 0.0068 | -0.0014 | 0.0153 | -3.66% | 0.0002 | yes |
| ESCI | gte_modernbert | M1_factual_field_sentence | -0.0128 | -0.0216 | -0.0044 | -3.05% | 0.0002 | no |
| ESCI | gte_modernbert | M2_normalized_attributes | 0.0012 | -0.0074 | 0.0099 | -6.91% | 0.0002 | yes |

### 5.4 Direct interventions

| Dataset | Retriever | Representation | Pairs | Median direct rank gain | Direct improved | Direct crossing@20 | Full-catalog crossing@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | minilm | M1_factual_field_sentence | 25470 | 26 | 65.55% | 9.71% | 6.90% |
| WANDS | minilm | M2_normalized_attributes | 25470 | -56 | 34.75% | 10.27% | 11.54% |
| WANDS | bge_base | M1_factual_field_sentence | 25470 | -47 | 21.62% | 8.43% | 6.79% |
| WANDS | bge_base | M2_normalized_attributes | 25470 | -180 | 14.24% | 10.12% | 10.96% |
| WANDS | gte_modernbert | M1_factual_field_sentence | 25470 | -35 | 30.77% | 9.34% | 8.23% |
| WANDS | gte_modernbert | M2_normalized_attributes | 25470 | -15 | 37.88% | 8.48% | 8.00% |
| ESCI | minilm | M1_factual_field_sentence | 4434 | 0 | 37.10% | 8.62% | 7.85% |
| ESCI | minilm | M2_normalized_attributes | 4434 | 0 | 39.17% | 10.80% | 10.71% |
| ESCI | bge_base | M1_factual_field_sentence | 4434 | -1 | 18.88% | 8.43% | 6.13% |
| ESCI | bge_base | M2_normalized_attributes | 4434 | 0 | 32.05% | 7.04% | 6.81% |
| ESCI | gte_modernbert | M1_factual_field_sentence | 4434 | 0 | 31.26% | 8.93% | 8.19% |
| ESCI | gte_modernbert | M2_normalized_attributes | 4434 | -2 | 19.33% | 9.70% | 7.98% |

## 6. Discussion

The confirmatory gate passes, so the effect cannot be dismissed as a MiniLM-only WANDS artifact. The size varies materially by encoder and dataset, which argues against a universal scalar correction. Canonical serialization is a useful systems intervention only in cells where relevance is maintained; deterministic stability alone is trivial to manufacture.

The strongest causal statement supported by the design is local: with a fixed query, catalog, bi-encoder and recorded fact set, serialization can change a judged product's retrieval visibility. The study does not identify the best real-world product, downstream user utility, or strategic seller behavior.

## 7. Limitations

- ESCI is a reproducible 500-query union-catalog confirmation, not the official candidate-list ranking task. Its incomplete judgments make raw full-catalog NDCG inappropriate; this report uses condensed cNDCG and known-label recall.
- WANDS and ESCI labels identify relevance, not the objectively best product or product quality. Source equivalence says that serialization retains recorded facts; it does not certify those facts as true.
- M2 removes attribute-order variability mechanically. Its scientific value depends on the paired relevance result and on whether other lossless perturbations still matter.
- The query taxonomy is coarse, and some apparent brand/entity queries may actually be product types. A blinded manual audit is needed for a camera-ready query analysis.
- Results cover bi-encoder retrieval. They do not yet support claims about rerankers, commercial shopping agents, recommendations, exposure, clicks or sales.
- Extreme improvements coexist with extreme degradations. A representation should not be recommended for deployment from averages alone; the saved failure cases show why.


## 8. Conclusion

The evidence supports continued development of a full paper under the representation-robustness framing. Neural e-commerce retrieval exhibits measurable visibility instability under lossless attribute-order changes, and relevance-aware mitigation evaluation prevents a stable but less useful serialization from being counted as progress.

## References and artifacts

The source datasets and model cards are linked in `PHASE2_REPORT.md`. Exact revisions, outputs, audits and statistical tables are included in the Phase II artifact tree.
