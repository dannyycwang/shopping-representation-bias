# Phase II Report

## Same Product, Different Visibility: Representation Sensitivity and Relevance–Visibility Alignment in E-Commerce Retrieval

Protocol version: `phase2-frozen-2026-09-04-v1`. Decision: **GO toward a WWW-style full paper.**

## Executive finding

The confirmatory Phase II-A gate passed. Across the three frozen dense retrievers, mean highest-relevance micro VI@20 was 11.33% on WANDS and 5.78% on held-out ESCI. A VI event means that the same judged product entered and left the first 20 results when only the order of its unchanged structured attributes changed. This result therefore generalizes beyond the Phase I MiniLM observation and survives the prespecified position controls.

The Phase II-B result is more qualified. Canonicalization can remove attribute-order instability by construction, but it counts as a successful mitigation only when query-paired relevance is not detectably worse. The table below reports that tradeoff for every model and dataset. Single-product interventions show the direct target effect separately from changes produced when the whole catalog is transformed.

## Go / no-go audit

- PASS — two models two datasets
- PASS — reasonable chunk pool controls
- PASS — broad query types
- PASS — highest relevance affected
- PASS — mitigation success on both datasets

The recommended framing is **representation robustness and relevance–visibility alignment in neural product retrieval**. The evidence does not establish bias by a deployed shopping agent, objective product quality, merchant harm, or downstream purchase effects.

## Experimental setup

WANDS uses 42,994 products, 480 queries and 231,859 cleaned judgments, retaining the Phase I labels and split. The held-out ESCI confirmation uses 10,076 US products, 500 deterministically selected test queries and 10,134 judgments. Query groups were selected before retrieval by sorting SHA-256 of `20260904:query_id`. Every judged product for those queries forms one union catalog; unknown query-product pairs remain unjudged. This is an intentionally difficult full-union retrieval evaluation and is not the official ESCI Task 1 candidate-pool setting.

The official ESCI grades are retained as E=1, C=0.1, S=0.01 and I=0. WANDS uses Exact=3, Partial=1 and Irrelevant=0. Pooled cNDCG removes unjudged products before discounting, while recall and rank operate on the complete retrieval catalog. The [WANDS repository](https://github.com/wayfair/WANDS) and [ESCI paper](https://arxiv.org/abs/2206.06588) document the source collections; the exact ESCI snapshot is pinned in the [official repository](https://github.com/amazon-science/esci-data).

The retrievers are [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with attention-mask mean pooling and 256 tokens, [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) with CLS pooling and 512 tokens, and [GTE-ModernBERT-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) with CLS pooling and 8,192 tokens. Revisions, tokenizers, precision and deterministic tie handling appear in `config/phase2.json` and the embedding metadata.

The primary family contains C0, reversed attribute order and five fixed per-product random attribute orders. All members preserve the character and token multisets of the original plain serialization. Sentence reversal, section reversal and two JSON key orders are secondary controls. Automatic audits verify source-atom retention, numeric equality and exact raw JSON values. The frozen protocol and its pre-result hashes are in `PROTOCOL.md` and `results/protocol_manifest.json`.

## Primary representation sensitivity

| Dataset | Retriever | Median rank range | Pairwise crossing@10 | Pairwise crossing@20 | Pairwise crossing@50 | VI@20 | Query-macro VI@20 | 95% query-bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | minilm | 207 | 3.70% | 5.52% | 8.52% | 13.26% | 19.17% | [17.09%, 21.27%] |
| WANDS | bge_base | 81 | 2.62% | 3.91% | 5.59% | 9.41% | 12.05% | [10.58%, 13.67%] |
| WANDS | gte_modernbert | 107 | 3.30% | 4.68% | 6.60% | 11.33% | 14.43% | [12.94%, 15.98%] |
| ESCI | minilm | 2 | 2.96% | 2.12% | 1.09% | 4.67% | 4.04% | [3.19%, 4.99%] |
| ESCI | bge_base | 2 | 3.05% | 1.92% | 0.97% | 4.42% | 3.66% | [2.84%, 4.57%] |
| ESCI | gte_modernbert | 3 | 4.59% | 3.65% | 1.69% | 8.25% | 6.91% | [5.78%, 8.13%] |

Micro VI treats judged highest-relevance pairs equally. Query-macro VI first averages pairs within a query; its interval uses 10,000 query bootstrap replicates. The go/no-go cell threshold was fixed at micro VI@20 >=3% and query-macro lower CI >1%.

### Secondary fact-equivalent controls

| Dataset | Retriever | Control | Delta cNDCG@10 | Delta highest recall@20 |
| --- | --- | --- | --- | --- |
| WANDS | minilm | C3 | 0.0025 | -0.00% |
| WANDS | minilm | C4 | -0.1040 | -21.66% |
| WANDS | minilm | C5asc | -0.1271 | -23.48% |
| WANDS | minilm | C5desc | -0.0150 | -8.98% |
| WANDS | bge_base | C3 | -0.0009 | -0.15% |
| WANDS | bge_base | C4 | -0.1021 | -21.15% |
| WANDS | bge_base | C5asc | -0.1042 | -22.82% |
| WANDS | bge_base | C5desc | 0.0005 | -0.29% |
| WANDS | gte_modernbert | C3 | 0.0020 | 0.12% |
| WANDS | gte_modernbert | C4 | -0.0117 | -3.27% |
| WANDS | gte_modernbert | C5asc | -0.0196 | -4.01% |
| WANDS | gte_modernbert | C5desc | -0.0097 | -1.76% |
| ESCI | minilm | C3 | -0.0030 | -1.14% |
| ESCI | minilm | C4 | -0.0106 | -3.27% |
| ESCI | minilm | C5asc | -0.0413 | -10.29% |
| ESCI | minilm | C5desc | -0.0264 | -5.38% |
| ESCI | bge_base | C3 | -0.0041 | -0.23% |
| ESCI | bge_base | C4 | -0.0139 | -3.22% |
| ESCI | bge_base | C5asc | -0.0176 | -3.75% |
| ESCI | bge_base | C5desc | -0.0047 | -0.93% |
| ESCI | gte_modernbert | C3 | -0.0007 | -0.32% |
| ESCI | gte_modernbert | C4 | -0.0059 | -2.39% |
| ESCI | gte_modernbert | C5asc | -0.0054 | 0.79% |
| ESCI | gte_modernbert | C5desc | -0.0093 | -0.35% |

These controls are prespecified diagnostics rather than members of the primary seven-variant family. C3 reverses complete description sentences, C4 reverses major sections, and C5 changes JSON key order while preserving decoded values. Large negative or positive shifts are reported as sensitivity evidence, not as mitigation gains.

## Query types

The largest prespecified coarse query-type cells are shown below. The taxonomy is a deterministic lexical heuristic and should be validated manually before submission.

| Dataset | Retriever | Query type | Queries | VI@20 | Median range |
| --- | --- | --- | --- | --- | --- |
| WANDS | minilm | long_descriptive | 10 | 29.13% | 35 |
| WANDS | gte_modernbert | long_descriptive | 10 | 25.88% | 28 |
| WANDS | minilm | attribute_constraint | 56 | 21.21% | 520 |
| WANDS | minilm | brand_entity_proxy | 78 | 18.83% | 103 |
| WANDS | minilm | product_type | 174 | 18.50% | 197 |
| WANDS | minilm | multi_constraint | 48 | 18.27% | 177 |
| WANDS | minilm | style | 13 | 16.96% | 1218 |
| WANDS | bge_base | long_descriptive | 10 | 16.14% | 16 |
| WANDS | gte_modernbert | product_type | 174 | 16.07% | 98 |
| WANDS | gte_modernbert | attribute_constraint | 56 | 13.66% | 284 |
| WANDS | bge_base | attribute_constraint | 56 | 13.31% | 220 |
| WANDS | gte_modernbert | style | 13 | 12.92% | 298 |

## Relevance strata

`results/phase2_tables/relevance_stratified_native.csv` reports every available label separately. Highest-minus-low-relevance query-paired contrasts use 10,000 sign-permutation draws and Holm correction across the three retrievers within dataset. These contrasts test disproportionate instability; the absolute highest-relevance VI values in Table 1 answer whether meaningful products are affected even when the contrast is small or negative.

| Dataset | Retriever | Contrast | Paired queries | Delta VI@20 | CI low | CI high | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WANDS | minilm | Exact-minus-Irrelevant query-macro VI@20 | 346 | 17.01% | 14.66% | 19.42% | 0.0003 |
| WANDS | bge_base | Exact-minus-Irrelevant query-macro VI@20 | 346 | 10.90% | 9.32% | 12.55% | 0.0003 |
| WANDS | gte_modernbert | Exact-minus-Irrelevant query-macro VI@20 | 346 | 12.62% | 10.81% | 14.45% | 0.0003 |
| ESCI | minilm | E-minus-I query-macro VI@20 | 277 | 0.51% | -1.09% | 2.15% | 1.0000 |
| ESCI | bge_base | E-minus-I query-macro VI@20 | 277 | -0.41% | -2.40% | 1.43% | 1.0000 |
| ESCI | gte_modernbert | E-minus-I query-macro VI@20 | 277 | 1.63% | -0.60% | 3.86% | 0.4707 |

## Chunking and position controls

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

The controls vary chunk length, overlap, mean pooling and CLS pooling on the full WANDS catalog for MiniLM and BGE. GTE's native 8,192-token window provides the long-context no-chunking control. Persistence is judged by the same magnitude rule; a failed cell is retained as a boundary condition.

## Relevance–visibility alignment

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

Here `RelevantHidden@20` is one minus recall of the highest relevance label. SAR is computed as per-query cNDCG@10 minus lambda times query-level VI@20 for every prespecified lambda in {0, .1, .25, .5, 1}; all results are in `stability_adjusted_relevance_native.csv`, with no lambda selected after seeing the data.

## Mitigation

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

M1 uses fixed factual field sentences while retaining source strings. M2 labels fields and sorts raw attributes canonically, retaining duplicates. A method is marked successful when its VI point estimate falls and its paired cNDCG interval does not establish a loss. Delta is mitigation minus C0; intervals and sign-permutation p-values are query-paired, and the two methods are Holm-corrected within dataset and retriever.

## Single-product intervention

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

For each judged relevant target, this experiment leaves every competitor in C0 and replaces only the target embedding. The corresponding full-catalog rank uses the same alternative representation for every product. The difference separates a direct representation effect from changes in the competitive score distribution.

## Qualitative evidence and failures

Figure A uses one deterministically selected Top-20 rescue from each dataset, selected only after quantitative analysis and only for illustration. Source descriptions and raw attributes are stored in `results/phase2_qualitative/strong_examples.csv`; the five largest direct degradations per dataset are retained in `failure_cases.csv`. The Phase I Santee example remains valid as prior illustration: for query *acrylic clear chair*, the unchanged source contained `acrylic` and `clear`, while factual framing moved the MiniLM chunked rank from 545 to 2. It is not reused as Phase II proof.

## Statistical interpretation

The primary generalization claim rests on a prespecified magnitude threshold and query-bootstrap uncertainty, rather than a post-hoc null test against exactly zero. Relevance-stratum and mitigation contrasts use paired sign permutations with Holm correction in their predefined families. Query-type cells and the qualitative extremes are exploratory. Confidence intervals quantify query sampling under these fixed catalogs and encoders; they do not cover dataset, model-family or annotation uncertainty.

## Failure modes and limitations

- ESCI is a reproducible 500-query union-catalog confirmation, not the official candidate-list ranking task. Its incomplete judgments make raw full-catalog NDCG inappropriate; this report uses condensed cNDCG and known-label recall.
- WANDS and ESCI labels identify relevance, not the objectively best product or product quality. Source equivalence says that serialization retains recorded facts; it does not certify those facts as true.
- M2 removes attribute-order variability mechanically. Its scientific value depends on the paired relevance result and on whether other lossless perturbations still matter.
- The query taxonomy is coarse, and some apparent brand/entity queries may actually be product types. A blinded manual audit is needed for a camera-ready query analysis.
- Results cover bi-encoder retrieval. They do not yet support claims about rerankers, commercial shopping agents, recommendations, exposure, clicks or sales.
- Extreme improvements coexist with extreme degradations. A representation should not be recommended for deployment from averages alone; the saved failure cases show why.

## Recommendation

**GO toward a WWW-style full paper.** The next paper version should keep the causal intervention narrow: fixed facts, fixed query and fixed model, with serialization as the manipulated variable. If the decision is GO, the highest-value additions are an official ESCI candidate-pool replication, one e-commerce-trained encoder, a manual query-type audit, and a reranker robustness experiment. Proprietary shopping-agent APIs and prompt-heavy agent simulations remain premature.

## Artifact index

- Configuration: `config/phase2.json`
- Frozen protocol: `PROTOCOL.md`
- Main tables: `results/phase2_tables/`
- Figures A–E: `results/phase2_figures/`
- Per-query outputs: `results/phase2_per_query/`
- Judged-pair ranks: `results/phase2_pair_ranks/`
- Single-product outputs: `results/phase2_single_product/`
- Qualitative cases: `results/phase2_qualitative/`
- Equivalence audits: `results/equivalence/`
