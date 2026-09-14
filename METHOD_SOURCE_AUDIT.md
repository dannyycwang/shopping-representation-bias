# Method source audit for the optimization-robustness extension

Frozen source review: 2026-09-07. Sources were read before evaluating the new test sample. This document distinguishes original methods from adaptations used in this repository.

## E-GEO

- Primary paper: Bagga, Farias, Korkotashvili, Peng, and Wu, *E-GEO: A Testbed for Generative Engine Optimization in E-Commerce*, arXiv:2511.20867v2, 14 July 2026.
- Official repository: `https://github.com/psbagga17/E-GEO`, inspected at commit `1016f8adf1ea578129d9aa81d21d924173058c28` (commit date 2026-07-03). A read-only shallow copy is retained in `phase5/vendor/E-GEO/`.
- Original setting: the rewriter sees one product description but not the query. One target product is rewritten within a fixed ten-product candidate set, and an LLM reranker measures rank change. The retrieval set is cached and fixed. The paper therefore studies reranking-stage GEO, not whether rewriting preserves first-stage inclusion.
- Original generation/ranking setup: heuristic prompts use GPT-4.1 as rewriter; the prompt meta-optimizer also uses GPT-4.1, evaluates candidates on 1,000 training and 500 validation queries for two epochs, and selects by mean validation score across four training rerankers. Five LLM rerankers are used for test evaluation. The released code uses the `OPTIMIZE_SYSTEM_PROMPT` plus the selected user prompt, temperature 0.5 for supported models, and model-dependent completion limits.
- Factuality: the paper frames rewriting as fact-preserving and its defended LLM ranker flags questionable descriptions. This is not the same as deterministic field-level fact verification. We do not claim that E-GEO ignores factuality.

Two released prompts are frozen for this extension without looking at extension test outcomes:

1. `heuristic_authoritative`: the complete hand-crafted `authoritative` prompt in `src/all_init_prompts.py`. It explicitly asks not to add or remove core information and keeps the original structure. It is selected as a representative public fact-preservation-oriented heuristic, not because of its E-GEO or extension test rank.
2. `optimized_technical`: the complete meta-optimized `technical` prompt in `src/optimized_prompts.json`. It explicitly requires strict preservation, approximately unchanged length, and structured sections. It is selected before extension evaluation because it is complete, public, accepts an entire product record, and does not require reviews, ratings, competitors, or the query. We do not select the strongest released test result.

### Adaptation in this repository

Both prompts receive the complete WANDS/ESCI product text and no query, label, score, or rank. They run with the same local `Qwen/Qwen2.5-0.5B-Instruct` model at pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`, rather than GPT-4.1/OpenRouter. The original E-GEO system prompt is retained. Primary generation uses greedy decoding and an explicit output budget; a separately frozen stochastic replicate study uses fixed seeds. WANDS/ESCI short keyword queries, dense first-stage retrieval, seven input serializations, and our rule-based fact checker also differ from E-GEO. Every result is therefore labelled an **adapted E-GEO baseline**, not a faithful reproduction of E-GEO scores.

We do not run E-GEO prompt meta-optimization, its ten-candidate LLM reranking evaluation, or its red-team benchmark. We do not add unavailable reviews, ratings, comparison information, or query-specific text.

## SAGEO Arena

- Paper: Kim, Jeong, Kim, Lee, and Lee, *SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization*, arXiv:2602.12187v2, 7 August 2026; KDD 2026.
- Official repository: `https://github.com/happysnail06/SAGEO_Arena`, inspected at commit `e88e2be0b60050ba6c4a72cf68f457f4d4ab33bb` (2026-05-28), retained read-only under `phase5/vendor/SAGEO_Arena/`.
- The paper and code explicitly model retrieval, reranking, and generation over a large web corpus and report that body-text optimization can damage retrieval. It also preserves structural fields and studies stage-specific optimization. We therefore do not claim to be the first end-to-end GEO study. Our narrower difference is a controlled, fact-equivalent product-attribute order intervention, target-level relevant inclusion, generation-noise controls, and raw-fact verification.

## Representational Stability

- Paper: Bhandari, Singh, Gao, Dan, and Gupta, *Improving Robustness of Tabular Retrieval via Representational Stability*, arXiv:2604.24040v2, 28 April 2026.
- Official repository: `https://github.com/KBhandari11/Centroid-Aligned-Table-Retrieval`, inspected at commit `760649a026879fd49057fbcf4f82f0f2c7ebd43b` (2026-03-24), retained under `phase5/vendor/Centroid-Aligned-Table-Retrieval/`.
- It establishes serialization sensitivity for tables and studies centroid targets plus a learned residual adapter. The existing product-centroid comparisons in this repository are a platform-side control and are not claimed as a new pooling idea.

## CHARM

- Paper: Freymuth, Liu, Ricatte, and Mansour, *Hierarchical Multi-field Representations for Two-Stage E-commerce Retrieval*, arXiv:2501.18707v1, 30 January 2025.
- CHARM learns hierarchical field representations with block-triangular attention and combines an aggregated first stage with field-level reranking on multilingual ESCI data. No author code link is given in the paper version inspected. We cite the method but do not reproduce its training. Our existing centroid/max and set-mean baselines remain clearly labelled as simpler platform controls.

## Primary source locations

- E-GEO paper: https://arxiv.org/abs/2511.20867
- E-GEO code/prompts: https://github.com/psbagga17/E-GEO
- SAGEO Arena paper: https://arxiv.org/abs/2602.12187
- SAGEO Arena code: https://github.com/happysnail06/SAGEO_Arena
- Representational Stability paper: https://arxiv.org/abs/2604.24040
- Representational Stability code: https://github.com/KBhandari11/Centroid-Aligned-Table-Retrieval
- CHARM paper: https://arxiv.org/abs/2501.18707
