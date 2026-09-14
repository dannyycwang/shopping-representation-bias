# Frozen retrieval-first extension protocol

Frozen before new method evaluation, 2026-09-07. This is a post-hoc extension, not a preregistration of the original study. Previous WANDS/ESCI test results have been inspected. Do not modify strategies after evaluating them on held-out/external labels.

## Populations and selection

Reuse the Phase III SHA256 `phase3-dev:{query_id}` WANDS split: lowest 20% development, remainder held-out. Store explicit IDs and source hashes in `phase4/config/splits.json`. No product-supervised strategy or category-specific tuning is planned; this is query holdout, not product-disjoint generalization. Keep the historical fixed complete WANDS catalog and ESCI 500-query union for external evaluation. All methods in a cell share exactly the same catalog.

Reserve 100 additional US small-version test ESCI queries, excluding all historical query IDs, in SHA256 `20260907-prospective:{query_id}` order. Freeze IDs before retrieval. If historical usage cannot be excluded, label them newly sampled fixed evaluation, not unquestionably unseen. Use their product union plus the original ESCI union as one new common catalog. No direct causal comparison of its absolute recall with the old catalog. Evaluate only fixed selected methods; no selection on new-query results.

## Methods and permissions

A0: Original C0; historical canonical M2; existing set mean (alpha .5); BM25 k1=1.2,b=.75, lowercase alphanumeric bag of tokens, no truncation; full-ranking RRF(60) BM25+Original. Keep empty query prefix and pinned encoders for comparability.

A1: Four query-independent priority rules over existing complete raw atoms: type/style; color/appearance; use/compatibility; dimension/material. Match field keys in WANDS and literal matching prefixes in unkeyed ESCI atoms; never create field names or change text. Priority keyword lists are frozen in code/config before evaluation. Within matched/unmatched groups use canonical lexical ordering with duplicate multiplicity retained. Title/free text/sections/separators unchanged. Choose exactly one global rule by WANDS development macro highest-label Recall@100, then cNDCG@10, then fixed rule ID. No category-specific follow-up in this bounded run. Report catalog-wide and target-only results for this rule, with competitors C0.

A2/A3: Canonical raw atom list shuffled with product-ID plus seeds 20260971–20260977. Nested first m=2/4/7 views, same m for all products. Each complete view is encoded and L2-normalized. Centroid averages vectors and then normalizes; max uses the maximum cosine and returns K unique product IDs. The deterministic canonical-based view generator is input-order independent, but finite-view averaging is not claimed to exhaust all permutations or yield general semantic invariance. Save actual unique text-view counts. Initial screening m=2/4; m=7 is a fixed saturation check, not a chance to retune.

Select best added strategy among four rules, centroid2/4 and max2/4 using the same recall/cNDCG/simplicity order (one-vector before multi-vector, smaller m, then ID). Freeze a selection JSON before held-out scoring. m=7 cannot replace the winner. Evaluate all screening results on development and preserve negatives. Transfer the selected rule and selected method unchanged to MiniLM and ESCI. No encoder-specific retuning.

## Outcomes and inference

Primary: query-macro highest-label Recall@100. Secondary: Recall@20/50/200, cNDCG@10, VI@20 and cost. K sweep 20/50/100/200/500/1000 using saved ranks. Highest-label-free queries have undefined recall, not zero; cNDCG includes all positive-IDCG queries through independent query tables. Official ESCI gains E/S/C/I=1/.1/.01/0; historical gains also recomputed for reconciliation. Discount linear gains, matching existing evaluator.

For each K and query: rescued/newly missed/net pairs relative to Original, divided by that query's highest-label count; also raw counts. For the seven original views report always/sometimes/never retrieval. VI is measured on equal seven-member input families where applicable; deterministic canonical-based methods may be invariant by construction and are judged primarily by recall.

Paired query bootstrap: 10,000 samples, seed 20260907. Primary contrasts: selected method–Original, selected method–hybrid, selected rule–Original, set mean–canonical (Recall@100 and cNDCG@10). Report all effects/CIs; if p-values are used, sign-randomization with 10,000 draws and Holm within each evaluation cell across the fixed contrasts. No non-inferiority or equivalence claim.

## Candidate cost and reranking

Original, hybrid and selected new method: all K above. Exact scoring with stable catalog-index ties; K unique products. Log index-vector bytes, number of document vectors, encoding seconds, query scoring/sorting timing excluding embedding separately from query-encoding cost. CPU exact scoring fixed four threads, warm-up then five repeats on first 50 query IDs, report median/p95; timing is local and not commercial-scale latency. Compare Original larger K as both recall and downstream cost control. Dense matrix scoring cost itself does not increase with K in this exact implementation; reranking cost does.

Canonical-input cross-encoder: pinned historical model, same 256-token joint budget, K=50/100/200, output top20; candidate-union caching avoids duplicate inference. Record end-to-end within-candidate cNDCG@10 definition and highest-label Recall@20. Keep historical varying-text reranker as a separate condition. No K>=500 reranking unless development recall gain from 200 to500 exceeds .02 absolute and the measured local cost permits it; otherwise mark not triggered.

## Diagnosis C

Use old ranks with source attribute count, duplicate count, tokenizer lengths/truncation, query token length, lexical overlap. Stratify original rank into 1–10,11–20,21–50,51–100,101–500,501+; compare highest and irrelevant labels only within shared strata, report support and query bootstrap. Include fully fitting and truncated subsets. Interpret as association, not an attention mechanism. No position-specific causal claim is planned, so query-aware matched-atom movement is conditional future work rather than required for the selected claims.

## Budget and stopping

Use the existing RTX4060 and caches, no paid resources. Cap new encoding/reranking at eight GPU-hours; checkpoint each artifact. Reserve priority for m2/4 selection, two-dataset/two-encoder transfer, canonical reranking and gain repair. m7/new-query confirmation may be explicitly pending only if the budget/access prevents completion. Do not turn missing work into a positive conclusion. All code, source hashes, timings, ranks and selection decisions remain inspectable.
