# Frozen optimization-robustness protocol

Frozen before extension test generation/evaluation on 2026-09-07. Existing Phase I–IV results and the development smoke outputs were already observed; this is a post-hoc extension, not a preregistration or pristine confirmation. The test sample, prompts, model, decoding, contrasts, fallback, and stopping rule below must not be changed after test ranks are read.

## Question and estimand

For each product, the seven existing C0/C1/C2s1–C2s5 inputs retain identical raw product facts and change only attribute order. `C` is raw-atom lexical sorting with no field labels or M2 framing. For each frozen optimizer `G`, evaluate Raw `E(x_s)`, Canonical `E(C(x_s))`, Optimize `E(G(x_s))`, and Canonical→Optimize `E(G(C(x_s)))`.

The main intervention is target-only. Every highest-label target is independently removed from its C0 position and reinserted with the counterfactual score while every competitor remains C0. The primary endpoint is query-macro target-intervention inclusion at K=100; K=20 is secondary. This is not a single catalog Recall. No full-catalog optimization is planned because generating all 42,994 WANDS products would exceed the local budget.

## Frozen sample

Within the existing WANDS held-out and historical ESCI query lists, retain queries with at least one highest label, sort IDs by SHA-256 of `phase5-sample:{dataset}:{query_id}`, and take the first eight per dataset. Include every Exact (WANDS) or E (ESCI) target for each selected query. Selection does not use old or new ranks, crossing, text length, or optimization outcome. Exact IDs, pair counts, unique products, and source hashes are written once to `phase5/config/sample.json` by `freeze.py`.

## Methods and generation

The two methods and source rationale are fixed in `METHOD_SOURCE_AUDIT.md`: E-GEO's hand-crafted `authoritative` prompt and released meta-optimized `technical` prompt. Both use the complete released prompt plus E-GEO's optimizer system prompt. They receive only the complete product text. They never receive a query, label, score, rank, competitor, or retrieval result.

Both are adapted baselines using `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, fp16, Transformers 4.55.4. Primary deployment generation is greedy (`do_sample=false`), max 512 new tokens, with an explicit recorded seed 20260907. Identical canonical inputs reuse one content-addressed output. The cache key includes actual input, system and user prompt, model ID/revision, all decoding settings, and replicate ID. Empty/error/length-truncated outputs use raw canonical text as the execution fallback and remain counted as generation failures.

## Generation-noise control

Select eight unique products by SHA-256 of `phase5-noise:{dataset}:{product_id}` across the frozen sample. For each optimizer, independently generate all seven raw orders and canonical input at temperature 0.7, top-p 0.9, max 512 new tokens, under seeds 20260981–20260983. Compute seven-order VI separately within each replicate. Report same-input C0 and canonical variation across replicates separately; never pool 21 raw generations into an order-effect family. Primary greedy results and stochastic controls are separate conditions.

## Fact verification and guard

Raw order equality is verified exactly by source-atom, character, token, and numeric multisets using the existing Phase II audits. Free-text rewrites are not called semantically equivalent. A frozen rule-based checker compares the complete raw inventory against each output: all source numbers and units, colors, negated attributes, and at least 80% of tokens in every nonempty structured value must remain; new numbers/units/colors are failures. It records omissions, additions, conflicts, empty output, and generation truncation. This conservative checker is incomplete and is not human ground truth.

Report (1) operational generated outputs with fact-failure rates and no claim that all changes are semantically irrelevant; (2) the common-valid subset where all seven outputs pass for a method, on identical query-target support; and (3) a prespecified guarded pipeline that replaces any failed output with raw canonical text. The guard never selects by rank. Source key conflicts are retained; the generator must retain both values or fail.

## Metrics and inference

For each method/model/dataset: per-order pair-micro and query-macro inclusion@20/100; seven-order mean; VI@20/100; always/sometimes/never; finite-family worst-observed robust coverage; schedule-min macro inclusion; rank spread; rescues/new misses/net relative to Raw on identical support; fact pass/fallback/truncation rates; source/output/retriever token lengths; generation and embedding time. Canonical and cached canonical→G have structural VI=0 and are evaluated primarily by inclusion.

Primary query-paired contrasts at K=100, fixed within each dataset/encoder cell: G1−Raw, C→G1−G1, C→G1−C, G2−Raw, C→G2−G2, C→G2−C. Use 10,000 query-cluster bootstrap samples, seed 20260907. If p-values are reported, use 10,000 paired sign-flips and Holm correction across these six contrasts per cell. Confidence intervals containing zero do not establish equivalence. Target-only cNDCG is undefined and is not reported.

## Transfer, cost, and stopping

Run WANDS and historical ESCI with frozen BGE; reuse exactly the same generated texts with MiniLM for encoder transfer. Existing hybrid, centroid/max, set-mean, canonical M2, candidate-budget, reranking, and gain-repair artifacts remain platform comparisons rather than merchant rewrites.

The extension may use at most eight GPU-hours after protocol freeze. Checkpoint every generation and embedding batch. Do not enlarge the sample or replace a prompt in response to test outcomes. If the complete frozen matrix reaches the cap, finish all BGE cells first and mark MiniLM transfer incomplete rather than changing the population. No paid API or automatic submission is authorized.
