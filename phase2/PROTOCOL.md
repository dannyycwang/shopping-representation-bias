# Phase II frozen confirmatory protocol

Frozen on 2026-09-04 before reading any ESCI retrieval result. The machine-readable specification is `config/phase2.json`. Dataset counts may be added after deterministic preparation; representation definitions, model identities, seeds, primary metrics, and go/no-go thresholds must not be changed in response to ESCI outcomes.

## Scope and order of decisions

Phase II-A runs first. It tests the seven-member attribute-order family (`C0`, `C1`, and five deterministic `C2` permutations) as the primary fact-equivalent family. Sentence order, section order, and the two JSON orders are prespecified secondary controls. Phase II-B runs only if the primary Top-20 instability threshold is met by at least two retrievers on both datasets.

WANDS retains all Phase I products, queries, and cleaned labels. ESCI is held-out confirmation: use official US English `small_version == 1`, `split == test`, deterministically select 500 complete query groups by the SHA-256 rule in the JSON configuration, and rank against the union of every product judged for those queries. This creates a manageable retrieval catalog with deliberately incomplete judgments; unknown query-product pairs stay unknown. It is not the official per-query candidate-pool ranking task, and both pooled cNDCG and full-union known-label recall must be labeled accordingly.

The official ESCI Task 1 gain schedule is retained: E=1, C=0.1, S=0.01, I=0. E is the highest-relevance stratum, S is reported separately as substitute relevance, and C and I are never silently merged. WANDS grades remain Exact=3, Partial=1, Irrelevant=0.

## Primary endpoint and gate

For every judged highest-relevance query-product pair, `VI@K=1` when at least one primary-family serialization is inside Top-K and another is outside. Report product-pair micro VI and the mean of within-query VI proportions with a 10,000-replicate query bootstrap interval. Rank range, rank standard deviation, reciprocal-rank variance, and crossing at K in {10,20,50} are secondary endpoints.

A dataset-model cell meets the meaningful-instability gate when micro VI@20 is at least 3% and the query-macro 95% interval lower bound exceeds 1%. Phase II-A confirms generalization only if at least two of the three frozen models meet that gate on WANDS and at least two meet it on ESCI. No p-value threshold will replace this magnitude-and-uncertainty rule.

Relevance-stratified VI is compared within query where both strata exist. Paired sign-permutation tests use 10,000 draws. Multiplicity correction is Holm within each prespecified family of three retrievers. Exploratory query-type results do not change the Phase II-A gate.

## Models and position controls

The three frozen encoders are MiniLM-L6 with native attention-mask mean pooling and a 256-token limit, BGE-base-en-v1.5 with native CLS pooling and a 512-token limit, and GTE-ModernBERT-base with native CLS pooling and an 8,192-token limit. Model revisions are pinned in the JSON file. Base results use each model's native maximum length and native pooling, so GTE does not receive unnecessary chunking.

Position controls are prespecified for the attribute-order family: native encoding, mean-pooled 126-token chunks with 32-token overlap, mean-pooled 254-token chunks without overlap, mean-pooled 254-token chunks with 64-token overlap, and CLS-pooled 254-token chunks without overlap. Because the full factorial is computationally large, controls first run on WANDS for MiniLM and BGE; GTE native long-context output supplies the no-chunking architectural control. Any reduced control subset must be selected by the same query hash rule and labeled exploratory.

## Frozen mitigations

`M1_factual_field_sentence` states each available field in deterministic factual sentences and retains every source value without inference. `M2_normalized_attributes` emits explicit field labels and a canonical, case-insensitive attribute order while retaining duplicates and raw values. Neither method receives a query or label. These are fixed from Phase I R3 and R5 before ESCI evaluation.

If Phase II-B opens, mitigation success requires relevance to be maintained within its paired uncertainty interval and VI@20 to fall. Query-paired bootstrap and sign-permutation tests compare M1 and M2 with C0, with Holm correction across the two methods within each dataset-model family. Stability-adjusted relevance is reported for every lambda in {0, .1, .25, .5, 1}; no single lambda is selected after results.

Single-product analysis changes one judged relevant target at a time while all competitors remain C0, and contrasts that direct rank with the corresponding whole-catalog intervention. It cannot be interpreted as a merchant optimization recommendation.
