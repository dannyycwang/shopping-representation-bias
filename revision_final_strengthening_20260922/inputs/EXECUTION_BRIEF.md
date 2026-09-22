# Final evidence strengthening before Chapter 6
## Execution brief

Repository: https://github.com/dannyycwang/shopping-representation-bias
Reviewed baseline: 29a699794746b4ecc916f69e75e7d05f737677f7 (merged PR #4, 22 September 2026).

Implement and RUN this bounded final evidence package, validate it, and return manuscript-ready evidence. Do not stop after proposing a plan or writing scripts. This is the last planned empirical supplement before the author and ChatGPT finalize Chapter 6. New forward passes and attribution are explicitly in scope. A new training study is not.

Use the current checkout, read applicable AGENTS.md instructions, and establish its relationship to the reviewed baseline. Preserve unrelated working changes and all historical outputs. If later commits already implement an item, audit and reuse them instead of repeating it. Put new outputs under revision_final_strengthening_20260922/ (choose a new non-overwriting suffix if that directory already belongs to another run).

Do not rewrite the author's manuscript in place. Prepare an evidence map, figure captions, and short result-based English insertion candidates for review. Preserve the three-RQ structure.

## 0. What is already finished

Read, rather than rerun:
- revision_graded_20260922/README.md
- revision_graded_20260922/GRADED_METRIC_AUDIT.md
- revision_graded_20260922/CLAIM_VERDICTS.md
- revision_graded_20260922/PROTOCOL.json
- revision_graded_20260922/tables/effectiveness_wide.csv
- revision_graded_20260922/tables/paired_contrasts.csv
- revision_graded_20260922/data/per_query.parquet
- revision_graded_20260922/data/rq2_membership_ndcg_per_query.parquet
- revision_completion_20260921/CLAIM_VERDICTS.md
- revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv
- revision_evidence_20260917/scripts/verify_target_rank_sources.py
- revision_evidence_20260917/figure_revision/data/figure1_case_audit.json

P0/P1 from the previous request are complete: 122 conditions, 49,227 query records, corrected label-derived graded metrics, and 12 canonical–BM25 hybrids. The optional permutation expansion was NOT run.

Carry forward these findings without trying to force a more favorable story:
1. ESCI direct gains are E=1, S=.1, C=.01, I=0. WANDS direct gains are Exact=3, Partial=1, Irrelevant=0; no exponential transform. Phase IV had already repaired some historical graded outputs. Do not claim that all historical reports or training were affected.
2. The gain correction leaves frozen ranks and highest-label inclusion unchanged.
3. Among queries with changed Top-20 highest-label membership, |delta nDCG20|>.01 occurs in 64.1–79.2% of WANDS cases and 81.1–96.2% of ESCI cases across the raw contrasts. The threshold is descriptive. A blanket "effectiveness barely changes" claim is unsupported.
4. All 12 canonical hybrids have positive uncorrected paired nDCG20 intervals versus BM25; nine do versus their own dense branch. Most contrasts versus the raw-hybrid seven-schedule mean remain inconclusive. WANDS/BGE ascending has a negative Recall20 contrast. No universal equivalence, superiority, or effectiveness–instability trade-off follows.
5. Canonical controls have VI=0 by construction under the permitted input transformations and deterministic retrieval. Present that as an expected property, not a newly discovered algorithmic achievement.
6. Full-ranking nDCG, condensed cNDCG, and JudgedCoverage must retain their distinct definitions and incomplete-judgment caveats.

Do not repeat the full nDCG, canonical hybrid, M=2/4/7, or model-training experiments. Do not silently promote any legacy Phase V/VI trained checkpoint into a matched comparison.

## 1. Freeze inputs and scope before new analysis

Primary populations remain:
- WANDS: 308 eligible queries, 21,299 Exact pairs, full 42,994-product catalog.
- ESCI: 499 eligible queries, 4,434 E pairs, fixed 10,076-product pooled catalog.

Use the exact support IDs, catalog order, saved query embeddings, model revisions, tokenizer limits, pooling, prefixes, dtype profiles, and deterministic tie rule in the audited packages. The historical query prefixes are empty; do not introduce an instruction prefix.

Useful existing sources include:
- phase2/src/representations.py
- phase2/src/encoding.py
- phase2/results/phase2_pair_ranks/
- phase3/results/target_only_permutations/
- revision_evidence_20260917/data/all_seven_token_lengths.csv
- revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv
- revision_graded_20260922/data/input_manifest.json
- revision_graded_20260922/data/persistent_states_per_query.parquet

Inventory exact cache payloads and hashes. Some full embeddings/scores are ignored by Git but were available in the author's execution environment. Restore the audited payloads if needed. Do not regenerate missing historical arrays under a different revision and call them the original experiment.

Write PROTOCOL.json and its hash before new outcomes, including selection rules, seeds, precision checks, attribution baselines, schedule families, and compute estimates. This is a frozen analysis plan for a follow-up to already observed results, not a prospective preregistration of the whole study.

Maintain four separately reported workstreams:
A. Existing-score boundary analysis.
B. Query-conditioned XAI cases.
C. Product-state transitions under fixed controls.
D. Bounded permutation coverage.

## A. Score variation and the Top-K boundary

Purpose: connect order changes to candidate inclusion through actual scores and competitors. The algebra connecting margins to ranks is a validation identity, not a novel empirical discovery.

### A1. Reconstruct the actual target-only setting

Use all existing eligible highest-label pairs, both datasets, all three frozen encoders, original seven schedules, K=20 and K=100. This stage should require no new encoding.

For each query q and target p:
- Fix every competitor at its raw C0 representation.
- REMOVE the old target entry before evaluating its alternative representation. Catalog size after replacement must remain unchanged.
- Let t_K(q,p) be the Kth highest competitor score among u != p.
- Let m_s(q,p)=z(q,x_s(p))-t_K(q,p).
- Resolve equality using the original catalog-index tie rule. Margin sign alone is insufficient at exact score ties.
- Reconstruct rank with the audited replacement function and compare with authoritative target-only ranks.

Use the historical full C0 competitor scores and saved target scores as their audits prescribe. Do not mix catalog-wide target ranks with target-only thresholds. Do not treat separate target-only counterfactual indexes as one coherent query ranking.

There are already documented floating-point reconstruction differences. Preserve authoritative saved ranks, record discrepancy IDs and their score/tie context, and flag unresolved boundary records. Do not widen a numerical tolerance until contradictions disappear. Use exact comparisons for the saved FP32 ranking protocol; any epsilon used to describe numerical uncertainty is a separate diagnostic.

Export pair/schedule records including:
dataset, model, query_id, product_id, schedule, K, original catalog index, score, competitor threshold score/ID/index, signed margin, reconstructed rank, saved rank, inclusion, token length, fully-fitting status, precision flags.

### A2. Summarize quantities that add empirical information

For each pair export:
- source-order margin and inclusion;
- min/max score, score range, max absolute deviation from C0;
- best/worst rank and number of included schedules;
- persistent inclusion / crossing / persistent omission;
- maximum embedding cosine distance from C0 only where compatible saved embeddings exist.

Report distributions by state on the full and all-seven-fitting supports separately. Query-macro weighting is primary; pair-micro is secondary. For pair distributions, explicitly define whether queries are weighted equally before pairs, rather than accidentally allowing queries with many judgments to dominate.

Use query-cluster bootstrap intervals for population summaries where useful (10,000 draws, seed 2026091701). At most one supplementary distribution figure is needed. Do not claim that large rank changes necessarily imply large embedding changes, or that absence of truncation identifies a unique internal mechanism.

Keep the seven-schedule population results intact. Later 32-schedule fitting filters must not retroactively change this support.

### A3. Existing illustrative cases

Audit and reuse the following as explicitly post-hoc illustrations:
- WANDS / MiniLM / query 359 "french molding" / product 12575: historical all-seven-fitting case, ranks 15–174, 139 tokens. Source: revision_evidence_20260917/data/posthoc_fully_fitting_illustrative_case.json.
- WANDS / GTE / query 409 "teal chair" / product 24318: audited target-only ranks 11–939, 1,059 tokens including special tokens, below the 8,192 cap. Source: revision_evidence_20260917/figure_revision/data/figure1_case_audit.json.

Confirm every number from the pinned artifacts. Do not substitute an older turquoise-chair case or silently change the Introduction figure. The GTE case can use saved scores for a boundary illustration; full GTE attribution is not required.

## B. Query-conditioned XAI, with controls

Purpose: describe whether attribution to product content changes along with the retrieval score. Do not assume in advance that attribution redistribution will be large or consistent.

### B1. Fix a small case panel before computing attribution

Use WANDS MiniLM and BGE, fully fitting under all seven original schedules. Select 24 diagnostic pairs: per model, four crossing, four persistently included, and four persistently omitted at K=20.

Use deterministic hash ordering with seed 2026092201 within each stratum, choosing distinct queries within a stratum where possible. Preserve a manifest of all candidates, selections, and any quota shortfall. This stratified diagnostic panel is not a prevalence sample. Keep unsuccessful attribution cases in the audit; do not replace them because their heatmaps look unconvincing.

For every selected pair, explain its minimum- and maximum-score original schedules, breaking ties by the original schedule order. These are explicitly extrema among tested orders, not two random orders.

Additionally include the historical MiniLM "french molding" pair as a post-hoc illustrative case if it is not already selected. This makes at most 25 pairs / 50 input variants. Do not expand attribution to the full catalog.

Choose the display cases by a written rule before seeing heatmaps:
- one BGE crossing pair closest to the median rank span in the selected BGE crossing stratum;
- one BGE persistently included control closest in token length and attribute count to that pair (use a frozen deterministic distance and tie rule);
- the historical MiniLM case, in the supplement if main-text space is limited.

### B2. Attribute the actual retrieval score

The product encoder in this bi-encoder does not receive q. Raw product self-attention is not a query-conditioned explanation.

Use a differentiable scalar F_q(x)=dot(cached normalized query embedding, normalized product embedding). Replicate native pooling and normalization: MiniLM uses attention-mask mean pooling; BGE uses CLS. Keep the query vector fixed and only attribute the product input.

The existing encode path uses torch.inference_mode(). Do not try to backpropagate through its detached NumPy outputs. Implement a separate, verified gradient-enabled forward adapter with model.eval(), unchanged weights/tokenizer, and the same pooling/normalization.

Before attribution:
- Check input_ids versus inputs_embeds forward agreement.
- Compare adapted forward scores with cached/native scores; report actual errors and any changed inclusion decisions.
- If FP32 is needed for stable gradients, disclose it as a numerical diagnostic of the frozen encoder. Keep native retrieval outcomes authoritative.
- For a displayed boundary explanation, the diagnostic forward must preserve both native inclusion decisions, and each absolute native margin must exceed five times the observed diagnostic/native score discrepancy. If this fails, mark the case numerically ambiguous and show its native score/rank evidence without interpreting the attribution as an explanation of that crossing.
- Do not secretly reselect more favorable cases after this check.

Compute Integrated Gradients over the word-embedding inputs:
- Primary baseline: zero word-embedding vectors for nonspecial, nonpadding tokens.
- Sensitivity baseline: the tokenizer PAD embedding substituted at the same nonspecial positions.
- Preserve sequence length, position IDs, token-type IDs, attention mask, special tokens and padding in each comparison. The PAD-vector baseline is a mathematical reference, not a natural product description.
- Include title/description/fixed context in attribution, not only reordered attributes.
- Start at 64 integration steps, increase to 128 and then at most 256 if completeness fails.
- Predefine completeness acceptance as absolute residual <= max(1e-4, .01*abs(F(input)-F(baseline))). Retain failures and baseline-specific scores; do not suppress them.
- Completeness applies separately to each input/baseline. If baseline scores differ across serializations, attribution-sum differences alone are not the full score difference.

Map tokens to original attribute occurrences using serializer spans and tokenizer offsets. Preserve duplicate entries via occurrence IDs. Track boundary/separator/special tokens separately so the attribution sum is accounted for. Do not add new field labels or change the input to make the explanation easier.

Aggregate signed token attributions by entry; also retain absolute magnitudes. Align plots by entry identity, not token position. Use a common signed color scale for the two orders. Do not normalize each order independently and then interpret colors as comparable absolute contribution.

Compare the two baselines and report which qualitative patterns persist. Label any cross-case association as exploratory; do not report population prevalence from the stratified 24-pair panel.

Raw attention or attention rollout may be an optional appendix visualization only if it is already easy to extract. Do not make it a required deliverable, select heads/layers because their patterns look persuasive, or describe raw self-attention as the model attending to the query.

### B3. Required case figure

Create one compact main-text figure with a crossing case and its stable control, each connecting:
1. query and the same attribute entries in two input orders;
2. aligned, query-conditioned entry attribution;
3. all seven native target scores relative to the fixed competitor threshold, with native ranks/inclusion.

Use zero on the margin axis as the boundary; the x axis denotes schedules, not a time trend. Put the two attributed orders in context among all seven schedules. Include token counts/cap and the fact that competitors are fixed.

Put the third illustrative case and extra baseline/QA detail in the supplement. For long records, display a reproducibly selected set of entries and retain an "other content" total; export complete values. Do not choose entries by largest change after inspecting them.

Export PDF, SVG and PNG, plot source, source data, and concise captions. Inspect the rendered figure at manuscript scale. If attribution is unreliable, retain the native margin panels and report an inconclusive XAI result. A negative diagnostic is an acceptable scientific outcome.

Primary method reference: https://proceedings.mlr.press/v70/sundararajan17a.html
Attention interpretation context: https://aclanthology.org/N19-1357/ and https://aclanthology.org/D19-1002/

## C. Which products become stable under canonical controls?

This is a zero-encoding analysis of existing product identities. It addresses the limitation that VI=0 alone does not reveal whom a fixed policy includes or excludes.

For K=20 and K=100, both datasets, MiniLM/BGE, all three canonical rules:
1. Compare the raw dense seven-schedule family to its corresponding pure canonical dense index.
2. Compare the raw hybrid seven-schedule family to its corresponding canonical–BM25 index.

For each comparison construct a 3 x 2 transition table:
- Rows: reference persistent inclusion, reference crossing, reference persistent omission.
- Columns: fixed-control included, fixed-control omitted.

Recover identities from the existing per-query highest_support_ids and included_ids_K* fields, not by subtracting aggregate rates. Reuse revision_graded_20260922/data/per_query.parquet and the saved conditions.

Report all six cells as fractions of Hq, query-macro primary and pair-micro secondary, with paired query-cluster intervals. Separately report conditional inclusion within each reference state, with explicit denominators; undefined groups are NA, not zero. If using ratios of query-macro cell masses, name that weighting rather than calling it the mean per-query conditional rate.

The table should make visible:
- previously always-in products lost by the fixed control;
- previously always-out products newly included;
- which formerly crossing products become included versus omitted.

Check row totals against the reference state masses and column totals against fixed-control Recall. These are consistency checks, not findings. Do not claim that all previously unstable products were "rescued" when some became consistently omitted.

Deliver all rules; no test-set winner selection. A compact table or supplementary figure is enough. This does not require another family of mitigation methods.

## D. Bounded finite-permutation coverage

This finishes the previously deferred concern at a controlled cost. It is a TARGET-ONLY supplement, not a rerun of the full catalog-wide experiment.

### D1. WANDS: fixed 128-query supplement, two encoders

Before examining additional-order outcomes, select 128 of the 308 eligible query IDs by sorting SHA-256("coverage-20260922|" + decimal_query_id), with numeric query_id as a tie-breaker. Reuse exactly these queries for MiniLM and BGE, retain all their highest-label targets, and freeze the full C0 competitor catalog.

Use nested families of 7, 16 and 32 schedules:
- The first seven must be the exact existing C0, C1 and C2s1–C2s5 schedules.
- Add nine schedules using the existing product-ID-seeded _random_order function with seeds 2026092201 through 2026092209.
- Add sixteen more with seeds 2026092210 through 2026092225.
- Do not reseed until a result looks favorable. Keep duplicate views in schedule-weighted frequencies, but count and disclose distinct serialized strings separately.

Encode only new distinct target strings, once per frozen model profile, and reuse them across queries sharing a product. Reuse cached old strings where compatible. Do not re-encode the whole competitor catalog or introduce additional GTE runs.

Before new scoring, audit all 32 serialized inputs/token lengths. For the fitting curve, define a fixed per-model pair support that fits under ALL 32 schedules. Use that same support and its fixed nonempty query set at 7/16/32. The full-target support is a separate curve. Report both support sizes and do not compare a changing fitting population.

For each family and K=20/100 report persistent inclusion, VI, persistent omission, and tested-order inclusion frequency. The last statistic is an average over separate target-only counterfactuals; do not label it as Recall/nDCG of one coherent catalog ranking.

Report the incremental crossing mass 7->16 and 16->32, plus the original-7 states of newly crossing pairs. Bootstrap queries within this 128-query supplement. Clearly distinguish these results from the primary 308-query estimates.

VI must be nondecreasing and persistent inclusion/omission nonincreasing as schedules are added on a fixed support. Verify this; the monotonicity is structural. A small last increment does not prove all-permutation invariance or convergence.

### D2. ESCI: enumerate the actual small permutation family

For all 499 eligible queries and their E targets, MiniLM/BGE, enumerate every DISTINCT permutation of whole nonempty entries under the existing representation. The current schema has at most three entries, hence at most six distinct orders per product; audit this rather than splitting bullet text into artificial new entries.

Preserve duplicate multiplicities and fixed blocks. Reuse exact existing text embeddings when possible; encode only missing distinct texts. Keep the full raw C0 competitor catalog fixed.

Compare original-seven observed states with exhaustive PER-TARGET states at K=20/100. Show how many distinct orders the original seven covered per target and the additional crossing mass. Use a common all-permutation-fitting support for the fitting comparison.

Do not claim to exhaust joint catalog-wide combinations. If all distinct target orders were already covered, report a valid zero additional mass. Do not manufacture extra distinct views by changing separators, labels or content.

### D3. Compute boundaries and unavailable resources

Estimate distinct new texts, token workload, and runtime before running. Cache by text hash plus full encoder profile and make the run resumable. There is no new training, hyperparameter search, new dataset, or production ANN benchmark.

Use the author's available execution environment. If exact payloads or model access are unavailable, complete all independent work, record the precise missing artifacts and blocked cells, and supply a resume command. Do not mark unexecuted coverage as completed, replace it with synthetic numbers, or enlarge a reduced run to the full support in the prose.

## 2. Statistics, scientific claims, and figure budget

- Keep source C0 and seven-schedule means distinct.
- Preserve all three canonical rules; display selection must not become method selection.
- Use 0–1 metric units in machine-readable tables. Explicitly convert to percentage points in prose; .00624 is .624 pp.
- Use query-cluster resampling, keep schedules/methods for each query together, and disclose uncorrected intervals. A zero-spanning interval is not equivalence.
- Keep full-ranking nDCG, condensed cNDCG and judged coverage together in the evidence map. Do not turn incomplete judged labels into claims about all unjudged products.
- Distinguish empirical finite-schedule stability, structural input invariance, and per-target exhaustive coverage.
- Attribution is descriptive evidence about a selected scalar and baseline. It does not establish the unique causal mechanism, human-like attention, seller harm, consumer behavior or fairness.
- The competing PI-FT paper reports a different training setup. Do not claim its VI is nonzero or that these frozen models outperform it without a matched experiment. No PI-FT retraining is requested in this final package.
- Restrict main-text additions to at most one case figure and one compact table/extension of an existing table. Coverage and detailed XAI QA belong in the supplement unless they materially change a main conclusion.

## 3. Deliverables

Produce:
1. README.md with what ran, what was reused, and what remains blocked.
2. PROTOCOL.json plus hash, input/output manifests, exact support and selection IDs.
3. BOUNDARY_AUDIT.md, per-pair/schedule margin records, population summaries.
4. XAI_CASE_MANIFEST.csv, token/entry attribution data, forward/precision/completeness/baseline QA and failure records.
5. State-transition tables and per-query identities/records for all specified controls.
6. PERMUTATION_COVERAGE.md, nested/exhaustive support manifests, coverage tables and plots.
7. Figures in PDF/SVG/PNG with reproducible plotting scripts and source data.
8. CLAIM_VERDICTS.md with supported / bounded / inconclusive / unsupported / not evaluated labels, each tied to exact table cells or cases.
9. CHAPTER6_EVIDENCE_MAP.md:
   - 6.1 Attribute-Order Sensitivity: original population findings, fixed-competitor/fitting evidence, margin cases; coverage bounds.
   - 6.2 Aggregate Effectiveness and Product Membership: retain Recall cancellation and corrected nDCG results, including substantial query-level changes.
   - 6.3 Effectiveness and Consistent Inclusion under Controls: structural zeros, graded effects, product-state transitions and costs.
   Include a recommended figure/table placement, concise English captions, result-based insertion candidates, and claims the author should remove or qualify.
10. REPRODUCE.md with tested stage commands and a resume command. Optional stages cannot silently count as passes.

The English insertions should be concise, use inline mathematics where readable, and avoid LaTeX paragraph commands. Do not invent final numbers before execution or rewrite Chapters 1–5 wholesale.

## 4. Completion gate and stopping rule

The execution is complete when:
- Existing ranks/supports remain traceable and all new comparison mismatches are explained.
- Boundary analysis covers the stated existing supports.
- All selected attribution cases are accounted for, including inconclusive or failed cases.
- Both control-transition families are exported with identity-based checks.
- The bounded coverage tasks have run, or genuine resource blocks are precisely identified.
- Every displayed number maps to reproducible data; captions identify intervention, support and cutoff.
- No outcome-dependent method/seed/case replacement was used.
- The evidence map supports drafting Chapter 6 without another open-ended experiment proposal.

An inconclusive XAI result or zero added permutation effect is acceptable. Do not keep searching for a positive result. Do not add new methods or training after these gates.

Commit only the new evidence package and necessary scoped code on a dedicated branch. Follow the session's existing authorization for pushing that branch; do not merge into the default branch. Report the branch/commit, completion status, and three to five findings for the author.

