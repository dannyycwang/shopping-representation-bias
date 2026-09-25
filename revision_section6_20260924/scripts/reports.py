"""Evidence-based audit reports and a scoped Section 6 editorial replacement."""
from common import *

def main():
    ledger=pd.read_csv(HERE/'section6_claim_ledger.csv');mem=pd.read_csv(HERE/'tables/membership_all_contrasts.csv');delta=pd.read_csv(HERE/'tables/controls_paired_comparisons.csv');ctl=pd.read_csv(HERE/'tables/controls_complete.csv');case=read(HERE/'data/illustrative_case_selection.json')
    harmonized=(HERE/'qa/harmonized_validation.json').exists()
    def write(name,text):
        p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf8')
    def effect(ds,m,method,k,metric,comparator,scale=100,dp=3):
        r=delta[delta.dataset.eq(ds)&delta.model.eq(m)&delta.method.eq(method)&delta.K.eq(k)&delta.metric.eq(metric)&delta.comparator.eq(comparator)]
        if method in ['raw','raw_hybrid']:r=r[r.schedule.eq('seven_mean')]
        r=r.iloc[0];return f'{r.estimate*scale:+.{dp}f} [{r.ci_low*scale:.{dp}f}, {r.ci_high*scale:.{dp}f}]'
    hybrid=[]
    for ds,m in [('wands','minilm'),('wands','bge_base'),('esci','minilm'),('esci','bge_base')]:hybrid.append(f'| {ds.upper()} / {m} | {effect(ds,m,"raw_hybrid",100,"Recall","raw:seven_mean")} | {effect(ds,m,"raw_hybrid",100,"VI","raw:seven_mean")} |')
    wb=mem[mem.dataset.eq('wands')&mem.model.eq('bge_base')&mem.method.eq('raw')&mem.K.eq(20)]
    reverse=wb[wb.schedule.eq('C1')].iloc[0]
    if harmonized:
        hs=pd.read_csv(HERE/'tables/harmonized_summary.csv');hci=pd.read_csv(HERE/'tables/harmonized_increments.csv');hc=pd.read_csv(HERE/'tables/historical_to_harmonized.csv')
        lines=[]
        for r in hs[hs.metric.eq('VI')&hs.K.eq(20)].itertuples():lines.append(f'| {r.dataset} | {r.model} | {r.support} | {r.family} | {r.queries}/{r.pairs} | {r.estimate*100:.5f} [{r.ci_low*100:.5f}, {r.ci_high*100:.5f}] |')
        comparison=[]
        for r in hc[hc.metric.eq('VI')].itertuples():comparison.append(f'| {r.dataset} | {r.model} | {r.support} | {r.K} | {r.estimate*100:+.6f} [{r.ci_low*100:.6f}, {r.ci_high*100:.6f}] |')
        execution=pd.read_csv(HERE/'tables/harmonized_execution_differences.csv');execution_lines=[];repeat_lines=[]
        for r in execution.itertuples():execution_lines.append(f'| {r.dataset} | {r.model} | {r.K} | {r.changed_cutoff_decisions}/{r.original7_pair_schedule_records} | {r.queries_with_changed_cutoff_decision} | {r.target_pairs_with_changed_cutoff_decision} |')
        for m in ['minilm','bge_base']:
            probes=read(HERE/f'harmonized/{m}/repeated_input_checks.json')
            repeat_lines.append(f'{m}: {len(probes)} focused checks, {sum(r["bitwise_identical"] for r in probes)} vectors with exactly equal numeric components, maximum component difference {max(r["max_abs_vector_error"] for r in probes):.17g}; {sum(bool(r.get("decision20_changed",False) or r.get("decision100_changed",False)) for r in probes)} changed repeated cutoff decisions.')
        expansion_status='PASS: harmonized supplement completed separately; historical mixed-execution estimates remain historical.'
        expansion_sentence='On the frozen 128-query WANDS supplement, expanding the harmonized family from seven to 32 schedules increases target-only VI@20 by '
        vals=[]
        for m in ['minilm','bge_base']:
            r=hci[hci.dataset.eq('wands')&hci.model.eq(m)&hci.support.eq('full')&hci.K.eq(20)&hci.metric.eq('VI')&hci.contrast.eq('32 minus 7')].iloc[0]
            e,l,h=r.estimate,r.ci_low,r.ci_high
            vals.append(f'{100*e:.2f} pp (95\\% CI [{100*l:.2f}, {100*h:.2f}]) for '+('MiniLM' if m=='minilm' else 'BGE'))
        expansion_sentence += ' and '.join(vals)+', with the full execution comparison and common fitting supports in the supplement.'
        write('harmonized_supplement.md',f'''# Harmonized expansion supplement

{expansion_status}

The WANDS cohort is the existing 128 query IDs and all 7,125 highest-label pairs (6,444 unique products): the original selection keeps the first 128 eligible IDs sorted by SHA256(`coverage-20260922|decimal query ID`), breaking ties numerically. The exact frozen IDs and pairs are reused. ESCI uses the existing 499 queries and 4,434 E pairs. No cohort, schedule, seed or product was selected using the new outcomes. Each encoder uses its own single pinned profile for every query, source-order competitor, original-seven target and extra order. MiniLM and BGE profiles are in `harmonized/<model>/profile.json`; every alias and vector is addressable in the adjacent manifests. This is a follow-up, not a new preregistration.

| Dataset | Encoder | Support | Family | Queries/pairs | VI@20 % [95% CI] |
|---|---|---|---|---|---|
{chr(10).join(lines)}

The all-32-fitting WANDS supports are held fixed across 7/16/32. The ESCI fitting support fits all distinct target orders. `tables/harmonized_increments.csv` reports paired 16-minus-7, 32-minus-16 and 32-minus-7 increments, and exhaustive-minus-7 for ESCI, at K20 and K100. Each estimate has the same complete per-query support as its comparator. `data/harmonized_added_crossing_identities.parquet` retains the affected identities, with each contrast labelled separately. Nested VI monotonicity follows structurally; the amounts and identities are empirical. There is no saturation claim.

| Dataset | Encoder | Support | K | Harmonized7 minus historical7 VI, pp [95% CI] |
|---|---|---|---|---|
{chr(10).join(comparison)}

`tables/historical_to_harmonized.csv` also compares persistent inclusion and omission. Exact per-schedule rank and cutoff differences are in `harmonized/<model>/original7_rank_differences.csv`; they are not silently attributed to order. Focused repeated-input and changed-decision forward checks are in each model directory. No score tolerance changes ranking or membership.

| Dataset | Encoder | K | Changed decisions / pair-schedule records | Affected queries | Affected target pairs |
|---|---|---|---|---|---|
{chr(10).join(execution_lines)}

These counts compare historical and harmonized execution of the **same** seven orders. They are record counts, not query-macro rates; `tables/harmonized_execution_differences.csv` also includes exact-rank changes. Additional-order increments use harmonized arrays throughout.

{chr(10).join(repeat_lines)}

These are focused sampled repeat checks, not an assertion that every possible batch shape or hardware environment produces identical output. The tested repeated forwards keep the documented batch shape, padding bucket and numerical settings. The recorded `bitwise_identical` field is defined by `numpy.array_equal`: it checks exact numeric component equality, without a tolerance, but does not distinguish signed zeros. Byte integrity of saved vector payloads is checked separately by SHA256; no stronger byte-level repeat claim is inferred from that field name.

Scoring consistently uses FP32 elementwise products and a fixed-axis FP32 sum for every source and target, followed by descending score and ascending catalog index. A target's original C0 entry is removed. The source and repeated target input resolve to the same cached vector. New source competitors and queries were encoded under the same profile; no historical vector is mixed into the harmonized family. This is a separately versioned supplement, not a replacement of the primary historical seven-schedule results.

ESCI exhaustive means every distinct order of one target with all competitors fixed, not all joint catalog configurations. Its exhaustive state calculation uses the distinct E schedules; duplicate original-seven observations do not create extra distinct orders. This package reports inclusion states, not a synthetic coherent-catalog Recall for target-only counterfactuals.

The MiniLM process survived the reported computer closure and resumed after a long pause. Wall time and the affected block's elapsed time include the pause, so these timings are execution records, not a latency benchmark. Historical cost tables remain the cost evidence.
''')
    else:
        expansion_status='UNRESOLVED while the separately harmonized runs are still executing; do not use the historical expansion as a strong order-only conclusion.'
        expansion_sentence='The expanded-order experiment is retained as a historical supplement pending harmonized execution; its mixed-runtime increase is not used here as an order-only conclusion.'
    write('section6_scope_and_provenance.md',f'''# Section 6 scope and provenance

Audit of the local 17-page manuscript version (11), titled *Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval*. Audited source revision: `{read(HERE/'qa/starting_state.json')['revision']}`. The directory keeps the request date, 20260924; actual execution continued on 2026-09-25. Starting branch, dirty-file hashes and runtime are in `qa/starting_state.json` and `qa/environment.json`. The main paper checkout contains older results and pre-existing figure/introduction edits; those are preserved.

The local `Downloads/chapter6_threepage_package.zip` contains a matching Section 6 component. Its exact component bytes and ZIP hash are recorded under `inputs/reference_section6/`. Its prose matches v11; the duplicate table and paragraph occur in the integrated PDF, not this component. No full LaTeX checkout matching v11 was found, so final full-paper pagination remains an integration check. The original PDF hash and all inherited evidence/source hashes are in `data/source_inventory.csv`.

## Populations and development history

WANDS retains all 42,994 catalog products. `phase4/scripts/freeze.py` reuses the Phase III ordering of the 480 query IDs by SHA256(`phase3-dev:{{query_id}}`): the first 96 are development, the remaining 384 held out. The primary 308 are exactly those held-out queries with at least one Exact judgment after contradictory pairs are excluded and duplicate identical labels collapsed; they contribute 21,299 distinct Exact pairs. `data/wands_sampling_audit.csv` records every query's development/eligibility status. The development split is a query split; catalog products are shared.

The historical appearance-priority strategy was selected on development macro highest-label Recall@100, then cNDCG@10 and fixed rule ID, as described in `phase4/config/EXPERIMENT_PROTOCOL.frozen.md`. It is distinct from the three pure canonical rules retained here. Ascending, descending and type priority are all reported; none is selected as a test winner. Later follow-up decisions and this presentation revision occurred after earlier outcomes had been inspected. “Held out from the historical development selection” must not become “blind independent confirmation.” Independent view seeds prevent direct reuse of evaluation schedules, but do not undo development feedback or prior test inspection.

ESCI filters the pinned dataset to US locale, `small_version == 1`, and test split. Sort eligible query IDs by SHA256(`20260904:{{query_id}}`) and keep the first 500. Retain all judgments for these queries and the union of their judged product IDs, sorted lexically, producing 10,076 products. Of the 500 queries, 499 have at least one E label, contributing 4,434 E pairs. This is the existing evaluation sample, not a newly sampled blind test. `data/esci_sampling_audit.csv` and the frozen protocol give exact IDs; P0 rechecked the sample against the raw examples parquet.

## Serialization, native support and ranks

Seven original schedules are C0/source, C1/reverse and five product-dependent shuffles, seeds 20260911–20260915. WANDS retains each full pipe-separated entry; ESCI retains nonempty brand, color and the whole bullet-point field (at most three entries). Every duplicate occurrence, non-attribute field, section placement and label is preserved. `qa/original_serialized_strings.csv` verifies the actual saved strings against the original serializer, rather than merely assuming losslessness from equal lengths.

MiniLM/BGE/GTE revisions and model limits are in `phase2/config/phase2.json`; native caps are 256/512/8192 including special tokens. Query and product prefixes are empty. MiniLM uses attention-mask mean pooling, BGE/GTE CLS; embeddings are unit normalized, with historical FP16 model and FP32 pooling/scoring. `data/historical_execution_matrix.csv` records the exact query/product arrays, sidecar profiles, hashes and any unavailable per-array library metadata. `revision_evidence_20260917/data/encoder_profiles_and_rank_sources.csv` is the inherited array-to-rank provenance reconciliation. Saved catalog and query axis mappings are in `revision_graded_20260922/data/*_ordering.csv`, and their order was checked against processed records.

Rank ties use descending exact FP32 score then ascending stable catalog index, independent of schedule. Target-only ranking replaces one entry and removes its old source representation; it never leaves an extra competitor copy. The original tied-score algorithm passed 11,600 historical and 12,700 fresh explicit-replacement checks. One inherited WANDS/BGE deep-rank discrepancy remains: query 252/product 385/Shuffle 3 has saved rank 479 versus reconstructed 480, with unchanged K20/K100 decisions. No tolerance was introduced to hide it.

`data/method_rank_provenance.csv` covers all 122 saved conditions: 82 from the original condition list and 40 later raw/canonical-hybrid conditions recorded by the authoritative per-query companion. Every rank payload matches its historical input or output manifest hash. Available embedding sidecars are linked; missing per-array library versions are explicit. Matching query support and verified file identity do not certify identical historical numerical execution across every method. Only the new expansion supplement claims its separately pinned uniform execution profile.

Fully-fitting support tests the complete native input under all seven schedules, including special tokens and prefixes. It filters evaluated targets, not competitors, and then drops queries with no eligible target. WANDS MiniLM: 118 queries/925 pairs; BGE: 239/6,797; GTE: 308/21,299. ESCI MiniLM: 460/2,578; BGE: 488/3,587; GTE: 499/4,434. GTE full and fitting populations coincide. Full-versus-fitting contrasts are descriptive population changes, not causal truncation effects or architecture comparisons.

## Metric and inference conventions

Every macro inclusion rate first divides by the query's eligible highest-label support and then gives queries equal weight. Pair-micro quantities are separate inherited outputs. Persistent inclusion + crossing (VI) + persistent omission equals one. Fixed controls are invariant by construction: their persistent inclusion equals Recall, omission equals one minus Recall, and VI is structurally zero. Source-order effectiveness and the raw seven-schedule mean are different references. In `controls_complete.csv`, `schedule` selects effectiveness; state metrics always refer to the method's complete seven-schedule family (or its fixed index), including rows whose effectiveness reference is C0.

Fixed-view invariance has a concrete implementation basis: `phase4/scripts/common.py::attrs` first sorts whole attributes by `(casefold(entry), entry)`, then applies the fixed product-ID-seeded shuffle. The resulting view family does not depend on incoming attribute order. This is not a claim that arbitrary finite random-view averaging is invariant. Set-Mean pools independently encoded atoms; the lexical branch uses position-independent term counts. These properties explain zero VI separately from the measured effectiveness.

Graded gains are used directly, without exponentiation: WANDS Exact/Partial/Irrelevant = 3/1/0, ESCI E/S/C/I = 1/0.1/0.01/0. nDCG discounts by 1/log2(rank+1) at original catalog ranks. Unjudged items contribute zero *for the numerical DCG calculation*, but are not asserted irrelevant. IDCG orders all available judged gains and takes the first K. Condensed cNDCG removes unjudged items before assigning discount positions; its IDCG uses the same judged support. JudgedCoverage is judged count in the original top K divided by K. Neither convention completes the judgments. Historical Phase II ESCI S/C grades were swapped; the corrected label-derived evaluation is reused, with all 49,227 original records checked and K100 graded measures newly computed from the same saved ranks. No ranks or highest-label support were changed to repair gains.

Intervals use 10,000 paired query-cluster percentile resamples, seed 2026091701, on sorted eligible query IDs. Each draw retains every product, condition and schedule within its query. They quantify query-sampling uncertainty conditional on these fixed catalogs/models, not training or annotation uncertainty. No equivalence/noninferiority margin is retrofitted. The revised prose removes counts of “significant” outcomes as an inferential basis; no adjusted tests are inferred from CI endpoints. Individual unadjusted intervals remain explicitly named and do not establish global equivalence.

## Historical versus harmonized expansion

{expansion_status}

Historical extension records combine cached original-seven representations/queries/C0 competitors with new target encodings under Transformers 4.46.2; prior records document 4.55.4 in historical execution. The BGE extension changed its maximum batch from 48 to 24 after 51,200 WANDS inputs. Equal model and tokenizer revisions alone do not establish identical numerical execution. Exact inherited profiles and the adjustment remain in the earlier manifests; unavailable per-array library fields are not guessed.

`harmonized/<model>/profile.json` pins the new supplement's single execution recipe, software, hardware, pooling, normalization, precision, empty prefixes, eager attention, fixed padding buckets and fixed batch size. Query vectors, the full source catalog and all target orders are newly encoded with that recipe. Identical complete native input IDs/masks resolve to one vector; full-text hashes are retained separately. All original schedules and the frozen 128 WANDS IDs/7,125 pairs are reused. This is the only new inference authorized and performed by this revision. New model training, new data, PI-FT, architecture sweeps and attribution repair are outside scope.

## Artifact authority

`data/effectiveness_per_query.parquet` is the authoritative graded/schedule companion; `data/inclusion_per_query.parquet` covers both interventions, both supports, every encoder/dataset and K20/K100; `data/membership_per_query.parquet` preserves integer sets and all six contrasts at each cutoff; `data/controls_per_query.parquet` and `data/transitions_per_query.parquet` supply method and transition estimates. Full-precision CSV equivalents and plots' exact selectors are included. `tables/unavailable_method_cells.csv` marks unevaluated methods rather than borrowing another encoder's results. Historical byte hashes and pre-existing edits remain unchanged. The user separately authorized GitHub delivery on 2026-09-25; that delivery is scoped to this revision directory. No manuscript submission or external message was performed.
''')
    write('section6_audit.md',f'''# Section 6 audit

**PASS** for the recomputed saved evidence and accounting checks. **NEEDS_CORRECTION** for v11 presentation and expansion provenance wording. **UNRESOLVED**: final integrated manuscript pagination/references until the full matching LaTeX is available. {expansion_status}

## Completed / reused / recomputed / unresolved

- Completed P0: support selection, stable catalog axes, complete-entry permutations, unchanged labels, complete-input fitting support and single-target replacement tie checks.
- Reused: historical seven-schedule ranks, all encoder/tokenizer pins, all three canonical rules, BM25, raw/canonical hybrids, Set-Mean, all centroid/multi-vector budgets and cost records; prior validation and numerical failure records retained.
- Recomputed: both-intervention full/fitting states at K20/K100; all 49,227 graded records; additional nDCG/cNDCG/Coverage@100 from saved judged ranks; 48,420 query/contrast/cutoff membership records; method effects with named paired comparators; six-cell transition checks.
- Completed presentation: compact vector Figure 3/4, main controls-table candidate, exactly one main transition table, hash-selected factual replacement case, full plotted CSVs and scripts.
- New forward passes: separately versioned harmonized expansion only. {expansion_status}
- Unavailable: encoder/method cells listed in `tables/unavailable_method_cells.csv`; full matching integrated v11 LaTeX. These are not imputed.
- Deferred/out of scope: IG/attention repair, PI-FT reproduction, new architectures/data/user studies/agent evaluations/latency studies. Historical records remain available.

## Quantitative reconciliation

`section6_claim_ledger.csv` has {len(ledger)} numerical assertion/cell/mark rows, including the actual 144 Table 1 cells and both copies of the 12-cell transition table. All checked manuscript numeric displays reproduce at their printed precision; four historical expanded-family assertions require a provenance-qualified replacement despite numerical agreement. Counts/ranges use their stated populations and are not adjusted to force agreement.

WANDS/BGE reversal, full precision: gained {reverse.gain_share*100:.12f}%, lost {reverse.loss_share*100:.12f}%, turnover {reverse.turnover*100:.12f}%, net Recall {reverse.delta_recall*100:+.12f} pp, 95% CI [{reverse.delta_recall_ci_low*100:.12f}, {reverse.delta_recall_ci_high*100:.12f}]. The displayed 2.357 + 2.263 versus 4.621 is ordinary rounding, not a failed identity.

Equal positive gains/losses occur in {int(wb.equal_positive_queries.min())}–{int(wb.equal_positive_queries.max())} of 308 WANDS/BGE queries per contrast ({100*wb.equal_positive_share_all.min():.6f}–{100*wb.equal_positive_share_all.max():.6f}%). Reverse specifically has {int(reverse.equal_positive_queries)}/{int(reverse.queries)} ({100*reverse.equal_positive_share_all:.6f}% of all, {100*reverse.equal_positive_share_changed:.6f}% of changed queries). Six contrasts remain separate; these counts cannot be summed into a union.

The illustrative query `{case['query']}` (ID {case['query_id']}) exchanges {case['gain_count']} Exact products in each direction. Both numerators are {case['source_numerator']} over {case['denominator']}; exact Recall@20 is unchanged. Complete names/IDs/labels/ranks/orders are in `data/illustrative_replacement_case.csv`. The hash rule, eligible case list and selection time are disclosed; no all-fitting eligibility was imposed and no user-harm claim is made.

## Named common-support method contrasts

The factual replacement case uses catalog-wide Source versus Reverse rankings on the same 42,994-product catalog. It is separate from the single-target, fixed-competitor intervention in RQ1.

Raw hybrid minus raw dense seven-schedule means, pp [95% unadjusted paired CI]:

| Dataset / encoder | Recall@100 | VI@100 |
|---|---|---|
{chr(10).join(hybrid)}

WANDS/BGE ascending **canonical hybrid** minus **raw-hybrid seven-schedule mean** Recall@20 is {effect('wands','bge_base','canonical_hybrid_lexical_ascending',20,'Recall','raw_hybrid:seven_mean')} pp. This names the comparator hidden by the abbreviated v11 sentence. Pure ascending dense versus raw source order is a different contrast, retained separately. Intervals spanning zero do not establish preservation.

All six transition cells, row margins and control inclusion/omission margins pass on identical query support. The WANDS/MiniLM ascending dense illustration includes both the 0.23% persistent-inclusion-to-omission cell and the 0.43% persistent-omission-to-inclusion cell. Their difference is not total Recall change, because crossing products contribute too. Crossing-to-omission is unconditional accounting mass, not a newly caused loss probability.

## Corrections and limits

Remove the duplicate Table 3 and p8 duplicate paragraph. Replace tDCG with the fully specified nDCG/cNDCG conventions. Clarify that fully-fitting support is a secondary target restriction, not the entire primary evaluation. Correct the canonical-rule cross-reference to the methods section. Remove attribution pointers when omitting the corresponding analysis from the proposed submission fragment; the old outputs/failures are untouched. Centroid construction uses multiple encodings but retains/scores one vector; multi-vector stores/scores M vectors. Historical timings are warm-cache CPU exact scoring, exclude query encoding, and are not ANN deployment latency.

The manuscript can support order sensitivity, membership information beyond averages, and empirically evaluated invariant controls. It does not establish a universal robustness/effectiveness trade-off, necessary canonicalization harm, architecture/length causation, saturation at 32, complete relevance judgments, or downstream user/agent harm.
''')
    handoff=[
      ('p6 opening, lines 623–627','rewrite','Use one short reporting-convention sentence; move repeated bootstrap definitions to setup.','section6_claim_ledger.csv; section6_scope_and_provenance.md'),
      ('p6 RQ1 P1, lines 631–635','keep','Catalog-wide VI ranges; retain finite-family interpretation.','tables/inclusion_complete.csv: catalog_wide/full/K20/VI'),
      ('p6 RQ1 P2, lines 636–647','rewrite','Separate fixed-competitor evidence from fully-fitting evidence and keep population warning.','tables/inclusion_complete.csv: target_only/K20; Figure 3'),
      ('p6 RQ1 P3, lines 648–656','appendix','At most one main sentence from harmonized results; retain historical/runtime comparison and all fitting supports in supplement. Remove saturation/causal-runtime conflation.','harmonized_supplement.md; tables/historical_to_harmonized.csv'),
      ('p7 Figure 3','rewrite','One GTE point labelled Full = fully fitting; color-independent markers, counts, explicit different scales, no connecting line.','figures/figure3_target_fitting.pdf; data/figure3_plotted.csv'),
      ('p6 RQ2 P1, lines 660–670','rewrite','Separate macro cancellation from exact within-query equality. Give 73/308 reversal count and all-six range; never a union.','tables/membership_all_contrasts.csv; data/membership_per_query.parquet'),
      ('p7 Figure 4','rewrite','Display Reverse uniformly in every setting; mark it as an editorial choice after outcomes. Include exact replacement counts beside bars. Keep all shuffles in companion.','figures/figure4_membership_replacement.pdf; data/figure4_plotted.csv'),
      ('RQ2 new compact illustration','appendix','Hash-selected anti fatigue mat case: exact unchanged 20/307 with five gained/five lost IDs. All orders in CSV. No all-fitting claim.','figures/illustrative_replacement_case.pdf; data/illustrative_case_selection.json'),
      ('p6 RQ2 P2, lines 671–680','rewrite','Keep conditional absolute nDCG and unconditional signed means distinct. Remove the 5-of-36 significant-count sentence; use descriptive intervals without global inference.','tables/membership_all_contrasts.csv; tables/controls_paired_comparisons.csv'),
      ('p6 RQ3 P1, lines 685–690','keep','State structural invariance once; effectiveness and identities remain empirical.','tables/controls_complete.csv; method metadata'),
      ('p8 Table 1','rewrite','Keep raw dense, all three canonical rules, BM25, raw hybrid, all canonical hybrids and Set-Mean. Move centroid/max budgets to appendix. Explicit candidate-vs-ranking cutoff rationale.','tables/main_table_candidate.tex; tables/method_complete_companion.csv'),
      ('p6 RQ3 transition P2, lines 691–696; p8 lines 848–852','rewrite','Add 0.43% persistent omission to inclusion alongside 0.23% opposite cell; explain crossing mass prevents subtracting only these cells for net Recall.','tables/transitions_complete.csv; data/transitions_per_query.parquet'),
      ('p8 duplicate transition paragraph, lines 853–856','omit-from-submission','Delete duplicate beginning with a stray period.','v11 PDF; inputs/section6_v11_raw.txt'),
      ('p9 Table 2','keep','Exactly one main six-cell dense/hybrid table, same convention-based ascending example.','tables/transition_single.tex; data/table2_plotted.csv'),
      ('p9 Table 3','omit-from-submission','Delete duplicate; update main text reference to the retained transition-table label.','v11 duplicate cells checked in claim ledger'),
      ('p8 RQ3 effectiveness P3, lines 857–862','rewrite','Name raw-seven means; report effect sizes and paired intervals without inferring a general trade-off.','section6_audit.md named contrasts; tables/controls_paired_comparisons.csv'),
      ('p8 RQ3 canonical hybrid P4, lines 863–909 across columns','rewrite','Replace significance counting with named comparator effects; retain ascending HYBRID Recall@20 loss against raw-hybrid seven mean. No equivalence inference.','tables/controls_paired_comparisons.csv'),
      ('p8 RQ3 cost P5, lines 910–916','rewrite','Centroid: M construction encodings, one stored/scored vector; max: M stored/scored vectors. Preserve warm-cache CPU/excludes-query/exact/not-ANN caveats.','revision_completion_20260921/data/strategy_costs.csv; controls metadata'),
      ('p6 setup, lines 581–596 and 599–620','rewrite','Primary full population plus secondary fitting restriction; canonical rules are in methods Section 4; replace tDCG, specify direct gains/IDCG/coverage and prior feedback.','manuscript/setup_corrections.tex; section6_scope_and_provenance.md'),
      ('Appendix native score-margin example','appendix','Optional; retain existing numerical guard and historical discrepancy. No expansion of IG/attention.','revision_final_strengthening_20260922/BOUNDARY_AUDIT.md'),
      ('Appendix C.4 attribution plus main pointers','omit-from-submission','Proposed submission fragment omits attribution analysis and its main pointers. Preserve all historical outputs and failure records; do not claim a unique mechanism.','revision_final_strengthening_20260922/XAI_AUDIT.md'),
      ('Appendix view budgets and costs','appendix','Keep every method/budget and all named comparisons in complete outputs; do not pick best-per-column synthetic methods.','tables/method_complete_companion.csv; inherited cost tables')]
    csv(pd.DataFrame(handoff,columns=['original_location','action','editorial_instruction','verified_reference']),HERE/'data/editorial_handoff.csv')
    write('section6_editorial_handoff.md','# Paragraph-level editorial handoff\n\nScope: v11 pages 6–9 and directly connected setup/appendix pointers. The matching component source is preserved in `inputs/reference_section6/`; a scoped replacement is in `manuscript/section6_revised.tex`. No entire-manuscript rewrite or overwrite is made.\n\n| Original paragraph or asset | Action | Instruction | Verified result |\n|---|---|---|---|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in handoff)+'\n\nFull-paper page count and float placement remain an author integration check because the full matching LaTeX is unavailable. The standalone ACM proof verifies local references and supplied asset geometry only. No citations have been invented.\n\nOptional future work (not run): a previously uninspected confirmation cohort would answer generalization beyond inspected queries; a matched same-support input-length intervention would address truncation causation; deployment measurements would address ANN/agent latency. None is needed to claim the finite-family inclusion and membership results here.')
    # A scoped revised section, with no dangling appendix references or duplicate input.
    asc_ndcg=effect('wands','bge_base','canonical_hybrid_lexical_ascending',20,'nDCG','BM25:fixed',1,4)
    asc_dense=effect('wands','bge_base','canonical_hybrid_lexical_ascending',20,'nDCG','lexical_ascending:fixed',1,4)
    tex=r'''\section{Results}
\label{sec:results}
Unless stated otherwise, we report query-macro percentages on the primary populations, percentage-point (pp) differences, and nDCG on the 0--1 scale.

\subsection{Attribute-Order Sensitivity (RQ1)}
Across the seven catalog-wide schedules, VI@20 ranges from 11.93--19.58\% on WANDS and 3.66--6.91\% on ESCI. These estimates describe crossing within the tested family, not a probability of omission under future orders.

With every competitor fixed at source order, changing only the target yields VI@20 of 10.83--19.45\% on WANDS and 2.42--5.88\% on ESCI. The interventions answer complementary questions; their difference does not decompose target and competitor effects. Sensitivity remains when every tested target input fits completely: fully-fitting VI@20 ranges from 7.16--14.55\% and 2.55--5.88\%, respectively (Figure~\ref{fig:rq1_target_fitting}). GTE's full and fitting populations coincide. MiniLM and BGE fitting subsets differ from the full populations and from one another, so these comparisons do not isolate a causal truncation or architecture effect.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/figure3_target_fitting.pdf}
\caption{Target-only VI@20 with 95\% query-cluster intervals and query/pair support. GTE shows one estimate for identical full and fitting populations. Panel axis ranges differ explicitly; competitors remain fixed.}
\label{fig:rq1_target_fitting}
\Description{Separate markers show full and fitting estimates, without a connecting line implying a truncation intervention.}
\end{figure*}

EXPANSION_SENTENCE

\subsection{Aggregate Effectiveness and Product Membership (RQ2)}
Gains and losses can be substantially larger than their net Recall change. Reversing WANDS attributes with BGE newly includes 2.357\% and newly omits 2.263\% of highest-label support. The full-precision sum rounds to 4.621\%, while Recall changes by only $+0.094$ pp (95\% CI $[-0.409,0.563]$). Each share is divided by $|H_q|$ within a query before macro averaging.

Exact replacement also occurs within individual queries. For this reversal, 73 of 308 WANDS/BGE queries (23.70\%) have equal positive gain and loss counts. Across the six contrasts, counts range from 67 to 78 (21.75--25.32\% per contrast), not a union of distinct queries. Figure~\ref{fig:rq2_membership} uses reversal as the same definition-based editorial contrast in every setting, selected after historical outcomes; all original schedules remain in companion outputs. In the same catalog-wide reversal, an illustrative hash-selected query, ``anti fatigue mat,'' exchanges five Exact products in each direction while Recall@20 remains exactly $20/307$. The case companion records every changed ID, name, label, rank and order.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/figure4_membership_replacement.pdf}
\caption{Catalog-wide Reverse versus source at $K=20$. Bars show query-macro gained/lost shares of $H_q$; losses are nonnegative quantities drawn left only for display. Diamonds show net Recall differences with 95\% paired intervals. Counts require equal positive integer gains and losses and use all eligible queries as denominator.}
\label{fig:rq2_membership}
\Description{Relevant membership turnover can exceed net Recall changes, and individual queries can have unchanged Recall with different relevant products.}
\end{figure*}

Graded effectiveness is complementary. Conditional on changed Top-20 relevant membership, median absolute nDCG@20 differences range from 0.0226--0.0399 on WANDS and 0.0388--0.0482 on ESCI. Signed means over all eligible queries range from $-0.01101$ to $+0.00151$ and $-0.00393$ to $+0.00092$, respectively. These summarize different populations and quantities. The companion retains each named contrast, its unadjusted interval, condensed nDCG and judged coverage; intervals containing zero do not establish equality.

\subsection{Effectiveness and Consistent Inclusion under Controls (RQ3)}
Canonical, set-based, fixed-view and lexical controls remove incoming-order dependence by construction. Their zero VI is structural; effectiveness and which products are persistently included remain empirical questions. Table~\ref{tab:strategy_joint} retains all three canonical rules without choosing a test winner. Raw dense and raw hybrid effectiveness average seven schedules; source-order results remain separately labelled in the companion.

\begin{table*}[t]
\centering\normalsize\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.05}
\caption{Common-support controls. Recall (R) and VI are percentages; nDCG uses 0--1. R@100/VI@100 assess candidate inclusion, whereas nDCG@20 assesses ranking near the top. All fixed controls have structural VI=0. The companion includes R@20, paired intervals, source-order references, GTE and every view budget.}
\label{tab:strategy_joint}
\input{tables/main_table_candidate.tex}
\end{table*}

Simple controls have setting-dependent effectiveness. Set-Mean's Recall@100 exceeds the raw dense seven-schedule mean on WANDS/MiniLM and is lower in the other three MiniLM/BGE settings. Raw hybrid minus raw dense changes WANDS/BGE mean Recall@100 by $+4.153$ pp (95\% CI $[2.550,5.821]$); full paired comparisons for all settings accompany the table. These observations do not establish a general effectiveness--instability trade-off.

For a convention-based canonical example, WANDS/BGE ascending canonical hybrid changes nDCG@20 by ASC_NDCG versus BM25 and ASC_DENSE versus its corresponding ascending dense branch. Against the raw-hybrid seven-schedule mean, its Recall@20 changes by $-0.624$ pp (95\% CI $[-1.370,-0.084]$). This counterexample remains visible even though the main table reports R@100. Each interval is unadjusted; neither an interval crossing zero nor structural invariance establishes universal preservation.

\begin{table*}[t]
\centering\small
\caption{One transition illustration: WANDS/MiniLM ascending at $K=20$. Cells are query-macro percentages of all highest-label support. Dense and hybrid use their own raw families; each six-cell block sums to 100\% before rounding.}
\label{tab:control_destinations}
\input{tables/transition_single.tex}
\end{table*}

The fixed order can place a previously crossing product on either side of the cutoff (Table~\ref{tab:control_destinations}). In the dense example, 7.57\% of all support moves from crossing to inclusion and 12.01\% to omission. The control also omits 0.23\% that was persistently included and includes 0.43\% that was persistently omitted. These are unconditional accounting shares, not newly caused losses or failure probabilities. The two small stable-state cells alone do not recover total Recall change because crossing products also contribute. The canonical order may lie outside the seven raw orders.

Cost provides a separate comparison. Canonicalization retains one dense vector; centroid construction encodes $M$ views but retains and scores one vector, whereas multi-vector retrieval stores and scores $M$ vectors. Hybrid adds sparse storage and branch scoring. Existing cost measurements use warm-cache CPU exact retrieval and exclude query encoding; they are not ANN deployment latency. Effectiveness, persistent inclusion/omission, identities and cost therefore accompany VI.
'''
    # Use recomputed named effects rather than hard-coded updated quantities.
    rawhy=effect('wands','bge_base','raw_hybrid',100,'Recall','raw:seven_mean')
    match=rawhy.split(' [');rawhytex='$'+match[0]+'$ pp (95\\% CI $['+match[1].rstrip(']')+']$)'
    tex=tex.replace('$+4.153$ pp (95\\% CI $[2.550,5.821]$)',rawhytex)
    tex=tex.replace('EXPANSION_SENTENCE',expansion_sentence).replace('ASC_NDCG','$'+asc_ndcg.replace(' [', '$ (95\\% CI $[')+']$)' if False else '$'+asc_ndcg.split(' [')[0]+'$ (95\\% CI $['+asc_ndcg.split(' [')[1].rstrip(']')+']$)').replace('ASC_DENSE','$'+asc_dense.split(' [')[0]+'$ (95\\% CI $['+asc_dense.split(' [')[1].rstrip(']')+']$)')
    write('manuscript/section6_revised.tex',tex)
    write('manuscript/setup_corrections.tex',r'''% Targeted setup/cost corrections; no new citations are added.
% Section 5.1: restore the historical development/heldout selection disclosure.
WANDS reserves the lowest 96 of 480 query IDs under SHA256(`phase3-dev:query_id') for development; the other 384 form the historical held-out set. The 308 with an Exact judgment contribute 21,299 pairs. ESCI uses the first 500 US small-version test queries sorted by SHA256(`20260904:query_id'), with the union of judged products forming the 10,076-product catalog; 499 queries have E labels. Earlier outcomes were inspected during development and follow-up, so this is not a blind confirmation claim.

% Section 5.2: primary support is FULL; fully-fitting is a separate restriction.
Primary analyses use full eligible support. Fully-fitting diagnostics retain targets whose complete native inputs, including special tokens and any prefixes, fit under every evaluated schedule; competitors are never filtered. The three canonical rules are defined in the methods section, not the problem-definition section.

% Section 5.3: replace tDCG and explicitly define graded evaluation.
Graded gains are used directly, without exponentiation: WANDS Exact/Partial/Irrelevant $=3/1/0$, and ESCI E/S/C/I $=1/0.1/0.01/0$. nDCG uses $1/\log_2(r+1)$ at full-catalog ranks and normalizes by the top-$K$ ideal ordering of all available judged gains. Unjudged items contribute zero to this calculation only. Condensed cNDCG removes unjudged items before assigning discount positions; JudgedCoverage is the judged fraction of the original top-$K$ list. Neither convention completes the judgments.

Query-macro estimates average within-query proportions before averaging queries. Intervals use 10,000 paired query-cluster percentile resamples, seed 2026091701, retaining every product and compared condition in the selected query. They are unadjusted and conditional on fixed catalogs and models. An interval containing zero does not demonstrate equality, equivalence or noninferiority. Source-order and raw seven-schedule means are separately labelled.

% Section 4.2.2: replace the claim of no online overhead for both aggregators.
Both strategies construct view embeddings offline. A centroid retains and scores one vector after $M$ construction encodings; multi-vector retrieval retains and scores all $M$ vectors. Historical warm-cache CPU exact-scoring timings exclude query encoding and do not establish approximate-index deployment latency.
''')
    write('REPRODUCE.md',r'''# Reproduce this scoped revision

Run from the repository root in PowerShell. The reference environment and exact packages are in `qa/environment.json`; each harmonized model's full profile pins CUDA/GPU, libraries, model/tokenizer commits, attention, precision, deterministic settings, padding and batching. Python 3.10, numpy, pandas, pyarrow, scipy/threadpoolctl, matplotlib, PyTorch/Transformers and the pinned local Hugging Face snapshots are required. Poppler and MiKTeX/acmart provide PDF QA/proof. No paid API is used.

```powershell
# Preserve the existing P0 snapshot; do not overwrite the audit baseline.
python revision_section6_20260924/scripts/verify_sources.py
python revision_section6_20260924/scripts/trace_rank_sources.py
python revision_section6_20260924/scripts/analyze_saved.py
python revision_section6_20260924/scripts/ledger.py
python revision_section6_20260924/scripts/figures.py
python revision_section6_20260924/scripts/harmonize.py --model minilm
python revision_section6_20260924/scripts/harmonize.py --model bge_base
python revision_section6_20260924/scripts/summarize_harmonized.py
python revision_section6_20260924/scripts/ledger.py
python revision_section6_20260924/scripts/reports.py
Push-Location revision_section6_20260924
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex
pdftoppm -r 144 -png qa/section6_acm_proof.pdf qa/proof
Pop-Location
# Inspect all current figure PDFs/PNGs and every proof page, then record
# that review in qa/visual_review.json before the final validation.
python revision_section6_20260924/scripts/validate_final.py
python revision_section6_20260924/scripts/package_review.py
```

On the audited Windows host, the exact interpreter is `C:\Users\ycw\AppData\Local\Programs\Python\Python310\python.exe`; use that executable in place of `python` if another environment is active. Stop on any command failure instead of consuming stale downstream outputs. The initial snapshot in `qa/starting_state.json` belongs to the original audit; preserve it with the package when making a separate reproduction copy. `p0_audit.py` created that snapshot initially and is not a resume command. Source verification accepts the audited commit or descendant delivery commits confined to this revision directory, while continuing to require all original source and pre-existing edit hashes. A fresh clone needs those local historical artifacts restored before it can reproduce the complete local audit.

Run GPU models sequentially. CPU saved-output analysis may run while encoding. `harmonize.py` verifies the immutable profile, planned aliases and every completed vector block before reuse; it recomputes only unfinished blocks. It refuses a changed profile in the same directory. A new scientific profile needs a new output version. Do not relabel newly generated vectors as historical payloads. If the device sleeps, completed blocks remain saved; block elapsed times may include the pause. Do not interpret them as benchmark latency.

The initial 82 rank/label conditions are enumerated in `revision_graded_20260922/data/conditions.json`; the authoritative `data/per_query.parquet` in that historical revision also records 40 later raw/canonical-hybrid conditions, for 122 total. The new provenance table merges both sources and checks every rank payload against the earlier input or output hash manifest. Original representations are `phase2/data/representations/<dataset>/C*.jsonl.gz`; native target-only ranks are in `phase3/results/target_only_permutations/`. Pinned expansion plan and cohort are under `revision_final_strengthening_20260922/data/`. `data/source_inventory.csv` and `data/historical_execution_matrix.csv` contain exact source/array hashes and explicit unavailable metadata. Large NPY payloads/model snapshots may be local/ignored; Git presence is not a guarantee of payload availability. Restore original missing artifacts using their hashes, never regenerate and label them original.

`data/method_rank_provenance.csv` maps all 122 saved method/schedule conditions to their exact rank hashes and available embedding sidecars, with historical input-manifest agreement and unavailable runtime fields distinguished. It supplements the raw seven-schedule execution matrix instead of treating a shared model name as complete execution provenance.

Some historical control sidecars lack exact per-array library versions. Common query support and checked output hashes do not establish bitwise matched execution across every historical method. Those comparisons retain their historical provenance; only the separately harmonized order-expansion supplement establishes the new uniform execution profile.

All generated evidence is confined to this revision directory. Original files, manuscript sources, historical results, prior failure records and dirty working-tree edits must remain byte-identical. Existing equivalent analyses are reused; the extra native implementation is limited to the harmonized supplement. No training, model sweep or attribution repair is a hidden dependency.

To resume after the graded stage only (same verified files):

```powershell
python revision_section6_20260924/scripts/analyze_saved.py --resume-controls
```

For the local ACM proof, run pdflatex twice from this revision directory with `-interaction=nonstopmode -halt-on-error -output-directory=qa qa/section6_acm_proof.tex`. This validates the supplied section and assets, not the unavailable full matching manuscript. PDF/PNG figures are emitted directly by matplotlib at seven-inch width with vector text/marks. Exact plotted records are `data/figure3_plotted.csv`, `data/figure4_plotted.csv`, `data/table2_plotted.csv` and `tables/main_table_candidate.csv`.
''')
    write('README.md',f'''# Section 6 evidence revision

Read `section6_audit.md`, `section6_claim_ledger.csv`, `section6_scope_and_provenance.md`, and `section6_editorial_handoff.md` first. Saved evidence is fully recomputed and checked. Harmonized status: {expansion_status}

The two revised figures, main controls table, single transition table and factual replacement illustration are in `figures/` as vector PDFs plus PNG previews. Source scripts and full-precision plotted records are included. The scoped manuscript replacement is `manuscript/section6_revised.tex`; matching original component sources remain under `inputs/`. This package does not overwrite the historical manuscript.

Complete companions: `tables/inclusion_complete.csv`, `tables/membership_all_contrasts.csv`, `tables/method_complete_companion.csv`, `tables/transitions_complete.csv`, and the authoritative query-level records in `data/`. Unevaluated cells are explicitly unavailable. Reproduction commands and resume rules are in `REPRODUCE.md`.

`COMPANION_GUIDE.md` explains the keys, units, denominators, reference labels and unavailable cells. `qa/final_validation.json` records the final checks, and `output_manifest.csv` fingerprints the delivered package. Large vector payloads are local caches, with their hashes recorded in each harmonized model's `complete.json`.

GitHub delivery was separately authorized by the user on 2026-09-25 and is scoped to this revision directory. Local pre-existing edits outside it remain uncommitted. The local review ZIP, large vector caches and redundant QA renders are excluded from Git; all evidence records, provenance, scripts, final assets and the ACM proof are included. `output_manifest.csv` also inventories local QA intermediates. No manuscript submission or external message was performed.
''')
    print('REPORTS WRITTEN; harmonized complete =',harmonized)
if __name__=='__main__':main()
