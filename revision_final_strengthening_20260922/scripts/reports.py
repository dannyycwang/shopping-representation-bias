"""Generate completed-evidence reports from machine-readable outputs."""
from common import *
def put(name,text):(HERE/name).write_text(text.strip()+'\n',encoding='utf8')
def md(f):
 f=f.copy()
 for col in f:
  if pd.api.types.is_float_dtype(f[col]):f[col]=f[col].map(lambda x:'NA' if pd.isna(x) else f'{x:.6f}')
 return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in f.itertuples(index=False,name=None))
def cell(f,**kw):
 for k,v in kw.items():f=f[f[k].eq(v)]
 assert len(f)==1,(kw,len(f));return f.iloc[0]
def main():
 verify()
 ba=pd.read_csv(HERE/'qa/boundary_reconstruction.csv');bd=pd.read_csv(HERE/'tables/boundary_distributions.csv');xq=pd.read_csv(HERE/'qa/xai_forward_completeness.csv',dtype={'product_id':str});xb=pd.read_csv(HERE/'qa/xai_baseline_per_pair.csv',dtype={'product_id':str});xa=read(HERE/'qa/xai_accounting.json');cv=pd.read_csv(HERE/'tables/coverage_summary.csv');inc=pd.read_csv(HERE/'tables/coverage_increments.csv');est=pd.read_csv(HERE/'tables/coverage_compute_estimate.csv');sens=pd.read_csv(HERE/'qa/coverage_precision_sensitivity.csv')
 counts={m:read(HERE/f'qa/{m}_coverage_encoding_complete.json') for m in ['minilm','bge_base']}
 cp=xq[xq.display.eq('main_crossing')&xq.variant.eq('maximum')&xq.baseline.eq('zero')].iloc[0]
 xfail=xq[xq.status.ne('passed')][['model','query_id','product_id','variant','baseline','status','steps','completeness_residual','completeness_tolerance']]
 compact=bd[(bd.K==20)&bd.aggregation.eq('query_macro')&bd.quantity.eq('score_range')][['dataset','model','support','queries','support_pairs','state','state_pairs','state_mass','state_mass_ci_low','state_mass_ci_high','conditional_mean','median','p95']]
 compact.to_csv(HERE/'tables/boundary_compact_K20.csv',index=False)
 v=cv[(cv.metric=='VI')&cv.aggregation.eq('query_macro')].copy();v['mean_pp']=100*v['mean'];v['CI_low_pp']=100*v.ci_low;v['CI_high_pp']=100*v.ci_high
 vcols=['dataset','model','support','family','K','queries','pairs','mean_pp','CI_low_pp','CI_high_pp']
 v[v.K.eq(20)].to_csv(HERE/'tables/coverage_display_K20.csv',index=False)
 d=inc[inc.aggregation.eq('query_macro')&inc.quantity.eq('incremental_crossing')].copy();d['delta_pp']=d['mean']*100;d['CI_low_pp']=d.ci_low*100;d['CI_high_pp']=d.ci_high*100
 distinct_esci=pd.read_csv(HERE/'data/esci_distinct_order_counts.csv');distinct_wands=pd.read_csv(HERE/'data/wands_distinct_order_counts.csv')
 distinct_table=distinct_esci.groupby(['distinct_full_family_texts','distinct_old_seven_texts']).size().rename('target_products').reset_index();distinct_table.to_csv(HERE/'tables/esci_distinct_order_coverage.csv',index=False)
 newtotal=sum(r['new_inputs'] for r in counts.values());seconds=sum(r['seconds'] for r in counts.values())
 def vi(ds,m,fam,scope='full',k=20):return cell(cv,dataset=ds,model=m,family=fam,support=scope,K=k,metric='VI',aggregation='query_macro')['mean']*100
 names={'minilm':'MiniLM','bge_base':'BGE'}
 wline='; '.join(f"{names[m]}: {vi('wands',m,'7'):.2f}% → {vi('wands',m,'16'):.2f}% → {vi('wands',m,'32'):.2f}%" for m in ['minilm','bge_base'])
 eline='; '.join(f"{names[m]}: {vi('esci',m,'7'):.2f}% → {vi('esci',m,'exhaustive_target'):.2f}%" for m in ['minilm','bge_base'])
 put('README.md',f"""
# Final evidence strengthening — 22 September 2026

**A–D completed**, with attribution failures and numerical ambiguity retained as results. No resource-blocked cells, training, new GTE forward passes, manuscript edits or default-branch merge. Dedicated branch: codex/final-strengthening-20260922.

- A: all 21,299 WANDS Exact pairs / 308 queries and 4,434 ESCI E pairs / 499 queries, three encoders, seven schedules, K=20/100: {int(ba.records.sum()):,} boundary records. No Top-K reconstruction contradictions. One previously documented deep BGE rank discrepancy remains authoritative.
- B: 24 hash-stratified diagnostic pairs plus French molding; 50 variants, 100 baseline runs. {xa['completeness_passes']} completeness passes, {xa['failures']} retained failures. The main crossing fails its precision guard; the stable control has a primary-baseline completeness failure. Neither is replaced.
- C: 48 dense/hybrid comparisons, every canonical rule, both K values, all six identity transitions with query-macro/pair-micro intervals and conditional denominators.
- D: actual encoding of {newtotal:,} new distinct model inputs. WANDS 128 frozen queries / 7,125 pairs, 7/16/32 schedules; ESCI all 499 queries / 4,434 E pairs, every distinct whole-entry target order. Full-support Top-20 crossing: {wline}. ESCI: {eline}.

P0/P1's 122 conditions / 49,227 records, corrected graded metrics, canonical hybrids, previous M=2/4/7 results and costs were read and reused, not rerun.

Read BOUNDARY_AUDIT.md, XAI_AUDIT.md, CONTROL_TRANSITIONS.md and PERMUTATION_COVERAGE.md for the four workstreams. CLAIM_VERDICTS.md and CHAPTER6_EVIDENCE_MAP.md provide review-ready conclusions, English insertions, captions and placements. REPRODUCE.md, manifests and qa/final_validation.json give reproducibility and completion gates. One main case figure and one compact table are proposed; coverage/extra QA are supplementary.

Reviewed baseline 29a699794746b4ecc916f69e75e7d05f737677f7 and starting HEAD/merge-base 914f5ec36a75e8bcfeeef5234c86b2468ff761dd have identical tracked trees; the baseline adds two merge-history commits. Existing manuscript/figure edits were hashed and preserved. No applicable AGENTS.md was found.

Protocol SHA256: {sha(HERE/'PROTOCOL.json')}. This is a frozen follow-up after observed historical results, not whole-study preregistration. INPUT_MANIFEST.json lists 185 exact payload/source files. COVERAGE_EXECUTION_ADDENDUM.json documents input-identity deduplication before new outcomes without changing supports, schedules or target strings.

Git includes scientific pair/schedule scores/ranks, XAI data, figures and cache metadata. Large generated vectors/score arrays remain locally available but ignored; CACHE_MANIFEST.json inventories hashes. Some original arrays/model snapshots also require restoration from pinned hashes outside Git. Never relabel newly generated arrays as original artifacts.
""")
 put('BOUNDARY_AUDIT.md',f"""
# A. Score boundary audit

All competing products remain at C0. The old target is removed before replacement, preserving catalog size 42,994 or 10,076. t_K is the Kth competitor excluding the target. A higher score wins; equality uses the smaller original catalog index. Exact saved FP32 comparisons are used with no epsilon; margin sign alone is not the tie rule.

C0 scores come from the complete cached normalized query/product matrix; alternative target scores are saved Phase II pair scores. Authoritative ranks are Phase III target-only ranks, never catalog-wide ranks. Separate replacements do not form one coherent query ranking. qa/boundary_algorithm_tests.json verifies the inherited algorithm against brute-force tied-score replacements.

{md(ba)}

## Precision reconciliation

The sole discrepancy, WANDS/BGE q252/product385/C2s3, is saved rank479 versus reconstructed480, exactly the inherited target_rank_precision_differences.csv record. Target score .7454689145088196; competitor30099 score .7454689741134644: one FP32 ULP (5.960464477539063e-8) higher under current matrix multiplication. Neighboring IDs/scores appear in qa/boundary_discrepancy_neighbors.csv. There is no exact tie or Top-20/100 change. The old computation's precise rounding source is not retroactively assigned; saved rank479 remains authoritative. No tolerance is widened. All unresolved-boundary flags are false.

## Distributions

The full tables/boundary_distributions.csv covers K20/K100, three states, full/all-seven-fitting supports, score ranges, maximum C0 deviation, C0 margins, cosine distances, best/worst ranks and included-schedule counts. Query-macro weights each nonempty query equally and divides its mass among its pairs; conditioning on state then renormalizes those weights. It is not the equal-query average over only queries with that state. Pair-micro weights pairs equally. Cluster intervals keep queries whole.

Top-20 state masses (0–1) and conditional score-range summaries:

{md(compact)}

Saved compatible embeddings supply maximum cosine distance from C0. The margin/rank relationship is an algebraic validation identity; the distributions are empirical information. Complete records include catalog/threshold indices and IDs, scores, margins, saved/reconstructed ranks, tokens, fitting flags and precision status.

## Historical illustrations

data/historical_case_boundary.csv verifies MiniLM q359 “french molding” / product12575: ranks15–174, source rank19, 139 tokens including specials, all seven fit under256. GTE q409 “teal chair” / product24318: ranks11–939, source rank11, 1,059 tokens under8,192. These are explicitly post-hoc illustrations, not prevalence estimates. The author's Introduction figure is untouched.
""")
 displays=xb[xb.display.isin(['main_crossing','main_stable_control','supplement_historical'])]
 put('XAI_AUDIT.md',f"""
# B. Query-conditioned attribution

The seed-2026092201 hash panel has four crossing, four always-in and four always-out all-seven-fitting pairs per MiniLM/BGE, with distinct queries within each stratum where possible. No quota is short. French molding is an added post-hoc case: 25 pairs / 50 variants. All candidates, selections and failures are retained, without replacement after inspecting results.

Variants are minimum/maximum native-score schedules, ties by original order; these are extrema, not random views. Main crossing BGE q235/product7253 is nearest the selected stratum's median rank span. Stable control q260/product41359 minimizes the frozen token/attribute-count distance. Figures use the first eight source occurrence IDs plus an exact other-content sum, not largest-change entries.

## Scalar, adapter and accounting

F_q(x)=dot(cached normalized query vector, normalized product vector). The query stays fixed. MiniLM uses attention-mask mean pooling, BGE CLS. IG varies word embeddings only at nonspecial nonpadding positions, preserving sequence length, positions, types, mask and special/padding embeddings. Fixed title/description fields remain included. Serializer spans and tokenizer offsets preserve duplicate occurrence IDs and assign all tokens to entries, fixed fields, separators, boundaries or specials.

Native retrieval uses FP16 model weights with FP32 pooling/normalization. The gradient adapter represents the same FP16-rounded frozen parameters in FP32, eval mode, TF32 disabled. It is a numerical diagnostic, not new authoritative retrieval. Actual runtime is torch2.11.0+cu128, transformers4.46.2 on RTX4060 Laptop; inherited metadata sometimes report transformers4.55.4. Fresh FP16 and FP32 scores are separately compared with saved scores; no bitwise runtime compatibility is assumed.

Maximum input_ids/inputs_embeds error: {xa['max_forward_ids_embeds_error']:.9g}; maximum diagnostic/saved score error: {xa['max_diagnostic_native_error']:.9g}; changed diagnostic inclusions: {xa['changed_diagnostic_inclusions']}/50. Token and entry sums agree to1e-10. Signed sums, absolute token magnitudes and absolute feature magnitudes are separately exported.

## Retained main-case limitations

The crossing maximum-order margin is {cp.native_margin:.12g}, versus diagnostic/native discrepancy {cp.diagnostic_native_error:.12g}. Both inclusion decisions remain unchanged, but margin >5×error fails. Its native extrema ranks20 and39 document a crossing; IG cannot reliably explain it.

The stable control extrema ranks19 and14 stay included. Its maximum-order primary zero-baseline IG fails completeness at256 nodes. The figure explicitly labels these limitations and retains native margins. Neither a changed baseline nor another case is substituted to obtain a favorable explanation.

## Completeness and baseline sensitivity

Primary reference: zero nonspecial word vectors. Sensitivity reference: PAD vectors at the same positions, not a natural product description. Gauss–Legendre64→128→256, stopping at |sumIG−(F(input)−F(baseline))| <= max(1e-4,.01|score difference|). Completeness is checked separately for each variant/reference. Baseline scores differ across orders, so attribution-sum differences alone need not equal input-score differences.

{xa['completed_runs']}/100 runs completed; {xa['completeness_passes']} pass, {xa['failures']} retained failures. Node counts: {json.dumps(xa['steps_counts'],sort_keys=True)}.

{md(xfail)}

data/xai contains every attempt, score, residual, token and occurrence value. qa/xai_failures.json retains failures. Aligned maximum-minus-minimum entry changes across both references are exported. Descriptive baseline comparisons for displayed cases:

{md(displays[['model','query_id','product_id','display','zero_pad_change_pearson','change_sign_agreement','zero_change_l1','pad_change_l1','both_variant_boundary_guard','all_completeness_passed']])}

Across the selected panel, median change-sign agreement is {xb.change_sign_agreement.median():.3f} (range {xb.change_sign_agreement.min():.3f}–{xb.change_sign_agreement.max():.3f}); median change correlation is {xb.zero_pad_change_pearson.median():.3f}. These are exploratory selected-panel diagnostics, not population prevalence. Correlation cannot rescue failed guards or completeness.

For the historical French-molding case, both baselines preserve positive maximum-minus-minimum changes for width (zero +.021282, PAD +.015322), product type (+.008897/+.008801) and material (+.018395/+.014716), and a negative change for molding use (−.012439/−.013483). Color changes sign (−.024187/+.000349), showing a baseline-sensitive detail. These exact occurrence-level changes are in xai_aligned_entry_changes.parquet; they describe attribution redistribution without identifying a unique mechanism.

Figures use one common signed color scale, no independent per-order normalization. Full serialized texts, spans, displayed values and all seven native margins/ranks accompany the plots.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) motivates the method; see also [Jain–Wallace](https://aclanthology.org/N19-1357/) and [Wiegreffe–Pinter](https://aclanthology.org/D19-1002/) on attention interpretation. No raw attention study was added. IG describes a scalar, path and baseline, not a unique causal mechanism, human attention, seller harm, behavior or fairness.
""")
 put('CONTROL_TRANSITIONS.md',"""
# C. Identity transitions under fixed controls

All48 dataset×encoder×family×rule×K comparisons retain every canonical rule, both dense and hybrid families, and K20/K100. Reference rows are always-in/crossing/always-out across the respective raw catalog-wide seven-schedule family. Columns are included/omitted in the corresponding fixed index. Unlike A/D, these references use coherent saved query rankings.

Highest-support and inclusion identities are recovered from the graded per-query records. Each product falls in exactly one of six cells; row totals equal reference state masses and the included column reproduces fixed-control highest-label Recall. Full product identities and per-query cells are exported.

All six cells are fractions of Hq, query macro primary and pair micro secondary, with10,000 uncorrected query-cluster bootstrap draws. Conditional inclusion is a ratio of query-macro cell masses, not the mean per-query within-state rate. Explicit pair/query denominators are supplied; zero-denominator conditions are NA. The full table includes K100 and every interval.

The proposed compact K20 main table shows always-in products lost, always-out gained and both destinations of crossing products, retaining both families and all rules. Values are percentage points:
"""+(HERE/'tables/main_control_transitions.md').read_text()+"""
A deterministic zero VI does not specify which products the policy consistently includes. Loss/gain columns refer to each family's own raw reference and do not estimate seller harm or human utility. No rule is chosen from these outcomes. Retain graded effectiveness and cost evidence alongside the identity table.
""")
 put('PERMUTATION_COVERAGE.md',f"""
# D. Completed permutation coverage

WANDS uses the128 queries selected by SHA256("coverage-20260922|"+decimalID), numeric tie-breaker:7,125 pairs,6,444 products, identical queries for both encoders. Exact old seven schedules are followed by seeds2026092201–2209 then2210–2225 through the original product-ID-seeded random-order function. Families are nested7/16/32. Duplicates remain in schedule-weighted frequencies and distinct texts are separately counted.

ESCI uses all499 queries/4,434 E pairs. Every distinct permutation of whole nonempty entries is enumerated, maximum3 entries/6orders, preserving duplicate multiplicities, fixed blocks and separators. This exhausts per-target orders with competitors atC0, not joint catalog combinations.

The original seven schedules left missing target orders for {int((distinct_esci.distinct_full_family_texts>distinct_esci.distinct_old_seven_texts).sum()):,} of {len(distinct_esci):,} distinct E-target products. Per-target coverage is below; three distinct orders arise when a three-entry multiset contains duplicates. Counts here are products, not queries or query–product pairs. Global unique text counts in the compute table additionally deduplicate identical strings across products.

{md(distinct_table)}

For WANDS, {int((distinct_wands.distinct_full_family_texts<32).sum())} of {len(distinct_wands):,} target products have fewer than 32 distinct full strings; the average is {distinct_wands.distinct_full_family_texts.mean():.6f}. Complete per-product old/full counts are in data/wands_distinct_order_counts.csv and data/esci_distinct_order_counts.csv.

Only missing distinct target model inputs are encoded. Exact old query/competitor payloads are reused; no whole-catalog/GTE encoding or training. Original-seven token lengths match historical values. Each model's fixed common fitting support fits every32 WANDS schedule or every ESCI permutation; nonempty query sets stay constant across the curve. Full supports are separate; the128-query supplement is not the308-query primary population.

## Frozen estimates and measured computation

{md(est[['dataset','model','products','schedule_records','distinct_full_texts','distinct_model_inputs','new_model_inputs','new_capped_tokens','runtime_estimate_seconds','full_queries','full_pairs','fully_fitting_queries','fully_fitting_pairs']])}

Counts/estimates were frozen before new scores in COVERAGE_PLAN_FROZEN.json. Full-text hashes resolve through explicit complete-token-input aliases: identical input_ids/types/masks after truncation share one vector. The execution addendum records this before new outcomes. No strings, schedules or support pairs are dropped.

Actually encoded {newtotal:,} new inputs. Block encoding/compression time totals {seconds:.1f}s ({seconds/60:.1f}min), excluding token planning, loading, forward audit, scoring and reporting:

The recorded total also excludes an interrupted partial block. BGE initially used native batch 48; at the 8 GiB GPU's memory limit, the run resumed after 50 complete WANDS blocks (51,200 inputs) with batches capped at 24 and unused allocator memory released between blocks. ENCODING_MEMORY_ADDENDUM.json documents this resource-driven adjustment before remaining BGE scores. Completed vectors were hash-verified and reused; no model, input, seed, support or outcome selection changed. Batch-dependent numerical differences are part of the disclosed forward audit, not assumed zero.

{md(pd.DataFrame([dict(model=m,inputs=r['new_inputs'],blocks=r['blocks'],seconds=r['seconds']) for m,r in counts.items()]))}

Each block metadata file retains input hashes, full encoder fingerprint, payload hash and elapsed time. Large payloads are local/ignored, metadata and scientific results tracked. Resume verifies hashes/profile; cache keys resolve full-text hash plus profile through the explicit model-input aliases.

## Results

Top20 query-macro crossing mass, percentage points with uncorrected95% intervals:

{md(v[v.K.eq(20)][vcols])}

All K100, persistent inclusion/omission, tested-order inclusion frequency and pair-micro results are in coverage_summary.csv. Frequency averages separate target counterfactuals, not coherent-catalog Recall/nDCG. WANDS weights requested schedules including duplicates. ESCI retains seven original observations plus each missing distinct text once: frequency weights that observation multiset, not uniformly unique permutations; inclusion states cover all distinct target orders.

Incremental crossing, percentage points:

{md(d[['dataset','model','support','K','contrast','queries','pairs','delta_pp','CI_low_pp','CI_high_pp']])}

Original-seven always-in/out origins are separately exported with all identities. Already-crossing pairs are not counted as new. Query-cluster bootstrap uses10,000 draws, seed2026091701, uncorrected percentile intervals, conditional on fixed catalogs/encoders. It does not quantify training or annotation uncertainty.

All per-pair monotonicity checks pass: VI nondecreasing, persistent inclusion/omission nonincreasing. This follows structurally from nesting. A small last increment is not convergence or all-permutation invariance. Zero additional ESCI mass is valid, without inventing extra views.

## Numerical sensitivity

New native inference uses pinned tokenizer/weights, empty prefixes, FP16 model/FP32 pooling, original encoder and cached query vectors. Current transformers4.46.2 differs from4.55.4 in some historical metadata; original arrays/ranks are never regenerated. Original-seven ranks remain authoritative.

Each dataset/encoder audits16 hash-fixed C0 products against cached vectors and all cached queries. Forward audit CSVs show actual vector/score errors. The sensitivity table identifies added-crossing witnesses inside the largest observed audit error; this small audit is not a certified global error bound or replacement tolerance. Ranking decisions remain exact.

The WANDS/BGE 16-product probe matches cached vectors exactly, yielding a sampled error threshold of zero. Its “beyond audit error” count therefore does not establish numerical robustness. Separate single-input BGE forwards in the frozen XAI panel differ from saved native scores by up to {xq[xq.model.eq('bge_base')].fresh_fp16_native_error.max():.9g}; this illustrates why a zero in one batch/sample cannot certify all batch shapes or target representations. All coverage estimates remain conditional on the documented native execution profile.

{md(sens[sens.support.eq('full')&sens.K.eq(20)][['dataset','model','contrast','observed_audit_score_error','new_crossing_pairs','pairs_with_witness_beyond_audit_error','pairs_near_observed_error']])}

coverage_validation.json confirms original ranks, boundary identities and monotonicity. coverage_plan_validation.json verifies serializers, token lengths and whole-entry multiplicities. No resource block remains. Coverage plots/source data are supplementary.
""")
 from manuscript_reports import claims_and_map
 claims_and_map(put,cp,wline,eline)
 extras=[GRADED/'GRADED_METRIC_AUDIT.md',GRADED/'tables/rq2_compact_evidence.csv',GRADED/'tables/canonical_hybrid_costs.csv',GRADED/'tables/inherited_strategy_costs.csv',GRADED/'tables/canonical_hybrid_claim_verdicts.csv',OLD/'data/target_rank_precision_differences.csv']
 dump(HERE/'SUPPLEMENTAL_SOURCE_MANIFEST.json',[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size,purpose='read-only report/precision context') for p in extras])
 from polish_reports import polish
 for name in ['README.md','BOUNDARY_AUDIT.md','XAI_AUDIT.md','CONTROL_TRANSITIONS.md','PERMUTATION_COVERAGE.md','CLAIM_VERDICTS.md','CHAPTER6_EVIDENCE_MAP.md']:
  path=HERE/name;path.write_text(polish(path.read_text(encoding='utf8')),encoding='utf8')
 print('REPORTS COMPLETE',flush=True)
if __name__=='__main__':main()
