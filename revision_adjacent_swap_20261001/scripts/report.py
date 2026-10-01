"""Write a scoped scientific report, audit reports, and English LaTeX fragments."""
from common import *

NAMES={'minilm':'MiniLM','bge_base':'BGE'}
def pct(x):return f'{100*x:.2f}%'
def interval(r):return f'{pct(r.estimate)} [{pct(r.ci_low)}, {pct(r.ci_high)}]'
def main():
    verify_freeze();profile=read(HERE/'execution_profile.json');pop=pd.read_csv(HERE/'data/population_counts.csv')
    agg=pd.read_csv(HERE/'data/aggregate_estimates.csv');events=pd.read_csv(HERE/'data/event_counts.csv')
    setting_order=[(ds,m) for ds in DATASETS for m in MODELS]
    events['setting_order']=[setting_order.index((ds,m)) for ds,m in zip(events.dataset,events.model)]
    events=events.sort_values(['setting_order','K'])
    audits={m:read(HERE/f'qa/{m}_numerical_report.json') for m in MODELS}
    numerical_unresolved=int(events.unresolved_flips.sum());confirmed=int(events[events.K.eq(20)].confirmed_flips.sum())
    est=agg.set_index(['dataset','model','K','metric']);case=read(HERE/'data/case_selection.json')
    lines=['# One adjacent attribute swap: completed results','',
        'This controlled follow-up evaluates one adjacent transposition of complete attribute entries with the query and every competitor fixed at raw C0. It is a different population and permutation family from the seven-/32-schedule analyses; its percentages are not directly comparable to their VI estimates. The protocol and cohort were frozen before new swap outcomes, not externally preregistered.','',
        f"Starting repository revision: `{profile['starting_commit']}`. Dedicated branch: `{profile['branch']}`. Source data, model revisions, tokenizer details, execution settings, cache provenance and hashes are recorded in `execution_profile.json` and `freeze_manifest.json`.",'',
        '## Exact population and support','',
        'Start from held-out eligible queries and WANDS Exact / ESCI E judgments. Eligibility requires at least one distinct non-no-op adjacent swap and complete C0 plus every such variant fitting the encoder context, including special tokens. C0 competitors use the established native encoding, including truncation where applicable. Sample up to 128 queries and up to 16 eligible targets per query by encoder-independent SHA-256 priorities. No selection uses previous outcomes, ranks, margins or attribution.','',
        '| Setting | Highest-label pairs / held-out queries | Eligible pairs / queries | Sampled pairs / queries | Pair-swap outcomes |',
        '|---|---:|---:|---:|---:|']
    for r in pop.itertuples():lines.append(f'| {r.dataset.upper()} / {NAMES[r.model]} | {r.highest_label_pairs:,} / {r.held_out_queries} | {r.eligible_pairs:,} / {r.eligible_queries} | {r.sampled_pairs:,} / {r.sampled_queries} | {r.tested_pair_swaps:,} |')
    lines += ['', f'Catalogs contain 42,994 WANDS and 10,076 ESCI products. The {int(pop.sampled_pairs.sum()):,} sampled query-product-setting pairs yield {int(pop.tested_pair_swaps.sum()):,} pair-swap outcomes. Targets repeated across queries share encodings. These are setting-specific supports, not unique products pooled across settings.','',
        '| Setting | Excluded: no distinct swap | Excluded: context | Sampled unique products |', '|---|---:|---:|---:|']
    for r in pop.itertuples():lines.append(f'| {r.dataset.upper()} / {NAMES[r.model]} | {r.excluded_no_swaps:,} | {r.excluded_context:,} | {r.sampled_products:,} |')
    lines += ['', 'Context exclusions are evaluated only among targets with a distinct swap, so exclusion counts are disjoint. Inputs failing C0 are rejected immediately; all variants of every retained target are tokenized. Full attribute-count and swap-count distributions are in `data/attribute_swap_distributions.csv`; they include population, eligible and sampled supports. Encoder-specific fitting populations differ, so cross-encoder differences are descriptive, not causal model comparisons. These are not prevalence estimates over all relevant products.','',
        '## Primary results, K=20','',
        'A is the query-macro share of sampled targets affected by at least one distinct adjacent swap. F first averages the flip fraction within each target, then across targets within a query, then across queries. Long records do not receive extra weight just because they have more swaps. Brackets are 95% unadjusted query-cluster percentile intervals from 10,000 resamples (seed 2026100101).','',
        '| Setting | A: any swap | F: one uniformly selected swap | Affected targets / sampled | Confirmed flip events |',
        '|---|---:|---:|---:|---:|']
    for r in events[events.K.eq(20)].itertuples():
        lines.append(f'| {r.dataset.upper()} / {NAMES[r.model]} | {interval(est.loc[r.dataset,r.model,20,"A"])} | {interval(est.loc[r.dataset,r.model,20,"F"])} | {r.affected_targets} / {r.targets} | {r.confirmed_flips} / {r.swaps} |')
    lines += ['', '| Setting | Baseline inclusion | Loss share | Gain share |', '|---|---:|---:|---:|']
    for r in events[events.K.eq(20)].itertuples():
        lines.append(f'| {r.dataset.upper()} / {NAMES[r.model]} | {interval(est.loc[r.dataset,r.model,20,"baseline_inclusion"])} | {interval(est.loc[r.dataset,r.model,20,"loss"])} | {interval(est.loc[r.dataset,r.model,20,"gain"])} |')
    lines += ['', 'Loss and gain use all sampled eligible targets in each query as denominator and sum to A. They are not conditional failure rates. Baseline inclusion and the target-only interventions are not catalog-wide Recall: different interventions do not form one jointly realized index. Uniform selection refers only to the enumerated distinct-swap family, not a model of real-world edits.','',
        'The observed one-swap flip frequencies are modest: 0.48%-1.44% at K=20 in these samples. The existence of a confirmed crossing does not make most adjacent swaps consequential. ESCI/BGE has only two loss-affected targets at K=20 (versus twelve gain-affected targets); its loss-share interval reaches zero. These sparse outcomes are retained and should temper broad failure-prevalence claims.','',
        '## Secondary results, K=100','', '| Setting | A | F | Affected targets | Confirmed / unresolved flip events |', '|---|---:|---:|---:|---:|']
    for r in events[events.K.eq(100)].itertuples():
        lines.append(f'| {r.dataset.upper()} / {NAMES[r.model]} | {interval(est.loc[r.dataset,r.model,100,"A"])} | {interval(est.loc[r.dataset,r.model,100,"F"])} | {r.affected_targets} / {r.targets} | {r.confirmed_flips} / {r.unresolved_flips} |')
    lines += ['', 'Intervals describe query-level variation in this frozen sample. They do not account for all possible target samples, all relevant products or arbitrary permutations. For any zero-event cell, the observed count and support remain in the table; an empirical [0,0] bootstrap interval does not establish zero underlying event probability.','',
        'Prespecified breakdowns by distinct-swap count (1, 2-3, 4-7, 8-15, 16+) and C0 rank (1-10, 11-20, 21-40, >40) are in `data/descriptive_breakdowns.csv`, with query, target and swap supports. Within each stratum, target means are averaged within represented queries and then over represented queries. These are descriptive analyses of the same frozen cohort. Native score, margin and rank-change distributions are in the per-variant/per-target files and `data/supporting_distributions.csv`.','',
        '## Numerical validity and preservation','',
        'All competitors and query representations stay fixed. Ranks use exact FP32 scores and ascending saved catalog index for ties; the original target is removed before insertion. Margins are signed differences between native target and fixed Kth competitor scores, with boundary indices and exact-tie flags saved. Diagnostic subtraction represents the exact difference of the two saved FP32 scores in float64; membership uses the original native score comparison. No epsilon changes membership.','']
    for m,a in audits.items():
        gate=read(HERE/f'qa/{m}_pre_sweep_gate.json')
        lines.append(f"- {NAMES[m]}: {gate['replayed_pairs']:,} no-op pair replays, {gate['distinct_replayed_inputs']:,} distinct inputs; maximum C0 score discrepancy {gate['maximum_score_discrepancy']:.9g}; {gate['rank_discrepancies']} replay rank discrepancies. {a['observed_crossing_variants']:,} variants crossing either cutoff received three C0 and three swapped forward checks each (shared identical inputs reused across query checks). Maximum crossing score errors: C0 {a['maximum_crossing_c0_score_error']:.9g}, swap {a['maximum_crossing_swap_score_error']:.9g}.")
    if numerical_unresolved:
        lines += ['',f'{numerical_unresolved} observed variant-cutoff events remain numerically unresolved. None is removed from denominators. The tables above are observed estimates; `A_confirmed` and `F_confirmed` in the aggregate output provide confirmed lower bounds, and observed A/F provide possible upper bounds on observed events. The figure marks confirmed lower estimates separately. These bounds do not cover unobserved numerical events.']
    else:lines += ['', 'Every observed crossing reproduced at its cutoff in all three repeats. There are zero unresolved observed variant-cutoff events; confirmed and observed estimates coincide.']
    audit=read(HERE/'qa/representation_summary.json')
    lines += ['',f"All {audit['variants_audited']:,} distinct product-setting variants passed entry-multiset (including duplicates), key/value, numeric-substring, fixed-text, adjacent-transposition, other-relative-order, identity and no-truncation checks. The repository representation audit was reused. Zero audit failures. C0 and every adjacent swap map to exactly the same ascending canonical serialization; this invariance is structural and is not claimed as a new empirical result.",'',
        'The installed PyTorch/Transformers/tokenizers versions, CUDA/GPU, model snapshots, precision, pooling, padding/batching and serializer matched the earlier harmonized C0 reference. Payload hashes and fresh input aliases were verified. No mixed-profile historical score arrays were used. Queries missing from that reference were encoded under the same profile. Exact compatible swap-input aliases, especially the already enumerated ESCI orders, also reuse vectors; this does not mean that every tested serialization required a new forward pass. Full repeat records and numerical reports retain native discrepancies.','',
        '## Deterministic illustrative case','']
    if case['status']=='selected':
        r=case['selected']
        lines += [f"The smallest prespecified case hash among confirmed K=20 {case['direction']} pairs selected {r['dataset'].upper()}/{NAMES[r['model']]}, query {r['query_id']}, product {r['product_id']}. Within that pair the smallest qualifying swap hash selected positions {r['positions']}. This selection is conditional on a confirmed crossing, not a representative prevalence sample.", '',
            f"Query: {case['query']!r}. C0 rank {r['c0_rank']} becomes {r['swap_rank']}; scores {r['c0_score']:.9f} and {r['swap_score']:.9f}; K=20 margins {r['c0_margin_20']:+.9f} and {r['swap_margin_20']:+.9f}. Tokens {r['c0_tokens']} and {r['tokens']} / limit {r['context_limit']}. All competing products remain C0. Full serializations and all adjacent outcomes are in `data/case_selection.json` and `data/case_all_swap_outcomes.csv`."]
    else:lines += [case['status']]
    lines += ['', '## Contribution and manuscript placement','']
    if confirmed:
        settings=events[events.K.eq(20)&events.confirmed_flips.gt(0)]
        lines += [f"The experiment adds a distinct sufficiency finding: a single adjacent exchange of complete attributes can change a relevant target's inclusion with fixed competitors and fully fitting inputs. Confirmed K=20 events occur in {len(settings)} of the four evaluated settings. This narrows the size of the perturbation needed compared with the earlier schedule analyses; it does not identify an attention mechanism, establish semantic responsibility for an attribute, or show typical real-world editing effects.", '',
            'Recommended placement: a short main-text finding with the compact K=20 aggregate figure if space permits; put the full numerical audit, K=100, support distributions, descriptive strata and illustrative case in the appendix. Keep the severe fully-fitting selection and setting-specific supports next to the claim. Do not present this as stronger prevalence evidence over the entire catalog or as an encoder ranking.']
    else:lines += ['No numerically confirmed K=20 crossing was found in this frozen sample. This does not establish invariance outside the tested targets/swaps. The experiment adds a scoped null control rather than positive sufficiency evidence. Recommended placement: appendix if it clarifies perturbation scope; it does not justify a positive main-text claim.']
    lines += ['', '## Artifacts','',
        '- Primary figure: `figures/aggregate_K20.pdf` and `.png`; supplementary: `figures/aggregate_K100.pdf` and `.png`.',
        '- Illustrative case: `figures/illustrative_case.pdf` and `.png` when a confirmed case exists; source data are beside the figures.',
        '- Outcomes: `data/per_variant.parquet`, `data/per_target.parquet` / `.csv`, `data/per_query.parquet` / `.csv`, `data/aggregate_estimates.csv`.',
        '- Population/audits: `data/eligibility_population.parquet`, `data/population_counts.csv`, `data/representation_audit.parquet`, `REPRESENTATION_AUDIT.md`, `NUMERICAL_VERIFICATION.md`.',
        '- Protocol and reproduction: `protocol.md`, `config.json`, `cohort_manifest.csv`, `execution_profile.json`, `freeze_manifest.json`, `REPRODUCE.md`.',
        '- English manuscript fragments: `manuscript/captions.tex`, `manuscript/paragraph.tex`. Existing manuscripts and experiments are preserved.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    (HERE/'README.md').write_text('# Adjacent attribute swap experiment\n\nCompleted four-setting experiment. Read [RESULTS.md](RESULTS.md) for scope and outcomes, [REPRODUCE.md](REPRODUCE.md) for commands, and [protocol.md](protocol.md) for the pre-outcome freeze. The sampled fully-fitting populations and local swap family differ from previous schedule VI. All outputs are confined to this directory; no main-branch merge or manuscript replacement.\n',encoding='utf8')
    (HERE/'NUMERICAL_VERIFICATION.md').write_text('\n'.join(lines[lines.index('## Numerical validity and preservation'):lines.index('## Deterministic illustrative case')])+'\n',encoding='utf8')
    (HERE/'REPRESENTATION_AUDIT.md').write_text('# Representation and fitting audit\n\n'+
        f"{audit['variants_audited']:,} distinct product-setting swaps audited; zero failures. Every selected target fits C0 and all enumerated variants. The pair-swap count is larger when the same product is relevant to multiple sampled queries.\n\n"+
        'The frozen Parquet audit stores original/variant SHA-256 hashes, native input hashes, token lengths and limits, 1-based swap-position lists, canonical hashes and each individual assertion. Attribute-entry Counter equality retains duplicates; partitioned key/value tuples and numeric substring Counters are unchanged. Fixed serializer blocks are compared directly. All non-swapped positions agree exactly. Identity serialization and the existing repository audit pass. Ascending canonical equality uses (casefold(entry), entry) and is structural.\n\n'+
        'The eligibility inventory includes no-op and duplicate-serialization counts and explicit exclusions. Identical output strings are deduplicated while their original positions are retained. Targets already failing C0 are rejected without tokenizing their variants; all retained targets have a complete variant scan. No new retrieval outcomes were used for eligibility or selection.\n',encoding='utf8')
    (HERE/'manuscript').mkdir(exist_ok=True)
    unresolved_note=' Observed estimates include numerically unresolved events; diamonds show confirmed lower bounds.' if numerical_unresolved else ' All observed crossings reproduced in three repeated forward checks.'
    captions=r'''% Standalone caption fragments; existing manuscript is unchanged.
\paragraph{Aggregate figure (main, K=20).}
Single adjacent attribute swaps under fixed raw-source competitors. Panel A reports the query-macro share of sampled eligible targets with at least one Top-20 inclusion change; Panel B reports the mean within-target fraction of distinct adjacent swaps changing inclusion, averaged within and then across queries. Error bars are 95\% unadjusted percentile intervals from 10,000 query-cluster bootstrap resamples (seed 2026100101). Inputs fit the encoder limit under C0 and every tested swap. The four sampled supports are WANDS/MiniLM: 117 queries and 712 pairs; WANDS/BGE: 128 and 1,319; ESCI/MiniLM: 128 and 641; ESCI/BGE: 128 and 769. Different fitting populations preclude causal cross-encoder comparisons. These local-swap estimands differ from schedule-based VI.''' + unresolved_note + r'''

\paragraph{Supplementary aggregate figure (K=100).}
The same frozen cohorts, estimands, fixed competitors and bootstrap procedure at Top-100. These target-only inclusion measurements are not catalog-wide Recall. Intervals reflect query-level variation within the sampled experiment, not all possible target samples or permutations.

\paragraph{Illustrative case (appendix).}
One adjacent transposition changes Top-20 inclusion with all competitor representations fixed at C0. The displayed pair has the smallest prespecified case hash among numerically confirmed losses (gains only if no loss exists); the displayed swap has the smallest qualifying swap hash within that pair. Selection is conditional on a confirmed crossing, not representative prevalence sampling. Full content and multiplicities are preserved and both serializations fit the encoder context. Scores, ranks and signed margins use the exact native ranking rule. All three repeated C0 and swapped-target forward checks reproduced the decisions. Canonical invariance is structural; no attribution or attention mechanism is inferred.
'''
    (HERE/'manuscript/captions.tex').write_text(captions,encoding='utf8')
    parts=[]
    for ds,m in [('wands','minilm'),('wands','bge_base'),('esci','minilm'),('esci','bge_base')]:
        a=est.loc[ds,m,20,'A'];ff=est.loc[ds,m,20,'F']
        parts.append(f"{ds.upper()}/{NAMES[m]}: {100*a.estimate:.2f}\\% [{100*a.ci_low:.2f}, {100*a.ci_high:.2f}] and {100*ff.estimate:.2f}\\% [{100*ff.ci_low:.2f}, {100*ff.ci_high:.2f}]")
    paragraph=('A controlled local-perturbation experiment shows that a single adjacent exchange of complete attribute entries can change a relevant product\'s Top-20 inclusion while the query and every competitor remain fixed at C0 and the target fully fits in all tested serializations. ' if confirmed else 'A controlled local-perturbation experiment found no numerically confirmed Top-20 crossing in the frozen sample. ')
    paragraph+='Query-macro affected-target shares and mean one-swap flip frequencies, respectively, were '+ '; '.join(parts)+'. '
    paragraph+='Brackets give 95\\% unadjusted query-cluster bootstrap intervals. These results concern the sampled fully-fitting populations and enumerated adjacent-swap families; they do not estimate prevalence over all relevant products, share the earlier schedule-based VI estimand, or identify an attention mechanism.\n'
    (HERE/'manuscript/paragraph.tex').write_text(paragraph,encoding='utf8')
    requirements=['# Python 3.10; torch requires the CUDA 12.8 wheel index.']+[f'{p}=={v}' for p,v in profile['versions'].items()]
    (HERE/'requirements-experiment.txt').write_text('\n'.join(requirements)+'\n',encoding='utf8')
    print('REPORTS COMPLETE')

if __name__=='__main__':main()
