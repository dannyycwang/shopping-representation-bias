"""Write caption sources, accessible descriptions, and insertion-only LaTeX."""
import pandas as pd
from io_common import HERE, DATA, NAMES, dump, load

CI = ('Intervals reuse the existing 95% paired query-cluster percentile bootstrap endpoints '
      '(10,000 draws; seed 2026091701). Products and comparisons from a sampled query remain '
      'together. Intervals are descriptive and uncorrected for multiple comparisons; inclusion '
      'of zero does not establish equivalence.')
POP = ('WANDS uses 308 eligible queries and 21,299 Exact pairs from its existing held-out split '
       '(384 requested queries). ESCI uses 499 eligible queries and 4,434 E pairs from its '
       'existing 500-query evaluation sample, not a newly held-out sample.')
FAMILY = ('The fixed finite family is C0 (source order), C1 (reverse attribute order), and '
          'C2s1-C2s5 (the five existing product-dependent permutations; seeds 20260911-20260915).')
CAPTIONS = {
 'figure2_intervention_and_states': {
  'short': 'Attribute-order interventions and candidate-inclusion states over a finite family of orderings.',
  'full': ('Schematic illustration. (A) Catalog-wide intervention permutes intact attribute entries '
           'for every product. Target-only intervention replaces the target\'s old C0 entry while '
           'every competitor retains raw C0. The query, encoder, catalog membership, non-attribute '
           'text, section placement, entry contents, and entry multiplicities stay fixed. Each target '
           'defines a separate counterfactual index, with its former entry removed. Letters denote '
           'complete entries, not characters or fields that may be deleted. (B) The three schematic '
           'rows 111, 101, and 000 denote persistent inclusion, cutoff crossing, and persistent omission, '
           'respectively; 1 means rank ≤ K and 0 means rank > K. Three columns illustrate the definitions '
           'and do not replace the seven-order experimental family or depict empirical frequencies.'),
  'description': ('Two intervention diagrams each contain a target and two other products. Colored '
                  'letter blocks denote complete attribute entries. Every product is reordered in the '
                  'catalog-wide panel; only the target is reordered in the Target-only intervention '
                  'panel, and both competitors stay at raw C0. Below, binary rows 111, 101, and 000 '
                  'show persistent inclusion, cutoff crossing, and persistent omission. One denotes '
                  'rank at most K. The diagram contains no measured proportions.'),
  'placement': 'Section 3, adjacent to the intervention and candidate-inclusion state definitions.', 'width': 'double'},
 'rq1_vi_top20_main': {
  'short': 'Target-only inclusion instability for full and fully-fitting target populations, with fixed competitors.',
  'full': ('At K=20, dots and triangles show query-macro VI for full and fully-fitting highest-relevance '
           'target populations, respectively. VI is the fraction of targets whose inclusion crosses '
           'the cutoff within the seven-order family, averaged within query and then across eligible '
           'queries. Fitting requires every complete serialized input, including special tokens, to '
           'fit under all seven schedules at that encoder\'s cap (MiniLM 256, BGE 512, GTE 8192). The '
           'full raw C0 competitor catalog is fixed: 42,994 WANDS or 10,076 ESCI products before target '
           'replacement. Non-fitting competitors are retained. Each target is replaced independently '
           'in its own index. Values and q/pair denominators appear beside every point. GTE full and '
           'fitting supports coincide, so identical estimates and intervals are separated vertically. '
           'The WANDS and ESCI panels use different horizontal ranges. Differences between full and '
           'fitting populations are descriptive, not causal truncation effects. Encoder-specific fitting '
           'supports differ; these comparisons do not identify intrinsic cross-encoder sensitivity. ' + POP + ' ' + CI),
  'description': ('Two panels show Top-20 VI with horizontal 95% intervals for WANDS and ESCI, each '
                  'with MiniLM, BGE, and GTE. Filled circles denote full targets, open triangles fully-fitting '
                  'targets. Each estimate has a numeric VI percentage and query/pair count. Full/fitting '
                  'VI percentages are WANDS MiniLM 19.45/10.59, BGE 10.83/7.16, GTE 14.55/14.55; '
                  'ESCI MiniLM 2.91/3.20, BGE 2.42/2.55, GTE 5.88/5.88. GTE points are separated '
                  'vertically despite identical support and intervals. All competitors remain raw C0.'),
  'placement': 'Results / RQ1, main text.', 'width': 'double'},
 'rq1_states_top20_top100_appendix': {
  'short': 'Persistent inclusion, cutoff crossing, and persistent omission for full and fully-fitting targets.',
  'full': ('Four panels cross WANDS/ESCI with K=20/100. Always, VI, and Never are query-macro target '
           'shares on identical support and sum to 100%. Every VI value appears in a separate numeric '
           'column, including small segments; q/pair counts accompany every bar. Fitting means fully-fitting '
           'under all seven complete inputs at each encoder\'s cap, including special tokens. The full '
           'raw C0 competitor catalog remains fixed, including non-fitting competitors. Full and fitting '
           'GTE supports coincide. Target-only interventions define separate indexes for separate targets; '
           'their averaged inclusion is not Recall from one shared index. Full/fitting differences are '
           'descriptive population differences, not causal truncation effects or proof of intrinsic '
           'cross-model sensitivity. ' + POP + ' ' + FAMILY),
  'description': ('Four stacked-bar panels compare two target supports for each encoder at Top-20 '
                  'and Top-100 on WANDS and ESCI. Teal is Always, rust is VI, and gray is Never. A numeric '
                  'VI column makes every cutoff-crossing proportion visible, including WANDS BGE fully-fitting '
                  '7.16 percent at Top-20. Each row also lists eligible queries and highest-relevance pairs. '
                  'The complete raw C0 competitor catalog stays fixed.'),
  'placement': 'Appendix, with a cross-reference from Results / RQ1.', 'width': 'double'},
 'rq2_cancellation_top20_table': {
  'short': 'Exact within-query cancellation across six attribute-order comparisons at Top-20.',
  'full': ('Exact within-query cancellation means newly included = newly omitted > 0, using integer '
           'counts of highest-relevance products for a query. Thus its Recall is exactly unchanged '
           'although its retrieved relevant-product identities change. Each range is the minimum and '
           'maximum proportion of all eligible queries across C1 and C2s1-C2s5 versus C0, under '
           'catalog-wide interventions at K=20. WANDS denominators are 308 and ESCI denominators are '
           '499 for every encoder and comparison. These are six-comparison min-max ranges, not '
           'confidence intervals. The same query may appear in several comparisons; counts are never '
           'summed into unique-query totals. The gains/losses figure shows aggregate cancellation; '
           'this table directly demonstrates within-query substitutions with exactly unchanged Recall. '
           'The appendix reports every numerator, eligible-query denominator, changed-query denominator, '
           'and conditional percentage at K=20 and K=100.'),
  'description': ('A six-row table lists dataset, encoder, eligible queries, and the Top-20 proportion '
                  'of queries with equal positive gains and losses. WANDS ranges are MiniLM 16.88-21.75, '
                  'BGE 21.75-25.32, and GTE 19.81-26.30 percent, with denominator 308. ESCI ranges are '
                  'MiniLM 1.40-2.81, BGE 2.40-3.21, and GTE 3.01-4.21 percent, with denominator 499. '
                  'Ranges span six existing comparisons and are not confidence intervals.'),
  'placement': 'Results / RQ2, beside the Top-20 membership figure. Prefer the supplied native LaTeX table.', 'width': 'single'},
 'figure1_teal_chair_candidate': {
  'short': 'Example of target-only attribute-order sensitivity: the same fully retained product input ranks 11th or 939th, crossing the Top-20 cutoff.',
  'full': ('Post-hoc illustrative case, preselected in the revision request: WANDS query 409, '
           '\"teal chair\", and product 24318, with the existing Exact relevance judgment and GTE '
           '(gte_modernbert). Query 409 belongs to the existing held-out split. Target-only ranks '
           'are 11 for C0 and 939 for C2s5; the catalog-wide C2s5 rank of 956 is not used. The old '
           'target entry is removed and its alternative replaces it, while all 42,993 competitors '
           'stay at raw C0. The cards show only a display excerpt: the first four C0 attribute entries '
           'in each condition\'s relative order. Identical colors identify identical complete entries; '
           'line wrapping is for display. All 104 attribute entries with their exact multiplicities, and '
           'all fixed title, class, category, description, separators, and section order were preserved '
           'and encoded. Every one of the seven existing inputs is 1,059 tokens including two special '
           'tokens, below GTE\'s actual 8,192-token limit. The full source record includes other color '
           'entries that are not shown in this excerpt; the judgment is inherited without relabeling. '
           'This is a comparison of representations, not a temporal deployment event, a typical-product '
           'claim, or an estimate of general effect size. Results support population prevalence and '
           'fully-fitting analyses. This candidate does not replace the current Introduction figure.'),
  'description': ('Two independent cards compare the same Exact-labeled WANDS product for query teal '
                  'chair with GTE. C0 has rank 11, inside Top-20; C2s5 has rank 939, outside Top-20. '
                  'The cards contain the same four complete attribute entries with matching colors: '
                  'seat height 19.5, estimated setup time 20, design armchair, and upholstery color teal '
                  'faux leather. Their order changes. The display is an excerpt of 104 entries; both '
                  'conditions encode the complete 1,059-token record. The figure is labeled post-hoc '
                  'illustrative case and contains no deployment timeline or product photograph.'),
  'placement': 'Candidate for Introduction Figure 1 after author review; keep the existing turquoise-chair figure and manuscript reference unchanged.', 'width': 'single'}
}
for k, suffix in [(20, 'main'), (100, 'appendix')]:
    CAPTIONS[f'rq2_membership_top{k}_{suffix}'] = {
      'short': f'Relevant-product gains and losses under attribute permutations at Top-{k}. Diamonds show net Recall changes with 95% query-bootstrap intervals.',
      'full': (f'At K={k}, six panels cross WANDS/ESCI with MiniLM/BGE/GTE. Every panel retains all '
               'six catalog-wide attribute-order comparisons C1 and C2s1-C2s5 against raw C0. '
               'For each query, newly included and newly omitted highest-relevance products are '
               'divided by its number of highest-relevance judgments, then macro-averaged across '
               'the same eligible queries. Losses extend left and gains right, in relevant-product '
               'share (%). Black diamonds and intervals show net Recall change in percentage points, '
               'equal to gained share minus lost share. All panels and both cutoff figures share '
               'the -7 to +7 scale. This figure displays aggregate cancellation; it does not alone '
               'establish exact cancellation within individual queries. The companion table supplies '
               'that integer-count evidence. ' + POP + ' ' + CI),
      'description': (f'Six Top-{k} panels cross two datasets with three encoders. Each has six horizontal '
                      'bar pairs, C1 and C2s1 through C2s5 versus C0. Rust bars extend left for omitted '
                      'relevant-product shares and teal bars extend right for included shares, in percent. '
                      'Black diamonds show net Recall changes in percentage points with paired 95% '
                      'intervals. Every panel uses the same minus seven to plus seven scale. This is '
                      'catalog-wide membership evidence; exact within-query cancellation is reported '
                      'separately in the companion table.'),
      'placement': 'Results / RQ2, main text.' if k == 20 else 'Appendix, with a cross-reference from Results / RQ2.', 'width': 'double'}

ORDER = ['figure1_teal_chair_candidate', 'figure2_intervention_and_states', 'rq1_vi_top20_main',
         'rq1_states_top20_top100_appendix', 'rq2_membership_top20_main', 'rq2_membership_top100_appendix', 'rq2_cancellation_top20_table']

def tex(s):
    return s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('≤', r'$\leq$')

def main():
    dump(DATA / 'captions.json', CAPTIONS)
    lines = ['# Captions, accessible descriptions, and placement\n',
             'Short captions may be used with the shared setup below. Full captions are self-contained.\n',
             '## Shared setup\n', POP + ' ' + FAMILY + ' ' + CI + '\n',
             'RQ1 uses Target-only intervention against fixed raw C0 competitors. RQ2 uses catalog-wide intervention. No new statistical estimates or interval endpoints were computed.\n']
    snippets = ['% Candidate insertions only. This file is not included by the manuscript.',
                '% Requires graphicx. Paths assume compilation from paper_www2027/.',
                '% Short captions below require the shared setup in SETUP_SNIPPET.tex.',
                '% Full standalone caption alternatives are in CAPTIONS_AND_ACCESSIBILITY.md.']
    for name in ORDER:
        c = CAPTIONS[name]
        lines += [f'## {name}\n', '**Short caption.** ' + c['short'] + '\n', '**Full caption.** ' + c['full'] + '\n',
                  '**ACM Description.** ' + c['description'] + '\n', '**Suggested placement.** ' + c['placement'] + '\n']
        env = 'figure*' if c['width'] == 'double' else 'figure'
        width = r'\textwidth' if c['width'] == 'double' else r'\columnwidth'
        snippets += ['', '% ' + c['placement'], f'\\begin{{{env}}}[t]', r'\centering',
                     '\\includegraphics[width=' + width + ']{../revision_evidence_20260917/figure_revision/figures/' + name + '.pdf}',
                     r'\caption{' + tex(c['short']) + '}', r'\Description{' + tex(c['description']) + '}',
                     r'\label{fig:revision-' + name.replace('_', '-') + '}', f'\\end{{{env}}}']
    (HERE / 'CAPTIONS_AND_ACCESSIBILITY.md').write_text('\n'.join(lines), encoding='utf-8')
    (HERE / 'LATEX_INSERTIONS.tex').write_text('\n'.join(snippets) + '\n', encoding='utf-8')
    setup = (POP + ' ' + FAMILY + ' ' + CI + '\n\n' +
             'RQ1 uses Target-only intervention: each target replaces its old entry independently, '
             'with the full raw C0 competitor catalog fixed. Fitting requires all seven serialized '
             'inputs including special tokens to fit at the encoder cap (256/512/8192 for MiniLM/BGE/GTE). '
             'Non-fitting competitors remain in the catalog. Full/fitting differences are descriptive '
             'population differences, not causal truncation effects or intrinsic cross-model sensitivity.\n\n' +
             'RQ2 uses catalog-wide interventions and reports query-macro gains/losses divided by each '
             'query\'s full highest-relevance set. Gains/losses are shares in percent; their difference '
             'is net Recall change in percentage points. Aggregate cancellation alone does not establish '
             'within-query substitutions. The companion table uses exact integer equality of positive '
             'gains and losses, with every eligible query in the denominator. Ranges span all six '
             'comparisons, not confidence intervals, and repeated queries are not summed across comparisons.\n\n' +
             'Figure 1 is a post-hoc illustrative case, not a typical-product or population-magnitude '
             'claim. Figure 2 is schematic; its three illustrative order columns are not measured frequencies.')
    (HERE / 'SETUP_SNIPPET.tex').write_text('% Optional setup prose; not inserted into the manuscript automatically.\n' + tex(setup) + '\n', encoding='utf-8')
    # Native LaTeX alternative for the small main table.
    table = pd.read_csv(DATA / 'cancellation_main_top20.csv')
    t = [r'\begin{table}[t]', r'\centering', r'\caption{' + tex(CAPTIONS['rq2_cancellation_top20_table']['short']) + '}',
         r'\label{tab:revision-cancellation}', r'\small', r'\begin{tabular}{llrr}', r'\toprule',
         r'Dataset & Encoder & Eligible $q$ & Queries (\%) \\', r'\midrule']
    for i, r in table.iterrows():
        if i == 3:
            t.append(r'\midrule')
        t.append(f'{r.dataset.upper()} & {NAMES[r.model]} & {r.eligible_queries} & {r.percent_min:.2f}--{r.percent_max:.2f}' + r' \\')
    t += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
          r'\begin{minipage}{\columnwidth}\small',
          'Newly included = newly omitted $>0$. Six-comparison min--max ranges, not confidence intervals; denominators include all eligible queries. Counts are not summed across comparisons.',
          r'\end{minipage}', r'\end{table}']
    (HERE / 'CANCELLATION_MAIN_TABLE.tex').write_text('\n'.join(t) + '\n', encoding='utf-8')
    app = pd.read_csv(DATA / 'cancellation_appendix_all_comparisons.csv')
    md = ['# Exact within-query cancellation: all comparisons\n',
          'Criterion: newly included = newly omitted > 0. Each row is a distinct C0-relative catalog-wide comparison. Proportions are descriptive. Queries can recur across rows; do not sum numerators into unique-query counts.\n',
          '| Dataset | Encoder | K | Alternative | Exact / eligible | % eligible | Exact / changed | % changed |',
          '|---|---|---:|---|---:|---:|---:|---:|']
    tx = ['% Requires longtable and booktabs; standalone appendix table source.', r'\begin{longtable}{llrlrrrr}',
          r'\caption{Exact within-query cancellation for every existing C0-relative comparison.}\\', r'\toprule',
          r'Dataset & Encoder & $K$ & Order & Exact/eligible & \% & Exact/changed & \% \\', r'\midrule', r'\endfirsthead',
          r'\toprule Dataset & Encoder & $K$ & Order & Exact/eligible & \% & Exact/changed & \% \\ \midrule', r'\endhead']
    for _, r in app.iterrows():
        a = f'{int(r.numerator_exact_nonzero_cancellation)}/{int(r.denominator_all_eligible)}'
        b = f'{int(r.numerator_exact_nonzero_cancellation)}/{int(r.denominator_changed_queries)}'
        values = [r.dataset.upper(), NAMES[r.model], str(r.K), r.alternative, a, f'{r.all_eligible_percent:.2f}', b, f'{r.changed_queries_percent:.2f}']
        md.append('| ' + ' | '.join(values) + ' |')
        tx.append(' & '.join(values) + r' \\')
    tx += [r'\bottomrule', r'\end{longtable}']
    (HERE / 'CANCELLATION_APPENDIX.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (HERE / 'CANCELLATION_APPENDIX.tex').write_text('\n'.join(tx) + '\n', encoding='utf-8')
    print('Wrote seven caption/description records, LaTeX insertions, and six-row / 72-row cancellation tables.')

if __name__ == '__main__':
    main()
