"""Freeze old-file hashes, copy verified estimates, and format cancellation counts.

No statistics are re-estimated. Ranges are arithmetic summaries of six existing
comparison-specific integer counts, never sums of query counts.
"""
import shutil
import subprocess
import csv
import numpy as np
import pandas as pd
from io_common import BASE, ROOT, HERE, DATA, MODELS, NAMES, dump, load, source, sha

SOURCES = ['rq1_target_fully_fitting_plot_data.csv', 'rq2_membership_plot_data.csv', 'rq2_cancellation_companion.csv']

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    snapshot = DATA / 'preserved_files.json'
    if not snapshot.exists():
        tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', 'revision_evidence_20260917'], cwd=ROOT).decode('utf-8').split('\0')
        paths = {ROOT / p for p in tracked if p and not p.startswith('revision_evidence_20260917/figure_revision/')}
        # Preserve the user's existing manuscript, original figure, and build state.
        for directory in [ROOT / 'paper_www2027', ROOT / 'phase3/scripts']:
            paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        dump(snapshot, {'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                        'files': [source(p) for p in sorted(paths)]})
    for item in load(snapshot)['files']:
        assert sha(ROOT / item['path']) == item['sha256'], ('Preserved file changed', item['path'])
    for name in SOURCES:
        shutil.copyfile(BASE / 'data' / name, DATA / name)
    fonts = HERE / 'assets/fonts'
    fonts.mkdir(parents=True, exist_ok=True)
    for p in (BASE / 'assets/fonts').glob('*.ttf'):
        shutil.copyfile(p, fonts / p.name)
    rq1 = pd.read_csv(DATA / SOURCES[0])
    rq2 = pd.read_csv(DATA / SOURCES[1])
    assert len(rq1) == 24 and len(rq2) == 72
    assert rq1.aggregation.eq('query_macro').all() and rq2.aggregation.eq('query_macro').all()
    assert rq1.intervention.eq('target_only').all() and rq2.intervention.eq('catalog_wide').all()
    # Retain the original decimal strings in filtered plotting CSVs, including CIs.
    for name, prefix, cutoffs in [(SOURCES[0], 'rq1_main_top', [20]), (SOURCES[1], 'rq2_top', [20, 100])]:
        with (DATA / name).open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fields, original_rows = reader.fieldnames, list(reader)
        for k in cutoffs:
            with (DATA / f'{prefix}{k}.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(r for r in original_rows if r['K'] == str(k))
    c = pd.read_csv(DATA / SOURCES[2])
    assert len(c) == 72 and not c.duplicated(['dataset', 'model', 'K', 'alternative']).any()
    assert (c.eligible_queries == c.cancellation_all_denominator).all()
    assert (c.queries_with_any_change == c.cancellation_conditional_denominator).all()
    numerator = c.queries_with_exact_zero_net_and_changes
    assert np.allclose(numerator / c.eligible_queries, c.within_query_cancellation_fraction, rtol=0, atol=1e-15)
    assert np.allclose(numerator / c.queries_with_any_change, c.within_query_cancellation_conditional, rtol=0, atol=1e-15)
    appendix = c[['dataset', 'model', 'K', 'alternative']].copy()
    appendix['numerator_exact_nonzero_cancellation'] = numerator
    appendix['denominator_all_eligible'] = c.eligible_queries
    appendix['all_eligible_percent'] = 100 * numerator / c.eligible_queries
    appendix['denominator_changed_queries'] = c.queries_with_any_change
    appendix['changed_queries_percent'] = 100 * numerator / c.queries_with_any_change
    appendix.to_csv(DATA / 'cancellation_appendix_all_comparisons.csv', index=False)
    rows = []
    for ds in ['wands', 'esci']:
        for model in MODELS:
            g = appendix[(appendix.dataset == ds) & (appendix.model == model) & (appendix.K == 20)]
            assert len(g) == 6
            assert g.denominator_all_eligible.nunique() == 1
            rows.append(dict(dataset=ds, model=model, K=20, comparisons=6,
                             eligible_queries=int(g.denominator_all_eligible.iloc[0]),
                             numerator_min=int(g.numerator_exact_nonzero_cancellation.min()),
                             numerator_max=int(g.numerator_exact_nonzero_cancellation.max()),
                             percent_min=float(g.all_eligible_percent.min()), percent_max=float(g.all_eligible_percent.max())))
    pd.DataFrame(rows).to_csv(DATA / 'cancellation_main_top20.csv', index=False)
    dump(DATA / 'schematic.json', {
        'schematic_only': True, 'entries_are_indivisible': True,
        'raw': [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'J']],
        'catalog_wide': [['C', 'A', 'B'], ['F', 'D', 'E'], ['J', 'G', 'H']],
        'target_only': [['C', 'A', 'B'], ['D', 'E', 'F'], ['G', 'H', 'J']],
        'states': {'Persistent inclusion': [1, 1, 1], 'Cutoff crossing': [1, 0, 1], 'Persistent omission': [0, 0, 0]}})
    dump(DATA / 'plot_sources.json', {'source_files': [source(BASE / 'data' / n) for n in SOURCES],
                                    'statistics': 'Existing query-macro estimates and interval endpoints reused without recomputation.'})
    print('Copied 24 RQ1 rows, 72 RQ2 rows; derived 6 main cancellation ranges and 72 appendix rows.')

if __name__ == '__main__':
    main()
