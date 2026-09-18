"""Verify copied evidence, exact cancellation counts, and vector deliverables."""
import csv
import importlib.metadata
import platform
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pypdf import PdfReader
from io_common import ROOT, BASE, HERE, DATA, FIG, STEMS, MODELS, dump, load, sha, source

def csvrows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def main():
    preserved = load(DATA / 'preserved_files.json')['files']
    for item in preserved:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    copies = ['rq1_target_fully_fitting_plot_data.csv', 'rq2_membership_plot_data.csv', 'rq2_cancellation_companion.csv']
    for name in copies:
        assert (BASE / 'data' / name).read_bytes() == (DATA / name).read_bytes()
    for name, prefix, ks in [(copies[0], 'rq1_main_top', [20]), (copies[1], 'rq2_top', [20, 100])]:
        original = csvrows(BASE / 'data' / name)
        for k in ks:
            assert csvrows(DATA / f'{prefix}{k}.csv') == [r for r in original if r['K'] == str(k)]
    # Confirm every small-table numerator independently from existing per-query
    # integer membership counts. This is a check, not a new bootstrap or estimate.
    perquery_path = BASE / 'data/membership_changes_per_query.csv'
    per = pd.read_csv(perquery_path)
    per = per[(per.population == 'primary_existing_evaluation') & (per.reference == 'C0') & per.alternative.isin(STEMS[1:])].copy()
    keys = ['dataset', 'model', 'K', 'alternative']
    assert not per.duplicated(keys + ['query_id']).any()
    per['verified_exact'] = (per.newly_included == per.newly_omitted) & (per.newly_included > 0)
    per['verified_changed'] = (per.newly_included + per.newly_omitted) > 0
    assert np.array_equal(per.verified_exact, per.exact_zero_net_with_changes)
    assert (per.loc[per.verified_exact, 'net_delta'] == 0).all()
    got = per.groupby(keys, sort=False).agg(numerator=('verified_exact', 'sum'), eligible=('query_id', 'size'), changed=('verified_changed', 'sum'))
    app = pd.read_csv(DATA / 'cancellation_appendix_all_comparisons.csv').set_index(keys)
    assert len(got) == len(app) == 72
    got = got.loc[app.index]
    assert np.array_equal(got.numerator, app.numerator_exact_nonzero_cancellation)
    assert np.array_equal(got.eligible, app.denominator_all_eligible)
    assert np.array_equal(got.changed, app.denominator_changed_queries)
    assert np.allclose(app.all_eligible_percent, 100*got.numerator/got.eligible, rtol=0, atol=1e-13)
    assert np.allclose(app.changed_queries_percent, 100*got.numerator/got.changed, rtol=0, atol=1e-13)
    main_table = pd.read_csv(DATA / 'cancellation_main_top20.csv')
    for r in main_table.itertuples():
        g = got.loc[(r.dataset, r.model, 20)]
        v = 100*g.numerator/g.eligible
        assert len(g) == 6 and g.eligible.eq(r.eligible_queries).all()
        assert np.isclose(v.min(), r.percent_min) and np.isclose(v.max(), r.percent_max)
    rq1 = pd.read_csv(DATA / copies[0])
    assert np.allclose(rq1[['Always', 'VI', 'Never']].sum(axis=1), 1, rtol=0, atol=1e-14)
    for ds in ['wands', 'esci']:
        for k in [20, 100]:
            g = rq1[(rq1.dataset == ds) & (rq1.model == 'gte_modernbert') & (rq1.K == k)]
            for col in ['VI', 'VI_ci_low', 'VI_ci_high', 'eligible_queries', 'highest_relevance_pairs']:
                assert g[col].nunique() == 1
    rq2 = pd.read_csv(DATA / copies[1])
    assert np.allclose(rq2.gain_fraction-rq2.loss_fraction, rq2.net_delta, rtol=0, atol=1e-14)
    assert (100*rq2[['gain_fraction', 'loss_fraction', 'net_delta_ci_low', 'net_delta_ci_high']].abs() <= 7).all().all()
    case = load(DATA / 'figure1_case_audit.json')
    assert case['status'] == 'PASS' and case['heldout'] and case['old_target_excluded']
    assert case['effective_native_token_limit'] == case['pinned_model_max_position_embeddings'] == 8192
    assert all(r['tokens_including_special'] == 1059 and r['special_tokens'] == 2 for r in case['tokens'])
    assert [r['saved_target_only_rank'] for r in case['ranks']] == [11, 468, 123, 301, 677, 435, 939]
    for item in case['sources']:
        assert sha(ROOT / item['path']) == item['sha256'], ('Case source changed', item['path'])
    geometry = load(HERE / 'qa/figure_geometry.json')
    assert len(geometry) == len(list(FIG.glob('*.pdf'))) == len(list(FIG.glob('*.svg'))) == 7
    pdf_checks = []
    for g in geometry:
        path = FIG / (g['name'] + '.pdf')
        pdf = PdfReader(path)
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        assert len(page.images) == 0, ('Raster image', path.name)
        width, height = float(page.mediabox.width)/72, float(page.mediabox.height)/72
        assert abs(width-g['width_inches']) < 1e-6 and abs(height-g['height_inches']) < 1e-6
        assert width in [3.45, 7.1] and g['minimum_font_pt'] >= 8 and not g['text_outside_page']
        # Relative ASCII filenames avoid legacy Windows Poppler path conversion.
        fonts = subprocess.check_output(['pdffonts', path.name], cwd=FIG).decode('utf-8')
        fontlines = fonts.splitlines()[2:]
        assert fontlines and all('yes yes yes' in line for line in fontlines), fonts
        assert all('TrueType' in line for line in fontlines), fonts
        svg = ET.parse(path.with_suffix('.svg')).getroot()
        ns = '{http://www.w3.org/2000/svg}'
        assert not list(svg.iter(ns+'image')) and svg.find(ns+'title') is not None and svg.find(ns+'desc') is not None
        assert svg.get('role') == 'img'
        pdf_checks.append(dict(name=g['name'], pages=1, width_inches=width, height_inches=height,
                               embedded_vector_fonts=True, raster_images=0, minimum_font_pt=g['minimum_font_pt'],
                               sha256=sha(path)))
    schematic = next(x for x in geometry if x['name'] == 'figure2_intervention_and_states')
    reduction = 100 * (1-schematic['height_inches']/4.8)
    assert 25 <= reduction <= 35
    dump(HERE / 'qa/validation_report.json', dict(status='PASS', preserved_files_checked=len(preserved),
          original_plot_csvs_byte_identical=3, filtered_csv_original_decimal_strings_preserved=True,
          existing_per_query_rows_checked=len(per), cancellation_comparisons_checked=72,
          exact_cancellation_criterion='newly_included == newly_omitted > 0',
          per_query_source=source(perquery_path), schematic_height_reduction_percent=reduction,
          case_audit='PASS', figures=pdf_checks))
    dump(HERE / 'qa/runtime.json', {'python': platform.python_version(),
         'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'matplotlib', 'pypdf']},
         'token_audit_runtime': 'phase2/.venv/Scripts/python.exe; transformers 4.55.4; pinned local tokenizer; no model inference'})
    print(f'PASS: {len(preserved)} preserved files; 72 cancellation comparisons from {len(per):,} query rows; seven one-page vector figures.')

if __name__ == '__main__':
    main()
