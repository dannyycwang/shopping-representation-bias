"""Validate the single-case artifacts without rerunning model inference."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import torch
from pypdf import PdfReader


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    out, repo = args.results.resolve(), args.repo.resolve()
    source = Path(__file__).resolve().parent
    checks = []

    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    report = read(out / 'run_report.json')
    fixture = read(source / 'case_inputs.json')
    preparatory = read(source / 'PREPARATION_CHECKS.json')
    check('official_captum_and_contrast', report['captum'] == '0.9.0'
          and report['method']['implementation'] == 'captum.attr.IntegratedGradients'
          and report['contrast'] == 'C1 minus C0')
    check('original_frozen_input_hashes', read(out / 'original_input_hash_checks.json')['passed'])
    check('existing_dependencies_unchanged', all(report['environment']['existing_dependencies_unchanged'].values()))
    check('four_runs_passed', len(report['checks']) == 4 and report['status'] == 'passed')
    tokens = rows(out / 'token_attributions.csv')
    entries = rows(out / 'entry_attributions.csv')
    aligned = rows(out / 'aligned_entry_changes.csv')
    attr_ids = [f'attr_{i:03d}' for i in range(10)]
    sequences = rows(out / 'attribute_sequences.csv')
    check('exact_C0_C1_attribute_occurrence_orders',
          [r['entry_id'] for r in sequences if r['schedule'] == 'C0'] == attr_ids
          and [r['entry_id'] for r in sequences if r['schedule'] == 'C1'] == attr_ids[::-1])
    full_texts = []
    completeness = []
    for schedule in ('C0', 'C1'):
        text = (out / (schedule + '_input.txt')).read_text(encoding='utf-8')
        full_texts.append(text)
        expected = next(x for x in preparatory['inputs'] if x['schedule'] == schedule)
        check(schedule + '_frozen_serialization', sha(out / (schedule + '_input.txt')) == expected['text_sha256'])
        encoded = read(out / (schedule + '_tokenized_input.json'))
        check(schedule + '_139_tokens_masks_and_positions',
              len(encoded['inputs']['input_ids'][0]) == 139
              and encoded['inputs']['attention_mask'][0] == [1] * 139
              and encoded['inputs']['position_ids'][0] == list(range(139))
              and encoded['inputs']['token_type_ids'][0] == [0] * 139)
        native = next(r for r in fixture['native_records'] if r['schedule'] == schedule)
        for baseline in ('zero', 'pad'):
            key = schedule + '_' + baseline
            qa = read(out / (key + '_qa.json'))
            tr = [r for r in tokens if r['schedule'] == schedule and r['baseline'] == baseline]
            er = [r for r in entries if r['schedule'] == schedule and r['baseline'] == baseline]
            values = [float(r['attribution']) for r in tr]
            check(key + '_full_tokens_and_attributes', len(tr) == 139 and
                  [int(r['token_index']) for r in tr] == list(range(139)) and
                  sorted(r['entry_id'] for r in er if r['entry_id'].startswith('attr_')) == attr_ids)
            mapping_ok = True
            for row in tr:
                i = int(row['token_index'])
                a, b = encoded['offsets'][i]
                if row['entry_id'].startswith('attr_'):
                    matches = [part for part in encoded['spans'] if part[2] == row['entry_id']]
                    mapping_ok &= len(matches) == 1 and matches[0][0] <= a < b <= matches[0][1]
                mapping_ok &= int(row['token_id']) == encoded['inputs']['input_ids'][0][i]
            check(key + '_token_mapping_by_occurrence', mapping_ok)
            check(key + '_signed_token_entry_sums', all(math.isclose(float(e['attribution']),
                  math.fsum(float(t['attribution']) for t in tr if t['entry_id'] == e['entry_id']
                            and t['entry_text'] == e['entry_text']), rel_tol=0, abs_tol=1e-14) for e in er))
            residual = math.fsum(values) - (qa['diagnostic_score'] - qa['baseline_score'])
            tolerance = max(1e-4, .01 * abs(qa['diagnostic_score'] - qa['baseline_score']))
            check(key + '_unchanged_completeness_rule', abs(residual) <= tolerance
                  and math.isclose(residual, qa['completeness_residual'], rel_tol=0, abs_tol=1e-14)
                  and tolerance == qa['completeness_tolerance'] and qa['completeness_passed'])
            check(key + '_captum_delta_consistency', abs(qa['captum_delta'] - residual) < 1e-7)
            check(key + '_native_and_diagnostic_cutoff', qa['fresh_native_included'] == native['inclusion']
                  == qa['diagnostic_included'] and qa['boundary_guard']
                  and abs(native['signed_margin']) > 5 * max(qa['fresh_native_error'], qa['diagnostic_native_error'])
                  and qa['input_ids_vs_embeds_error'] <= 1e-6)
            features = np.load(out / (key + '_features.npz'))
            attr = features['attribution'][0]
            special = np.array(encoded['special_tokens_mask'], dtype=bool)
            inp, base = features['input_embeddings'][0], features['baseline_embeddings'][0]
            # Replay the runner's exact CUDA FP32 reduction, not a different NumPy reduction tree.
            reduced = torch.from_numpy(attr).cuda().sum(-1).cpu().double().numpy()
            check(key + '_feature_to_token_sum', np.array_equal(reduced, np.array(values)))
            check(key + '_special_embeddings_and_attributions_fixed',
                  np.array_equal(inp[special], base[special]) and np.all(attr[special] == 0))
            check(key + '_baseline_word_vectors', np.all(base[~special] == 0) if baseline == 'zero'
                  else np.all(base[~special] == base[~special][0]))
            completeness.append({k: qa[k] for k in ['schedule', 'baseline', 'nodes', 'baseline_score',
                                 'completeness_residual', 'completeness_tolerance', 'captum_delta']})
    check('character_multiset_unchanged', Counter(full_texts[0]) == Counter(full_texts[1]))
    native_rows = {r['schedule']: r for r in fixture['native_records']}
    competitors = rows(out / 'fixed_competitors.csv')
    check('42993_fixed_C0_competitors', len(competitors) == 42993
          and len({r['product_id'] for r in competitors}) == 42993
          and all(r['product_id'] != '12575' for r in competitors))
    ids = np.array([int(r['catalog_index']) for r in competitors])
    scores = np.array([float(r['C0_score']) for r in competitors], dtype=np.float32)
    ranked = np.lexsort((ids, -scores))
    for schedule, expected_rank in [('C0', 19), ('C1', 58)]:
        native = native_rows[schedule]
        rank = 1 + int(np.sum((scores > native['score']) | ((scores == native['score']) & (ids < 12575))))
        check(schedule + '_rank_and_cutoff_exact', rank == expected_rank
              and int(ids[ranked[19]]) == 9834 and float(scores[ranked[19]]) == native['competitor_threshold_score'])
    qpath = Path(report['sources']['query']['path'])
    check('original_query_vector_bitwise_equal', np.array_equal(np.load(out / 'fixed_query_embedding.npy'),
                                                               np.load(qpath)[report['query_row_index']]))
    for r in aligned:
        a, b = [next(e for e in entries if e['baseline'] == r['baseline'] and
                e['schedule'] == s and e['entry_id'] == r['entry_id']) for s in ('C0', 'C1')]
        check('aligned_' + r['baseline'] + '_' + r['entry_id'],
              float(r['C0']) == float(a['attribution']) and float(r['C1']) == float(b['attribution'])
              and float(r['C1_minus_C0']) == float(r['C1']) - float(r['C0'])
              and r['entry_text'] == a['entry_text'] == b['entry_text'])
    signs = []
    for key in attr_ids:
        a, b = [next(r for r in aligned if r['entry_id'] == key and r['baseline'] == baseline)
                for baseline in ('zero', 'pad')]
        x, y = float(a['C1_minus_C0']), float(b['C1_minus_C0'])
        signs.append(dict(entry_id=key, entry_text=a['entry_text'], zero=x, pad=y,
                          sign_changes=bool(np.sign(x) != np.sign(y))))
    fig = read(out / 'figure_data.json')
    check('figure_contains_all_10_attributes_and_fixed_row',
          [r['entry_id'] for r in fig['rows']] == attr_ids + ['fixed_text_and_boundaries'])
    for r in fig['rows']:
        for baseline in ('zero', 'pad'):
            expected = math.fsum(float(v['C1_minus_C0']) for v in aligned if v['baseline'] == baseline
                                and (v['entry_id'] == r['entry_id'] if r['entry_id'] in attr_ids
                                     else v['entry_id'] not in attr_ids))
            check('figure_data_' + baseline + '_' + r['entry_id'], r[baseline + '_C1_minus_C0'] == expected)
    stem = out / 'french_molding_c0_c1_captum'
    doc = ET.parse(stem.with_suffix('.drawio'))
    cells = doc.findall('.//mxCell')
    draw_text = [c.get('value') for c in cells if c.get('value')]
    check('drawio_all_text_matches_renderer', draw_text == fig['figure_text'])
    svg = ET.parse(stem.with_suffix('.svg'))
    svg_text = [''.join(e.itertext()) for e in svg.findall('.//{http://www.w3.org/2000/svg}text')]
    check('svg_text_matches_drawio', Counter(svg_text) == Counter(draw_text))
    pdf = PdfReader(stem.with_suffix('.pdf'))
    pdf_text = normalized(' '.join(p.extract_text() for p in pdf.pages))
    # PDF glyph kerning may be extracted as spaces inside words (e.g. "T arget").
    # Compare the complete ordered non-whitespace characters of each required label.
    compact_pdf = re.sub(r'\s+', '', pdf_text)
    check('pdf_has_all_labels_and_legends', all(re.sub(r'\s+', '', r['label']) in compact_pdf for r in fig['rows'])
          and all(re.sub(r'\s+', '', s) in compact_pdf for s in ['Zero baseline', 'PAD baseline', 'Included', 'Omitted', 'C0', 'C1',
                                         '19', '58', '139 / 256', '42,993', 'Target-only reversal']))
    for baseline, color, shape in [('zero', '#24618c', 'ellipse'), ('pad', '#c27328', 'triangle')]:
        markers = []
        for c in cells:
            style = c.get('style', '')
            if f'shape={shape};' in style and f'fillColor={color};' in style:
                g = c.find('mxGeometry')
                markers.append((float(g.get('x')) + float(g.get('width')) / 2,
                                float(g.get('y')) + float(g.get('height')) / 2))
        for r in fig['rows']:
            x, y = r[baseline + '_marker_display']
            point = (x * fig['drawio_scale'], (fig['height_pixels'] - y) * fig['drawio_scale'])
            check('drawio_marker_' + baseline + '_' + r['entry_id'],
                  sum(max(abs(a-point[0]), abs(b-point[1])) < 1e-4 for a, b in markers) == 1)
    protected = read(source / 'protected_files_before.json')
    check('old_figures_and_existing_edits_unchanged', all(sha(repo / p) == digest for p, digest in protected.items()))
    result = dict(status='passed' if all(c['passed'] for c in checks) else 'failed',
                  check_count=len(checks), failed_checks=[c for c in checks if not c['passed']],
                  checks=checks, completeness=completeness, attribute_sign_comparison=signs,
                  scope='target-only reversal; baseline-dependent attribution diagnostic',
                  limitations=['No independent attribute causal effects or population prevalence.',
                               'Completeness is a global identity check, not a per-attribute sign uncertainty bound.',
                               'PAD height and installation differences are near zero; their tiny signs are not emphasized.'])
    (out / 'artifact_qa.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'check_count', 'failed_checks', 'attribute_sign_comparison']}, indent=2))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
