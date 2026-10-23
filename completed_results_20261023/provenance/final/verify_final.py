"""Read-only final scientific verification; writes evidence only in this audit."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

AUDIT = Path(__file__).resolve().parent
OLD = AUDIT.parents[1]
ROOT = OLD.parent
MODELS = {
    'embeddinggemma2': ('batch1_20261017', 768, '3f3004b166a855287ad7435ff0b9c278d983640bfb5f20a11bc2ec3ebeaa876c'),
    'harrier06b': ('harrier_20261019', 1024, '7153798d3c03023dba8fb62df34d04fcefd0618474afe44602c056690d3903a4'),
}
POP = {'esci': (499, 4434, 10076), 'wands': (308, 21299, 42994)}
RULES = {'lexical_ascending', 'lexical_descending', 'field_priority_type'}
RAW = {'C0', 'C1', 'C2s1', 'C2s2', 'C2s3', 'C2s4', 'C2s5'}
checked = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def source(path):
    path = Path(path).resolve()
    key = str(path)
    if key not in checked:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        checked[key] = dict(path=key, sha256=h.hexdigest(), bytes=path.stat().st_size)
    return checked[key]


def verify_entry(entry):
    actual = source(entry['path'])
    assert actual['sha256'] == entry['sha256'], entry['path']
    if 'bytes' in entry:
        assert actual['bytes'] == entry['bytes'], entry['path']
    return actual


def receipt(path, descriptor=None):
    d = read(path)
    assert d['status'] == 'completed', path
    if descriptor is not None:
        assert d['inputs_fingerprint'] == fingerprint(descriptor), path
    for entry in d.get('outputs', []):
        verify_entry(entry)
    return d


def array(path, shape):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    assert a.dtype == np.float32 and a.shape == shape, (path, a.shape, shape)
    for start in range(0, len(a), 8192):
        assert np.isfinite(a[start:start + 8192]).all(), path
    return a


def idle():
    excluded = {os.getpid(), *[p.pid for p in psutil.Process().parents()]}
    live = []
    for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'create_time']):
        try:
            d = p.info
            cmd = ' '.join(d.get('cmdline') or [])
            if d['pid'] not in excluded and 'python' in (d.get('name') or '').lower() and any(x in cmd for x in ('revision_recent_models_20261016', 'revision_jina_nano_20261019')):
                live.append(d)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    assert not live, live
    return live


def save(name, value):
    with (AUDIT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    idle()
    combined = read(OLD / 'manifest_multi_profile.json')
    assert combined['all_active_experiments_completed'] is True
    expected_cells = {m + '/' + ds for m in MODELS for ds in POP}
    assert set(combined['completed_primary_cells']) == expected_cells
    assert {x['model'] for x in combined['profiles']} == set(MODELS)
    for entry in combined['files']:
        verify_entry(entry)
    assert len(combined['files']) == 10
    summary = pd.read_csv(OLD / 'tables/summary.csv')
    main_table = pd.read_csv(OLD / 'tables/main_candidates.csv')
    cross = pd.read_csv(OLD / 'tables/cross_model_VI.csv')
    assert set(main_table['model'] + '/' + main_table['dataset']) == expected_cells and len(main_table) == 4
    assert (main_table.status == 'completed').all()
    cells, profiles = {}, {}
    for model, (folder, dim, protocol_hash) in MODELS.items():
        p = OLD / 'profiles' / folder
        protocol = read(p / 'PROTOCOL.json')
        assert source(p / 'PROTOCOL.json')['sha256'] == protocol_hash
        assert set(protocol['canonical_rules']) == RULES
        inspection = read(AUDIT / (model + '_inspection_stdout.json'))
        assert inspection['primary_complete'] and inspection['canonical_complete']
        assert all(all(v.values()) for v in inspection['cells'].values())
        manifest = read(p / 'manifest.json')
        assert manifest['protocol_sha256'] == protocol_hash
        for entry in [*protocol['source_files'], *manifest['source_files'], *manifest['files'], *manifest['completion_receipts']]:
            verify_entry(entry)
        for entry in manifest['completion_receipts']:
            receipt(entry['path'])
        for entry in combined['profiles']:
            if entry['model'] == model:
                verify_entry(entry['protocol']); verify_entry(entry['profile_manifest'])
                assert set(entry['verified_primary_datasets']) == set(POP)
                assert set(entry['verified_canonical_datasets']) == set(POP)
        profile_summary = pd.read_csv(p / 'tables/summary.csv')
        combined_subset = summary[summary.model == model].reset_index(drop=True)
        pd.testing.assert_frame_equal(combined_subset, profile_summary.reset_index(drop=True), check_dtype=False)
        assert read(p / 'audit/active_process.json')['status'] == 'idle'
        assert read(p / 'audit/supervisor.json')['status'] == 'finished'
        profiles[model] = dict(protocol=source(p / 'PROTOCOL.json'), manifest=source(p / 'manifest.json'), files_verified=len(manifest['files']), completion_receipts_verified=len(manifest['completion_receipts']), inspection=source(AUDIT / (model + '_inspection_stdout.json')))
        for ds, (queries, pairs, catalog) in POP.items():
            base = p / 'runs' / model / ds
            evidence = []
            for stage in ('encode', 'evaluate', 'boundary', 'canonical'):
                path = p / 'status' / f'{model}_{ds}_{stage}.json'
                receipt(path); evidence.append(source(path))
            for tag, rows in (('queries', queries), ('raw', 7 * catalog), ('canonical', 3 * catalog)):
                descriptor = read(base / tag / 'descriptor.json')
                assert descriptor['spec'] == protocol['models'][model]
                done = receipt(base / tag / 'complete.json', descriptor)
                assert done['total_inputs'] == rows
                array(base / tag / 'embeddings.npy', (rows, dim))
                evidence.append(source(base / tag / 'complete.json'))
            data = {}
            for tag, schedules in (('raw', RAW), ('canonical', RULES)):
                ev = base / 'evaluation' / tag
                done = receipt(ev / 'complete.json')
                assert (done['queries'], done['highest_pairs'], done['full_catalog']) == (queries, pairs, catalog)
                qm = pd.read_parquet(ev / 'query_metrics.parquet')
                pr = pd.read_parquet(ev / 'pair_ranks.parquet')
                assert set(qm.schedule) == schedules and set(qm.K) == {20, 100}
                assert len(qm) == queries * len(schedules) * 2
                assert not qm.duplicated(['query_id', 'schedule', 'K']).any()
                assert qm.groupby(['schedule', 'K']).query_id.nunique().eq(queries).all()
                assert len(pr) == pairs and not pr.duplicated(['query_id', 'product_id']).any()
                assert pr.query_id.nunique() == queries
                assert np.isfinite(qm[['Recall', 'nDCG', 'cNDCG', 'judged_coverage']]).all().all()
                for schedule in schedules:
                    array(ev / (schedule + '_scores.npy'), (queries, catalog))
                data[tag] = (qm, pr)
                evidence.append(source(ev / 'complete.json'))
            raw_q, raw_p = data['raw']; can_q, can_p = data['canonical']
            assert set(zip(raw_p.query_id, raw_p.product_id)) == set(zip(can_p.query_id, can_p.product_id))
            assert set(raw_q.query_id) == set(can_q.query_id)
            boundary = read(base / 'evaluation/raw/boundary_audit.json')
            assert boundary['status'] == 'completed' and boundary['original_rank_records_unchanged'] is True
            assert boundary['selected_pairs'] == 32 and boundary['repeat_count'] == 3
            evidence.append(source(base / 'evaluation/raw/boundary_audit.json'))
            for k in (20, 100):
                for metric in ('Recall', 'nDCG', 'cNDCG', 'judged_coverage'):
                    raw_mean = raw_q[raw_q.K == k].groupby('query_id')[metric].mean()
                    for rule in RULES:
                        canonical = can_q[(can_q.K == k) & (can_q.schedule == rule)].set_index('query_id')[metric]
                        actual = float((canonical - raw_mean).mean())
                        rows = summary[(summary.model == model) & (summary.dataset == ds) & (summary.K == k) & (summary.support == 'full') & (summary.kind == 'canonical_minus_raw_seven_mean') & (summary.schedule == rule) & (summary.metric == metric)]
                        assert len(rows) == 1, (model, ds, k, rule, metric, len(rows))
                        row = rows.iloc[0]
                        assert abs(float(row.estimate) - actual) < 1e-12
                        assert np.isfinite([row.ci_low, row.ci_high]).all() and row.ci_low <= row.ci_high
                        assert row.queries == queries and row.bootstrap_seed == 2026091701 and row.bootstrap_draws == 10000
            row = main_table[(main_table.model == model) & (main_table.dataset == ds)].iloc[0]
            assert row.queries == queries and row.pairs == pairs
            assert abs(row.C0_Recall - raw_q[(raw_q.K == 20) & (raw_q.schedule == 'C0')].Recall.mean()) < 1e-12
            assert abs(row.raw_mean_Recall - raw_q[raw_q.K == 20].Recall.mean()) < 1e-12
            for kind in ('catalog', 'target'):
                vi = raw_p.assign(crossing=raw_p[kind + '_state_K20'].eq('crossing')).groupby('query_id').crossing.mean().mean()
                assert abs(row[kind + '_VI'] - vi) < 1e-12
                c = cross[(cross.model == model) & (cross.dataset == ds) & (cross.kind == kind + '_states')]
                assert len(c) == 1 and abs(c.iloc[0].estimate - vi) < 1e-12
            cells[model + '/' + ds] = dict(queries=queries, highest_pairs=pairs, catalog=catalog, dimension=dim, canonical_rules=sorted(RULES), primary_schedules=sorted(RAW), fp32_finite=True, receipts=evidence, boundary_selected_pairs=boundary['selected_pairs'], boundary_numerical_ambiguities=boundary['numerical_ambiguities'], canonical_contrast_estimates_verified=24)
            print('Verified cell ' + model + '/' + ds, flush=True)
    # Fully reconcile the newly finished WANDS checkpoint stream to final vectors.
    p = OLD / 'profiles/harrier_20261019'
    directory = p / 'runs/harrier06b/wands/canonical'
    plan = read(directory / 'plan.json'); descriptor = read(directory / 'descriptor.json')
    final = np.load(directory / 'embeddings.npy', mmap_mode='r')
    aliases = np.array(plan['aliases'], dtype=np.int64)
    unique_count = len(plan['unique_text_hashes'])
    representatives = np.full(unique_count, -1, dtype=np.int64)
    for idx, alias in enumerate(aliases):
        if representatives[alias] == -1: representatives[alias] = idx
        assert descriptor['text_hashes'][idx] == plan['unique_text_hashes'][alias]
    assert (representatives >= 0).all()
    seen = np.zeros(unique_count, dtype=bool)
    end = count = 0; flags = sorted(directory.glob('block_*.json')); evidence = []
    for flag in flags:
        start = int(flag.stem.split('_')[1]); assert start == end
        batches = plan['batches'][start:start + 128]
        indices = [i for batch in batches for i in batch]
        expected = dict(profile=fingerprint(descriptor), batches=batches, text_hashes=[plan['unique_text_hashes'][i] for i in indices])
        d = receipt(flag, expected)
        assert not seen[indices].any()
        with np.load(d['outputs'][0]['path'], allow_pickle=False) as block:
            assert block['indices'].tolist() == indices
            assert block['embeddings'].dtype == np.float32 and np.isfinite(block['embeddings']).all()
            assert np.array_equal(final[representatives[indices]], block['embeddings'])
        seen[indices] = True; count += len(indices); end += len(batches); evidence.append(source(flag))
    assert len(flags) == 206 and end == 26306 and count == unique_count == 128525 and seen.all()
    assert len(aliases) == 128982
    for start in range(0, len(aliases), 1024):
        assert np.array_equal(final[start:start + 1024], final[representatives[aliases[start:start + 1024]]])
    baseline = read(OLD / 'audit/harrier_can_oom_20261022/checkpoints.json')
    for b in baseline['blocks']:
        verify_entry(b['receipt'])
    assert source(p / 'audit/canonical_after_all_primary_attempt_20261021.json')['sha256'] == '95b44696a3a03b1daadab6993a0c8f3b80653c05bd33569cd0ef8f9b0abd6ed9'
    attempt = read(OLD / 'audit/harrier_can_resume_20261022/attempt.json')
    assert attempt['status'] == 'phase_completed'
    supervisor = read(p / 'audit/supervisor.json')
    assert supervisor['all_active_experiments_completed'] is True and all(v == 0 for v in supervisor['refresh_exit_codes'].values())
    assert not (OLD / 'audit/harrier_can_resume_20261022/failure.json').exists()
    report = (OLD / 'REPORT_zh.md').read_text(encoding='utf-8')
    tex = (OLD / 'MANUSCRIPT_ADDITIONS.tex').read_text(encoding='utf-8')
    assert '4/4' in report and 'canonical 完成：True' in report
    for model in MODELS:
        for ds in POP:
            for rule in RULES:
                assert f'{model} / {ds} / {rule}' in report
                assert f'{model} on {ds.upper()}, {rule.replace("_", " ")}' in tex
    snapshot = AUDIT / 'report'; snapshot.mkdir()
    for entry in [source(OLD / 'manifest_multi_profile.json'), *combined['files']]:
        original = Path(entry['path']); dest = snapshot / original.relative_to(OLD)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, dest)
        assert source(dest)['sha256'] == entry['sha256']
    result = dict(status='both_models_whole_scope_verified_complete', checked_utc=datetime.now(timezone.utc).isoformat(), whole_two_model_scope_complete=True, active_models=list(MODELS), cells=cells, profiles=profiles, harrier_supervisor=source(p / 'audit/supervisor.json'), harrier_finished_utc=supervisor['finished_utc'], harrier_resume_attempt=source(OLD / 'audit/harrier_can_resume_20261022/attempt.json'), harrier_wands_checkpoints=dict(blocks=206, batches=26306, unique_inputs=128525, alias_inputs=128982, original_baseline_preserved=True, all_indices_fingerprints_hashes_finite_and_final_assembly_verified=True, receipts=evidence), combined_manifest=source(OLD / 'manifest_multi_profile.json'), combined_report_files_verified=10, snapshot=str(snapshot), live_experiment_processes=idle(), github_republished=False)
    save('verified_files.json', list(checked.values()))
    result['verified_files'] = source(AUDIT / 'verified_files.json')
    save('verification.json', result)
    print(json.dumps({'status':result['status'], 'files_verified':len(checked), 'verification':source(AUDIT / 'verification.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
