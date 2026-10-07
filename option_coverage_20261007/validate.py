"""Independent checks of saved records, summary denominators and preserved inputs."""
import gzip
import importlib.metadata
import io
import json
import unittest
import numpy as np
import pandas as pd
from analyze import HERE, ROOT, DATA, CFG, sha, dump, load_records, boot
from core import selection_hash


def main():
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_core.py')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (qa/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    assert result.wasSuccessful()
    manifest=json.loads((DATA/'input_manifest.json').read_text(encoding='utf-8'))
    serial=json.loads((DATA/'serialization_audit.json').read_text(encoding='utf-8'))
    protected=json.loads((DATA/'preexisting_state.json').read_text(encoding='utf-8'))
    for record in manifest['inputs']+serial['input_hashes']+protected['files']:
        assert sha(ROOT/record['path'])==record['sha256'],record['path']
    spec_sha=sha(HERE/'extraction_spec_v1.json')
    assert spec_sha==json.loads((DATA/'extraction_freeze.json').read_text())['spec_sha256']
    with gzip.open(DATA/'product_fields.jsonl.gz','rt',encoding='utf-8') as f:ex={(str(r['product_id']),r['field']):r for r in map(json.loads,f)}
    ms=load_records('data/membership.jsonl');fs=load_records('data/query_fields.jsonl')
    mindex={(m['comparison'],m['query_id']):m for m in ms}
    assert len(mindex)==len(ms)
    assert len({(r['comparison'],r['query_id'],r['field']) for r in fs})==len(fs)
    assert len(fs)==3*len(ms)
    for m in ms:
        before,after=set(m['before']),set(m['after'])
        assert set(m['lost_products'])==before-after
        assert set(m['gained_products'])==after-before
        assert m['cancellation']==(len(before-after)==len(after-before)>0)
        assert m['recall_equal']==(len(before)==len(after))
        assert m['recall_before']==len(before)/m['highest_n']
        assert m['recall_after']==len(after)/m['highest_n']
    eligible_total=0
    for r in fs:
        m=mindex[r['comparison'],r['query_id']]
        before,after=set(m['before']),set(m['after']);field=r['field']
        valid={p for p in before|after if ex[p,field]['status']=='valid'}
        eligible=bool(before and after and (before|after)<=valid)
        assert eligible==r['eligible']
        assert len(before-valid)==r['missing_before'] and len(after-valid)==r['missing_after']
        if eligible:
            eligible_total+=1
            a={v for p in before for v in ex[p,field]['values']};b={v for p in after for v in ex[p,field]['values']}
            assert a==set(r['options_before']) and b==set(r['options_after'])
            assert a-b==set(r['lost_options']) and b-a==set(r['gained_options'])
            assert r['n_lost']==len(a-b) and r['n_gained']==len(b-a)
            assert r['net_options']==len(b)-len(a)==r['n_gained']-r['n_lost']
            assert r['unchanged_options']==(a==b)
            if not m['membership_changed']:assert not r['option_changed']
        else:
            assert all(r[k] is None for k in ['any_lost','any_gained','n_lost','n_gained','lost_options','gained_options'])
            assert r['exclusion_reasons']
    s=pd.read_csv(DATA/'option_summary.csv')
    exc=pd.read_csv(DATA/'exclusion_summary.csv')
    for row in s.to_dict('records'):
        group=[r for r in fs if r['comparison']==row['comparison'] and r['field']==row['field'] and (row['subgroup']=='all' or r['cancellation'])]
        valid=[r for r in group if r['eligible']]
        assert row['subgroup_total']==len(group) and row['eligible_n']==len(valid)
        assert row['excluded_n']==len(group)-len(valid)
        for metric in ['option_changed','any_lost','any_gained','unchanged_options','n_lost','n_gained','n_options_before','n_options_after','net_options']:
            assert np.isclose(row[metric+'_mean'],np.mean([r[metric] for r in valid]),rtol=0,atol=1e-14)
            assert row[metric+'_ci_low']<=row[metric+'_ci_high']
            if metric+'_count' in row:assert row[metric+'_count']==sum(r[metric] for r in valid)
        assert row['option_changed_count']+row['unchanged_options_count']==len(valid)
        n=exc[exc.comparison.eq(row['comparison'])&exc.field.eq(row['field'])&exc.subgroup.eq(row['subgroup'])&exc.category.eq('mutually_exclusive_combination')]['n'].sum()
        assert n==row['excluded_n']
    # Direct independent full-array bootstrap agrees with chunked implementation.
    sample=np.array([[0.,1.],[1.,0.],[2.,0.]])
    rng=np.random.default_rng(CFG['seed'])
    independently=sample[rng.integers(0,3,size=(CFG['bootstrap_draws'],3))].mean(axis=1)
    means,lo,hi=boot(sample)
    np.testing.assert_array_equal(means,sample.mean(axis=0))
    np.testing.assert_array_equal(np.stack([lo,hi]),np.quantile(independently,[.025,.975],axis=0))
    z=boot([0,0,0,0]);assert all(float(v[0])==0 for v in z)
    assert boot([])==(None,None,None)
    cases=load_records('data/cases.jsonl')
    selection=json.loads((DATA/'case_selection.json').read_text())
    for pool,chosen in zip(selection['candidates'],selection['selected'][1:]):
        assert chosen['query_id']==min(pool['query_ids_in_hash_order'],key=lambda q:selection_hash(f'20261007:case:{q}'))
    for c in cases:
        qid=c['membership']['query_id'];assert c['membership']==mindex['raw_C0_vs_C1',qid]
        assert {str(p['product']['product_id']) for p in c['source_products']}==set(c['membership']['before'])|set(c['membership']['after'])
        for p in c['source_products']:
            assert (p['ranks']['C0']<=CFG['k'])==p['in_before']
            assert (p['ranks']['C1']<=CFG['k'])==p['in_after']
    anti=next(c for c in cases if c['membership']['query_id']==231)
    assert next(f for f in anti['query_fields'] if f['field']=='shape')['lost_options']==['semi-circle']
    plot=pd.read_csv(HERE/'figures/option_coverage_plot_data.csv')
    primary=s[s.comparison.eq('raw_C0_vs_C1')&s.subgroup.eq('cancellation')].reset_index(drop=True)
    for col in primary:
        if primary[col].dtype.kind in 'biufc':np.testing.assert_allclose(plot[col],primary[col],rtol=0,atol=1e-14)
        else:assert plot[col].tolist()==primary[col].tolist()
    tex=(HERE/'table_option_coverage.tex').read_text()
    for r in primary.to_dict('records'):
        assert f"{r['field'].capitalize()} & {r['eligible_n']}/{r['subgroup_total']} & {r['any_lost_count']}/{r['eligible_n']}" in tex
    report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    paragraph=next(line[2:] for line in report.splitlines() if line.startswith('> '))
    assert 80<=len(paragraph.split())<=120
    runtime={}
    for name in ['numpy','pandas','pyarrow','matplotlib','Pillow']:
        runtime[name]=importlib.metadata.version(name)
    dump('qa/validation.json',dict(passed=True,unit_tests=result.testsRun,unit_test_failures=0,independent_bootstrap_checks=3,
          input_hash_records_checked=len(manifest['inputs'])+len(serial['input_hashes']),protected_files_checked=len(protected['files']),
          product_fields=len(ex),membership_records=len(ms),query_field_records=len(fs),strict_eligible_query_fields=eligible_total,
          summary_rows_checked=len(s),case_count=len(cases),figure_table_same_saved_data=True,candidate_paragraph_words=len(paragraph.split()),
          extraction_spec_sha256=spec_sha,runtime_packages=runtime,source_and_manuscript_unchanged=True))
    outputs=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='output_manifest.json':
            outputs.append(dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    dump('output_manifest.json',dict(files=outputs,excluded=['output_manifest.json','__pycache__'],all_outputs_within_analysis_directory=True))
    print(json.dumps(dict(passed=True,unit_tests=result.testsRun,summary_rows=len(s),membership_records=len(ms),query_field_records=len(fs),protected_files=len(protected['files']),output_files=len(outputs)),indent=2))


if __name__=='__main__':main()
