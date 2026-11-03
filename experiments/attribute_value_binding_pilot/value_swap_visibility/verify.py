"""Artifact-level checks independent of forward execution and summary functions."""
from common import *
import unittest
import test_integrity
import test_summary

def main():
    freeze_check();preservation_check()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_integrity,test_summary]])
    test=unittest.TextTestRunner().run(suite)
    assert test.wasSuccessful()
    products,queries,judgments=load();counts=read(HERE/'data/sample_counts.json')
    selected=frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str})
    f=frame(HERE/'results/pilot/pairs.csv',dtype={'product_id':str})
    smoke=frame(HERE/'results/smoke/pairs.csv',dtype={'product_id':str})
    full=pd.read_parquet(HERE/'data/full_inputs.parquet').set_index('product_id')
    assert len(f)==counts['sampled_pairs']<=500 and f.query_id.nunique()<=100
    assert f.groupby('query_id').size().max()<=5 and not f.duplicated(['query_id','product_id']).any()
    cols=selected.columns.tolist()
    pd.testing.assert_frame_equal(selected.sort_values(['query_id','product_id']).reset_index(drop=True),
        f[cols].sort_values(['query_id','product_id']).reset_index(drop=True))
    eq=judgments[judgments.label.eq('Exact')];known=set(zip(eq.query_id,eq.product_id))
    assert set(zip(f.query_id,f.product_id))<=known
    for r in selected.drop_duplicates('product_id').to_dict('records'):
        p=products[r['catalog_index']];s,_=select_pair(p)
        assert all(r[k]==s[k] for k in s)
        a,c=swap(p,r);saved=full.loc[r['product_id']]
        assert saved.text_A==saved.model_input_A==a and saved.text_C==saved.model_input_C==c
        assert digest(a)==saved.text_sha256_A and digest(c)==saved.text_sha256_C
        assert r['tokens_A']<=SPEC['max_tokens'] and r['tokens_C']<=SPEC['max_tokens']
    assert (f.catalog_size==len(products)).all()
    assert np.array_equal(f.score_delta,f.score_C-f.score_A)
    assert np.array_equal(f.rank_delta,f.rank_C-f.rank_A)
    assert np.array_equal(f.direction,np.where(f.rank_delta<0,'improved',np.where(f.rank_delta>0,'declined','unchanged')))
    assert f.rank_A.between(1,len(products)).all() and f.rank_C.between(1,len(products)).all()
    assert np.array_equal(f.in_top20_A,f.rank_A.le(20)) and np.array_equal(f.in_top20_C,f.rank_C.le(20))
    assert np.array_equal(f.top20_entry,~f.in_top20_A&f.in_top20_C)
    assert np.array_equal(f.top20_exit,f.in_top20_A&~f.in_top20_C)
    assert f.all_competitor_scores_bitwise_unchanged.all()
    pd.testing.assert_frame_equal(smoke.sort_values(['query_id','product_id']).reset_index(drop=True),
        f.set_index(['query_id','product_id']).loc[list(zip(smoke.query_id,smoke.product_id))].reset_index()[smoke.columns].sort_values(['query_id','product_id']).reset_index(drop=True),check_exact=True)
    summary=frame(HERE/'results/pilot/summary.csv');s=summary.iloc[0]
    assert s.scope=='all_sampled_pairs' and s.pairs==len(f)
    for label,mask in [('improved',f.rank_delta<0),('declined',f.rank_delta>0),('unchanged',f.rank_delta==0)]:
        assert s[label+'_n']==mask.sum() and s[label+'_den']==len(f) and s[label+'_rate']==float(mask.mean())
    for label,n,d in [('entry',f.top20_entry.sum(),(~f.in_top20_A).sum()),
                       ('exit',f.top20_exit.sum(),f.in_top20_A.sum()),('crossing',(f.in_top20_A!=f.in_top20_C).sum(),len(f))]:
        assert s[label+'_n']==n and s[label+'_den']==d
        if d:assert float(s[label+'_rate'])==float(n/d)
        else:assert s[label+'_rate']==''
    for label in ['rank_delta','score_delta']:
        for q,name in zip([.25,.5,.75],['q25','median','q75']):assert s[label+'_'+name]==np.quantile(f[label],q)
    dist=frame(HERE/'results/pilot/baseline_rank_distribution.csv');assert dist.numerator.sum()==len(f)
    assert dist.iloc[0].numerator==f.in_top20_A.sum()
    cases=frame(HERE/'results/pilot/cases.csv',dtype={'product_id':str})
    for name,mask in [('top20_entry',f.top20_entry),('top20_exit',f.top20_exit)]:
        assert set(zip(cases[cases.case_kind.eq(name)].query_id,cases[cases.case_kind.eq(name)].product_id))==set(zip(f[mask].query_id,f[mask].product_id))
    ordinary=cases[cases.case_kind.eq('ordinary_hash_sample')]
    expected=sorted(zip(f.query_id,f.product_id),key=lambda x:pick('ordinary_case',*x))[:5]
    assert list(zip(ordinary.query_id,ordinary.product_id))==expected
    for stage,n in [('smoke',len(smoke)),('pilot',len(f))]:
        complete=read(HERE/f'qa/{stage}/complete.json');assert complete['passed']
        assert complete['selection_manifest_sha256']==sha(HERE/'selection_manifest.json')
        assert complete['run_sha256']==sha(HERE/'run.py')
        assert complete['baseline_manifest_sha256']==(sha(HERE/'baseline_manifest.json') if (HERE/'baseline_manifest.json').exists() else None)
        replay=frame(HERE/f'qa/{stage}/fresh_replay.csv');assert replay.exact_equal.all() and replay.max_abs_error.eq(0).all()
        expected_products=smoke.product_id.nunique() if stage=='smoke' else f.product_id.nunique()
        assert replay.role.eq('catalog').sum()==expected_products
        assert replay.role.eq('query').sum()==(smoke.query_id.nunique() if stage=='smoke' else f.query_id.nunique())
        audit=read(HERE/f'qa/{stage}/ranking_audit.json');assert audit['pairs']==n and audit['independent_count_checks']==2*n
        if stage=='smoke':assert audit['full_array_sort_checks']==2*n
        assert frame(HERE/f'qa/{stage}/swap_input_audit.csv').no_truncation.all()
    if (HERE/'baseline_manifest.json').exists():
        bm=read(HERE/'baseline_manifest.json')
        assert bm['selection_manifest_sha256']==sha(HERE/'selection_manifest.json')
        assert bm['plan_sha256']==sha(HERE/'baseline_rebuild_plan.json')
        for info in bm['arrays'].values():assert info['sha256']==sha(HERE/info['path'])
    dump('qa/final_verification.json',dict(passed=True,completed_utc=now(),unit_tests=test.testsRun,
        pairs=len(f),queries=f.query_id.nunique(),catalog_products=len(products),
        smoke_rows_match_pilot_exactly=True,original_pilot_and_preexisting_modified_files_preserved=True,
        frozen_inputs_valid=True,summary_recomputed_independently=True))
    dump('delivery_manifest.json',dict(created_utc=now(),files={p.relative_to(HERE).as_posix():sha(p)
        for p in HERE.rglob('*') if p.is_file() and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts
        and 'unused_encoder_cache' not in p.parts}))
    print('ALL CHECKS PASSED',len(f),'pairs',len(products),'catalog products',flush=True)

if __name__=='__main__':main()
