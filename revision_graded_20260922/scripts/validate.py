"""Independent numerical cross-checks and preservation audit."""
from common import *
import unittest

def main():
    verify_protocol();pq=pd.read_parquet(HERE/'data/per_query.parquet');checks={}
    keys=['dataset','model','method','schedule'];assert len(pq[keys].drop_duplicates())==122
    assert not pq.duplicated(keys+['query_id']).any()
    assert len(pq)==49227
    for key,f in pq.groupby(keys):
        ds=key[0];assert sorted(f.query_id)==support(ds)
        assert int(f.n_highest.sum())==(21299 if ds=='wands' else 4434)
        assert int(f.n_judged.sum())==(162218 if ds=='wands' else 10119)
        for col in ['nDCG@10','nDCG@20','cNDCG@10','cNDCG@20','JudgedCoverage@10','JudgedCoverage@20','Recall@20','Recall@100']:
            assert f[col].between(-1e-14,1+1e-14).all(),(key,col)
    checks['all_122_conditions_have_exact_frozen_support']=True
    # Independent full catalog gain-vector implementation for deterministic query samples.
    sampled=0
    for key,f in pq.groupby(keys):
        ds=key[0];pairs=pairread(ROOT/f.rank_source.iloc[0])
        ids=support(ds);sample=[ids[0],ids[len(ids)//2],ids[-1]]
        for qid in sample:
            g=pairs[pairs.query_id.eq(qid)];row=f[f.query_id.eq(qid)].iloc[0]
            gains=g.label.map(GAINS[ds]).to_numpy();full=np.zeros(int(row.catalog_size));full[g['rank'].to_numpy()-1]=gains
            for k in [10,20]:
                ideal=np.sort(gains)[::-1][:k];den=ideal@np.reciprocal(np.log2(np.arange(len(ideal))+2))
                observed=full[:k]@np.reciprocal(np.log2(np.arange(k)+2))/den
                assert abs(observed-row[f'nDCG@{k}'])<1e-14
                condensed=g.sort_values('rank').label.map(GAINS[ds]).to_numpy()[:k]
                assert abs(condensed@np.reciprocal(np.log2(np.arange(len(condensed))+2))/den-row[f'cNDCG@{k}'])<1e-14
            sampled+=1
    checks['independent_dense_gain_vector_queries']=sampled
    for (ds,model,s),g in pq[pq.method.eq('raw')].groupby(['dataset','model','schedule']):
        old=pd.read_csv(ROOT/f'phase2/results/phase2_per_query/{ds}_{model}_native_{s}.csv').set_index('query_id').loc[support(ds)]
        f=g.set_index('query_id').loc[support(ds)]
        for k in [10,20]:np.testing.assert_allclose(old[f'cNDCG@{k}'],f[f'old_cNDCG@{k}'],rtol=0,atol=1e-14)
    checks['historical_raw_old_gain_cndcg_reproduced']=42
    # Stored grades are irrelevant to the new evaluator; all metrics came from labels.
    correction=pd.read_csv(HERE/'tables/gain_correction.csv')
    assert (correction[correction.dataset.eq('wands')].delta==0).all()
    checks['wands_gain_correction_exactly_zero']=True
    summary=pd.read_csv(HERE/'tables/effectiveness_summary.csv')
    old=pd.read_csv(COMPLETE/'tables/strategy_joint_complete.csv')
    checked=0
    for r in old[old.aggregation.eq('query_macro')].itertuples():
        method=r.method;s='fixed'
        if method=='raw_C0':method='raw';s='C0'
        elif method=='raw_seven':method='raw';s='seven_mean'
        elif method=='hybrid_seven':method='raw_hybrid';s='seven_mean'
        m='lexical' if method=='BM25' else r.model
        found=summary[(summary.dataset==r.dataset)&(summary.model==m)&(summary.method==method)&(summary.schedule==s)&(summary.metric==f'Recall@{r.K}')]
        if not len(found):continue
        assert len(found)==1
        value=r.seven_schedule_mean_recall if s=='seven_mean' else r.source_order_recall
        assert abs(float(found['mean'].iloc[0])-value)<1e-14,(r.dataset,r.model,r.method,r.K)
        checked+=1
    assert checked>=70,checked
    checks['historical_summary_recall_cells_reproduced']=checked
    for ds in ['wands','esci']:
        w=np.load(HERE/f'data/{ds}_bootstrap_weights.npy');assert w.shape==(DRAWS,len(support(ds))) and np.all(w.sum(1)==w.shape[1])
        rng=np.random.default_rng(SEED);ix=rng.integers(0,w.shape[1],size=(250,w.shape[1]))
        expected=np.zeros((250,w.shape[1]),dtype='uint16');np.add.at(expected,(np.arange(250)[:,None],ix),1)
        assert np.array_equal(expected,w[:250])
    checks['bootstrap_seed_cluster_counts_verified']=True
    rq=pd.read_parquet(HERE/'data/rq2_membership_ndcg_per_query.parquet')
    assert len(rq[['dataset','model','method','schedule']].drop_duplicates())==60
    assert ((rq.gained_count+rq.lost_count>0)==rq.membership_changed).all()
    assert (rq.integer_recall_cancellation==((rq.gained_count==rq.lost_count)&(rq.gained_count>0))).all()
    assert (rq.numerical_zero_nDCG20==(rq.abs_delta_nDCG20<=ZERO_TOL)).all()
    checks['rq2_all_60_contrasts_membership_and_tolerance_verified']=True
    manifest=read(HERE/'data/input_manifest.json');extra=read(HERE/'data/supplemental_audit_inputs.json')
    for i,e in enumerate(manifest['files']+extra['files']):
        assert sha(ROOT/e['path'])==e['sha256'],('source changed',e['path'])
        if i%100==0:print('VERIFY HASH',i,flush=True)
    for records in [manifest['preexisting_files'],read(HERE/'qa/preexisting_tracked_changes.json')]:
        for path,digest in records.items():assert sha(ROOT/path)==digest,('preexisting edit changed',path)
    checks['input_hash_records_preserved']=len(manifest['files'])+len(extra['files'])
    checks['preexisting_working_changes_preserved']=True
    checks['head_unchanged']=git('rev-parse','HEAD')==manifest['git_head'];assert checks['head_unchanged']
    checks['protocol_sha256']=sha(HERE/'PROTOCOL.json')
    dump(HERE/'qa/validation.json',checks)
    print('VALIDATION PASSED',json.dumps(checks),flush=True)

if __name__=='__main__':main()
