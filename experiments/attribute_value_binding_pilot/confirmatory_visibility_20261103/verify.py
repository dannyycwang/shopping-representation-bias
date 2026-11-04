"""Independent final artifact, source-selection, full-catalog and bootstrap checks."""
from core import *
from bootstrap_stats import load_scores

def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14,equal_nan=True)

def main():
    t=Timer();freeze_check();counts=read(HERE/'data/sample_counts.json');n=counts['sampled_pairs']
    for name,h in read(HERE/'analysis_plan_snapshot.json')['files'].items():assert sha(HERE/name)==h
    amendment=read(HERE/'execution_efficiency_amendment.json');assert amendment['run_sha256']==sha(HERE/'run.py')
    assert sha(HERE/'results/formal/gte_scores.csv')==amendment['gte_formal_original_sha256']
    for name in ['column_scoring','allocator']:
        gate=read(HERE/f'qa/{name}_gate.json');assert gate['passed']
        assert sha(HERE/f'qa/{name}_gate.json')==amendment[f'{name}_gate_sha256']
    for info in read(HERE/'provenance/source_snapshot_manifest.json')['files']:
        assert sha(HERE/info['snapshot'])==info['sha256']
    preserved=read(HERE/'qa/preservation.json')
    for path,h in preserved.items():assert sha(ROOT/path)==h,('Existing artifact changed',path)
    assert read(HERE/'qa/design_validation.json')['passed'] and read(HERE/'qa/analysis_smoke.json')['passed']
    raw=load_scores();pairs=frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str});wanted=set(zip(pairs.query_id,pairs.product_id))
    fullsorts=0;baselinechecks=0
    for key,s in raw.groupby('model'):
        completed=read(HERE/f'qa/formal/{key}_complete.json');assert completed['passed'] and completed['run_sha256']==sha(HERE/'run.py')
        assert completed['sample_manifest_sha256']==sha(HERE/'sample_manifest.json')
        assert len(s)==n*11 and set(zip(s.query_id,s.product_id))==wanted
        assert s.groupby(['query_id','product_id']).variant.nunique().eq(11).all()
        assert s.catalog_size.eq(counts['catalog_products']).all() and s.all_competitor_scores_bitwise_unchanged.all()
        assert np.array_equal(s.in_top20,s['rank'].le(20)) and np.array_equal(s.in_top20_A,s.rank_A.le(20))
        assert np.array_equal(s.rank_delta,s['rank']-s.rank_A);close(s.score_delta,s.score-s.score_A)
        baseline=s[s.variant.eq('A')];assert baseline.rank_delta.eq(0).all() and baseline.score_delta.eq(0).all()
        replay=frame(HERE/f'qa/formal/{key}_replay.csv');assert replay.exact_equal.all() and replay.max_abs_error.eq(0).all()
        audit=frame(HERE/f'qa/formal/{key}_input_audit.csv');assert audit.complete_input.all() and len(audit)==counts['sampled_products']*10
        if key=='harrier':
            allowed=set(read(HERE/'qa/column_scoring_gate.json')['bitwise_compatible_blocks'])
            fast=s.catalog_index.map(lambda i:str((int(i)//256)*256)).isin(allowed)
            expected_crossings=int((fast&s.in_top20.ne(s.in_top20_A)).sum())
            assert read(HERE/'qa/formal/harrier_ranking.json')['observed_crossing_full_matrix_score_checks']==expected_crossings
        smoke=frame(HERE/f'results/smoke/{key}_scores.csv',dtype={'product_id':str})
        idx=['query_id','product_id','variant'];actual=s.set_index(idx).loc[smoke.set_index(idx).index].reset_index()
        pd.testing.assert_frame_equal(smoke.reset_index(drop=True),actual[smoke.columns].reset_index(drop=True),check_exact=True)
        plan=read(HERE/f'cache_plan_{key}.json')
        for info in plan['files'].values():assert sha(ROOT/info['path'])==info['sha256']
        model_hashes=plan['model_snapshot_hashes'] if 'model_snapshot_hashes' in plan else {'model.safetensors':plan['weights_sha256']}
        for name,h in model_hashes.items():assert sha(Path(plan['model_snapshot'])/name)==h
        cat=np.load(ROOT/plan['files']['product']['path'],mmap_mode='r')[:counts['catalog_products']]
        q=np.load(ROOT/plan['files']['query']['path'])
        assert cat.dtype==q.dtype==np.float32 and np.isfinite(cat).all() and np.isfinite(q).all()
        allbase=q@cat.T if key=='harrier' else None
        for qid,g in s.groupby('query_id'):
            qi=plan['query_map'][str(qid)]['baseline_index'];base=allbase[qi] if allbase is not None else (q[qi:qi+1]@cat.T)[0]
            order=np.argsort(-base,kind='stable');ranks=np.empty(len(base),dtype=int);ranks[order]=np.arange(1,len(base)+1)
            for r in g[g.variant.eq('A')].itertuples():
                ix=int(r.catalog_index);assert r.score==float(base[ix]) and r.rank==int(ranks[ix]);baselinechecks+=1
            # All observed crossings, in addition to the predetermined smoke,
            # receive an independent full-array replacement and stable sort.
            for r in g[g.in_top20.ne(g.in_top20_A)].itertuples():
                changed=base.copy();changed[int(r.catalog_index)]=np.float32(r.score)
                expected=int(np.flatnonzero(np.argsort(-changed,kind='stable')==int(r.catalog_index))[0]+1)
                assert r.rank==expected;fullsorts+=1
    choices=frame(HERE/'results/rq2_choices.csv',dtype={'product_id':str});assert len(choices)==2*2*3*n
    lookup=raw.set_index(['model','query_id','product_id','variant']);verified_choices=0
    for (source,target,condition,budget),g in choices.groupby(['source','target','condition','budget']):
        assert set(zip(g.query_id,g.product_id))==wanted and not g.duplicated(['query_id','product_id']).any()
        for r in g.itertuples():
            names=['A']+[f'{condition}{j}' for j in range(1,budget+1)]
            candidates=[lookup.loc[(source,r.query_id,r.product_id,v)] for v in names]
            # Python max keeps the first maximum: A, then ascending j.
            winner=max(range(len(names)),key=lambda k:candidates[k].score)
            assert r.chosen_variant==names[winner] and r.modified==(winner!=0)
            for ev,model in [('source',source),('target',target)]:
                expected=lookup.loc[(model,r.query_id,r.product_id,r.chosen_variant)]
                assert getattr(r,ev+'_score')==expected.score and getattr(r,ev+'_rank')==expected['rank']
                assert getattr(r,ev+'_inclusion')==expected.in_top20 and getattr(r,ev+'_inclusion_A')==expected.in_top20_A
                assert getattr(r,ev+'_net')==int(expected.in_top20)-int(expected.in_top20_A)
            assert r.source_score>=r.source_score_A and r.source_net>=0;verified_choices+=1
    for _,g in choices.groupby(['source','condition','query_id','product_id']):
        assert np.all(np.diff(g.sort_values('budget').source_score)>=0)
    # Independently re-estimate intervals by direct indexing, without the
    # analysis implementation's bootstrap multiplicity matrix.
    manifest=read(HERE/'results/bootstrap_manifest.json');draw=np.load(HERE/'results/bootstrap_indices.npy')
    qs=manifest['query_ids'];assert draw.shape==(10000,len(qs)) and sha(HERE/'results/bootstrap_indices.npy')==manifest['indices_sha256']
    expected=np.random.default_rng(int(pick('query_bootstrap'),16)).integers(0,len(qs),(10000,len(qs)),dtype=np.int32)
    assert np.array_equal(draw,expected)
    def estimate(f,num,den):
        arrays=pd.DataFrame({'query_id':f.query_id,'n':np.asarray(num,float),'d':np.asarray(den,float)}).groupby('query_id').sum().reindex(qs,fill_value=0)
        bn=arrays.n.to_numpy()[draw].sum(1);bd=arrays.d.to_numpy()[draw].sum(1);valid=bd>0
        bs=np.full(len(draw),np.nan);bs[valid]=bn[valid]/bd[valid]
        return arrays.n.sum()/arrays.d.sum(),np.quantile(bs[valid],[.025,.975]),bs
    summaries=frame(HERE/'results/rq1_summary.csv');averages=frame(HERE/'results/rq1_pair_averages.csv',dtype={'product_id':str});bootstrap_checks=0;d1={}
    for r in summaries.itertuples():
        f=averages[averages.model.eq(r.model)&averages.condition.eq(r.condition)&averages.stratum.eq(r.stratum)]
        den=~f.in_top20_A if r.metric=='entry' else f.in_top20_A if r.metric=='exit' else np.ones(len(f))
        point,ci,bs=estimate(f,f[r.metric],den);close([r.estimate,r.ci_low,r.ci_high],[point,*ci]);bootstrap_checks+=1
        if r.stratum=='all':d1[(r.model,r.condition,r.metric)]=(point,bs)
        selected=raw[raw.model.eq(r.model)&raw.condition.eq(r.condition)]
        if r.stratum=='all':
            expected_events=int(((~selected.in_top20_A)&selected.in_top20).sum()) if r.metric=='entry' else int((selected.in_top20_A&(~selected.in_top20)).sum()) if r.metric=='exit' else int(selected.in_top20.ne(selected.in_top20_A).sum()) if r.metric=='crossing' else int((selected.in_top20.astype(int)-selected.in_top20_A.astype(int)).sum())
            assert r.raw_events==expected_events and r.raw_trial_denominator==5*r.denominator_pairs
    for r in frame(HERE/'results/rq1_C_minus_O.csv').itertuples():
        c,cb=d1[(r.model,'C',r.metric)];o,ob=d1[(r.model,'O',r.metric)]
        close([r.estimate,r.ci_low,r.ci_high],[c-o,*np.nanquantile(cb-ob,[.025,.975])]);bootstrap_checks+=1
    d2={}
    for r in frame(HERE/'results/rq2_summary.csv').itertuples():
        f=choices[choices.source.eq(r.source)&choices.condition.eq(r.condition)&choices.budget.eq(r.budget)]
        a=f[r.evaluation+'_inclusion_A'];v=f[r.evaluation+'_inclusion']
        num=(v.astype(int)-a.astype(int)) if r.metric=='delta_inclusion' else ((~a)&v) if r.metric=='entry' else (a&(~v)) if r.metric=='exit' else a.ne(v) if r.metric=='crossing' else f.modified
        den=~a if r.metric=='entry' else a if r.metric=='exit' else np.ones(len(f))
        point,ci,bs=estimate(f,num,den);close([r.estimate,r.ci_low,r.ci_high],[point,*ci]);assert r.raw_events==int(num.sum())
        if r.metric=='delta_inclusion':d2[(r.source,r.evaluation,r.condition,r.budget)]=(point,bs)
        bootstrap_checks+=1
    for r in frame(HERE/'results/rq2_C_minus_O.csv').itertuples():
        c,cb=d2[(r.source,r.evaluation,'C',r.budget)];o,ob=d2[(r.source,r.evaluation,'O',r.budget)]
        close([r.estimate_pp,r.ci_low_pp,r.ci_high_pp],100*np.array([c-o,*np.nanquantile(cb-ob,[.025,.975])]))
        bootstrap_checks+=1
    timed('qa/final_verification.json',t,passed=True,preserved_files=len(preserved),formal_rows=len(raw),baseline_full_sort_checks=baselinechecks,
        crossing_full_sort_checks=fullsorts,source_only_choices_verified=verified_choices,bootstrap_estimates_and_contrasts_verified=bootstrap_checks,
        query_clusters=len(qs),draws=len(draw),sample_manifest_sha256=sha(HERE/'sample_manifest.json'))
    files={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and 'cache' not in p.relative_to(HERE).parts and '__pycache__' not in p.parts and p.name!='delivery_manifest.json' and not p.name.startswith('continuation_')}
    dump('delivery_manifest.json',dict(created_utc=now(),files=files,cache_not_bundled='Local forward arrays are reproducible from pinned prerequisites and text/context descriptors.',volatile_files_excluded='qa/continuation_* supervisor logs/status can continue updating after verification; they are operational, not analysis inputs.'))
    print('FINAL VERIFICATION PASSED',len(raw),'raw rows;',verified_choices,'source-only choices;',bootstrap_checks,'bootstrap checks',flush=True)

if __name__=='__main__':main()
