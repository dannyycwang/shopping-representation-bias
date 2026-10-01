"""Final independent checks of frozen selection, outcomes, uncertainty and delivery."""
from common import *
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-original-worktree',action='store_true',help='Also verify unrelated edits preserved in the original working checkout.')
    args=ap.parse_args()
    verify_freeze();checks={};profile=read(HERE/'execution_profile.json')
    if args.check_original_worktree:
        for p,h in profile['preexisting_tracked_edits'].items():assert sha(ROOT/p)==h,p
        checks['preexisting_tracked_edits_preserved']=True
    pop=pd.read_parquet(HERE/'data/eligibility_population.parquet')
    cohort=pd.read_csv(HERE/'cohort_manifest.csv',dtype={'product_id':str});plan=pd.read_parquet(HERE/'data/variant_plan.parquet')
    for (ds,m),g in pop.groupby(['dataset','model']):
        eligible=g[g.eligible].copy()
        assert all(eligible['all_variants_tokenized_'+m])
        assert (eligible['max_tokens_'+m]<=MODEL_SPECS[m]['max_tokens']).all()
        eligible['qh']=[selection_hash('query',ds,q) for q in eligible.query_id]
        eligible['th']=[selection_hash('target',ds,q,p) for q,p in zip(eligible.query_id,eligible.product_id)]
        qs=eligible[['query_id','qh']].drop_duplicates().sort_values(['qh','query_id']).head(128).query_id
        chosen=eligible[eligible.query_id.isin(qs)].sort_values(['qh','th','product_id']).groupby('query_id',sort=False).head(16)
        actual=cohort[cohort.dataset.eq(ds)&cohort.model.eq(m)]
        assert set(zip(chosen.query_id,chosen.product_id))==set(zip(actual.query_id,actual.product_id))
    checks['cohort_hash_selection_reconstructed']=True
    assert (plan.tokens<=plan.context_limit).all()
    assert not plan.duplicated(['dataset','model','product_id','text_sha256']).any()
    assert (plan.text.map(digest)==plan.text_sha256).all()
    # Independently reconstruct the *entire* expected string from source entries,
    # rather than accepting a multiset check as proof of fixed-block order.
    for ds in DATASETS:
        products,_,_=load(ds);lookup={str(p['product_id']):p for p in products}
        unique=plan[plan.dataset.eq(ds)].drop_duplicates(['product_id','text_sha256'])
        for r in unique.itertuples():
            product=lookup[r.product_id];positions=json.loads(r.positions)
            if r.variant=='C0':assert r.text==REP._plain(product)
            else:
                for position in positions:
                    attrs=list(product['attributes']);i=position-1
                    attrs[i],attrs[i+1]=attrs[i+1],attrs[i]
                    assert r.text==REP._plain(product,attrs),(ds,r.product_id,r.variant)
            assert r.c0_sha256==digest(REP._plain(product))
    checks['full_serializations_reconstructed_from_source']=True
    audited=pd.read_parquet(HERE/'data/representation_audit.parquet')
    flags=['entry_multiset','keys_values','numeric_multiset','fixed_text','adjacent_transposition','other_relative_order','identity','canonical','repository_audit','pass','no_truncation']
    assert audited[flags].all().all() and len(audited)==plan.variant.ne('C0').sum()
    checks['every_tested_serialization_audited_and_fitting']=True
    counts=plan[plan.variant.ne('C0')].groupby(['dataset','model','product_id']).size()
    for r in cohort.itertuples():assert counts.loc[r.dataset,r.model,r.product_id]==r.distinct_swaps
    f=pd.read_parquet(HERE/'data/per_variant.parquet');targets=pd.read_parquet(HERE/'data/per_target.parquet')
    q=pd.read_parquet(HERE/'data/per_query.parquet');est=pd.read_csv(HERE/'data/aggregate_estimates.csv')
    assert len(f)==int(cohort.distinct_swaps.sum())
    assert not f.duplicated(['dataset','model','query_id','product_id','variant']).any()
    assert not targets.duplicated(['dataset','model','query_id','product_id','K']).any()
    for k in [20,100]:
        assert ((f.c0_rank<=k)==f[f'c0_included_{k}']).all()
        assert ((f.swap_rank<=k)==f[f'swap_included_{k}']).all()
        assert (f[f'flip_{k}']==f[f'c0_included_{k}'].ne(f[f'swap_included_{k}'])).all()
        assert ((f[f'confirmed_{k}']|f[f'unresolved_{k}'])==f[f'flip_{k}']).all()
        for name in ['c0','swap']:
            expected=f[f'{name}_score']>f[f'threshold_score_{k}']
            expected |= f[f'{name}_score'].eq(f[f'threshold_score_{k}']) & f.catalog_index.lt(f[f'threshold_index_{k}'])
            assert (expected==f[f'{name}_included_{k}']).all()
        a=f.groupby(['dataset','model','query_id','product_id'])[f'flip_{k}'].agg(['max','mean'])
        t=targets[targets.K.eq(k)].set_index(['dataset','model','query_id','product_id']).loc[a.index]
        np.testing.assert_allclose(t.A,a['max']);np.testing.assert_allclose(t.F,a['mean'])
    assert np.allclose(targets.A,targets.loss+targets.gain)
    checks['ranks_ties_self_exclusion_and_estimands_consistent']=True
    for (ds,m,k),g in q.groupby(['dataset','model','K']):
        g=g.sort_values('query_id');rng=np.random.default_rng(2026100101)
        ix=rng.integers(0,len(g),size=(10000,len(g)))
        for metric in ['A','F','loss','gain','baseline_inclusion','A_confirmed','F_confirmed']:
            r=est[est.dataset.eq(ds)&est.model.eq(m)&est.K.eq(k)&est.metric.eq(metric)].iloc[0]
            value=g[metric].to_numpy();ci=np.quantile(value[ix].mean(axis=1),[.025,.975])
            np.testing.assert_allclose([r.estimate,r.ci_low,r.ci_high],[value.mean(),*ci],atol=2e-15,rtol=1e-12)
    checks['bootstrap_intervals_recomputed']=True
    for m in MODELS:
        rr=pd.read_parquet(HERE/f'data/{m}_numerical_repeats.parquet');g=f[f.model.eq(m)]
        crossing=g[g.flip_20|g.flip_100]
        assert len(rr)==3*len(crossing)
        if len(rr):
            assert (rr.groupby(['dataset','query_id','product_id','variant']).size()==3).all()
            merged=rr.merge(g,on=['dataset','model','query_id','product_id','variant'],validate='many_to_one',suffixes=('_repeat','_saved'))
            for k in [20,100]:
                reproduces=(merged[f'c0_included_{k}_repeat']==merged[f'c0_included_{k}_saved']) & (merged[f'swap_included_{k}_repeat']==merged[f'swap_included_{k}_saved'])
                all_reproduce=merged.assign(reproduces=reproduces).groupby(['dataset','query_id','product_id','variant']).reproduces.all()
                gg=crossing.set_index(['dataset','query_id','product_id','variant']).loc[all_reproduce.index]
                assert ((all_reproduce & gg[f'flip_{k}'])==gg[f'confirmed_{k}']).all()
        assert read(HERE/f'qa/{m}_pre_sweep_gate.json')['passed']
    checks['every_crossing_repeated_and_classified']=True
    case=read(HERE/'data/case_selection.json')
    candidates=f[f.confirmed_20 & f.c0_included_20.eq(1)]
    if candidates.empty:candidates=f[f.confirmed_20 & f.c0_included_20.eq(0)]
    if len(candidates):
        row=candidates.sort_values(['case_hash','swap_hash']).iloc[0]
        assert case['selected']['case_hash']==row.case_hash and case['selected']['swap_hash']==row.swap_hash
    else:assert case['status']!='selected'
    checks['case_hash_selection_verified']=True
    pdfs=list((HERE/'figures').glob('*.pdf'))
    assert len(pdfs)==(3 if len(candidates) else 2)
    for p in pdfs:
        info=subprocess.check_output(['pdfinfo',str(p)]).decode('utf8',errors='replace')
        assert re.search(r'Pages:\s+1\b',info)
        assert p.with_suffix('.png').exists()
        assert (HERE/'qa/rendered'/p.with_suffix('.png').name).exists()
    checks['pdfs_rendered']=True
    review=read(HERE/'qa/visual_review.json');assert review['passed']
    for p,h in review['pdf_sha256'].items():assert sha(HERE/p)==h,p
    checks['current_pdf_visual_review_recorded']=True
    inventory=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['output_manifest.csv','final_validation.json','cache_manifest.csv']:continue
        if 'cache' in p.relative_to(HERE).parts or p.suffix=='.log' or 'rendered' in p.parts:continue
        inventory.append(dict(path=str(p.relative_to(HERE)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)))
    csv(pd.DataFrame(inventory),HERE/'output_manifest.csv')
    cache=[]
    for p in sorted((HERE/'cache').rglob('*')):
        if p.is_file() and p.suffix in ['.npy','.npz']:
            cache.append(dict(path=str(p.relative_to(HERE)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)))
    csv(pd.DataFrame(cache),HERE/'cache_manifest.csv')
    dump(HERE/'qa/final_validation.json',dict(completed_utc=now(),passed=all(checks.values()),checks=checks,
        original_worktree_preservation_check=args.check_original_worktree,
        pairs=len(cohort),variant_rows=len(f),audit_rows=len(audited),figures=len(pdfs),tracked_inventory_files=len(inventory)))
    print('FINAL VALIDATION PASSED',checks)

if __name__=='__main__':main()
