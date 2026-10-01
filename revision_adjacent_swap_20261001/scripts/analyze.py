"""Prespecified query-macro estimands, cluster bootstrap, and hash-selected case."""
from common import *

METRICS=['A','F','loss','gain','baseline_inclusion','A_confirmed','F_confirmed','loss_confirmed','gain_confirmed']
def summarize_queries(frame,meta):
    q=frame.sort_values('query_id');values=q[METRICS].to_numpy(dtype=float)
    rng=np.random.default_rng(2026100101);indices=rng.integers(0,len(q),size=(10000,len(q)))
    samples=values[indices].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975],axis=0)
    point=values.mean(axis=0)
    return [dict(**meta,metric=metric,estimate=float(e),ci_low=float(l),ci_high=float(h),
        queries=len(q),targets=int(q.targets.sum()),swaps=int(q.swaps.sum()),bootstrap_seed=2026100101,
        bootstrap_resamples=10000,interval='95% unadjusted query-cluster percentile') for metric,e,l,h in zip(METRICS,point,lo,hi)]

def query_means(f):
    g=f.groupby('query_id',sort=True)
    q=g[METRICS].mean();q['targets']=g.size();q['swaps']=g.distinct_swaps.sum()
    return q.reset_index()

def main():
    verify_freeze();out=[]
    for m in MODELS:
        report=read(HERE/f'qa/{m}_numerical_report.json')
        assert sha(HERE/f'data/{m}_variants.parquet')==report['results_sha256']
        out.append(pd.read_parquet(HERE/f'data/{m}_variants.parquet'))
    f=pd.concat(out,ignore_index=True);f.to_parquet(HERE/'data/per_variant.parquet',index=False)
    rows=[]
    for (ds,m,q,p),g in f.groupby(['dataset','model','query_id','product_id'],sort=True):
        r=g.iloc[0];assert len(g)==r.distinct_swaps
        for k in [20,100]:
            flips=g[f'flip_{k}'];confirmed=g[f'confirmed_{k}'];unresolved=g[f'unresolved_{k}']
            assert ((confirmed|unresolved)==flips).all() and not (confirmed&unresolved).any()
            affected=bool(flips.any());confirmed_any=bool(confirmed.any());inside=bool(r[f'c0_included_{k}'])
            rows.append(dict(dataset=ds,model=m,query_id=int(q),product_id=p,K=k,distinct_swaps=len(g),
                attribute_count=int(r.attribute_count),c0_rank=int(r.c0_rank),c0_score=r.c0_score,c0_margin=r[f'c0_margin_{k}'],
                c0_tokens=int(r.c0_tokens),maximum_tokens=int(r.max_tokens),context_limit=int(r.context_limit),
                A=float(affected),F=float(flips.mean()),loss=float(affected and inside),gain=float(affected and not inside),
                baseline_inclusion=float(inside),A_confirmed=float(confirmed_any),F_confirmed=float(confirmed.mean()),
                loss_confirmed=float(confirmed_any and inside),gain_confirmed=float(confirmed_any and not inside),
                observed_flip_count=int(flips.sum()),confirmed_flip_count=int(confirmed.sum()),unresolved_flip_count=int(unresolved.sum()),
                minimum_swap_rank=int(g.swap_rank.min()),maximum_swap_rank=int(g.swap_rank.max()),
                mean_rank_change=float(g.rank_change.mean()),maximum_absolute_rank_change=int(g.rank_change.abs().max()),
                minimum_swap_margin=float(g[f'swap_margin_{k}'].min()),maximum_swap_margin=float(g[f'swap_margin_{k}'].max()),
                case_hash=r.case_hash))
    targets=pd.DataFrame(rows)
    targets['swap_count_bin']=pd.cut(targets.distinct_swaps,[0,1,3,7,15,np.inf],labels=['1','2-3','4-7','8-15','16+']).astype(str)
    targets['c0_rank_bin']=pd.cut(targets.c0_rank,[0,10,20,40,np.inf],labels=['1-10','11-20','21-40','>40']).astype(str)
    targets.to_parquet(HERE/'data/per_target.parquet',index=False);csv(targets,HERE/'data/per_target.csv')
    qframes=[];estimates=[];event_counts=[];breakdowns=[];descriptives=[]
    for (ds,m,k),g in targets.groupby(['dataset','model','K'],sort=True):
        meta=dict(dataset=ds,model=m,K=int(k));q=query_means(g);qframes.append(q.assign(**meta));estimates.extend(summarize_queries(q,meta))
        event_counts.append(dict(**meta,queries=g.query_id.nunique(),targets=len(g),swaps=int(g.distinct_swaps.sum()),
            affected_targets=int(g.A.sum()),confirmed_affected_targets=int(g.A_confirmed.sum()),
            loss_targets=int(g.loss.sum()),gain_targets=int(g.gain.sum()),confirmed_loss_targets=int(g.loss_confirmed.sum()),
            confirmed_gain_targets=int(g.gain_confirmed.sum()),observed_flips=int(g.observed_flip_count.sum()),
            confirmed_flips=int(g.confirmed_flip_count.sum()),unresolved_flips=int(g.unresolved_flip_count.sum()),
            unresolved_targets=int(g.unresolved_flip_count.gt(0).sum()),affected_queries=int(q.A.gt(0).sum())))
        for field in ['swap_count_bin','c0_rank_bin']:
            for value,sub in g.groupby(field,sort=True):
                means=query_means(sub)
                breakdowns.append(dict(**meta,breakdown=field,bin=value,queries=len(means),targets=len(sub),swaps=int(sub.distinct_swaps.sum()),
                    affected_targets=int(sub.A.sum()),confirmed_targets=int(sub.A_confirmed.sum()),
                    **{metric:float(means[metric].mean()) for metric in METRICS}))
        for field in ['attribute_count','distinct_swaps','c0_rank','c0_margin','mean_rank_change','maximum_absolute_rank_change']:
            a=g[field];descriptives.append(dict(**meta,variable=field,targets=len(g),minimum=float(a.min()),p25=float(a.quantile(.25)),
                median=float(a.median()),p75=float(a.quantile(.75)),maximum=float(a.max()),mean=float(a.mean())))
    queries=pd.concat(qframes,ignore_index=True);csv(queries,HERE/'data/per_query.csv');queries.to_parquet(HERE/'data/per_query.parquet',index=False)
    csv(pd.DataFrame(estimates),HERE/'data/aggregate_estimates.csv');csv(pd.DataFrame(event_counts),HERE/'data/event_counts.csv')
    csv(pd.DataFrame(breakdowns),HERE/'data/descriptive_breakdowns.csv');csv(pd.DataFrame(descriptives),HERE/'data/supporting_distributions.csv')
    # All qualifying settings are in the single prespecified case lottery.
    candidates=f[f.confirmed_20 & f.c0_included_20.eq(1)];direction='loss'
    if candidates.empty:candidates=f[f.confirmed_20 & f.c0_included_20.eq(0)];direction='gain'
    if candidates.empty:
        dump(HERE/'data/case_selection.json',dict(status='No numerically confirmed K=20 crossing; no illustrative crossing manufactured.'))
    else:
        pairs=candidates[['dataset','model','query_id','product_id','case_hash']].drop_duplicates().sort_values(['case_hash','dataset','model','query_id','product_id'])
        pair=pairs.iloc[0];eligible=candidates[(candidates.dataset==pair.dataset)&(candidates.model==pair.model)&(candidates.query_id==pair.query_id)&(candidates.product_id==pair.product_id)]
        chosen=eligible.sort_values(['swap_hash','variant']).iloc[0]
        selected=f[(f.dataset==pair.dataset)&(f.model==pair.model)&(f.query_id==pair.query_id)&(f.product_id==pair.product_id)]
        csv(selected,HERE/'data/case_all_swap_outcomes.csv')
        plan=pd.read_parquet(HERE/'data/variant_plan.parquet');pp=plan[(plan.dataset==pair.dataset)&(plan.model==pair.model)&(plan.product_id==pair.product_id)]
        serializations={r.variant:dict(text=r.text,text_sha256=r.text_sha256,tokens=int(r.tokens),positions=json.loads(r.positions)) for r in pp.itertuples()}
        ps,qs,_=load(pair.dataset);product=next(x for x in ps if str(x['product_id'])==pair.product_id)
        pos=json.loads(chosen.positions)[0]
        case=dict(status='selected',selection_conditional_on_confirmed_crossing=True,direction=direction,
            eligible_crossing_pairs=len(pairs),eligible_crossing_swaps=len(candidates),selected=chosen.to_dict(),
            query=qs.set_index('query_id').loc[pair.query_id,'query'],product_title=product['title'],
            original_entries=product['attributes'][pos-1:pos+1],swapped_entries=product['attributes'][pos-1:pos+1][::-1],
            all_serializations=serializations,competitors='All other catalog products remain at C0; target original entry removed before insertion.',
            unchanged='All fixed fields, all entry contents and multiplicities, separators, and the relative order of every other entry are unchanged.')
        dump(HERE/'data/case_selection.json',case)
    assert np.allclose(targets.A,targets.loss+targets.gain)
    assert np.allclose(queries.A,queries.loss+queries.gain)
    dump(HERE/'qa/analysis_validation.json',dict(completed_utc=now(),variant_rows=len(f),target_cutoff_rows=len(targets),query_cutoff_rows=len(queries),
        loss_plus_gain_equals_A=True,confirmation_partitions_observed_flips=True,
        query_cluster_bootstrap='10000 resamples, seed 2026100101, retained complete per-query measurements',
        unresolved_variant_cutoff_events=int(sum(f[f'unresolved_{k}'].sum() for k in [20,100]))))
    print(pd.DataFrame(estimates).query('K == 20 and metric in ["A","F"]')[['dataset','model','metric','estimate','ci_low','ci_high']].to_string(index=False))

if __name__=='__main__':main()
