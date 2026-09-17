"""Recompute all states and inclusion from exact ranks, including K=100."""
from common import *

def states(f,k):
    inc=f[STEMS].to_numpy()<=k
    z=f[['query_id','product_id']].copy()
    z['Always']=inc.all(1).astype(float);z['VI']=(inc.any(1)&~inc.all(1)).astype(float);z['Never']=(~inc.any(1)).astype(float)
    z['mean_schedule_inclusion']=inc.mean(1);z['C0_inclusion']=inc[:,0].astype(float)
    assert np.all(z[['Always','VI','Never']].sum(1)==1)
    return z

def main():
    fits=pd.read_csv(DATA/'all_seven_token_lengths.csv',dtype={'product_id':str})
    rows=[];pqout=[];audit=[];sources=[]
    for ds in ['wands','esci']:
      p=products(ds)
      for model in ['minilm','bge_base','gte_modernbert']:
        raw=rawwide(ds,model)
        path=ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet'
        tgt=pairread(path).set_index(['query_id','product_id']).sort_index()
        raw=raw.set_index(['query_id','product_id']).sort_index()
        assert raw.index.equals(tgt.index)
        assert np.array_equal(raw.C0,tgt.C0)
        for s in STEMS: assert np.array_equal(raw[s],tgt['whole_'+s])
        audit.append(dict(dataset=ds,model=model,pairs=len(raw),C0_rank_exact_match=True,all_saved_whole_ranks_exact_match=True,
                          competitors='raw C0',catalog_size=len(p),target_old_entry_removed='verified in source code and tied-score tests'))
        sources.append(source(path))
        ids=set(fits.loc[(fits.dataset==ds)&(fits.model==model)&fits.fully_fits,'product_id'])
        for intervention,f in [('catalog_wide',raw.reset_index()),('target_only',tgt.reset_index())]:
          for pop,queryids in populations(ds):
            for scope in ['full','fully_fitting']:
              z=f[f.query_id.isin(queryids)].copy()
              if scope=='fully_fitting':z=z[z.product_id.isin(ids)]
              eligible=sorted(map(int,z.query_id.unique()));w=weights(eligible)
              for k in [20,100]:
                ps=states(z,k);q=ps.drop(columns='product_id').groupby('query_id',sort=True).mean()
                n=ps.groupby('query_id',sort=True).size().to_numpy()
                meta=dict(dataset=ds,model=model,encoder_profile='native',population=pop,target_support=scope,K=k,
                  intervention=intervention,competitors='raw C0, fixed' if intervention=='target_only' else 'same raw schedule as target',
                  catalog_size=len(p),eligible_queries=len(q),highest_relevance_pairs=len(z),bootstrap_draws=DRAWS,bootstrap_seed=SEED)
                pqout.append(q.reset_index().assign(**meta,relevant_count=n))
                for agg in ['query_macro','pair_micro']:
                  row={**meta,'aggregation':agg}
                  for col in q.columns:
                    v=q[col].to_numpy()
                    if agg=='query_macro':value=v.mean();b=w@v/len(q)
                    else:value=(v*n).sum()/n.sum();b=(w@(v*n))/(w@n)
                    label=col
                    if intervention=='catalog_wide':label={'C0_inclusion':'C0_Recall','mean_schedule_inclusion':'mean_over_seven_schedules_Recall'}.get(col,col)
                    row[label]=float(value);row[label+'_ci_low'],row[label+'_ci_high']=ci(b)
                  rows.append(row)
        print('STATES',ds,model,flush=True)
    pd.DataFrame(rows).to_csv(DATA/'inclusion_states_summary.csv',index=False)
    pd.concat(pqout,ignore_index=True).to_csv(DATA/'inclusion_states_per_query.csv',index=False)
    pd.DataFrame(audit).to_csv(DATA/'target_only_integrity.csv',index=False)
    dump(DATA/'inclusion_states_sources.json',dict(sources=sources,token_lengths=source(DATA/'all_seven_token_lengths.csv'),
      definitions='Always: all ranks <= K; VI: min rank <= K < max rank; Never: all ranks > K. Seven raw schedules. All three use the same aggregation and denominators.',
      scope='Filtering targets only; all original competitors remain. Full-minus-fitting is descriptive, not a causal effect.',
      target_inclusion='Mean across independent counterfactual target indexes; not Recall from a shared index.',
      phase_VI_scope='Phase VI method target sensitivity uses that method own C0 competitors. Its raw Original baseline agrees with raw C0; other methods are not pooled with this raw intervention.',
      outputs=[source(DATA/x) for x in ['inclusion_states_summary.csv','inclusion_states_per_query.csv','target_only_integrity.csv']]))

if __name__=='__main__':main()
