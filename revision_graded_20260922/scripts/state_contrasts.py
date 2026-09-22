"""Paired P1 persistent-state differences from completed per-query states."""
from common import *

def main():
    verify_protocol();f=pd.read_parquet(HERE/'data/persistent_states_per_query.parquet')
    cols=['persistent_inclusion','persistent_omission','VI','mean_recall','source_recall'];rows=[];records=[]
    for ds in ['wands','esci']:
        for model in ['minilm','bge_base']:
            for rule in RULES:
                method='canonical_hybrid_'+rule
                for k in [20,100]:
                    def get(m):return f[(f.dataset==ds)&(f.model==model)&(f.method==m)&(f.K==k)].set_index('query_id').loc[support(ds)]
                    alt=get(method)
                    for ref in [rule,'BM25','raw_hybrid']:
                        base=get(ref);assert alt.index.equals(base.index) and alt.n_highest.equals(base.n_highest)
                        d=alt[cols]-base[cols];n=alt.n_highest.to_numpy();v=d.to_numpy();w=weights(ds)
                        for agg in ['query_macro','pair_micro']:
                            point=v.mean(0) if agg=='query_macro' else n@v/n.sum()
                            boot=w@v/len(v) if agg=='query_macro' else (w@(v*n[:,None]))/(w@n)[:,None]
                            lo,hi=np.quantile(boot,[.025,.975],axis=0)
                            for col,m,l,h in zip(cols,point,lo,hi):rows.append(dict(dataset=ds,model=model,method=method,reference=ref,K=k,aggregation=agg,metric=col,delta=m,ci_low=l,ci_high=h,eligible_queries=len(d),highest_pairs=int(n.sum()),interval='uncorrected 95% paired query-cluster percentile',bootstrap_seed=SEED,bootstrap_draws=DRAWS))
                        for qid,row in d.iterrows():records.append(dict(dataset=ds,model=model,method=method,reference=ref,K=k,query_id=int(qid),n_highest=int(alt.loc[qid,'n_highest']),**row.to_dict()))
    pd.DataFrame(rows).to_csv(HERE/'tables/canonical_hybrid_state_contrasts.csv',index=False)
    pd.DataFrame(records).to_parquet(HERE/'data/canonical_hybrid_state_contrasts_per_query.parquet',index=False)
    print('STATE CONTRASTS',len(rows),'summary records',flush=True)

if __name__=='__main__':main()
