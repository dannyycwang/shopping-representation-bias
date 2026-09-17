from common import *
from verify_target_rank_sources import original_rank_function

fn=original_rank_function();audit=pd.read_csv(DATA/'target_rank_reconstruction_audit.csv');rows=[]
for ds,model in audit.loc[audit.reconstructed_rank_mismatches>0,['dataset','model']].drop_duplicates().itertuples(index=False):
    p=products(ds);pi={str(x['product_id']):i for i,x in enumerate(p)}
    q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv');qi={int(x):i for i,x in enumerate(q.query_id)}
    prov=next(x for x in PROVENANCE['sources'] if x['dataset']==ds and x['model']==model)
    scores=np.load(ROOT/prov['query_embedding'])@np.load(ROOT/prov['product_embedding']).T
    target=pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet').set_index(['query_id','product_id']).sort_index()
    for s in audit.loc[(audit.dataset==ds)&(audit.model==model)&(audit.reconstructed_rank_mismatches>0),'schedule']:
        raw=pairread(ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet').set_index(['query_id','product_id']).loc[target.index].reset_index()
        for qid,g in raw.groupby('query_id',sort=False):
            ix=np.array([pi[x] for x in g.product_id]);actual=fn(scores[qi[int(qid)]],ix,g.score.to_numpy(dtype='f4'))
            old=target[s].to_numpy()[g.index]
            for pos in np.flatnonzero(actual!=old):
                row=g.iloc[pos]
                rows.append(dict(dataset=ds,model=model,schedule=s,query_id=int(qid),product_id=row.product_id,
                  inherited_rank=int(old[pos]),reconstructed_rank=int(actual[pos]),
                  membership20_changed=bool((old[pos]<=20)!=(actual[pos]<=20)),
                  membership100_changed=bool((old[pos]<=100)!=(actual[pos]<=100)),
                  authority='Inherited ranks retained; current-runtime fp32 reconstruction is an audit only.'))
pd.DataFrame(rows).to_csv(DATA/'target_rank_precision_differences.csv',index=False)
print(rows,flush=True)
