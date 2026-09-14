from common import *
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def analyze(ds,model):
    p,q,j=load(ds);folder=OUT/ds;choice=json.loads((P4/'config/selection.json').read_text());names=list(dict.fromkeys(['Original','hybrid','canonical','canonical_raw','set_mean',choice['rule'],choice['method']]))
    base=pd.read_parquet(folder/f'{model}_Original_pairs.parquet');highest='Exact' if ds=='wands' else 'E';ids=q.loc[~dev_mask(q),'query_id'] if ds=='wands' else q.query_id
    rows=[];trans=[];metrics={}
    for method in names+['BM25']:
        stem='BM25' if method=='BM25' else f'{model}_{method}'
        pq=pd.read_csv(folder/f'{stem}_per_query.csv');pq=pq[pq.query_id.isin(ids)].set_index('query_id');metrics[method]=pq
        for metric in ['cNDCG@10',*[f'Recall@{k}' for k in KS]]:
            mu,lo,hi=boot(pq[metric]);rows.append({'dataset':ds,'encoder':model,'method':method,'metric':metric,'queries':pq[metric].notna().sum(),'mean':mu,'ci_low':lo,'ci_high':hi})
        f=pd.read_parquet(folder/f'{stem}_pairs.parquet');joined=base.merge(f[['query_id','product_id','rank']],on=['query_id','product_id'],suffixes=('_base','_new'),validate='one_to_one');h=joined[(joined.label.eq(highest))&joined.query_id.isin(ids)].copy()
        for k in KS:
            h['rescued']=(h.rank_base>k)&(h.rank_new<=k);h['newly_missed']=(h.rank_base<=k)&(h.rank_new>k);h['net']=h.rescued.astype(int)-h.newly_missed.astype(int)
            for qid,g in h.groupby('query_id'):
                trans.append({'dataset':ds,'encoder':model,'method':method,'query_id':qid,'K':k,'n_highest':len(g),'rescued':g.rescued.sum(),'newly_missed':g.newly_missed.sum(),'net':g.net.sum(),'rescued_fraction':g.rescued.mean(),'missed_fraction':g.newly_missed.mean(),'net_fraction':g.net.mean()})
    pd.DataFrame(rows).to_csv(OUT/f'{ds}_{model}_retrieval_summary.csv',index=False);pd.DataFrame(trans).to_csv(OUT/f'{ds}_{model}_transitions.csv',index=False)
    contrasts=[]
    for a,b in list(dict.fromkeys([(choice['method'],'Original'),(choice['method'],'hybrid'),(choice['rule'],'Original'),('set_mean','canonical')])):
        for metric in ['Recall@100','cNDCG@10']:
            joined=metrics[a][[metric]].join(metrics[b][[metric]],lsuffix='_a',rsuffix='_b');mu,lo,hi=boot(joined.iloc[:,0]-joined.iloc[:,1]);contrasts.append({'dataset':ds,'encoder':model,'method_a':a,'method_b':b,'metric':metric,'delta':mu,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(contrasts).to_csv(OUT/f'{ds}_{model}_primary_contrasts.csv',index=False)
    # Selected merchant-side ordering, one product at a time against Original competitors.
    sys.path.insert(0,str(ROOT/'phase3/scripts'))
    from analyze_target_only_permutations import rank_target
    selected=choice['rule'];f=pd.read_parquet(folder/f'{model}_{selected}_pairs.parquet');s=np.load(folder/f'{model}_Original_scores.npy');pi={str(x['product_id']):i for i,x in enumerate(p)};qi={int(v):i for i,v in enumerate(q.query_id)}
    target=base[base.label.eq(highest)].merge(f[['query_id','product_id','score']],on=['query_id','product_id'],suffixes=('_C0','_rule'),validate='one_to_one')
    tr=np.zeros(len(target),dtype=int)
    for qid,g in target.groupby('query_id'):
        idx=np.array([pi[str(v)] for v in g.product_id]);tr[g.index]=rank_target(s[qi[int(qid)]],idx,g.score_rule.to_numpy(dtype='f4'))
    target['target_rank']=tr;target.to_parquet(OUT/f'{ds}_{model}_selected_rule_target_only.parquet',index=False)
    ts=[]
    for k in KS:
        h=target[target.query_id.isin(ids)].copy();h['gain']=(h.target_rank<=k).astype(int)-(h['rank']<=k).astype(int);h['cross']=(h.target_rank<=k)!=(h['rank']<=k)
        mu,lo,hi=boot(h.groupby('query_id').gain.mean());ts.append({'dataset':ds,'model':model,'K':k,'net_macro_recall':mu,'ci_low':lo,'ci_high':hi,'crossing_micro':h.cross.mean()})
    pd.DataFrame(ts).to_csv(OUT/f'{ds}_{model}_target_only_summary.csv',index=False)
    print('COMPARISONS',ds,model,flush=True)

if __name__=='__main__':
    for ds in ['wands','esci']:
        for model in ['bge_base','minilm']:analyze(ds,model)
