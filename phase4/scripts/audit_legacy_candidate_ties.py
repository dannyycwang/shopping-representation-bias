from common import *
from threadpoolctl import threadpool_limits
threadpool_limits(4)
records=[]
for ds in ['wands','esci']:
    p,q,j=load(ds);qv,_=original_cache(ds,'bge_base');pi={str(x['product_id']):i for i,x in enumerate(p)};highest='Exact' if ds=='wands' else 'E';hg={qid:set(g.product_id.astype(str)) for qid,g in j[j.label.eq(highest)].groupby('query_id')}
    for stem in ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']:
        if stem=='C0':_,pv=original_cache(ds,'bge_base')
        else:
            fs=list((P2/'results/embeddings').glob(f'{ds}_bge_base_native_{stem}_*.npy'));assert len(fs)==1;pv=np.load(fs[0])
        score=qv@pv.T;order,_=ranks(score);old=pd.read_parquet(ROOT/f'phase3/results/phase3_reranking/{ds}_{stem}_dense_top100.parquet')
        for qi,qid in enumerate(q.query_id):
            g=old[old.query_id.eq(qid)].sort_values('dense_rank');ix=np.array([pi[str(x)] for x in g.product_id]);oldids=set(g.product_id.astype(str));newids={str(p[i]['product_id']) for i in order[qi,:100]};relevant=hg.get(qid,set());maxerr=float(np.max(abs(score[qi,ix]-g.dense_score.to_numpy())))
            records.append({'dataset':ds,'variant':stem,'query_id':int(qid),'set_difference':oldids!=newids,'order_difference':not np.array_equal(ix,order[qi,:100]),'relevant_added':len((newids-oldids)&relevant),'relevant_removed':len((oldids-newids)&relevant),'max_score_error':maxerr})
df=pd.DataFrame(records);df.to_csv(OUT/'legacy_candidate_reconciliation_per_query.csv',index=False)
df.groupby(['dataset','variant']).agg(set_differences=('set_difference','sum'),order_differences=('order_difference','sum'),relevant_added=('relevant_added','sum'),relevant_removed=('relevant_removed','sum'),max_score_error=('max_score_error','max')).to_csv(OUT/'legacy_candidate_reconciliation.csv')
print('Audited all fourteen legacy candidate cells')
