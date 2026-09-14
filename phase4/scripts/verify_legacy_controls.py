from common import *
from common import _plain
from threadpoolctl import threadpool_limits
threadpool_limits(4)
rows=[]
for ds in ['wands','esci']:
    p,q,j=load(ds);qv,pv=original_cache(ds,'bge_base');score=qv@pv.T;order,_=ranks(score)
    old=pd.read_parquet(ROOT/f'phase3/results/phase3_reranking/{ds}_C0_dense_top100.parquet');pi={str(x['product_id']):i for i,x in enumerate(p)};bad=0;different=0
    for qi,qid in enumerate(q.query_id):
        g=old[old.query_id.eq(qid)].sort_values('dense_rank');indices=np.array([pi[str(x)] for x in g.product_id]);bad+=int(set(indices)!=set(order[qi,:100]));different+=int(not np.array_equal(indices,order[qi,:100]))
    lexical0,bs=bm25([_plain(x) for x in p],q['query'].tolist());lexical1,_=bm25([_plain(x,list(reversed(x['attributes']))) for x in p],q['query'].tolist());assert np.array_equal(lexical0,lexical1)
    rows.append({'dataset':ds,'queries':len(q),'legacy_C0_top100_set_differences':bad,'legacy_C0_top100_order_differences':different,'BM25_reverse_max_score_difference':float(np.max(abs(lexical0-lexical1))),'no_highest_queries':int(sum(not (g.label.eq('Exact' if ds=='wands' else 'E')).any() for _,g in j.groupby('query_id'))),'no_positive_gain_queries':int(sum(not (g.grade>0).any() for _,g in j.groupby('query_id'))),'judgments':len(j),'catalog_unique_products':len({str(x['product_id']) for x in p})})
pd.DataFrame(rows).to_csv(OUT/'legacy_control_audit.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
