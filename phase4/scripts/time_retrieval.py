from common import *
from common import _plain
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def run(ds):
    p,q,j=load(ds);qv,pv=original_cache(ds,'bge_base');folder=OUT/ds;choice=json.loads((P4/'config/selection.json').read_text())['method'];records=[]
    def cache(key):
        fs=list((OUT/'embeddings').glob(f'{ds}_bge_base_{key}_*.npy'));assert len(fs)==1,(key,fs);return np.load(fs[0])
    if choice.startswith('rule'):vectors=[cache(choice)]
    else:
        m=int(choice.split('_')[-1]);views=[cache(f'view{i}') for i in range(1,m+1)];vectors=[norm(np.mean(views,axis=0))] if choice.startswith('centroid') else views
    # Lexical scoring is timed separately with the existing sparse BM25 implementation.
    # Saved BM25 scores alone cannot be used as a claim about online hybrid latency.
    _,lexbytes,matrix,vocab=bm25([_plain(x) for x in p],q['query'].tolist(),return_index=True)
    timed={'Original':[pv],choice:vectors,'hybrid':[pv]}
    allviews=[cache(f'view{i}') for i in range(1,8)]
    for m in [2,4,7]:
        timed[f'centroid_{m}']=[norm(np.mean(allviews[:m],axis=0))]
        timed[f'max_{m}']=allviews[:m]
    for method,vs in timed.items():
        for rep in range(6):
            for qi in range(min(50,len(q))):
                start=time.perf_counter();s=np.maximum.reduce([qv[qi]@v.T for v in vs])
                if method=='hybrid':
                    terms=[vocab[t] for t in sorted(set(tokens(str(q.iloc[qi]['query'])))) if t in vocab]
                    lexical=np.asarray(matrix[:,terms].sum(axis=1)).ravel() if terms else np.zeros(len(p))
                    r1=ranks(s[None,:])[1][0];r2=ranks(lexical[None,:])[1][0];s=1/(60+r1.astype('f4'))+1/(60+r2.astype('f4'))
                mid=time.perf_counter();order=np.argsort(-s,kind='stable');end=time.perf_counter()
                if rep:records.append({'dataset':ds,'method':method,'repeat':rep,'query_id':int(q.iloc[qi].query_id),'score_ms':1000*(mid-start),'sort_ms':1000*(end-mid),'retrieval_ms':1000*(end-start),'vector_bytes':sum(v.nbytes for v in vs),'lexical_sparse_bytes':lexbytes if method=='hybrid' else 0,'vectors_per_product':len(vs)})
    pd.DataFrame(records).to_csv(OUT/f'{ds}_exact_timing.csv',index=False)
    # Export view multiplicity, including duplicated text views for small attribute sets.
    for m in [2,4,7]:
        ts=[texts(p,f'view{i}') for i in range(1,m+1)]
        embeddings=[cache(f'view{i}') for i in range(1,m+1)];unique=np.ones(len(p),dtype=int)
        for i in range(1,m):
            duplicate=np.zeros(len(p),dtype=bool)
            for j in range(i):duplicate|=(embeddings[i]==embeddings[j]).all(axis=1)
            unique+=~duplicate
        counts=pd.DataFrame({'product_id':[x['product_id'] for x in p],'requested_views':m,'unique_views':[len({t[i] for t in ts}) for i in range(len(p))],'unique_encoded_vectors':unique});counts.to_parquet(OUT/f'{ds}_unique_views_m{m}.parquet',index=False)
    enc=DenseEncoder(next(x for x in CFG['models'] if x['key']=='bge_base'),CFG,OUT/'embeddings');qtime=[]
    for rep in range(6):
        for i in range(min(50,len(q))):
            start=time.perf_counter();enc._forward_native([str(q.iloc[i]['query'])],'model_native',1);elapsed=time.perf_counter()-start
            if rep:qtime.append({'dataset':ds,'query_id':int(q.iloc[i].query_id),'repeat':rep,'single_query_encode_ms':1000*elapsed})
    pd.DataFrame(qtime).to_csv(OUT/f'{ds}_query_encoding_timing.csv',index=False);enc.close()

if __name__=='__main__':run('wands');run('esci')
