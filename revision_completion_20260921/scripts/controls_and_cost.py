"""Verify common-query control caches, deterministic inputs, and storage accounting."""
from common import *
from collections import Counter
from scipy.sparse import csr_matrix

def canonical(attrs,rule):
    key=lambda v:(v.casefold(),v)
    if rule=='lexical_ascending':return sorted(attrs,key=key)
    if rule=='lexical_descending':return sorted(attrs,key=key,reverse=True)
    def priority(v):
        field=v.split(':',1)[0].lower() if ':' in v else v[:40].lower()
        return (0 if any(t in field for t in ['producttype','type','style']) else 1,v.casefold(),v)
    return sorted(attrs,key=priority)

def cache(ds,model,key):
    paths=list((ROOT/'phase4/results/embeddings').glob(f'{ds}_{model}_{key}_*.npy'))
    assert len(paths)==1,(ds,model,key,paths)
    return paths[0]

def main():
    costs=[];checks=[];struct=[];distinct=[];sourcepaths=set()
    provenance=pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv')
    for ds in ['wands','esci']:
        p,q,j=p4.load(ds);ncat=len(p);pids=[str(x['product_id']) for x in p];pi={v:i for i,v in enumerate(pids)};qi={int(v):i for i,v in enumerate(q.query_id)}
        h=relevant(j.assign(product_id=j.product_id.astype(str)),ds)
        a=np.array([qi[int(v)] for v in h.index.get_level_values(0)]);b=np.array([pi[str(v)] for v in h.index.get_level_values(1)])
        texts={f'view{i}':p4.texts(p,f'view{i}') for i in range(1,8)}
        # Equality of canonical base and entry multiplicities for all incoming raw schedules
        # is sufficient for deterministic canonical-seeded views and set-mean structures.
        fingerprints={rule:hashlib.sha256() for rule in RULES}
        for item in p:
            original=item['attributes'];expected={rule:canonical(original,rule) for rule in RULES}
            variants=[original,list(reversed(original))]+[p4.sys.modules['src.representations']._random_order(original,str(item['product_id']),seed) for seed in range(20260911,20260916)]
            for values in variants:
                assert Counter(values)==Counter(original)
                for rule in RULES:assert canonical(values,rule)==expected[rule]
            for rule in RULES:fingerprints[rule].update((p4._plain(item,expected[rule])+'\0').encode())
        for rule in RULES:struct.append(dict(dataset=ds,method=rule,products=ncat,incoming_schedules=7,all_inputs_identical=True,fingerprint=fingerprints[rule].hexdigest(),basis='complete entries, stable intrinsic key; multiplicities retained'))
        for method in ['set_mean','centroid','max','BM25']:struct.append(dict(dataset=ds,method=method,products=ncat,incoming_schedules=7,all_inputs_identical=True,fingerprint=None,basis='identical attribute multiset/canonical base verified for all incoming schedules; deterministic fixed index reused; BM25 source text fixed'))
        for model in MODELS:
            qr=provenance[(provenance.dataset==ds)&(provenance.model==model)&(provenance.schedule=='C0')].iloc[0]
            qv=np.load(ROOT/qr.query_embedding);dim=qv.shape[1];qm=read((ROOT/qr.query_embedding).with_suffix('.json'))
            for rule in RULES:
                _,s=canonical_source(ds,model,rule)
                assert s['query_embedding']['sha256']==sha(ROOT/qr.query_embedding)
                assert s['query_ids']==q.query_id.tolist()
                assert s['catalog_size']==ncat
            if model=='gte_modernbert':continue
            views=[];vp=[]
            for i in range(1,8 if model=='bge_base' else 3):
                path=cache(ds,model,f'view{i}');v=np.load(path);meta=read(path.with_suffix('.json'))
                assert meta['source_sha256']==hashlib.sha256('\0'.join(texts[f'view{i}']).encode()).hexdigest()
                assert meta['model']==qm['model'] and v.dtype==np.float32 and v.shape==(ncat,dim)
                views.append(v);vp.append(path)
            unique=sorted({a for x in p for a in x['attributes']});lookup={a:i for i,a in enumerate(unique)}
            rows=[];cols=[];val=[]
            for i,x in enumerate(p):
                for atom in x['attributes']:rows.append(i);cols.append(lookup[atom]);val.append(1/len(x['attributes']))
            inc=csr_matrix((np.array(val,dtype='float32'),(rows,cols)),shape=(ncat,len(unique)))
            nt=['\n'.join(t for t in [x['title'],x.get('class',''),x.get('category',''),x.get('description','')] if t) for x in p]
            if model=='bge_base':
                folder=ROOT/'phase3/results/phase3_invariant_method/embeddings'
                tp=next(path for path in folder.glob(f'{ds}_title_description_*.npy') if read(path.with_suffix('.json'))['source_sha256']==hashlib.sha256('\0'.join(nt).encode()).hexdigest())
                ap=list(folder.glob(f'{ds}_unique_attributes_*.npy'));assert len(ap)==1;ap=ap[0]
            else:tp=cache(ds,model,'nonattributes');ap=cache(ds,model,'atoms')
            assert read(ap.with_suffix('.json'))['source_sha256']==hashlib.sha256('\0'.join(unique).encode()).hexdigest()
            sv=p4.norm(.5*np.load(tp)+.5*p4.norm(inc@np.load(ap)))
            reconstructed={'set_mean':qv@sv.T};encpaths={'set_mean':[tp,ap]}
            for m in [2,4,7] if model=='bge_base' else [2]:
                centroid=p4.norm(np.mean(views[:m],axis=0))
                reconstructed[f'centroid_{m}']=qv@centroid.T
                reconstructed[f'max_{m}']=np.maximum.reduce([qv@v.T for v in views[:m]])
                encpaths[f'centroid_{m}']=vp[:m];encpaths[f'max_{m}']=vp[:m]
                tv=[len({texts[f'view{i}'][j] for i in range(1,m+1)}) for j in range(ncat)]
                uv=np.ones(ncat,dtype=int)
                for i in range(1,m):
                    dup=np.zeros(ncat,dtype=bool)
                    for k in range(i):dup|=np.all(views[i]==views[k],axis=1)
                    uv+=~dup
                pd.DataFrame(dict(product_id=pids,requested_views=m,unique_text_views=tv,unique_vectors=uv)).to_parquet(DATA/f'{ds}_{model}_distinct_views_M{m}.parquet',index=False)
                distinct.append(dict(dataset=ds,model=model,M=m,products=ncat,mean_distinct_text_views=float(np.mean(tv)),mean_distinct_vectors=float(uv.mean()),duplicate_views_retained=True))
            for method,score in reconstructed.items():
                saved=np.load(ROOT/f'phase4/results/{ds}/{model}_{method}_scores.npy');err=float(np.max(np.abs(score-saved)))
                _,r=p4.ranks(score);old=relevant(pairread(ROOT/f'phase4/results/{ds}/{model}_{method}_pairs.parquet'),ds)
                assert old.index.equals(h.index)
                mismatch=int(np.sum(r[a,b]!=old['rank'].to_numpy()))
                assert mismatch==0,(ds,model,method,mismatch)
                checks.append(dict(dataset=ds,model=model,method=method,common_query_hash=sha(ROOT/qr.query_embedding),full_score_max_error=err,highest_rank_mismatches=mismatch,catalog_order_verified=True))
            for method in ['raw_C0','raw_seven',*RULES,'set_mean','centroid_2','max_2','BM25','hybrid_seven']+(['centroid_4','max_4','centroid_7','max_7'] if model=='bge_base' else []):
                m=int(method.rsplit('_',1)[1]) if method.startswith(('max_','centroid_')) else 1
                vectors=0 if method=='BM25' else m if method.startswith('max_') else 1
                paths=[]
                if method in encpaths:paths=encpaths[method]
                elif method in ['raw_C0','raw_seven','hybrid_seven']:paths=[ROOT/qr.product_embedding]
                elif method in RULES:
                    _,s=canonical_source(ds,model,method)
                    key='product_embedding_metadata'
                    if key in s:paths=[(ROOT/s[key]['relative_path']).with_suffix('.npy')]
                    elif 'embedding_metadata' in s:paths=[(ROOT/s['embedding_metadata']['relative_path']).with_suffix('.npy')]
                seconds=[];items=[]
                for path in paths:
                    sourcepaths.add(path.as_posix());sourcepaths.add(path.with_suffix('.json').as_posix())
                    meta=read(path.with_suffix('.json'));seconds.append(meta['seconds']);items.append(meta['shape'][0])
                sparse=read(ROOT/f'phase4/results/{ds}/BM25_cost.json')['bytes'] if method in ['BM25','hybrid_seven'] else 0
                dv=next((x for x in distinct if x['dataset']==ds and x['model']==model and x['M']==m),None) if m>1 else None
                costs.append(dict(dataset=ds,model=model,method=method,catalog_size=ncat,dimension=dim,dtype='float32',requested_view_budget=m,index_vectors=ncat*vectors,vectors_per_product=vectors,dense_array_bytes=ncat*dim*4*vectors,sparse_array_bytes=sparse,mean_distinct_text_views=dv['mean_distinct_text_views'] if dv else (1 if method not in ['set_mean','BM25'] else np.nan),mean_distinct_vectors=dv['mean_distinct_vectors'] if dv else (1 if vectors else np.nan),offline_cache_construction_seconds=sum(seconds) if seconds else np.nan,encoded_items=sum(items) if items else np.nan,offline_scope='historical cache construction sum, not matched cold end-to-end build; aggregation not included; raw/hybrid per schedule',query_encoding_cache_seconds=qm.get('seconds') if vectors else np.nan,source_ids=';'.join(str(path.relative_to(ROOT)) for path in paths),unmeasured='matched cold build; ANN latency; sparse vocabulary/metadata overhead'))
            print('CONTROL/COST',ds,model,flush=True)
    pd.DataFrame(checks).to_csv(DATA/'control_query_compatibility.csv',index=False)
    pd.DataFrame(struct).to_csv(DATA/'structural_invariance_audit.csv',index=False)
    pd.DataFrame(distinct).to_csv(DATA/'distinct_views_summary.csv',index=False)
    cost=pd.DataFrame(costs)
    timing=[]
    for ds in ['wands','esci']:
        t=pd.read_csv(ROOT/f'phase4/results/{ds}_exact_timing.csv')
        for method,g in t.groupby('method'):
            timing.append(dict(dataset=ds,model='bge_base',method={'Original':'raw_C0','hybrid':'hybrid_seven'}.get(method,method),score_ms_median=g.score_ms.median(),score_ms_p95=g.score_ms.quantile(.95),sort_ms_median=g.sort_ms.median(),sort_ms_p95=g.sort_ms.quantile(.95),exact_retrieval_ms_median=g.retrieval_ms.median(),exact_retrieval_ms_p95=g.retrieval_ms.quantile(.95),timing_queries=g.query_id.nunique(),timing_repeats=g['repeat'].nunique(),timing_scope='historical BGE first 50 queries, four CPU threads, C0 index, five warm repeats; not held-out-only or ANN'))
        qt=pd.read_csv(ROOT/f'phase4/results/{ds}_query_encoding_timing.csv')
        cost.loc[(cost.dataset==ds)&(cost.model=='bge_base')&(cost.method!='BM25'),'single_query_encode_ms_median']=qt.single_query_encode_ms.median()
        cost.loc[(cost.dataset==ds)&(cost.model=='bge_base')&(cost.method!='BM25'),'single_query_encode_ms_p95']=qt.single_query_encode_ms.quantile(.95)
    cost.merge(pd.DataFrame(timing),on=['dataset','model','method'],how='left').to_csv(DATA/'strategy_costs.csv',index=False)
    dump(DATA/'additional_cost_sources.json',dict(files=[dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(sourcepaths)]))

if __name__=='__main__':main()
