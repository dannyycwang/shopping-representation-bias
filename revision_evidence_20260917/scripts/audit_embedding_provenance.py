"""Identify historical encoder metadata from score agreement, without changing ranks."""
from common import *

rows=[];details=[]
for ds in ['wands','esci']:
    p=products(ds);pi={str(x['product_id']):i for i,x in enumerate(p)}
    q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv');qi={int(v):i for i,v in enumerate(q.query_id)}
    for model in ['minilm','bge_base','gte_modernbert']:
        prov=next(x for x in PROVENANCE['sources'] if x['dataset']==ds and x['model']==model)
        qpath=ROOT/prov['query_embedding'];qv=np.load(qpath,mmap_mode='r')
        qmeta=json.loads(qpath.with_suffix('.json').read_text())
        for s in STEMS:
            rankpath=ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet'
            f=pairread(rankpath);sample=f.iloc[np.linspace(0,len(f)-1,min(400,len(f)),dtype=int)]
            a=np.array([qi[int(v)] for v in sample.query_id]);b=np.array([pi[str(v)] for v in sample.product_id])
            candidates=[]
            for path in (ROOT/'phase2/results/embeddings').glob(f'{ds}_{model}_native_{s}_*.npy'):
                pv=np.load(path,mmap_mode='r');meta=json.loads(path.with_suffix('.json').read_text())
                predicted=np.einsum('ij,ij->i',qv[a],pv[b]);error=float(np.max(np.abs(predicted-sample.score.to_numpy())))
                candidates.append((error,path,meta))
            assert candidates,(ds,model,s)
            candidates.sort(key=lambda t:t[0]);error,path,meta=candidates[0]
            assert error<2e-6,(ds,model,s,error)
            spec=next(x for x in CFG['models'] if x['key']==model)
            assert meta['model']==qmeta['model']==spec
            assert meta['profile']==qmeta['profile']==next(x for x in CFG['chunk_controls'] if x['key']=='native')
            rows.append(dict(dataset=ds,model=model,schedule=s,model_name=spec['name'],revision=spec['revision'],
              profile='native',pooling=spec['native_pooling'],tokenizer_cap=spec['max_tokens'],query_prefix='',product_prefix='',
              implementation=meta['implementation'],precision=meta['precision'],catalog_size=len(p),query_count=len(q),
              rank_source=rankpath.relative_to(ROOT).as_posix(),product_embedding=path.relative_to(ROOT).as_posix(),
              query_embedding=qpath.relative_to(ROOT).as_posix(),sampled_judged_pairs=len(sample),max_sample_score_error=error,
              embedding_candidates=len(candidates),candidates_under_tolerance=sum(x[0]<2e-6 for x in candidates)))
            details.append(dict(dataset=ds,model=model,schedule=s,query_metadata=source(qpath.with_suffix('.json')),
               product_metadata=source(path.with_suffix('.json')),product_embedding=source(path),
               candidate_score_errors=[dict(path=z.relative_to(ROOT).as_posix(),max_error=e) for e,z,_ in candidates]))
        print('EMBEDDING PROVENANCE',ds,model,flush=True)
pd.DataFrame(rows).to_csv(DATA/'encoder_profiles_and_rank_sources.csv',index=False)
dump(DATA/'embedding_provenance_sources.json',dict(cells=details,interpretation='Score agreement audits encoder provenance; it does not tune or select a method. All empirical ranks remain inherited exact rank artifacts. Multiple agreeing cache candidates, if present, remain disclosed.'))
