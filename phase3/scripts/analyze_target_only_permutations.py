from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; P2=ROOT/'phase2'; OUT=ROOT/'phase3/results/target_only_permutations'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'phase3/scripts'))
from run_reranker import bootstrap
STEMS=['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
def rank_target(base,indices,values):
    # Stable source-order ties; exclude the target's former C0 entry exactly.
    order=np.argsort(-base,kind='stable'); keys=np.empty(len(base),dtype=[('negative','f4'),('index','i8')]); keys['negative']=-base[order]; keys['index']=order
    targets=np.empty(len(indices),dtype=keys.dtype); targets['negative']=-values; targets['index']=indices
    before=np.searchsorted(keys,targets)
    remove=(base[indices]>values) # same target has equal source index: equal scores are not ahead
    return 1+before-remove.astype(int)
def main():
 rows=[];sources=[];candidates=[]
 for ds in ['wands','esci']:
  with gzip.open(P2/f'data/representations/{ds}/C0.jsonl.gz','rt',encoding='utf8') as f: recs=[json.loads(l) for l in f]
  ids=[str(r['product_id']) for r in recs]; pi={p:i for i,p in enumerate(ids)}
  queries=pd.read_csv(P2/f'data/processed/{ds}_queries.csv'); qi={int(q):i for i,q in enumerate(queries.query_id)}
  for model in ['minilm','bge_base','gte_modernbert']:
   frames=[pd.read_parquet(P2/f'results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet') for s in STEMS]
   base=frames[0]; sample=base.iloc[np.linspace(0,len(base)-1,400,dtype=int)]
   ix=np.array([pi[str(p)] for p in sample.product_id]); iq=np.array([qi[int(q)] for q in sample.query_id])
   choices=[]
   for qf in (P2/'results/embeddings').glob(f'{ds}_{model}_native_queries_*.npy'):
    q=np.load(qf,mmap_mode='r')
    for pf in (P2/'results/embeddings').glob(f'{ds}_{model}_native_C0_*.npy'):
     p=np.load(pf,mmap_mode='r'); err=np.max(np.abs(np.einsum('ij,ij->i',q[iq],p[ix])-sample.score.to_numpy()))
     choices.append((err,str(qf),str(pf)))
   _,qf,pf=min(choices); q=np.load(qf); p=np.load(pf); scores=np.asarray(q,dtype='f4')@np.asarray(p,dtype='f4').T
   alliq=np.array([qi[int(v)] for v in base.query_id]); allip=np.array([pi[str(v)] for v in base.product_id]); diff=np.max(abs(scores[alliq,allip]-base.score.to_numpy()))
   assert diff<2e-6,(ds,model,diff)
   highest=base[base.label.eq('Exact' if ds=='wands' else 'E')].copy(); out=highest[['query_id','product_id','label']].copy()
   for s,f in zip(STEMS,frames):
    f=out[['query_id','product_id']].merge(f[['query_id','product_id','score','rank']],on=['query_id','product_id'],validate='one_to_one')
    ranks=np.empty(len(f),dtype=int)
    for qid,g in f.groupby('query_id',sort=False):
     ind=np.array([pi[str(v)] for v in g.product_id]); vals=g.score.to_numpy(dtype='f4')
     if s=='C0': vals=scores[qi[int(qid)],ind]
     ranks[g.index]=rank_target(scores[qi[int(qid)]],ind,vals)
    out[s]=ranks; out[f'whole_{s}']=f['rank'].to_numpy(); out[f'delta_{s}']=ranks-highest['rank'].to_numpy()
   assert (out.C0.to_numpy()==highest['rank'].to_numpy()).all(),(ds,model,'C0 mismatch')
   rr=out[STEMS].to_numpy();out['rank_range']=rr.max(1)-rr.min(1)
   row={'dataset':ds,'retriever':model,'highest_pairs':len(out),'rank_range_median':float(out.rank_range.median()),'max_C0_score_error':float(diff)}
   for k in [10,20,50]:
    out[f'VI@{k}']=(rr.min(1)<=k)&(rr.max(1)>k); macro=out.groupby('query_id')[f'VI@{k}'].mean();mu,lo,hi=bootstrap(macro)
    row.update({f'VI@{k}_micro':float(out[f'VI@{k}'].mean()),f'VI@{k}_query_macro':mu,f'VI@{k}_ci_low':lo,f'VI@{k}_ci_high':hi})
   out.to_parquet(OUT/f'{ds}_{model}_pairs.parquet',index=False);out.groupby('query_id')[[f'VI@{k}' for k in [10,20,50]]].mean().to_csv(OUT/f'{ds}_{model}_per_query.csv')
   cand=out[out['VI@20']].sort_values('rank_range',ascending=False).head(30).copy();cand['dataset']=ds;cand['retriever']=model;cand=cand.merge(queries[['query_id','query']],on='query_id');candidates.append(cand)
   sources.append({'dataset':ds,'model':model,'query_embedding':str(Path(qf).relative_to(ROOT)),'product_embedding':str(Path(pf).relative_to(ROOT)),'max_score_error':float(diff),'C0_rank_exact_match':True})
   rows.append(row);print(ds,model,row['VI@20_micro'],flush=True)
 pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.concat(candidates).to_csv(OUT/'candidates.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'intervention':'One target changes; every competitor fixed in C0','tie_rule':'descending fp32 score then original catalog index','seed':20260963,'bootstrap_samples':10000,'sources':sources},indent=2))
if __name__=='__main__':main()
