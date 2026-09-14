"""Explain tiny historical/new baseline differences without overwriting history."""
from common import *
from threadpoolctl import threadpool_limits
threadpool_limits(4)
sys.path.insert(0,str(ROOT/'phase3/scripts'))
from run_invariant_method import structures
rows=[]
for ds in ['wands','esci']:
    p,q,j=load(ds);unique,inc,text,counts=structures(p);old=ROOT/'phase3/results/phase3_invariant_method/embeddings'
    sha=hashlib.sha256('\0'.join(text).encode()).hexdigest()
    t=next(f for f in old.glob(f'{ds}_title_description_*.npy') if json.loads(f.with_suffix('.json').read_text())['source_sha256']==sha)
    a=next(old.glob(f'{ds}_unique_attributes_*.npy'));oq=next(old.glob(f'{ds}_queries_*.npy'))
    sv=norm(.5*np.load(t)+.5*norm(inc@np.load(a)));newq,_=original_cache(ds,'bge_base');oldq=np.load(oq)
    oldpairs=pd.read_parquet(ROOT/f'phase3/results/phase3_invariant_method/{ds}_set_mean_pairs.parquet')
    qi={int(v):i for i,v in enumerate(q.query_id)};pi={str(v['product_id']):i for i,v in enumerate(p)}
    x=np.array([qi[int(v)] for v in oldpairs.query_id]);y=np.array([pi[str(v)] for v in oldpairs.product_id]);s_old=oldq@sv.T;s_new=newq@sv.T
    rows.append(dict(dataset=ds,query_embedding_max_abs_difference=float(abs(oldq-newq).max()),reconstructed_historical_score_max_error=float(abs(s_old[x,y]-oldpairs.score.to_numpy()).max()),shared_query_score_max_difference=float(abs(s_new[x,y]-oldpairs.score.to_numpy()).max()),scope='historical set-mean used separately encoded queries; extension shares the frozen Original query matrix across methods; no historical ranks overwritten'))
pd.DataFrame(rows).to_csv(OUT/'shared_query_cache_audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
