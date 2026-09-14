from pathlib import Path
import sys, json, gzip, hashlib, random, re, time
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2]; P2=ROOT/'phase2'; P4=ROOT/'phase4'; OUT=P4/'results'
sys.path.insert(0,str(P2))
from src.encoding import DenseEncoder
from src.representations import _plain, tokens
CFG=json.loads((P2/'config/phase2.json').read_text())
CFG['encoding']['batch_size_bge_base']=16
CFG['encoding']['batch_size_minilm']=64
KS=[20,50,100,200,500,1000]
RULES={'rule_type':['producttype','type','style'], 'rule_appearance':['color','finish','pattern','upholstery'], 'rule_use':['use','compatible','suitable','intended'], 'rule_dimension':['dimension','height','width','depth','length','material','weight']}
SEEDS=list(range(20260971,20260978))
PROFILE={'key':'native','chunk_tokens':None,'overlap':0,'pooling':'model_native'}
if (P4/'config/splits.json').exists():
    frozen=json.loads((P4/'config/splits.json').read_text())
    assert RULES==frozen['rules'] and SEEDS==frozen['view_seeds'], 'Do not mutate frozen strategy definitions'
    assert hashlib.sha256((P4/'config/EXPERIMENT_PROTOCOL.frozen.md').read_bytes()).hexdigest()==(P4/'config/protocol.sha256').read_text().strip(), 'Frozen protocol changed'

def load(ds):
    data=P4/'data' if ds=='esci_new' else P2/'data/processed'
    with gzip.open(data/f'{ds}_products.jsonl.gz','rt',encoding='utf8') as f:p=list(map(json.loads,f))
    q=pd.read_csv(data/f'{ds}_queries.csv');j=pd.read_csv(data/f'{ds}_judgments.csv')
    assert not j.duplicated(['query_id','product_id']).any()
    return p,q,j

def dev_mask(q):
    ids=set(json.loads((P4/'config/splits.json').read_text())['wands_dev'])
    return q.query_id.isin(ids).to_numpy()

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def attrs(p,method):
    a=sorted(p['attributes'],key=lambda v:(v.casefold(),v))
    if method.startswith('view'):
        seed=SEEDS[int(method[4:])-1];rng=random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{p["product_id"]}'.encode()).digest()[:8],'big'));rng.shuffle(a)
    elif method in RULES:
        words=RULES[method]
        def key(v):
            field=v.split(':',1)[0].lower() if ':' in v else v[:40].lower()
            return (0 if any(t in field for t in words) else 1,v.casefold(),v)
        a=sorted(a,key=key)
    assert Counter(a)==Counter(p['attributes'])
    return a

def texts(p,method):
    return [_plain(x,attrs(x,method)) for x in p]

def original_cache(ds,model):
    if ds=='esci_new':return np.load(OUT/f'esci_new/{model}_queries.npy'),np.load(OUT/f'esci_new/{model}_products.npy')
    prov=json.loads((ROOT/'phase3/results/target_only_permutations/provenance.json').read_text())
    entry=next(x for x in prov['sources'] if x['dataset']==ds and x['model']==model)
    return np.load(ROOT/entry['query_embedding']),np.load(ROOT/entry['product_embedding'])

def encode(enc,text,name):
    sha=hashlib.sha256('\0'.join(text).encode()).hexdigest()
    return np.asarray(enc.encode(text,name,PROFILE,sha),dtype='f4')

def ranks(scores):
    order=np.argsort(-scores,axis=1,kind='stable').astype('i4');r=np.empty_like(order)
    np.put_along_axis(r,order,np.broadcast_to(np.arange(1,order.shape[1]+1),order.shape),axis=1)
    return order,r

def pair_metrics(pairs,q,ds):
    highest='Exact' if ds=='wands' else 'E'
    gains={'Exact':3,'Partial':1,'Irrelevant':0} if ds=='wands' else {'E':1,'S':.1,'C':.01,'I':0}
    rows=[]
    for qid,g in pairs.groupby('query_id',sort=False):
        h=g[g.label.eq(highest)];d={'query_id':int(qid),'n_highest':len(h),'n_judged':len(g)}
        for k in KS:d[f'Recall@{k}']=float((h['rank']<=k).mean()) if len(h) else np.nan
        sort=g.sort_values('rank',kind='stable');v=sort.label.map(gains).to_numpy();ideal=np.sort(v)[::-1];n=min(10,len(v));discount=1/np.log2(np.arange(2,n+2));den=ideal[:n]@discount
        d['cNDCG@10']=float(v[:n]@discount/den) if den else np.nan
        rows.append(d)
    result=q[['query_id']].merge(pd.DataFrame(rows),on='query_id',how='left',validate='one_to_one')
    assert len(result)==len(q)
    return result

def evaluate(scores,p,q,j,ds,name,output_ds=None):
    folder=OUT/(output_ds or ds);folder.mkdir(exist_ok=True)
    order,r=ranks(scores);qi={int(v):i for i,v in enumerate(q.query_id)};pi={str(v['product_id']):i for i,v in enumerate(p)}
    a=np.array([qi[int(v)] for v in j.query_id]);b=np.array([pi[str(v)] for v in j.product_id]);pairs=j[['query_id','product_id','label']].copy();pairs['rank']=r[a,b];pairs['score']=scores[a,b]
    pq=pair_metrics(pairs,q,ds)
    pairs.to_parquet(folder/f'{name}_pairs.parquet',index=False);pq.to_csv(folder/f'{name}_per_query.csv',index=False)
    np.save(folder/f'{name}_top1000.npy',order[:,:min(1000,len(p))]);np.save(folder/f'{name}_scores.npy',scores)
    return pq

def bm25(text,queries,return_index=False):
    vocab={};rows=[];cols=[];counts=[];lengths=[]
    for i,t in enumerate(text):
        c=Counter(tokens(t));lengths.append(sum(c.values()))
        for w,n in c.items():rows.append(i);cols.append(vocab.setdefault(w,len(vocab)));counts.append(n)
    tf=sparse.csr_matrix((np.asarray(counts,dtype='f4'),(rows,cols)),shape=(len(text),len(vocab)));df=np.bincount(tf.indices,minlength=len(vocab));idf=np.log(1+(len(text)-df+.5)/(df+.5));dl=np.asarray(lengths,dtype='f4');den=1.2*(.25+.75*dl/max(dl.mean(),1));weighted=tf.copy();weighted.data=((2.2*tf.data/(tf.data+np.repeat(den,np.diff(tf.indptr))))*idf[tf.indices]).astype('f4');matrix=weighted.tocsc();score=np.zeros((len(queries),len(text)),dtype='f4')
    for i,q in enumerate(queries):
        ids=[vocab[w] for w in sorted(set(tokens(q))) if w in vocab]
        if ids:score[i]=np.asarray(matrix[:,ids].sum(axis=1)).ravel()
    size=int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes)
    return (score,size,matrix,vocab) if return_index else (score,size)

def boot(values,seed=20260907):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return [np.nan]*3
    rng=np.random.default_rng(seed);b=[]
    for _ in range(20):b.extend(a[rng.integers(0,len(a),(500,len(a)))].mean(1))
    return [float(a.mean()),*map(float,np.quantile(b,[.025,.975]))]
