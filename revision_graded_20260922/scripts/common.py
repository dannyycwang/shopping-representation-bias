"""Read frozen historical inputs; write only revision_graded_20260922."""
from pathlib import Path
import hashlib, json, gzip, subprocess, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OLD = ROOT / 'revision_evidence_20260917'
COMPLETE = ROOT / 'revision_completion_20260921'
STEMS = ['C0', 'C1', 'C2s1', 'C2s2', 'C2s3', 'C2s4', 'C2s5']
MODELS = ['minilm', 'bge_base', 'gte_modernbert']
RULES = ['lexical_ascending', 'lexical_descending', 'field_priority_type']
GAINS = {'wands': {'Exact':3., 'Partial':1., 'Irrelevant':0.}, 'esci': {'E':1., 'S':.1, 'C':.01, 'I':0.}}
OLD_GAINS = {'wands':GAINS['wands'], 'esci':{'E':1., 'S':.01, 'C':.1, 'I':0.}}
SEED, DRAWS, ZERO_TOL = 2026091701, 10000, 1e-12

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')

def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf8').strip()
def load(ds):
    with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz','rt',encoding='utf8') as f: p = list(map(json.loads,f))
    q = pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
    j = pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv', dtype={'product_id':str})
    return p,q,j

def pairread(path):
    f = pd.read_parquet(path); f['product_id'] = f.product_id.astype(str); f['query_id'] = f.query_id.astype(int)
    assert not f.duplicated(['query_id','product_id']).any(), path
    return f

def support(ds): return read(HERE/'PROTOCOL.json')['populations'][ds]['eligible_ids']
def highest(ds): return 'Exact' if ds=='wands' else 'E'
def sortedpairs(f): return f.set_index(['query_id','product_id']).sort_index()
def ranks(scores):
    order = np.argsort(-scores, axis=1, kind='stable').astype('int32')
    rank = np.empty_like(order)
    np.put_along_axis(rank,order,np.broadcast_to(np.arange(1,order.shape[1]+1,dtype='int32'),order.shape),axis=1)
    return order,rank

def canonical(attrs,rule):
    key = lambda v:(v.casefold(),v)
    if rule=='lexical_ascending': return sorted(attrs,key=key)
    if rule=='lexical_descending': return sorted(attrs,key=key,reverse=True)
    def priority(v):
        field=v.split(':',1)[0].lower() if ':' in v else v[:40].lower()
        return (0 if any(t in field for t in ['producttype','type','style']) else 1,v.casefold(),v)
    return sorted(attrs,key=priority)

def verify_protocol():
    assert sha(HERE/'PROTOCOL.json')==(HERE/'PROTOCOL.sha256').read_text().strip()

_weights={}
def weights(ds):
    if ds not in _weights:
        ids=support(ds); n=len(ids); rng=np.random.default_rng(SEED); w=np.zeros((DRAWS,n),dtype='uint16')
        for start in range(0,DRAWS,250):
            ix=rng.integers(0,n,size=(min(250,DRAWS-start),n))
            np.add.at(w,(np.arange(start,start+len(ix))[:,None],ix),1)
        _weights[ds]=w.astype(float)
        path=HERE/f'data/{ds}_bootstrap_weights.npy'; np.save(path,w)
        dump(path.with_suffix('.json'),dict(query_ids=ids,seed=SEED,draws=DRAWS,sha256=sha(path)))
    return _weights[ds]

def interval(values,ds):
    v=np.asarray(values,dtype=float); assert v.shape[0]==len(support(ds)) and np.isfinite(v).all()
    boot=weights(ds)@v/len(v)
    return np.mean(v,axis=0),*np.quantile(boot,[.025,.975],axis=0)
