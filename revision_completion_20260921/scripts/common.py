"""Read historical artifacts; write only this completion package."""
from pathlib import Path
import hashlib, json, gzip, sys, importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OLD = ROOT / 'revision_evidence_20260917'
DATA = HERE / 'data'
for name in ['data', 'figures', 'tables', 'qa', 'inputs', 'manuscript']:
    (HERE/name).mkdir(exist_ok=True)
STEMS = ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
MODELS = ['minilm','bge_base','gte_modernbert']
RULES = ['lexical_ascending','lexical_descending','field_priority_type']
SEED, DRAWS = 2026091701, 10000
CFG = json.loads((ROOT/'phase2/config/phase2.json').read_text())
SPLITS = json.loads((ROOT/'phase4/config/splits.json').read_text())
spec = importlib.util.spec_from_file_location('p4common', ROOT/'phase4/scripts/common.py')
p4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p4)

def dump(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE.resolve()), path
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest_ids(values):
    return hashlib.sha256(json.dumps(list(map(str,values)),separators=(',',':')).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def primary(ds): return sorted(SPLITS['wands_heldout' if ds=='wands' else 'esci_old'])
def pairread(path):
    f=pd.read_parquet(path); f['product_id']=f.product_id.astype(str); f['query_id']=f.query_id.astype(int)
    assert not f.duplicated(['query_id','product_id']).any(),path
    return f

def relevant(f,ds):
    return f[f.label.eq('Exact' if ds=='wands' else 'E') & f.query_id.isin(primary(ds))].set_index(['query_id','product_id']).sort_index()

def rawwide(ds,model):
    base=None
    for s in STEMS:
        f=relevant(pairread(ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet'),ds)
        if base is None: base=f[['label']].copy()
        assert base.index.equals(f.index)
        base[s]=f['rank'].astype('int32')
    assert len(base)==(21299 if ds=='wands' else 4434)
    return base

_weights={}
def weights(ids):
    ids=tuple(map(int,ids)); assert list(ids)==sorted(ids)
    if ids not in _weights:
        n=len(ids); rng=np.random.default_rng(SEED); w=np.zeros((DRAWS,n),dtype='float64')
        for start in range(0,DRAWS,250):
            ix=rng.integers(0,n,size=(min(250,DRAWS-start),n))
            np.add.at(w,(np.arange(start,start+len(ix))[:,None],ix),1)
        _weights[ids]=w
        key=digest_ids(ids)[:16]
        np.save(DATA/f'bootstrap_weights_{key}.npy',w.astype('uint16'))
        dump(DATA/f'bootstrap_weights_{key}.json',dict(query_ids=list(ids),draws=DRAWS,seed=SEED,
             unit='paired query cluster; all products and schedules retained',weights_sha256=sha(DATA/f'bootstrap_weights_{key}.npy')))
    return _weights[ids]

def summarize(q,cols,meta):
    """Same paired resamples for macro and pair-micro; q must be sorted."""
    w=weights(q.index); n=q.n_highest.to_numpy(dtype=float); v=q[cols].to_numpy(dtype=float)
    rows=[]
    for agg in ['query_macro','pair_micro']:
        estimate=v.mean(0) if agg=='query_macro' else n@v/n.sum()
        boots=w@v/len(q) if agg=='query_macro' else (w@(v*n[:,None]))/(w@n)[:,None]
        lo,hi=np.quantile(boots,[.025,.975],axis=0)
        row=dict(**meta,aggregation=agg,eligible_queries=len(q),highest_label_pairs=int(n.sum()),bootstrap_seed=SEED,bootstrap_draws=DRAWS)
        for col,e,l,h in zip(cols,estimate,lo,hi): row.update({col:float(e),col+'_ci_low':float(l),col+'_ci_high':float(h)})
        rows.append(row)
    return rows

def aggregate_pairs(f, cols):
    q=f.groupby(level='query_id',sort=True)[cols].mean()
    q['n_highest']=f.groupby(level='query_id',sort=True).size()
    return q

def membership(base,alt,index):
    f=pd.DataFrame(index=index)
    for name,val in [('retained',base&alt),('newly_included',~base&alt),('newly_omitted',base&~alt),('absent_in_both',~base&~alt)]: f[name]=val.astype('int32')
    assert (f.sum(axis=1)==1).all()
    q=f.groupby(level='query_id',sort=True).sum(); q['n_highest']=f.groupby(level='query_id',sort=True).size()
    for col in f.columns:q[col+'_fraction']=q[col]/q.n_highest
    q['delta_recall']=(q.newly_included-q.newly_omitted)/q.n_highest
    q['changed']=(q.newly_included+q.newly_omitted)>0
    q['exact_cancellation']=(q.newly_included==q.newly_omitted)&(q.newly_included>0)
    assert (q[['retained','newly_included','newly_omitted','absent_in_both']].sum(axis=1)==q.n_highest).all()
    return f,q

def canonical_source(ds,model,rule):
    path=OLD/f'experiments/{ds}_{model}_{rule}_source.json'
    s=read(path); r=ROOT/s['rank_source']['relative_path']
    assert sha(r)==s['rank_source']['sha256']
    return r,s
