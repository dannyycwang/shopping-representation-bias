"""Section 6 revision. Historical inputs are read-only; all writes stay here."""
from pathlib import Path
import hashlib, json, gzip, importlib.util, subprocess, sys, datetime
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OLD = ROOT / 'revision_evidence_20260917'
COMPLETE = ROOT / 'revision_completion_20260921'
GRADED = ROOT / 'revision_graded_20260922'
FINAL = ROOT / 'revision_final_strengthening_20260922'
STEMS = ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
MODELS = ['minilm','bge_base','gte_modernbert']
RULES = ['lexical_ascending','lexical_descending','field_priority_type']
SEED, DRAWS = 2026091701, 10000
CFG = json.loads((ROOT/'phase2/config/phase2.json').read_text())
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def digest(t): return hashlib.sha256(t.encode('utf8')).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):
    p=Path(p); assert p.resolve().is_relative_to(HERE.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf8')
def csv(f,p):
    p=Path(p); assert p.resolve().is_relative_to(HERE.resolve())
    p.parent.mkdir(parents=True,exist_ok=True); f.to_csv(p,index=False,float_format='%.17g')
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()
def source_revision():
    """Allow delivery commits only when they preserve the audited source tree."""
    baseline=read(HERE/'qa/starting_state.json')['revision'];head=git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',baseline,head],cwd=ROOT,check=True)
    changed=git('diff','--name-only',baseline,head).splitlines()
    assert all(p.startswith(HERE.name+'/') for p in changed),changed
    return dict(audited_revision=baseline,current_revision=head,delivery_changes_confined_to_revision=True)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
REP=module('section6_serializer',ROOT/'phase2/src/representations.py')
def load(ds):
    with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz','rt',encoding='utf8') as f:p=list(map(json.loads,f))
    q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
    j=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv',dtype={'product_id':str})
    return p,q,j
def support(ds): return read(GRADED/'PROTOCOL.json')['populations'][ds]['eligible_ids']
def pairread(p):
    f=pd.read_parquet(p); f['product_id']=f.product_id.astype(str);f['query_id']=f.query_id.astype(int);return f
def attrs(p,s):
    if s=='C0':return list(p['attributes'])
    if s=='C1':return list(reversed(p['attributes']))
    if s.startswith('E'):
        import itertools
        return list(sorted(set(itertools.permutations(p['attributes'])))[int(s[1:])])
    seed=20260910+int(s[-1]) if s.startswith('C2s') else int(s[1:])
    return REP._random_order(p['attributes'],str(p['product_id']),seed)
def text_for(p,s):return REP._plain(p,attrs(p,s))
_weights={}
def weights(ids):
    ids=tuple(map(int,ids));assert list(ids)==sorted(ids)
    if ids not in _weights:
        rng=np.random.default_rng(SEED);w=np.zeros((DRAWS,len(ids)),dtype='float64')
        for start in range(0,DRAWS,250):
            ix=rng.integers(0,len(ids),size=(min(250,DRAWS-start),len(ids)))
            np.add.at(w,(np.arange(start,start+len(ix))[:,None],ix),1)
        _weights[ids]=w
    return _weights[ids]
def ci(v,ids):
    v=np.asarray(v,dtype=float);assert len(v)==len(ids) and np.isfinite(v).all()
    b=weights(ids)@v/len(ids)
    return np.mean(v,axis=0),*np.quantile(b,[.025,.975],axis=0)
def summarize(q,cols,meta):
    q=q.sort_values('query_id');e,l,h=ci(q[cols],q.query_id)
    return [dict(**meta,metric=c,estimate=float(a),ci_low=float(b),ci_high=float(d),queries=len(q),pairs=int(q.n_highest.sum()),weighting='query_macro',bootstrap_seed=SEED,bootstrap_resamples=DRAWS,interval='95% unadjusted paired query-cluster percentile') for c,a,b,d in zip(cols,e,l,h)]
