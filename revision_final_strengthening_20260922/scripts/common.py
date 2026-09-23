from pathlib import Path
import json,hashlib,gzip,subprocess,sys,importlib.util,ast
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
GRADED=ROOT/'revision_graded_20260922';OLD=ROOT/'revision_evidence_20260917'
STEMS=['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5'];MODELS=['minilm','bge_base','gte_modernbert'];RULES=['lexical_ascending','lexical_descending','field_priority_type']
SEED=2026091701;DRAWS=10000
CFG=json.loads((ROOT/'phase2/config/phase2.json').read_text())
spec=importlib.util.spec_from_file_location('representations_frozen',ROOT/'phase2/src/representations.py');rep=importlib.util.module_from_spec(spec);spec.loader.exec_module(rep)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,v):
 p=Path(p);assert p.resolve().is_relative_to(HERE.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def textsha(t):return hashlib.sha256(t.encode()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').rstrip()
def load(ds):
 with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz','rt',encoding='utf8') as f:p=list(map(json.loads,f))
 q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv');return p,q
def support(ds):return read(GRADED/'PROTOCOL.json')['populations'][ds]['eligible_ids']
def pairread(path):
 f=pd.read_parquet(path);f['product_id']=f.product_id.astype(str);f['query_id']=f.query_id.astype(int);return f
def target(ds,model):return pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet').query('query_id in @support(ds)')
def target_frame(ds,model):
 f=pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet');return f[f.query_id.isin(support(ds))].sort_values(['query_id','product_id']).reset_index(drop=True)
def provenance():return pd.read_csv(OLD/'data/encoder_profiles_and_rank_sources.csv')
def verify():assert sha(HERE/'PROTOCOL.json')==(HERE/'PROTOCOL.sha256').read_text().strip()
def ordered_attrs(p,s):
 if s=='C0':return list(p['attributes'])
 if s=='C1':return list(reversed(p['attributes']))
 seed=20260910+int(s[-1]) if s.startswith('C2s') else int(s[1:])
 return rep._random_order(p['attributes'],str(p['product_id']),seed)
def serialize(p,s):return rep._plain(p,ordered_attrs(p,s))
def old_rank_function():
 path=ROOT/'phase3/scripts/analyze_target_only_permutations.py';tree=ast.parse(path.read_text(encoding='utf8'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rank_target');d={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),d);return d['rank_target']
def replacement(base,indices,values,order=None):
 if order is None:order=np.argsort(-base,kind='stable')
 keys=np.empty(len(base),dtype=[('negative','f4'),('index','i8')]);keys['negative']=-base[order];keys['index']=order
 vals=np.empty(np.shape(values),dtype=keys.dtype);vals['negative']=-values;vals['index']=np.broadcast_to(indices,np.shape(values))
 return 1+np.searchsorted(keys,vals)-(base[indices]>values).astype(int)
_weights={}
def weights(ids):
 ids=tuple(map(int,ids))
 if ids not in _weights:
  rng=np.random.default_rng(SEED);w=np.zeros((DRAWS,len(ids)),dtype='float64')
  for start in range(0,DRAWS,250):
   ix=rng.integers(0,len(ids),size=(min(250,DRAWS-start),len(ids)));np.add.at(w,(np.arange(start,start+len(ix))[:,None],ix),1)
  _weights[ids]=w
 return _weights[ids]
def ci(v,ids):
 v=np.asarray(v);b=weights(ids)@v/len(ids);return np.mean(v,axis=0),*np.quantile(b,[.025,.975],axis=0)
def model_spec(model):return next(m for m in CFG['models'] if m['key']==model)
