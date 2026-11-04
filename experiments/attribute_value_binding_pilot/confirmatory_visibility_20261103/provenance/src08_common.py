"""Read-only inherited sources, revision-contained outputs and verified checkpoints."""
from pathlib import Path
import ast, gzip, hashlib, importlib.util, json, os, subprocess, sys, time, tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('HF_HOME', str(HERE/'cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
STEMS = ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
RULES = ['lexical_ascending','lexical_descending','field_priority_type']
SEED, DRAWS = 2026091701, 10000
GAINS = {'wands': {'Exact':3., 'Partial':1., 'Irrelevant':0.}, 'esci':{'E':1.,'S':.1,'C':.01,'I':0.}}

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def now(): return datetime.now(timezone.utc).isoformat()
def digest(x): return hashlib.sha256(x.encode('utf8')).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def fingerprint(x): return digest(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')))
def dump(p,x):
    p=Path(p); assert p.resolve().is_relative_to(HERE.resolve()), p
    p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix='j',suffix='.tmp',dir=p.parent)
    tmp=Path(name)
    with os.fdopen(fd,'w',encoding='utf8') as handle:handle.write(payload)
    for attempt in range(12):
        try:
            tmp.replace(p);return
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.25*2**attempt,5.))

def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()
def source(p):
    p=Path(p); return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def inherited_functions(relative,names,namespace=None):
    """Compile original function ASTs without executing historical mkdir/output side effects."""
    path=ROOT/relative; tree=ast.parse(path.read_text(encoding='utf8'))
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    ns={} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

spec=importlib.util.spec_from_file_location('recent_inherited_serializer',ROOT/'phase2/src/representations.py')
serializer=importlib.util.module_from_spec(spec); spec.loader.exec_module(serializer)
canonical_attrs=inherited_functions('revision_evidence_20260917/scripts/common.py',['canonical_attrs'])['canonical_attrs']
rank_target=inherited_functions('phase3/scripts/analyze_target_only_permutations.py',['rank_target'],{'np':np})['rank_target']
CFG=read(ROOT/'phase2/config/phase2.json')
SPLITS=read(ROOT/'phase4/config/splits.json')
def highest(ds): return 'Exact' if ds=='wands' else 'E'
def load(ds):
    with gzip.open(ROOT/f'phase2/data/processed/{ds}_products.jsonl.gz','rt',encoding='utf8') as f: p=list(map(json.loads,f))
    q=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv',keep_default_na=False)
    j=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv',dtype={'product_id':str})
    return p,q,j
def support(ds):
    p,q,j=load(ds); ids=SPLITS['wands_heldout' if ds=='wands' else 'esci_old']
    ids=sorted(set(ids)&set(j.loc[j.label.eq(highest(ds)),'query_id'].astype(int)))
    return p,q[q.query_id.isin(ids)].sort_values('query_id').reset_index(drop=True),j[j.query_id.isin(ids)].copy()
def attributes(p,stem):
    if stem=='C0': return list(p['attributes'])
    if stem=='C1': return list(reversed(p['attributes']))
    return serializer._random_order(p['attributes'],str(p['product_id']),CFG['attribute_permutation_seeds'][STEMS.index(stem)-2])
def texts_for(products,condition):
    if condition in RULES: return [serializer._plain(p,canonical_attrs(p,condition)) for p in products]
    return [serializer.build(p,CFG['primary_variant_family'][STEMS.index(condition)],CFG['attribute_permutation_seeds']) for p in products]
def protocol():
    p=HERE/'PROTOCOL.json'; assert sha(p)==(HERE/'PROTOCOL.sha256').read_text().strip(),'Protocol changed'
    x=read(p)
    for entry in x['source_files']:
        assert sha(entry['path'])==entry['sha256'],('frozen source changed',entry['path'])
    return x
def status(stage,model='all',dataset='all',state='completed',**extra):
    dump(HERE/f'status/{model}_{dataset}_{stage}.json',dict(stage=stage,model=model,dataset=dataset,status=state,updated_utc=now(),**extra))
def complete(path,inputs,outputs,**extra):
    dump(path,dict(status='completed',completed_utc=now(),inputs_fingerprint=fingerprint(inputs),outputs=[source(p) for p in outputs],**extra))
def valid(path,inputs):
    if not Path(path).exists(): return False
    x=read(path)
    return x.get('status')=='completed' and x.get('inputs_fingerprint')==fingerprint(inputs) and all(Path(f['path']).is_file() and sha(f['path'])==f['sha256'] for f in x['outputs'])

