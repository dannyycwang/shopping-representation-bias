"""Full-record value misfiling pilot: generic lossless parsing and inherited GTE."""
from __future__ import annotations
import ast
import collections
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
HERE=Path(__file__).resolve().parent
PILOT=HERE.parent
ROOT=PILOT.parents[1]
SEED=20261030
CFG=json.loads((ROOT/'phase2/config/phase2.json').read_text(encoding='utf8'))
SPEC=next(x for x in CFG['models'] if x['key']=='gte_modernbert')
NATIVE=next(x for x in CFG['chunk_controls'] if x['key']=='native')
MISSING={'','-','--','---','?','n/a','n.a.','na','none','null','nan','unknown',
         'unspecified','not specified','missing','not available','not applicable','no data'}

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m
REP=module(ROOT/'phase2/src/representations.py','value_swap_serializer')
ENC=module(ROOT/'phase2/src/encoding.py','value_swap_encoder')
# Reuse the existing target-only formula without importing scripts with output side effects.
_path=ROOT/'phase3/scripts/analyze_target_only_permutations.py'
_tree=ast.parse(_path.read_text(encoding='utf8'));_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='rank_target']
_ns={'np':np};exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(_path),'exec'),_ns)
rank_target=_ns['rank_target']

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode('utf8')).hexdigest()
def pick(purpose,*ids):return digest(canonical([str(SEED),'value_swap_visibility:'+purpose,*map(str,ids)]))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,x):
    path=HERE/path;assert path.resolve().is_relative_to(HERE.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf8')
def csv(path,rows):
    path=HERE/path;assert path.resolve().is_relative_to(HERE.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g')
def frame(path,**kwargs):return pd.read_csv(path,keep_default_na=False,float_precision='round_trip',**kwargs)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()
def load():
    with gzip.open(ROOT/'phase2/data/processed/wands_products.jsonl.gz','rt',encoding='utf8') as f:p=list(map(json.loads,f))
    q=frame(ROOT/'phase2/data/processed/wands_queries.csv')
    j=frame(ROOT/'phase2/data/processed/wands_judgments.csv',dtype={'product_id':str})
    return p,q,j
def norm(s):return ' '.join(s.strip().casefold().split())

def select_pair(product):
    """Only first colon is structural; preserve raw substrings, including whitespace."""
    parsed=[];reasons=collections.Counter()
    counts=collections.Counter(norm(a.partition(':')[0]) for a in product['attributes'] if ':' in a)
    for i,a in enumerate(product['attributes']):
        key,sep,value=a.partition(':');nk,nv=norm(key),norm(value)
        if not sep:reasons['no_colon']+=1
        elif nk in MISSING or not any(c.isalnum() for c in key) or any(c in key for c in '\r\n\t'):
            reasons['unreliable_key']+=1
        elif counts[nk]!=1:reasons['ambiguous_selected_key']+=1
        elif nv in MISSING:reasons['empty_or_missing_value']+=1
        else:parsed.append((i,key,value,nv))
    options=[(a,b) for a in range(len(parsed)) for b in range(a+1,len(parsed)) if parsed[a][3]!=parsed[b][3]]
    if not options:
        reason='fewer_than_two_unambiguous_nonmissing_entries' if len(parsed)<2 else 'all_eligible_values_equal'
        return None,dict(reason=reason,entry_rejections=dict(reasons),usable_entries=len(parsed))
    seed=pick('attribute_pair',product['product_id'])
    a,b=random.Random(int(seed,16)).choice(options)
    i,k1,v1,_=parsed[a];j,k2,v2,_=parsed[b]
    return (dict(product_id=str(product['product_id']),entry_index1=i,entry_index2=j,
        key1=k1,key2=k2,value1_A=v1,value2_A=v2,value1_C=v2,value2_C=v1,
        pair_selection_hash=seed,eligible_entry_count=len(parsed),eligible_pair_count=len(options)),
        dict(reason='eligible',entry_rejections=dict(reasons),usable_entries=len(parsed)))

def swap(product,s):
    attrs=list(product['attributes']);i,j=int(s['entry_index1']),int(s['entry_index2'])
    k1,sep1,v1=attrs[i].partition(':');k2,sep2,v2=attrs[j].partition(':')
    assert k1==s['key1'] and k2==s['key2'] and v1==s['value1_A'] and v2==s['value2_A']
    assert norm(k1)!=norm(k2) and norm(v1)!=norm(v2) and norm(v1) not in MISSING and norm(v2) not in MISSING
    attrs[i]=k1+sep1+v2;attrs[j]=k2+sep2+v1
    assert [x.partition(':')[0] for x in attrs]==[x.partition(':')[0] for x in product['attributes']]
    assert all(attrs[n]==a for n,a in enumerate(product['attributes']) if n not in [i,j])
    before=REP._plain(product);after=REP._plain(product,attrs)
    assert [(k,v) for k,v in REP._blocks(product) if k!='attributes']==[(k,v) for k,v in REP._blocks(product,attrs) if k!='attributes']
    assert collections.Counter(before)==collections.Counter(after) and before!=after
    return before,after

def batching(lengths,base=None):
    """Exact phase2 _encode_native schedule, including dynamic GTE batch tiers."""
    cap=SPEC['max_tokens'];base=base or CFG['encoding']['batch_size_gte_modernbert']
    order=np.argsort(np.minimum(lengths,cap),kind='stable');plan=[];pos=0
    while pos<len(order):
        n=int(min(lengths[order[pos]],cap));bs=base
        if n>1500:bs=1
        elif n>1024:bs=min(bs,2)
        elif n>768:bs=min(bs,4)
        elif n>500:bs=min(bs,8)
        plan.append(order[pos:pos+bs].tolist());pos+=bs
    return plan

def previous_snapshot():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in PILOT.rglob('*') if p.is_file()
        and not p.is_relative_to(HERE) and '__pycache__' not in p.parts}
def preservation_start():
    if not (HERE/'qa/preservation.json').exists():
        dump('qa/preservation.json',dict(previous_pilot_files=previous_snapshot(),
            preexisting_modified_files={p:sha(ROOT/p) for p in git('diff','--name-only').splitlines() if (ROOT/p).is_file()},
            git_status=git('status','--short'),cancelled_round2='preserved as-is; no round2 scores executed',started_utc=now()))
def preservation_check():
    old=read(HERE/'qa/preservation.json');assert old['previous_pilot_files']==previous_snapshot()
    assert all(sha(ROOT/p)==h for p,h in old['preexisting_modified_files'].items())
def freeze_check():
    f=read(HERE/'selection_manifest.json')
    for p,h in f['files'].items():assert sha(HERE/p)==h,('Frozen input changed',p)
    for p,h in f['source_files'].items():assert sha(ROOT/p)==h,('Source changed',p)
    return f

def ratio(n,d):return n/d if d else None
def quantile(values,prefix):
    return {prefix+'_'+name:float(v) for name,v in zip(['q25','median','q75'],np.quantile(values,[.25,.5,.75]))}
