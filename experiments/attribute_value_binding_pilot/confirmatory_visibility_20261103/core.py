"""Frozen-data helpers; lossless C/O interventions; suspend-aware timing."""
from __future__ import annotations
import ast, collections, ctypes, datetime, gzip, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
threadpool_limits(4)
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PILOT=HERE.parent/'value_swap_visibility'
HAR=ROOT/'revision_recent_models_20261016/profiles/harrier_20261019'
SEED=20261103
def mod(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m
OLD=mod(PILOT/'common.py','previous_visibility_helpers')
REP=OLD.REP;norm=OLD.norm;MISSING=OLD.MISSING;rank_target=OLD.rank_target
read=OLD.read;sha=OLD.sha;digest=OLD.digest;canonical=OLD.canonical;load=OLD.load;frame=OLD.frame
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pick(purpose,*ids):return digest(canonical([str(SEED),'confirmatory_visibility:'+purpose,*map(str,ids)]))
def dump(name,value):
    p=HERE/name;assert p.resolve().is_relative_to(HERE.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf8')
def csv(name,rows):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def valid_pairs(p):
    entries=[];reasons=collections.Counter()
    counts=collections.Counter(norm(a.partition(':')[0]) for a in p['attributes'] if ':' in a)
    for i,a in enumerate(p['attributes']):
        k,sep,v=a.partition(':');nk,nv=norm(k),norm(v)
        if not sep:reasons['no_colon']+=1
        elif nk in MISSING or not any(c.isalnum() for c in k) or any(c in k for c in '\r\n\t'):reasons['unreliable_key']+=1
        elif counts[nk]!=1:reasons['ambiguous_selected_key']+=1
        elif nv in MISSING:reasons['empty_or_missing_value']+=1
        else:entries.append((i,k,v,nv))
    pairs=[(a,b) for ai,a in enumerate(entries) for b in entries[ai+1:] if a[3]!=b[3]]
    return pairs,dict(reasons)
def choose_five(p):
    pairs,reasons=valid_pairs(p);old,_=OLD.select_pair(p)
    assert (old is not None)==bool(pairs)
    if old:assert old['eligible_pair_count']==len(pairs)
    if len(pairs)<5:return [],dict(reason='fewer_than_five_valid_pairs',valid_pairs=len(pairs),entry_rejections=reasons)
    chosen=random.Random(int(pick('attribute_pairs',p['product_id']),16)).sample(pairs,5)
    rows=[]
    for j,(a,b) in enumerate(chosen,1):
        i,k1,v1,_=a;l,k2,v2,_=b
        rows.append(dict(product_id=str(p['product_id']),j=j,entry_index1=i,entry_index2=l,key1=k1,key2=k2,
            value1_A=v1,value2_A=v2,value1_C=v2,value2_C=v1,involves_yes_no=norm(v1) in {'yes','no'} or norm(v2) in {'yes','no'},
            valid_pair_count=len(pairs),swap_set_hash=pick('attribute_pairs',p['product_id'])))
    return rows,dict(reason='eligible',valid_pairs=len(pairs),entry_rejections=reasons)
def variants(p,swaps):
    a=REP._plain(p);rows=[dict(product_id=str(p['product_id']),variant='A',condition='A',j=0,text=a,text_sha256=digest(a))]
    for s in swaps:
        aa,c=OLD.swap(p,s);assert aa==a
        attrs=list(p['attributes']);i,k=s['entry_index1'],s['entry_index2'];attrs[i],attrs[k]=attrs[k],attrs[i]
        assert collections.Counter(attrs)==collections.Counter(p['attributes'])
        assert all(attrs[n]==x for n,x in enumerate(p['attributes']) if n not in (i,k))
        assert attrs[i]==p['attributes'][k] and attrs[k]==p['attributes'][i]
        o=REP._plain(p,attrs)
        assert collections.Counter(o)==collections.Counter(a) and o!=a
        for condition,text in [('C',c),('O',o)]:rows.append(dict(product_id=str(p['product_id']),variant=f'{condition}{s["j"]}',condition=condition,j=s['j'],text=text,text_sha256=digest(text)))
    assert len(rows)==11
    return rows
def profiles():
    p=read(HAR/'PROTOCOL.json');g=read(PILOT/'config.json')
    return {'gte':dict(model=g['model'],profile=g['model_profile'],runtime=g['runtime'],prefixes={'query':g['query_prefix'],'document':g['document_prefix']},max_length=g['max_length']),
        'harrier':dict(model=p['models']['harrier06b'],runtime=p['software'],prefixes=p['models']['harrier06b']['prefixes'],max_length=p['models']['harrier06b']['max_length'])}
def harrier_adapter():
    # Execute the existing adapter verbatim, with its own existing common module.
    old=sys.modules.get('common');hcommon=mod(HAR/'scripts/common.py','original_harrier_common')
    sys.modules['common']=hcommon
    try:adapter=mod(HAR/'scripts/adapters.py','original_harrier_adapter')
    finally:
        if old is None:sys.modules.pop('common',None)
        else:sys.modules['common']=old
    return adapter.NativeAdapter(profiles()['harrier']['model'])
def awake_seconds():
    # Windows unbiased interrupt time explicitly excludes sleep/hibernation.
    if os.name=='nt':
        value=ctypes.c_ulonglong();fn=ctypes.windll.kernel32.QueryUnbiasedInterruptTime
        if not fn(ctypes.byref(value)):raise ctypes.WinError()
        return value.value/10_000_000
    return None
class Timer:
    def __init__(self):self.start=(time.monotonic(),time.time(),awake_seconds(),time.process_time());self.utc=now()
    def result(self):
        mono,wall,awake,cpu=self.start;m=time.monotonic()-mono;w=time.time()-wall;a=awake_seconds()
        active=a-awake if a is not None else None
        return dict(started_utc=self.utc,finished_utc=now(),monotonic_elapsed_seconds=m,wall_elapsed_seconds=w,
            awake_elapsed_seconds=active,process_cpu_seconds=time.process_time()-cpu,
            possible_suspended_seconds=max(0,m-active) if active is not None else None,
            wall_clock_jump_seconds=w-m,not_pure_gpu_time=True,
            note='Awake elapsed excludes Windows sleep/hibernation but includes CPU, I/O, scheduling and GPU wait; not kernel-only GPU time.')
def timed(name,timer,**extra):dump(name,dict(**timer.result(),**extra))
def freeze_check():
    m=read(HERE/'sample_manifest.json')
    for n,h in m['files'].items():assert sha(HERE/n)==h,('Frozen input changed',n)
    for n,h in m['source_files'].items():assert sha(ROOT/n)==h,('Source changed',n)
    return m
