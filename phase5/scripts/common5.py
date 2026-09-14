from pathlib import Path
import sys, json, gzip, hashlib, random, re, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
P2=ROOT/'phase2'; P3=ROOT/'phase3'; P4=ROOT/'phase4'; P5=ROOT/'phase5'
OUT=P5/'results'; CACHE=P5/'cache'; CONFIG=P5/'config'; DATA=P5/'data'
for p in [OUT,CACHE,CONFIG,DATA]:p.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(P2));sys.path.insert(0,str(P4/'scripts'))
from src.representations import build as build_rep, _plain
from src.encoding import DenseEncoder
from common import load, original_cache, norm

STEMS=['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']
CONDITIONS={'C0':'C0_original','C1':'C1_reverse_attribute_order',**{f'C2s{i}':f'C2_random_attribute_order_s{i}' for i in range(1,6)}}
SEEDS=json.loads((P4/'config/splits.json').read_text())['view_seeds']
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'; MODEL_REV='7ae557604adf67be50417f59c2c2f167def9a775'
SYSTEM_PROMPT="""You are an expert ml researcher having previous background in SEO and search engines in general.
You are working on novel research ideas for next generation of e-commerce websites.
These websites will have language models augmented with search engines, with the task of recommending products based on the catalog of products backed by the search engine.
This new set of systems will be collectively called language engines (generative search engines).
This will require e-commerce businesses to update their SEO techniques so that the product ranks higher in the llm generated answer.
Specifically they will use GEO (Generative Engine Optimization) techniques to boost their visibility in the final rankings outputted by the Language Engine."""

def sha(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def stable_ids(ids,prefix):return sorted(map(int,ids),key=lambda x:sha(f'{prefix}:{x}'))
def record_map(ds):
    p,_,_=load(ds);return {str(x['product_id']):x for x in p}
def raw_text(r,stem):return build_rep(r,CONDITIONS[stem],SEEDS)
def canonical_text(r):return _plain(r,sorted(r['attributes'],key=lambda x:(x.casefold(),x)))
def prompt_sources():
    import importlib.util
    src=P5/'vendor/E-GEO/src'
    spec=importlib.util.spec_from_file_location('egeo_init',src/'all_init_prompts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    opt=json.loads((src/'optimized_prompts.json').read_text(encoding='utf8'))
    return {'heuristic_authoritative':m.INITIAL_PROMPTS['authoritative'].strip(),'optimized_technical':opt['technical'].strip()}
def cache_key(actual_input,prompt,decoding,replicate):
    return sha(json.dumps({'input':actual_input,'prompt':prompt,'system':SYSTEM_PROMPT,'model':MODEL_ID,'revision':MODEL_REV,'decoding':decoding,'replicate':replicate},sort_keys=True))

COLORS={'black','blue','brown','clear','gold','gray','grey','green','orange','pink','purple','red','silver','tan','turquoise','white','yellow','beige','bronze','ivory','multicolor'}
UNITS={'mm','cm','m','in','inch','inches','ft','feet','oz','ounce','ounces','lb','lbs','kg','g','gb','tb','mah','v','w','hz','hour','hours','year','years','pack'}
WORD=re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*",re.I)
NUM=re.compile(r"(?<![a-z])\d+(?:\.\d+)?",re.I)
def fact_check(record,source,output,finish_reason='ok'):
    sl=source.lower();ol=(output or '').lower();st=WORD.findall(sl);ot=WORD.findall(ol);sc=Counter(st);oc=Counter(ot)
    src_nums=Counter(NUM.findall(sl));out_nums=Counter(NUM.findall(ol))
    new_nums=list((out_nums-src_nums).elements());missing_nums=list((src_nums-out_nums).elements())
    src_colors=sorted(set(st)&COLORS);out_colors=set(ot)&COLORS
    missing_colors=sorted(set(src_colors)-out_colors);new_colors=sorted(out_colors-set(src_colors))
    src_units=sorted(set(st)&UNITS);out_units=set(ot)&UNITS
    missing_units=sorted(set(src_units)-out_units);new_units=sorted(out_units-set(src_units))
    title_tokens={x for x in WORD.findall(record.get('title','').lower()) if len(x)>2};title_fraction=sum(x in oc for x in title_tokens)/len(title_tokens) if title_tokens else 1.0
    src_models={x for x in st if len(x)>=4 and re.search('[a-z]',x) and re.search(r'\d',x)};missing_models=sorted(src_models-set(ot));new_models=sorted({x for x in ot if len(x)>=4 and re.search('[a-z]',x) and re.search(r'\d',x)}-src_models)
    attrs=[]
    for a in record.get('attributes',[]):
        val=a.split(':',1)[-1].strip(); toks=[x for x in WORD.findall(val.lower()) if len(x)>1]
        if toks:attrs.append((a,sum(oc[t]>0 for t in set(toks))/len(set(toks))))
    missing_attrs=[a for a,c in attrs if c<.8]
    by_key=defaultdict(set)
    for a in record.get('attributes',[]):
        if ':' in a:
            k,v=a.split(':',1);by_key[k.strip().casefold()].add(v.strip().casefold())
    conflicts={k:sorted(v) for k,v in by_key.items() if k and len(v)>1}
    neg=[]
    for a in record.get('attributes',[]):
        if re.search(r':\s*(?:no|not|false)\b',a,re.I):
            key=[x for x in WORD.findall(a.split(':',1)[0].lower()) if len(x)>2]
            if key and not (all(x in oc for x in key) and ('no' in oc or 'not' in oc or 'false' in oc)):neg.append(a)
    empty=not bool((output or '').strip());truncated=finish_reason=='length'
    passed=not(empty or truncated or new_nums or missing_nums or missing_colors or new_colors or missing_units or new_units or missing_models or new_models or missing_attrs or neg or title_fraction<.8)
    return {'pass':passed,'empty':empty,'truncated_generation':truncated,'new_numbers':new_nums,'missing_numbers':missing_nums,'missing_colors':missing_colors,'new_colors':new_colors,'missing_units':missing_units,'new_units':new_units,'missing_models':missing_models,'new_models':new_models,'title_pass_fraction':float(title_fraction),'missing_attributes':missing_attrs,'attribute_pass_fraction':float(1-len(missing_attrs)/len(attrs)) if attrs else 1.0,'missing_negations':neg,'source_conflicts':conflicts,'source_has_conflict':bool(conflicts),'source_tokens':len(st),'output_tokens':len(ot)}

def rank_target(base,indices,values):
    order=np.argsort(-base,kind='stable');keys=np.empty(len(base),dtype=[('negative','f4'),('index','i8')]);keys['negative']=-base[order];keys['index']=order
    targets=np.empty(len(indices),dtype=keys.dtype);targets['negative']=-values;targets['index']=indices
    before=np.searchsorted(keys,targets);remove=(base[indices]>values)
    return 1+before-remove.astype(int)

def boot(a,seed=20260907,n=10000):
    a=np.asarray(a,float);a=a[np.isfinite(a)];rng=np.random.default_rng(seed)
    if not len(a):return (np.nan,np.nan,np.nan)
    chunks=[]
    for _ in range(20):chunks.append(a[rng.integers(0,len(a),(n//20,len(a)))].mean(1))
    b=np.concatenate(chunks);return float(a.mean()),*map(float,np.quantile(b,[.025,.975]))
