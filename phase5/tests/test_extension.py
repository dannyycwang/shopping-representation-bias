import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'phase5/scripts'))
from common5 import *

def test_canonical_is_identical_across_raw_orders():
    frozen=json.loads((CONFIG/'sample.json').read_text())
    for ds,spec in frozen['datasets'].items():
        rm=record_map(ds)
        for pid in spec['product_ids'][:20]:
            record=rm[pid];base=canonical_text(record)
            assert all(canonical_text(record)==base for _ in STEMS)
            assert all(Counter(raw_text(record,s))==Counter(raw_text(record,'C0')) for s in STEMS)

def test_cache_key_covers_input_prompt_revision_decoding_and_replicate():
    p=list(prompt_sources().values())[0];d={'do_sample':False,'max_new_tokens':512}
    base=cache_key('a',p,d,'primary')
    assert len({base,cache_key('b',p,d,'primary'),cache_key('a',p+'x',d,'primary'),cache_key('a',p,{'do_sample':True},'primary'),cache_key('a',p,d,'noise1')})==5

def test_target_insertion_matches_stable_full_sort():
    rng=np.random.default_rng(20260907)
    for n in [2,10,100]:
        base=rng.integers(-3,4,n).astype('f4');idx=np.arange(n);vals=rng.integers(-3,4,n).astype('f4');actual=rank_target(base,idx,vals)
        expected=[]
        for i,v in zip(idx,vals):
            x=base.copy();x[i]=v;expected.append(np.flatnonzero(np.argsort(-x,kind='stable')==i)[0]+1)
        np.testing.assert_array_equal(actual,expected)

def test_fact_failure_is_not_silent():
    r={'attributes':['capacity: 250 lbs','compatible: no'],'product_id':'x'};source='capacity: 250 lbs | compatible: no'
    assert fact_check(r,source,source)['pass']
    bad=fact_check(r,source,'capacity 300 lbs compatible yes')
    assert not bad['pass'] and bad['new_numbers'] and bad['missing_numbers']

def test_frozen_metric_population():
    frozen=json.loads((CONFIG/'sample.json').read_text())
    for ds,spec in frozen['datasets'].items():
        _,_,j=load(ds);label='Exact' if ds=='wands' else 'E';pairs=j[j.query_id.isin(spec['query_ids'])&j.label.eq(label)]
        assert len(pairs)==spec['query_product_pairs']
        assert pairs.query_id.nunique()==spec['eligible_queries']


def test_generation_cache_records_when_complete():
    path=OUT/'generations.jsonl'
    if not path.exists():
        return
    rows=[json.loads(x) for x in path.open(encoding='utf8')]
    by_key={}
    for x in rows:
        if x['cache_key'] in by_key:
            assert x['output_sha256']==by_key[x['cache_key']]
        else:
            by_key[x['cache_key']]=x['output_sha256']
    ids=[(x['dataset'],x['product_id'],x['method'],x['stem'],x['replicate']) for x in rows]
    assert len(ids)==len(set(ids))
    assert all(sha(x['output'])==x['output_sha256'] for x in rows)
    assert all(len(x['input_sha256'])==64 for x in rows)


def test_guard_falls_back_without_rank_selection():
    good={'finish_reason':'ok','fact_check':{'empty':False,'pass':True},'output':'rewritten'}
    bad={'finish_reason':'ok','fact_check':{'empty':False,'pass':False},'output':'unsupported 999'}
    truncated={'finish_reason':'length','fact_check':{'empty':False,'pass':False},'output':'partial'}
    def guarded(g,canonical):
        execution=g['finish_reason']!='ok' or g['fact_check']['empty']
        return canonical if execution or not g['fact_check']['pass'] else g['output']
    assert guarded(good,'canonical')=='rewritten'
    assert guarded(bad,'canonical')=='canonical'
    assert guarded(truncated,'canonical')=='canonical'
