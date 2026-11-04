"""Resumable native matched-context embeddings and independent full-catalog ranks."""
from core import *
import argparse, importlib.metadata as md

def initialize(key):
    cm=read(HERE/'cache_manifest.json');assert cm['sample_manifest_sha256']==sha(HERE/'sample_manifest.json')
    assert cm['plans'][key]==sha(HERE/f'cache_plan_{key}.json');plan=read(HERE/f'cache_plan_{key}.json')
    expected=read(HERE/f'qa/runtime_{key}.json')['versions']
    assert all(md.version(p)==v for p,v in expected.items()),'Runtime version mismatch'
    for info in plan['files'].values():assert sha(ROOT/info['path'])==info['sha256']
    if key=='gte':
        engine=OLD.ENC.DenseEncoder(OLD.SPEC,OLD.CFG,HERE/'cache/gte_unused')
        cfg=read(PILOT/'config.json')['runtime'];torch=engine.torch
        assert torch.backends.cuda.matmul.allow_tf32==cfg['tf32_matmul'] and torch.backends.cudnn.allow_tf32==cfg['tf32_cudnn']
        assert torch.are_deterministic_algorithms_enabled()==cfg['deterministic_algorithms']
        def forward(texts,role):return engine._forward_native(texts,OLD.NATIVE['pooling'],len(texts))
        def inputids(texts,role):return engine.tokenizer(texts,truncation=False,verbose=False)['input_ids']
        def truncatedids(texts,role):return engine.tokenizer(texts,truncation=True,max_length=OLD.SPEC['max_tokens'])['input_ids']
    else:
        engine=harrier_adapter()
        forward=engine.encode
        allocator_gate=HERE/'qa/allocator_gate.json'
        if allocator_gate.exists() and read(allocator_gate)['enabled']:
            ag=read(allocator_gate)
            assert ag['sample_manifest_sha256']==sha(HERE/'sample_manifest.json') and ag['script_sha256']==sha(HERE/'allocator_check.py')
            def forward(texts,role):
                result=engine.encode(texts,role)
                engine.torch.cuda.empty_cache()
                return result
        def getids(texts,role,truncate):
            x=engine.inputs(texts,role,truncate)
            return [i[m.bool()].tolist() for i,m in zip(x['input_ids'],x['attention_mask'])]
        inputids=lambda texts,role:getids(texts,role,False)
        truncatedids=lambda texts,role:getids(texts,role,True)
    torch=engine.torch
    dump(f'qa/execution_profile_{key}.json',dict(profile=profiles()[key],gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),torch_threads=torch.get_num_threads(),
        model_snapshot=plan['model_snapshot'],baseline_files=plan['files'],execution_script_sha256=sha(Path(__file__)),
        computational_gate_checksums={name:sha(HERE/f'qa/{name}.json') for name in ['column_scoring_gate','allocator_gate'] if key=='harrier' and (HERE/f'qa/{name}.json').exists()}))
    cat=np.load(ROOT/plan['files']['product']['path'],mmap_mode='r')[:plan['source_catalog_slice'][1]]
    qemb=np.load(ROOT/plan['files']['query']['path'],mmap_mode='r')
    return plan,engine,forward,inputids,truncatedids,cat,qemb

def cached_forward(key,forward,texts,role,context,stage):
    descriptor=dict(model=key,profile=profiles()[key],cache_plan_sha256=sha(HERE/f'cache_plan_{key}.json'),role=role,
        text_sha256=[digest(x) for x in texts],context=context)
    name=digest(canonical(descriptor));p=HERE/f'cache/{key}/{name}.npy';marker=p.with_suffix('.json')
    if marker.exists():
        m=read(marker);assert m['descriptor']==descriptor and sha(p)==m['sha256'];return np.load(p),True
    t=Timer();arr=forward(texts,role);assert np.isfinite(arr).all() and arr.dtype==np.float32
    p.parent.mkdir(parents=True,exist_ok=True);np.save(p,arr)
    dump(marker.relative_to(HERE),dict(descriptor=descriptor,sha256=sha(p),first_stage=stage,timing=t.result()))
    return arr,False

def replay(key,forward,plan,pairs,catalog,qemb,stage):
    rows=[]
    for role,mapping,batches,needed,baseline in [
        ('document',plan['product_map'],plan['catalog_batches'],set(pairs.product_id),catalog),
        ('query',plan['query_map'],plan['query_batches'],set(map(str,pairs.query_id)),qemb)]:
        bybatch=collections.defaultdict(list)
        for identity in needed:bybatch[mapping[identity]['batch']].append(identity)
        for bn,(bi,identities) in enumerate(sorted(bybatch.items(),key=lambda kv:int(kv[0]))):
            c=batches[bi];arr,reused=cached_forward(key,forward,c['texts'],role,dict(batch=bi,variant='A'),stage)
            for identity in identities:
                loc=mapping[identity];x=arr[loc['offset']];ref=baseline[loc['baseline_index']]
                rows.append(dict(role=role,identity=identity,exact_equal=np.array_equal(x,ref),max_abs_error=float(abs(x-ref).max()),reused=reused))
            if bn%50==0:print(key,stage,'REPLAY',role,bn+1,'/',len(bybatch),flush=True)
    csv(f'qa/{stage}/{key}_replay.csv',rows)
    ok=all(x['exact_equal'] for x in rows);dump(f'qa/{stage}/{key}_replay_gate.json',dict(passed=ok,checked=len(rows),max_error=max(x['max_abs_error'] for x in rows)))
    assert ok,'Cache replay mismatch: stop; never mix profiles. Rebuild only necessary source-order baseline before continuing.'

def encode(key,forward,inputids,truncatedids,plan,pairs,stage,catalog):
    f=pd.read_parquet(HERE/'data/full_inputs.parquet');f=f[f.product_id.isin(set(pairs.product_id))]
    audit=frame(HERE/f'data/token_audit_{key}.csv',dtype={'product_id':str}).set_index(['product_id','variant'])
    vectors={};checks=[];packing_checks=[];t=Timer()
    gate=read(HERE/f'qa/packing_{key}_gate.json')
    assert gate['sample_manifest_sha256']==sha(HERE/'sample_manifest.json')
    assert gate['packing_script_sha256']==sha(HERE/'packing.py')
    for pn,(pid,g) in enumerate(f.groupby('product_id',sort=False)):
        loc=plan['product_map'][pid];context=plan['catalog_batches'][loc['batch']]
        packed=None;pa=dict(eligible=False,reason='packing_disabled')
        if gate['enabled']:
            lengths=[int(audit.loc[(pid,v),'tokens']) for v in g.variant]
            if len(set(lengths))!=1:pa=dict(eligible=False,reason='variant_token_lengths_differ')
            else:
                from packing import packed_product
                packed,pa=packed_product(key,forward,inputids,context,g,catalog[loc['baseline_index']],stage)
        packing_checks.append(dict(product_id=pid,**pa))
        for r in g.itertuples():
            ids=inputids([r.text],'document')[0];a=audit.loc[(pid,r.variant)]
            assert len(ids)==a.tokens and digest(canonical(ids))==a.token_ids_sha256
            assert ids==truncatedids([r.text],'document')[0]
            if r.variant=='A':continue
            if packed is not None:
                vectors[(pid,r.variant)]=packed[(pid,r.variant)];reused='recorded_per_packed_forward'
            else:
                texts=list(context['texts']);texts[loc['offset']]=r.text
                arr,reused=cached_forward(key,forward,texts,'document',dict(batch=loc['batch'],variant=r.variant,product_id=pid),stage)
                vectors[(pid,r.variant)]=arr[loc['offset']]
            checks.append(dict(product_id=pid,variant=r.variant,tokens=int(a.tokens),complete_input=True,reused=reused,encoding_mode='same_shape_packing' if packed is not None else 'original_companions'))
        if pn%10==0 or pn+1==f.product_id.nunique():
            timing=t.result();print(key,stage,'VARIANTS',pn+1,'/',f.product_id.nunique(),'awake seconds',round(timing['awake_elapsed_seconds'],1),flush=True)
            dump(f'qa/{stage}/{key}_progress.json',dict(completed_products=pn+1,total_products=f.product_id.nunique(),timing=timing))
    csv(f'qa/{stage}/{key}_input_audit.csv',checks);csv(f'qa/{stage}/{key}_packing_audit.csv',packing_checks)
    timed(f'qa/{stage}/{key}_encoding_timing.json',t)
    return vectors

def score(key,plan,pairs,catalog,qemb,vectors,stage):
    t=Timer();work=np.array(catalog,copy=True,order='C');n=len(work);all_indices=np.arange(n);rows=[];sortchecks=0
    # Harrier's pinned profile explicitly uses the entire 308-query matrix.
    # GEMV/one-query scoring is NOT bitwise equivalent to that FP32 GEMM.
    full_base=qemb@work.T if key=='harrier' else None
    score_gate=None;score_modes=collections.Counter();crossing_full_matrix_checks=0
    if key=='harrier' and (HERE/'qa/column_scoring_gate.json').exists():
        score_gate=read(HERE/'qa/column_scoring_gate.json')
        assert score_gate['enabled'] and score_gate['script_sha256']==sha(HERE/'column_scoring.py')
        assert score_gate['sample_manifest_sha256']==sha(HERE/'sample_manifest.json') and score_gate['cache_plan_sha256']==sha(HERE/'cache_plan_harrier.json')
    maps=frame(HERE/'data/selected_swaps.csv',dtype={'product_id':str}).set_index(['product_id','j'])
    for qn,(qid,g) in enumerate(pairs.groupby('query_id',sort=True)):
        qi=plan['query_map'][str(qid)]['baseline_index'];query=qemb[qi:qi+1]
        base=full_base[qi] if full_base is not None else (query@work.T)[0]
        for r in g.to_dict('records'):
            ix=int(r['catalog_index']);pid=r['product_id'];scores=[base[ix]];names=['A']
            for condition in ['C','O']:
                for j in range(1,6):
                    variant=f'{condition}{j}'
                    try:
                        work[ix]=vectors[(pid,variant)]
                        if full_base is not None and score_gate is not None:
                            from column_scoring import target_column
                            column,mode=target_column(qemb,work,full_base,ix,score_gate);score_modes[mode]+=1
                            changed=base.copy();changed[ix]=column[qi]
                            new_rank=rank_target(base,np.array([ix]),np.array([column[qi]],dtype='float32'))[0]
                            old_rank=rank_target(base,np.array([ix]),np.array([base[ix]],dtype='float32'))[0]
                            if mode=='certified_column_block' and ((new_rank<=20)!=(old_rank<=20)):
                                reference=(qemb@work.T)[:,ix]
                                assert np.array_equal(reference,column),'Observed crossing must also match original full-matrix scoring'
                                crossing_full_matrix_checks+=1
                        elif full_base is not None:changed=(qemb@work.T)[qi]
                        else:changed=(query@work.T)[0]
                    finally:work[ix]=catalog[ix]
                    assert np.array_equal(base[:ix],changed[:ix]) and np.array_equal(base[ix+1:],changed[ix+1:])
                    scores.append(changed[ix]);names.append(variant)
            ranks=rank_target(base,np.repeat(ix,11),np.array(scores,dtype='float32'))
            for name,s,rank in zip(names,scores,ranks):
                independent=1+int(np.count_nonzero((all_indices!=ix)&((base>s)|((base==s)&(all_indices<ix)))))
                assert rank==independent
                if stage=='smoke':
                    changed=base.copy();changed[ix]=s
                    assert np.flatnonzero(np.argsort(-changed,kind='stable')==ix)[0]+1==rank;sortchecks+=1
                j=int(name[1:]) if name!='A' else 0;mapping=maps.loc[(pid,j)].to_dict() if j else {}
                rows.append(dict(**r,model=key,variant=name,condition=name[0],j=j,
                    involves_yes_no=mapping.get('involves_yes_no',''),score=float(s),rank=int(rank),in_top20=bool(rank<=20),
                    score_A=float(scores[0]),rank_A=int(ranks[0]),in_top20_A=bool(ranks[0]<=20),
                    score_delta=float(np.float64(s)-np.float64(scores[0])),rank_delta=int(rank-ranks[0]),
                    catalog_size=n,all_competitor_scores_bitwise_unchanged=True))
        if qn%50==0:print(key,stage,'FULL CATALOG RANKS',qn+1,'queries',flush=True)
    assert np.array_equal(work,catalog);csv(f'results/{stage}/{key}_scores.csv',rows)
    dump(f'qa/{stage}/{key}_ranking.json',dict(rows=len(rows),catalog_size=n,independent_count_checks=len(rows),full_sort_checks=sortchecks,competitors_bitwise_unchanged=True,
        scoring_query_matrix_rows=len(qemb) if full_base is not None else 1,numerical_scoring='inherited full-query FP32 GEMM; certified column blocks with full-catalog fallback' if score_gate is not None else 'inherited full-query-matrix FP32 GEMM' if full_base is not None else 'inherited pilot query-row FP32 matmul',
        score_computation_modes=dict(score_modes),observed_crossing_full_matrix_score_checks=crossing_full_matrix_checks))
    timed(f'qa/{stage}/{key}_scoring_timing.json',t)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['gte','harrier'],required=True);ap.add_argument('--stage',choices=['smoke','formal'],required=True);a=ap.parse_args()
    freeze_check();key=a.model;stage=a.stage;t=Timer()
    if stage=='formal':
        for model in ['gte','harrier']:
            m=read(HERE/f'qa/smoke/{model}_complete.json');assert m['passed'] and m['sample_manifest_sha256']==sha(HERE/'sample_manifest.json')
            assert m['run_sha256']==sha(Path(__file__)) and m['cache_manifest_sha256']==sha(HERE/'cache_manifest.json')
    pairs=frame(HERE/('data/smoke_pairs.csv' if stage=='smoke' else 'data/selected_pairs.csv'),dtype={'product_id':str})
    attempt_name=f'qa/attempts/{key}_{stage}_{time.time_ns()}.json'
    timed(attempt_name,t,status='started',model=key,stage=stage,run_sha256=sha(Path(__file__)),
        absolute_timer_start=dict(monotonic=t.start[0],awake=t.start[2],wall=t.start[1]))
    engine=None
    try:
        plan,engine,forward,inputids,truncatedids,catalog,qemb=initialize(key)
        replay(key,forward,plan,pairs,catalog,qemb,stage)
        vectors=encode(key,forward,inputids,truncatedids,plan,pairs,stage,catalog)
        score(key,plan,pairs,catalog,qemb,vectors,stage)
    except BaseException as exc:
        timed(attempt_name,t,status='interrupted_or_failed',model=key,stage=stage,error=repr(exc),run_sha256=sha(Path(__file__)));raise
    finally:
        if engine is not None:engine.close()
    timed(attempt_name,t,status='complete',model=key,stage=stage,run_sha256=sha(Path(__file__)))
    timed(f'qa/{stage}/{key}_complete.json',t,passed=True,sample_manifest_sha256=sha(HERE/'sample_manifest.json'),
        run_sha256=sha(Path(__file__)),cache_manifest_sha256=sha(HERE/'cache_manifest.json'))
    print('COMPLETE',key,stage,len(pairs),'pairs',flush=True)

if __name__=='__main__':main()
