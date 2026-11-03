"""Matched replay, independent C encodings, exact full-catalog replacement ranking."""
from common import *
import argparse

def initialize(use_rebuilt=True):
    cfg=read(HERE/'config.json')
    import torch,transformers,tokenizers
    for k,v in dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
                    gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda).items():
        assert cfg['runtime'][k]==v,('Runtime mismatch',k,v)
    assert torch.backends.cuda.matmul.allow_tf32==cfg['runtime']['tf32_matmul']
    assert torch.backends.cudnn.allow_tf32==cfg['runtime']['tf32_cudnn']
    assert torch.are_deterministic_algorithms_enabled()==cfg['runtime']['deterministic_algorithms']
    for p,h in cfg['model_snapshot_hashes'].items():assert sha(Path(cfg['model_snapshot'])/p)==h
    for c in cfg['cache'].values():assert sha(ROOT/c['path'])==c['sha256']
    engine=ENC.DenseEncoder(SPEC,CFG,HERE/'cache/unused_encoder_cache')
    assert engine.device=='cuda' and engine.attention_implementation=='sdpa'
    catalog=np.load(ROOT/cfg['cache']['product']['path'],mmap_mode='r')
    qemb=np.load(ROOT/cfg['cache']['query']['path'],mmap_mode='r')
    if use_rebuilt and (HERE/'baseline_manifest.json').exists():
        manifest=read(HERE/'baseline_manifest.json')
        assert manifest['selection_manifest_sha256']==sha(HERE/'selection_manifest.json')
        assert manifest['config_sha256']==sha(HERE/'config.json')
        assert manifest['encoder_sha256']==sha(ROOT/'phase2/src/encoding.py')
        assert manifest['plan_sha256']==sha(HERE/'baseline_rebuild_plan.json')
        for info in manifest['arrays'].values():assert info['sha256']==sha(HERE/info['path'])
        catalog=np.load(HERE/manifest['arrays']['product']['path'],mmap_mode='r')
        qemb=np.load(HERE/manifest['arrays']['query']['path'],mmap_mode='r')
        print('Using compatible rebuilt source-only baseline; historical arrays excluded',flush=True)
    return cfg,engine,catalog,qemb

def forward_cache(engine,texts,role,indices,stage_key,context):
    """Cache only within this experiment, pinned to exact profile/text/batch context."""
    descriptor=dict(config_sha256=sha(HERE/'config.json'),encoder_sha256=sha(ROOT/'phase2/src/encoding.py'),
        role=role,indices=indices,text_sha256=[digest(t) for t in texts],context=context)
    key=digest(canonical(descriptor));folder=HERE/'cache/forwards';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.npy');marker=path.with_suffix('.json')
    if marker.exists():
        old=read(marker);assert old['descriptor']==descriptor and old['sha256']==sha(path)
        return np.load(path),True
    vectors=engine._forward_native(texts,NATIVE['pooling'],len(texts))
    assert np.isfinite(vectors).all() and vectors.dtype==np.float32
    np.save(path,vectors)
    dump(marker.relative_to(HERE),dict(descriptor=descriptor,sha256=sha(path),first_stage=stage_key,completed_utc=now()))
    return vectors,False

def replay(engine,products,queries,pairs,catalog,qemb,stage):
    texts=[REP._plain(p) for p in products];plan=read(HERE/'data/baseline_batch_plan.json')
    chosen=set(pairs.catalog_index.astype(int));qchosen=set(pairs.query_id)
    checked=[];meta=[];start=time.time()
    for role,batches,alltexts,baseline,needed in [
        ('catalog',plan['catalog_batches'],texts,catalog,chosen),
        ('query',plan['query_batches'],queries['query'].tolist(),qemb,
         {i for i,q in enumerate(queries.query_id) if q in qchosen})]:
        selected_batches=[(bi,b) for bi,b in enumerate(batches) if set(b)&needed]
        for n,(bi,b) in enumerate(selected_batches):
            arr,reused=forward_cache(engine,[alltexts[i] for i in b],role,b,stage,dict(baseline_batch=bi,variant='A'))
            for offset,i in enumerate(b):
                if i not in needed:continue
                error=float(abs(arr[offset]-baseline[i]).max())
                checked.append(dict(role=role,index=i,product_id=str(products[i]['product_id']) if role=='catalog' else '',
                    query_id=int(queries.iloc[i].query_id) if role=='query' else '',
                    batch_index=bi,batch_size=len(b),exact_equal=np.array_equal(arr[offset],baseline[i]),max_abs_error=error,reused_local_fresh_replay=reused))
            if n%25==0:print(stage,'REPLAY',role,n+1,'/',len(selected_batches),round(time.time()-start,1),'sec',flush=True)
        meta.append(dict(role=role,needed_rows=len(needed),source_batches=len(selected_batches),forward_rows=sum(len(b) for _,b in selected_batches)))
    csv(f'qa/{stage}/fresh_replay.csv',checked)
    passed=all(r['exact_equal'] for r in checked)
    dump(f'qa/{stage}/replay_gate.json',dict(passed=passed,details=meta,maximum_vector_error=max(r['max_abs_error'] for r in checked),
        note='No C inference until all selected original A and query replay checks pass bitwise.',seconds=time.time()-start))
    assert passed,'Original replay differs from baseline. Do not mix embeddings; inspect audit before deciding source-only rebuild.'
    return texts,plan

def encode_swaps(engine,products,pairs,texts,plan,stage):
    full=pd.read_parquet(HERE/'data/full_inputs.parquet').set_index('product_id')
    batchmap={i:(bi,b) for bi,b in enumerate(plan['catalog_batches']) for i in b}
    vectors={};records=[];start=time.time()
    unique=pairs.drop_duplicates('product_id').sort_values('catalog_index')
    for n,r in enumerate(unique.itertuples()):
        pid=r.product_id;ix=int(r.catalog_index);row=full.loc[pid]
        a,c=swap(products[ix],r._asdict());assert a==row.text_A and c==row.text_C
        # Audit actual full inputs under the same tokenizer, including special tokens.
        tokens=engine.tokenizer([a,c],truncation=False,verbose=False)['input_ids']
        assert [len(x) for x in tokens]==[r.tokens_A,r.tokens_C]
        truncated=engine.tokenizer([a,c],truncation=True,max_length=SPEC['max_tokens'])['input_ids']
        assert tokens==truncated
        bi,b=batchmap[ix];context=[texts[i] for i in b];offset=b.index(ix);context[offset]=c
        arr,reused=forward_cache(engine,context,'catalog',b,stage,dict(baseline_batch=bi,variant='C',product_id=pid))
        vectors[pid]=arr[offset]
        records.append(dict(product_id=pid,source_batch=bi,batch_size=len(b),tokens_A=r.tokens_A,tokens_C=r.tokens_C,
            exact_values_swapped=True,all_keys_and_order_unchanged=True,unselected_content_unchanged=True,
            no_truncation=True,variant_reused=reused,vector_sha256=hashlib.sha256(arr[offset].tobytes()).hexdigest()))
        if n%25==0 or n+1==len(unique):print(stage,'C ENCODE',n+1,'/',len(unique),round(time.time()-start,1),'sec',flush=True)
    csv(f'qa/{stage}/swap_input_audit.csv',records)
    return vectors

def score(products,queries,pairs,catalog,qemb,swaps,stage):
    qi={int(q):i for i,q in enumerate(queries.query_id)};working=np.array(catalog,copy=True,order='C')
    assert len(working)==len(products)
    result=[];sort_checks=0;start=time.time()
    for n,(qid,g) in enumerate(pairs.groupby('query_id',sort=True)):
        query=qemb[qi[int(qid)]:qi[int(qid)]+1]
        base=(query@working.T)[0]
        assert np.isfinite(base).all()
        for r in g.to_dict('records'):
            ix=int(r['catalog_index']);pid=r['product_id'];assert str(products[ix]['product_id'])==pid
            sA=base[ix]
            try:
                working[ix]=swaps[pid]
                changed=(query@working.T)[0]
            finally:working[ix]=catalog[ix]
            # This also checks that every modification is a separate counterfactual.
            assert np.array_equal(changed[:ix],base[:ix]) and np.array_equal(changed[ix+1:],base[ix+1:])
            sC=changed[ix]
            ranks=rank_target(base,np.array([ix,ix]),np.array([sA,sC],dtype='float32'))
            for s,rank in zip([sA,sC],ranks):
                keep=np.arange(len(base))!=ix
                independent=1+np.count_nonzero(keep & ((base>s)|((base==s)&(np.arange(len(base))<ix))))
                assert rank==independent
            if stage=='smoke':
                for array,rank in [(base,ranks[0]),(changed,ranks[1])]:
                    assert np.flatnonzero(np.argsort(-array,kind='stable')==ix)[0]+1==rank
                    sort_checks+=1
            result.append(dict(**r,model=SPEC['key'],catalog_size=len(base),score_A=float(sA),score_C=float(sC),
                rank_A=int(ranks[0]),rank_C=int(ranks[1]),in_top20_A=bool(ranks[0]<=20),in_top20_C=bool(ranks[1]<=20),
                score_delta=float(np.float64(sC)-np.float64(sA)),rank_delta=int(ranks[1]-ranks[0]),
                direction='improved' if ranks[1]<ranks[0] else 'declined' if ranks[1]>ranks[0] else 'unchanged',
                top20_entry=bool(ranks[0]>20 and ranks[1]<=20),top20_exit=bool(ranks[0]<=20 and ranks[1]>20),
                all_competitor_scores_bitwise_unchanged=True))
        if n%20==0:print(stage,'FULL CATALOG SCORES',n+1,'queries',round(time.time()-start,1),'sec',flush=True)
    assert np.array_equal(working,catalog)
    csv(f'results/{stage}/pairs.csv',result)
    dump(f'qa/{stage}/ranking_audit.json',dict(pairs=len(result),catalog_size=len(products),all_competitor_scores_bitwise_unchanged=True,
        independent_count_checks=2*len(result),full_array_sort_checks=sort_checks,score_dtype='float32',
        scoring='same-shape FP32 matrix product for A and independently replaced C; unchanged query; target removed/replaced, never appended',seconds=time.time()-start))
    # Saved historical scores used a 480-row query GEMM. Expose numerical shape
    # differences without changing either pilot scores or selected observations.
    historical=pd.read_parquet(ROOT/'phase2/results/phase2_pair_ranks/wands_gte_modernbert_native_C0.parquet')
    historical['product_id']=historical.product_id.astype(str)
    compare=pd.DataFrame(result).merge(historical[['query_id','product_id','score','rank']],on=['query_id','product_id'])
    compare['score_error']=compare.score_A-compare.score;compare['rank_difference']=compare.rank_A-compare['rank']
    csv(f'qa/{stage}/historical_rank_comparison.csv',compare[['query_id','product_id','score_A','score','score_error','rank_A','rank','rank_difference']])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','pilot'],required=True)
    args=ap.parse_args();start=time.time();freeze_check();stage=args.stage
    if stage=='pilot':
        gate=read(HERE/'qa/smoke/complete.json')
        assert gate['passed'] and gate['selection_manifest_sha256']==sha(HERE/'selection_manifest.json')
        assert gate['run_sha256']==sha(HERE/'run.py')
        assert gate['baseline_manifest_sha256']==(sha(HERE/'baseline_manifest.json') if (HERE/'baseline_manifest.json').exists() else None)
    products,queries,_=load()
    pairs=frame(HERE/('data/smoke_pairs.csv' if stage=='smoke' else 'data/selected_pairs.csv'),dtype={'product_id':str})
    cfg,engine,catalog,qemb=initialize()
    try:
        texts,plan=replay(engine,products,queries,pairs,catalog,qemb,stage)
        swaps=encode_swaps(engine,products,pairs,texts,plan,stage)
        rows=score(products,queries,pairs,catalog,qemb,swaps,stage)
    finally:engine.close()
    preservation_check()
    dump(f'qa/{stage}/complete.json',dict(passed=True,pairs=len(rows),queries=pairs.query_id.nunique(),
        distinct_products=pairs.product_id.nunique(),selection_manifest_sha256=sha(HERE/'selection_manifest.json'),
        run_sha256=sha(HERE/'run.py'),baseline_manifest_sha256=sha(HERE/'baseline_manifest.json') if (HERE/'baseline_manifest.json').exists() else None,
        completed_utc=now(),seconds=time.time()-start))
    print('COMPLETE',stage,len(rows),'pairs',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
