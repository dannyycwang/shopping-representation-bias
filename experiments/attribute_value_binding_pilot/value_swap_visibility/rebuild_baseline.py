"""Necessary C0-only rebuild after documented historical cache replay mismatch."""
from common import *
from run import initialize
import shutil

def main():
    freeze_check();cfg,engine,oldp,oldq=initialize(use_rebuilt=False)
    oldaudit=HERE/'qa/historical_cache_replay'
    if not oldaudit.exists():shutil.copytree(HERE/'qa/smoke',oldaudit)
    gate=read(oldaudit/'replay_gate.json');assert not gate['passed']
    plan=dict(selection_manifest_sha256=sha(HERE/'selection_manifest.json'),created_utc=now(),
        reason='Historical cache passed source/profile checks but selected A/query fresh replay differed; exact historical numerical runtime cannot be recovered from sidecars.',
        historical_replay_max_abs_error=gate['maximum_vector_error'],
        rebuild_scope='source-order full catalog and original queries only; no other representation/model',
        model=SPEC,profile=NATIVE,runtime=cfg['runtime'],config_sha256=sha(HERE/'config.json'),
        reference_seconds=cfg['cache']['product']['metadata']['seconds']+cfg['cache']['query']['metadata']['seconds'],
        reference_catalog_products=cfg['cache']['product']['metadata']['shape'][0],
        script_sha256=sha(Path(__file__)),inherited_encoder_sha256=sha(ROOT/'phase2/src/encoding.py'))
    dump('baseline_rebuild_plan.json',plan)
    # Independent second execution: ensure new runtime is repeatable before rebuilding.
    checks=[];products,queries,_=load();texts=[REP._plain(p) for p in products]
    for marker in sorted((HERE/'cache/forwards').glob('*.json')):
        meta=read(marker);d=meta['descriptor']
        if d['context']['variant']!='A':continue
        alltexts=texts if d['role']=='catalog' else queries['query'].tolist()
        arr=engine._forward_native([alltexts[i] for i in d['indices']],NATIVE['pooling'],len(d['indices']))
        before=np.load(marker.with_suffix('.npy'))
        checks.append(dict(batch_key=marker.stem,role=d['role'],rows=len(arr),exact_equal=np.array_equal(arr,before),max_abs_error=float(abs(arr-before).max())))
    csv('qa/historical_cache_replay/current_runtime_repeatability.csv',checks)
    assert checks and all(c['exact_equal'] for c in checks),'Current runtime not stable; investigate before rebuild'
    print('CURRENT RUNTIME REPEATABILITY PASS',len(checks),'batches; rebuilding only source baseline',flush=True)
    started=time.time();output={}
    try:
        for role,input_texts,old in [('product',texts,oldp),('query',queries['query'].tolist(),oldq)]:
            t=time.time();arr,stats=engine._encode_native(input_texts,NATIVE['pooling'])
            assert arr.shape==old.shape and arr.dtype==np.float32 and np.isfinite(arr).all()
            path=HERE/'cache'/('rebuilt_source_'+role+'.npy');np.save(path,arr)
            output[role]=dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path),shape=list(arr.shape),
                metadata=stats,seconds=time.time()-t,maximum_difference_from_historical=float(abs(arr-old).max()))
            print('REBUILT',role,arr.shape,round(time.time()-t,1),'sec',flush=True)
    finally:engine.close()
    preservation_check()
    dump('baseline_manifest.json',dict(plan_sha256=sha(HERE/'baseline_rebuild_plan.json'),
        selection_manifest_sha256=sha(HERE/'selection_manifest.json'),config_sha256=sha(HERE/'config.json'),
        encoder_sha256=sha(ROOT/'phase2/src/encoding.py'),arrays=output,
        completed_utc=now(),seconds=time.time()-started,historical_cache_used_for_scoring=False))
    print('BASELINE REBUILD COMPLETE',round(time.time()-started,1),'sec',flush=True)

if __name__=='__main__':main()
