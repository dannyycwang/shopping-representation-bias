"""Resolve and audit existing source-only slices and exact original batch contexts."""
from core import *

def main():
    freeze_check();t=Timer();products,queries,_=load();n=len(products)
    selected=frame(HERE/'data/selected_pairs.csv',dtype={'product_id':str});needed=set(selected.product_id)
    qneeded=set(selected.query_id);original=[REP._plain(p) for p in products];plans={}
    gb=read(PILOT/'baseline_manifest.json');g=read(PILOT/'config.json')
    gp=read(PILOT/'data/baseline_batch_plan.json');gfiles={}
    for role,info in gb['arrays'].items():
        path=PILOT/info['path'];assert sha(path)==info['sha256'];gfiles[role]=dict(path=path.relative_to(ROOT).as_posix(),sha256=info['sha256'])
    gm={};gcontexts={};qcontexts={};qm={}
    for bi,b in enumerate(gp['catalog_batches']):
        targets=[i for i in b if str(products[i]['product_id']) in needed]
        if targets:
            gcontexts[str(bi)]=dict(indices=b,texts=[original[i] for i in b])
            for i in targets:gm[str(products[i]['product_id'])]=dict(batch=str(bi),offset=b.index(i),baseline_index=i)
    for bi,b in enumerate(gp['query_batches']):
        targets=[i for i in b if int(queries.iloc[i].query_id) in qneeded]
        if targets:
            qcontexts[str(bi)]=dict(indices=b,texts=queries.iloc[b]['query'].tolist())
            for i in targets:qm[str(int(queries.iloc[i].query_id))]=dict(batch=str(bi),offset=b.index(i),baseline_index=i)
    for p,h in g['model_snapshot_hashes'].items():assert sha(Path(g['model_snapshot'])/p)==h
    plans['gte']=dict(files=gfiles,catalog_batches=gcontexts,query_batches=qcontexts,product_map=gm,query_map=qm,
        source_catalog_slice=[0,n],model_snapshot=g['model_snapshot'],model_snapshot_hashes=g['model_snapshot_hashes'],
        catalog_native_truncated=int(frame(PILOT/'data/catalog_token_counts.csv').native_baseline_truncated.sum()))
    hfiles={};hc={};hm={};hq={};hqm={};checks=[]
    eligible_q=sorted(frame(HERE/'data/query_partition.csv').query_id)
    for role,sub,rawtexts,ids,needed_ids in [('product','raw',original,[str(p['product_id']) for p in products],needed),
                                           ('query','queries',queries.set_index('query_id').loc[eligible_q]['query'].tolist(),list(map(str,eligible_q)),set(map(str,qneeded)))]:
        directory=HAR/f'runs/harrier06b/wands/{sub}';complete=read(directory/'complete.json');desc=read(directory/'descriptor.json')
        assert desc['spec']==profiles()['harrier']['model']
        assert desc['adapter_sha256']==sha(HAR/'scripts/adapters.py')
        assert desc['software']==read(HAR/'PROTOCOL.json')['software']
        assert desc['text_hashes'][:len(rawtexts)]==[digest(x) for x in rawtexts]
        for info in complete['outputs']:
            assert sha(info['path'])==info['sha256'],info['path']
            checks.append(dict(role=role,path=str(Path(info['path']).relative_to(ROOT)),sha256=info['sha256']))
        cache=directory/'embeddings.npy';hfiles[role]=dict(path=cache.relative_to(ROOT).as_posix(),sha256=next(x['sha256'] for x in complete['outputs'] if Path(x['path']).name=='embeddings.npy'))
        plan=read(directory/'plan.json');aliases=plan['aliases'];targets={aliases[i]:[] for i,x in enumerate(ids) if x in needed_ids}
        for i,x in enumerate(ids):
            if x in needed_ids:targets[aliases[i]].append((x,i))
        contexts={};mapping={};texts_needed=set()
        for bi,b in enumerate(plan['batches']):
            chosen=set(b)&set(targets)
            if chosen:
                contexts[str(bi)]=dict(indices=b);texts_needed.update(b)
                for u in chosen:
                    for x,i in targets[u]:mapping[x]=dict(batch=str(bi),offset=b.index(u),baseline_index=i)
        texts={}
        with gzip.open(directory/'unique_inputs.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line);i=r['index']
                if i in texts_needed:
                    assert digest(r['text'])==plan['unique_text_hashes'][i];texts[i]=r['text']
        assert len(texts)==len(texts_needed)
        for c in contexts.values():c['texts']=[texts[i] for i in c['indices']]
        for x,loc in mapping.items():assert contexts[loc['batch']]['texts'][loc['offset']]==rawtexts[loc['baseline_index']]
        if role=='product':
            hc,hm=contexts,mapping;h_lengths=np.load(directory/'lengths.npy',mmap_mode='r')[:n]
        else:hq,hqm=contexts,mapping
    receipt=read(HAR/'audit/download_harrier06b.json');hp=Path(receipt['path'])
    assert sha(hp/'model.safetensors')==receipt['weights_sha256']
    plans['harrier']=dict(files=hfiles,catalog_batches=hc,query_batches=hq,product_map=hm,query_map=hqm,
        source_catalog_slice=[0,n],model_snapshot=str(hp),weights_sha256=receipt['weights_sha256'],
        catalog_native_truncated=int((h_lengths>profiles()['harrier']['max_length']).sum()),validated_source_files=checks)
    for key,plan in plans.items():
        assert len(plan['product_map'])==len(needed) and len(plan['query_map'])==len(qneeded)
        dump(f'cache_plan_{key}.json',plan)
    estimate={}
    for key,plan in plans.items():
        rows=sum(len(plan['catalog_batches'][v['batch']]['indices'])*10 for v in plan['product_map'].values())
        ips=42994/(gb['arrays']['product']['seconds']) if key=='gte' else read(HAR/'smoke/harrier06b/throughput_estimate.json')['inputs_per_second']
        estimate[key]=dict(variant_count=len(needed)*10,matched_context_forward_rows=rows,reference_inputs_per_second=ips,
            rough_variant_seconds=rows/ips,note='Historical throughput estimate, not a promise; preserves source batch companions and numerical profile.')
    dump('qa/cache_cost_estimate.json',estimate)
    dump('cache_manifest.json',dict(sample_manifest_sha256=sha(HERE/'sample_manifest.json'),plans={k:sha(HERE/f'cache_plan_{k}.json') for k in plans},created_utc=now()))
    timed('qa/cache_plan_timing.json',t);print('CACHE PLANS VERIFIED',canonical(estimate),flush=True)

if __name__=='__main__':main()
