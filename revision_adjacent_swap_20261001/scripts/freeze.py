"""Generate eligibility and a frozen cohort without computing retrieval outcomes."""
from common import *
import importlib.metadata, time

def main():
    assert not (HERE/'freeze_manifest.json').exists(), 'Frozen cohort exists; do not overwrite.'
    from transformers import AutoTokenizer
    import torch, transformers, tokenizers
    for d in ['data','qa','cache','figures']: (HERE/d).mkdir(parents=True, exist_ok=True)
    config = dict(version='adjacent-swap-v1',salt='20261001',query_cap=128,target_cap=16,
        primary_k=20,secondary_k=100,bootstrap_seed=2026100101,bootstrap_draws=10000,
        confirmation_repetitions=3,models={m:MODEL_SPECS[m] for m in MODELS},
        query_hash='["20261001","query",dataset,str(query_id)]',
        target_hash='["20261001","target",dataset,str(query_id),str(product_id)]',
        case_hash='["20261001","case",dataset,model,str(query_id),str(product_id)]',
        swap_hash='["20261001","case_swap",dataset,model,str(query_id),str(product_id),positions_json,text_sha256]',
        hash_encoding='UTF-8 compact JSON array; every element a string; SHA-256; lexicographic hex order',
        query_prefix='',document_prefix='',swap_count_bins=[1,3,7,15],rank_bins=[10,20,40],
        selection_uses_outcomes=False)
    dump(HERE/'config.json',config)
    assert torch.cuda.is_available(), 'Frozen execution needs the existing CUDA profile.'
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','torch','transformers','tokenizers','matplotlib','scipy','safetensors','threadpoolctl']}
    profiles={}; toks={}; source_files={}
    for path in [ROOT/'phase2/config/phase2.json',ROOT/'phase2/src/representations.py',ROOT/'revision_graded_20260922/PROTOCOL.json']:
        source_files[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    for m in MODELS:
        s=MODEL_SPECS[m]; tok=AutoTokenizer.from_pretrained(s['name'],revision=s['revision'],local_files_only=True); toks[m]=tok
        previous=read(HARM/m/'profile.json'); complete=read(HARM/m/'complete.json'); plan=read(HARM/m/'plan.json')
        checks=dict(torch=previous['torch']==torch.__version__,transformers=previous['transformers']==transformers.__version__,
            tokenizers=previous['tokenizers']==tokenizers.__version__,cuda=previous['cuda']==torch.version.cuda,
            gpu=previous['gpu']==torch.cuda.get_device_name(),serializer=previous['serializer_sha256']==sha(ROOT/'phase2/src/representations.py'),
            model=previous['model']==s,profile_complete=previous['fingerprint']==complete['profile'],
            vectors=sha(HARM/m/'vectors.npy')==complete['vectors_sha256'],
            inputs=sha(HARM/m/'inputs.parquet')==plan['inputs_sha256'],aliases=sha(HARM/m/'aliases.parquet')==plan['aliases_sha256'])
        # Validate profile semantics as well as versions and payload hashes.
        checks['settings']=all([previous['model_dtype']=='float16', previous['pooling_normalization_scoring_dtype']=='float32',
            previous['prefixes']=='',previous['attention']=='eager',previous['tf32'] is False,previous['deterministic'] is True,
            previous['batch_size']==(64 if m=='minilm' else 16),previous['cap']==s['max_tokens']])
        assert all(checks.values()), f'Incompatible reference; construct a consistent new C0 cache before freeze: {checks}'
        snapshots=Path.home()/'.cache/huggingface/hub'/('models--'+s['name'].replace('/','--'))/'snapshots'/s['revision']
        model_files={p.name:sha(p) for p in sorted(snapshots.iterdir()) if p.is_file()}
        profiles[m]=dict(reference_folder=str((HARM/m).relative_to(ROOT)).replace('\\','/'),reference_profile=previous,
            reference_complete=complete,reference_plan=plan,compatibility_checks=checks,model_snapshot_hashes=model_files,
            tokenizer=dict(class_name=tok.__class__.__name__,is_fast=tok.is_fast,padding_side=tok.padding_side,
                truncation_side=tok.truncation_side,model_max_length=tok.model_max_length,add_special_tokens=True,
                special_tokens=tok.special_tokens_map,do_lower_case=getattr(tok,'do_lower_case',None)),
            precision='FP16 model; FP32 pooling/L2 normalization/elementwise score reduction',batch_size=64 if m=='minilm' else 16)
        print('CACHE VERIFIED',m,flush=True)
    dirty=git('diff','--name-only').splitlines()
    execution=dict(frozen_at_utc=now(),starting_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        python=sys.version,executable=sys.executable,platform=platform.platform(),versions=versions,
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']).decode().strip(),
        profiles=profiles,deterministic_algorithms=True,tf32=False,attention='eager',cpu_threads=4,seed=20261001,
        cublas_workspace_config=':4096:8',query_instructions='',query_and_competitors_fixed=True,
        preexisting_tracked_edits={p:sha(ROOT/p) for p in dirty if (ROOT/p).is_file()},
        agents_md='No applicable AGENTS.md found in repository or ancestor directories.',
        writes_confined_to=HERE.name)
    dump(HERE/'execution_profile.json',execution)
    population=[]; selected=[]; planned=[]; query_rows=[]; audit_rows=[]; counts=[]; started=time.time()
    for ds in DATASETS:
        products, queries, judgments=load(ds); lookup={str(p['product_id']):p for p in products}
        for suffix in ['products.jsonl.gz','queries.csv','judgments.csv']:
            path=ROOT/f'phase2/data/processed/{ds}_{suffix}'; source_files[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
        held=support(ds); relevant=judgments[judgments.query_id.isin(held)&judgments.label.eq('Exact' if ds=='wands' else 'E')][['query_id','product_id']].drop_duplicates()
        assert len(relevant)==(21299 if ds=='wands' else 4434)
        wanted=set(relevant.product_id); metadata={}; catalog_index={str(p['product_id']):i for i,p in enumerate(products)}
        # C0 alone can prove ineligibility. Do not tokenize hundreds of variants
        # of an input that already fails; all retained inputs get full enumeration.
        original_lengths={m:{} for m in MODELS}; pids=sorted(wanted)
        for start in range(0,len(pids),256):
            part=pids[start:start+256];texts=[REP._plain(lookup[p]) for p in part]
            for m,tok in toks.items():
                enc=tok(texts,add_special_tokens=True,truncation=False,verbose=False)
                original_lengths[m].update(zip(part,map(len,enc['input_ids'])))
        for n,pid in enumerate(sorted(wanted)):
            p=lookup[pid]; vs,noops=variants(p); texts=[REP._plain(p)]+[v['text'] for v in vs]
            meta=dict(dataset=ds,product_id=pid,catalog_index=catalog_index[pid],attribute_count=len(p['attributes']),
                candidate_positions=max(0,len(p['attributes'])-1),no_op_positions=noops,distinct_swaps=len(vs),
                duplicate_serializations=sum(len(v['positions']) for v in vs)-len(vs),c0_sha256=digest(texts[0]))
            for m,tok in toks.items():
                c0length=original_lengths[m][pid]
                full_scan=c0length<=MODEL_SPECS[m]['max_tokens'] and len(vs)>0
                lengths=([c0length]+[len(x) for x in tok(texts[1:],add_special_tokens=True,truncation=False,verbose=False)['input_ids']]) if full_scan else [c0length]
                meta['all_variants_tokenized_'+m]=full_scan
                meta['c0_tokens_'+m]=lengths[0];meta['max_tokens_'+m]=max(lengths)
            metadata[pid]=meta
            if n%1000==0:print('ELIGIBILITY',ds,n,'/',len(wanted),'seconds',round(time.time()-started),flush=True)
        for m in MODELS:
            cap=MODEL_SPECS[m]['max_tokens']; pool=relevant.merge(pd.DataFrame(metadata.values()),on='product_id',validate='many_to_one')
            pool['model']=m;pool['eligible']=pool.distinct_swaps.gt(0)&pool['max_tokens_'+m].le(cap)
            pool['exclusion']=np.where(pool.distinct_swaps.eq(0),'no_distinct_non_noop_swap',np.where(pool['max_tokens_'+m].gt(cap),'one_or_more_inputs_exceed_context','eligible'))
            pool['query_hash']=[selection_hash('query',ds,q) for q in pool.query_id]
            pool['target_hash']=[selection_hash('target',ds,q,p) for q,p in zip(pool.query_id,pool.product_id)]
            eligible=pool[pool.eligible]; qsel=eligible[['query_id','query_hash']].drop_duplicates().sort_values(['query_hash','query_id']).head(128)
            sample=eligible[eligible.query_id.isin(qsel.query_id)].sort_values(['query_hash','target_hash','product_id']).groupby('query_id',sort=False).head(16).copy()
            sample['query_priority']=sample.query_id.map({q:i+1 for i,q in enumerate(qsel.query_id)})
            sample['target_priority']=sample.groupby('query_id',sort=False).cumcount()+1
            sample['case_hash']=[selection_hash('case',ds,m,q,p) for q,p in zip(sample.query_id,sample.product_id)]
            selected.append(sample); population.append(pool)
            qmap=queries.set_index('query_id')['query'].to_dict()
            query_rows.extend(dict(dataset=ds,model=m,query_id=int(q),query=str(qmap[q])) for q in qsel.query_id)
            for pid in sorted(set(sample.product_id)):
                p=lookup[pid]; vs,_=variants(p); text=REP._plain(p)
                texts=[text]+[v['text'] for v in vs]; enc=toks[m](texts,add_special_tokens=True,truncation=False,verbose=False)
                for i,t in enumerate(texts):
                    positions=[] if i==0 else vs[i-1]['positions']; n_tokens=len(enc['input_ids'][i])
                    row=dict(dataset=ds,model=m,product_id=pid,variant='C0' if i==0 else 'S'+str(positions[0]),
                        positions=canonical_json(positions),text=t,text_sha256=digest(t),input_sha256=input_hash(enc,i),
                        tokens=n_tokens,context_limit=cap,bucket=bucket(n_tokens,cap),c0_sha256=digest(text),
                        canonical_sha256=digest(REP._plain(p,sorted(p['attributes'],key=lambda x:(x.casefold(),x)))))
                    planned.append(row)
                    if i:
                        checks=audit(p,vs[i-1]); checks['no_truncation']=n_tokens<=cap and len(enc['input_ids'][0])<=cap
                        assert all(checks.values()), (ds,m,pid,positions,checks)
                        audit_rows.append({k:row[k] for k in row if k!='text'}|checks)
            counts.append(dict(dataset=ds,model=m,catalog_products=len(products),held_out_queries=len(held),highest_label_pairs=len(pool),
                eligible_queries=eligible.query_id.nunique(),eligible_pairs=len(eligible),eligible_products=eligible.product_id.nunique(),
                excluded_no_swaps=int(pool.distinct_swaps.eq(0).sum()),excluded_context=int((pool.distinct_swaps.gt(0)&pool['max_tokens_'+m].gt(cap)).sum()),
                sampled_queries=sample.query_id.nunique(),sampled_pairs=len(sample),sampled_products=sample.product_id.nunique(),
                tested_pair_swaps=int(sample.distinct_swaps.sum()),distinct_product_swaps=sum(r['model']==m and r['dataset']==ds and r['variant']!='C0' for r in planned)))
            print('COHORT',counts[-1],flush=True)
    pop=pd.concat(population,ignore_index=True); cohort=pd.concat(selected,ignore_index=True); plan=pd.DataFrame(planned)
    pop.to_parquet(HERE/'data/eligibility_population.parquet',index=False)
    csv(cohort,HERE/'cohort_manifest.csv'); plan.to_parquet(HERE/'data/variant_plan.parquet',index=False)
    csv(pd.DataFrame(query_rows),HERE/'data/queries.csv');csv(pd.DataFrame(counts),HERE/'data/population_counts.csv')
    pd.DataFrame(audit_rows).to_parquet(HERE/'data/representation_audit.parquet',index=False)
    dist=[]
    for scope,f in [('highest_label_population',pop),('eligible_population',pop[pop.eligible]),('sampled',cohort)]:
        for (ds,m,a,s),g in f.groupby(['dataset','model','attribute_count','distinct_swaps']):
            dist.append(dict(scope=scope,dataset=ds,model=m,attribute_count=int(a),distinct_swaps=int(s),pairs=len(g),queries=g.query_id.nunique()))
    csv(pd.DataFrame(dist),HERE/'data/attribute_swap_distributions.csv')
    dump(HERE/'qa/representation_summary.json',dict(variants_audited=len(audit_rows),failures=0,all_targets_fit_every_variant=True,
        canonical_invariance='Structural equality for every variant; no independent retrieval experiment',population_tokenized_before_selection=True))
    names=['protocol.md','config.json','cohort_manifest.csv','execution_profile.json','data/eligibility_population.parquet',
        'data/variant_plan.parquet','data/queries.csv','data/population_counts.csv','data/representation_audit.parquet','scripts/common.py','scripts/freeze.py']
    dump(HERE/'freeze_manifest.json',dict(frozen_at_utc=now(),new_swap_outcomes_examined=False,
        files={p:sha(HERE/p) for p in names},source_files=source_files,counts=counts,elapsed_seconds=time.time()-started))
    print('FREEZE COMPLETE',flush=True)

if __name__=='__main__':main()
