"""Read source/cache provenance; freeze generic swaps and samples before scores."""
from common import *
from transformers import AutoTokenizer

def main():
    preservation_start();start=time.time();products,queries,judgments=load()
    ids=[str(p['product_id']) for p in products];pindex={p:i for i,p in enumerate(ids)}
    assert len(set(ids))==len(ids) and [int(x) for x in ids]==sorted(map(int,ids))
    assert not judgments.duplicated(['query_id','product_id']).any()
    rawj=frame(ROOT/'data/raw/label.csv',sep='\t',dtype={'product_id':str})
    nlabels=rawj.groupby(['query_id','product_id']).label.nunique();bad=nlabels[nlabels>1].index
    cleaned=rawj[~rawj.set_index(['query_id','product_id']).index.isin(bad)].drop_duplicates(['query_id','product_id'])
    cols=['query_id','product_id','label']
    pd.testing.assert_frame_equal(cleaned[cols].sort_values(cols).reset_index(drop=True),judgments[cols].sort_values(cols).reset_index(drop=True))
    rawq=frame(ROOT/'data/raw/query.csv',sep='\t')
    pd.testing.assert_frame_equal(rawq[['query_id','query']].sort_values('query_id').reset_index(drop=True),queries[['query_id','query']].reset_index(drop=True))
    split=read(ROOT/'phase4/config/splits.json')['wands_heldout']
    eligible_q=sorted(set(split)&set(judgments.loc[judgments.label.eq('Exact'),'query_id']))
    assert eligible_q==read(ROOT/'revision_graded_20260922/PROTOCOL.json')['populations']['wands']['eligible_ids']
    high=judgments[judgments.query_id.isin(eligible_q)&judgments.label.eq('Exact')]
    # Resolve cache by recorded provenance, not by choosing a file with convenient scores.
    provenance=frame(ROOT/'revision_section6_20260924/data/historical_execution_matrix.csv')
    provenance=provenance[provenance.dataset.eq('wands')&provenance.model.eq(SPEC['key'])&provenance.schedule.eq('C0')]
    cache={}
    for role in ['product','query']:
        r=provenance[provenance.role.eq(role)].iloc[0]
        assert sha(ROOT/r.payload)==r.payload_sha256 and sha(ROOT/r.metadata)==r.metadata_sha256
        meta=read(ROOT/r.metadata)
        assert meta['model']==SPEC and meta['profile']==NATIVE and meta['precision']=='fp16_model_fp32_pooling'
        assert meta['implementation']=='phase2-encoder-v3-length-bucket-sdpa'
        cache[role]=dict(path=r.payload,sha256=r.payload_sha256,metadata_path=r.metadata,metadata_sha256=r.metadata_sha256,metadata=meta)
    assert cache['product']['metadata']['shape'][0]==len(products) and cache['query']['metadata']['shape'][0]==len(queries)
    catalog_order=frame(ROOT/'revision_graded_20260922/data/wands_catalog_ordering.csv',dtype={'catalog_id':str})
    query_order=frame(ROOT/'revision_graded_20260922/data/wands_queries_ordering.csv')
    assert catalog_order.catalog_id.tolist()==ids and query_order.queries_id.tolist()==queries.query_id.tolist()
    original=[REP._plain(p) for p in products];sourcehash=hashlib.sha256()
    with gzip.open(ROOT/'phase2/data/representations/wands/C0.jsonl.gz','rt',encoding='utf8') as f:
        for i,line in enumerate(f):
            sourcehash.update(line.encode());r=json.loads(line)
            assert str(r['product_id'])==ids[i] and r['text']==original[i]
    assert i+1==len(products) and sourcehash.hexdigest()==cache['product']['metadata']['source_sha256']
    assert digest('\0'.join(queries['query']))==cache['query']['metadata']['source_sha256']
    tok=AutoTokenizer.from_pretrained(SPEC['name'],revision=SPEC['revision'],local_files_only=True)
    chosen=[];excluded=[];entry_reasons=collections.Counter()
    for i,p in enumerate(products):
        s,audit=select_pair(p);entry_reasons.update(audit['entry_rejections'])
        if s:chosen.append(dict(**s,catalog_index=i,source_class=p['class']))
        else:excluded.append(dict(product_id=ids[i],stage='parse',reason=audit['reason'],details=canonical(audit)))
    print('Parsed',len(products),'products;',len(chosen),'have a preselected generic swap',flush=True)
    lengths_A=[]
    for start in range(0,len(original),256):
        lengths_A.extend(map(len,tok(original[start:start+256],add_special_tokens=True,truncation=False,verbose=False)['input_ids']))
        if start%8192==0:print('A tokens',start,'/',len(original),flush=True)
    lengths_C={};newtexts={}
    for start in range(0,len(chosen),128):
        part=chosen[start:start+128];texts=[]
        for s in part:
            a,c=swap(products[s['catalog_index']],s);assert a==original[s['catalog_index']]
            texts.append(c)
        nums=list(map(len,tok(texts,add_special_tokens=True,truncation=False,verbose=False)['input_ids']))
        for s,n in zip(part,nums):lengths_C[s['product_id']]=n
        if start%8192==0:print('C tokens',start,'/',len(chosen),flush=True)
    cap=SPEC['max_tokens'];eligible=[]
    for s in chosen:
        a=lengths_A[s['catalog_index']];c=lengths_C[s['product_id']]
        s.update(tokens_A=a,tokens_C=c)
        if a<=cap and c<=cap:eligible.append(s)
        else:excluded.append(dict(product_id=s['product_id'],stage='length',reason='A_or_C_exceeds_model_cap',details=canonical(dict(tokens_A=a,tokens_C=c,cap=cap))))
    eligible_map={s['product_id']:s for s in eligible}
    eh=high[high.product_id.isin(eligible_map)]
    qids=sorted(eh.query_id.unique(),key=lambda q:pick('query',int(q)))[:100]
    qmap=queries.set_index('query_id')['query'].to_dict();pairs=[]
    for qid in qids:
        targets=sorted(eh[eh.query_id.eq(qid)].product_id,key=lambda pid:pick('target',int(qid),pid))[:5]
        for pid in targets:
            pairs.append(dict(query_id=int(qid),query=qmap[qid],label='Exact',**eligible_map[pid],
                query_selection_hash=pick('query',int(qid)),target_selection_hash=pick('target',int(qid),pid)))
    selected_ids={r['product_id'] for r in pairs}
    actual=[]
    for s in eligible:
        if s['product_id'] not in selected_ids:continue
        a,c=swap(products[s['catalog_index']],s)
        actual.append(dict(**s,text_A=a,text_C=c,model_input_A=a,model_input_C=c,text_sha256_A=digest(a),text_sha256_C=digest(c)))
    csv('data/selected_pairs.csv',pairs);csv('data/full_inputs.csv',actual)
    pd.DataFrame(actual).to_parquet(HERE/'data/full_inputs.parquet',index=False)
    csv('data/preselected_product_swaps.csv',chosen);csv('data/eligible_products.csv',eligible);csv('data/excluded_products.csv',excluded)
    csv('data/entry_exclusion_counts.csv',[dict(reason=k,entries=v) for k,v in sorted(entry_reasons.items())])
    csv('data/catalog_token_counts.csv',[dict(product_id=pid,catalog_index=i,tokens_A=lengths_A[i],tokens_C=lengths_C.get(pid,''),native_baseline_truncated=lengths_A[i]>cap) for i,pid in enumerate(ids)])
    qlengths=list(map(len,tok(queries['query'].tolist(),truncation=False,verbose=False)['input_ids']))
    csv('data/query_token_counts.csv',[dict(query_id=int(r.query_id),query=r.query,tokens=n,selected=int(r.query_id) in qids) for r,n in zip(queries.itertuples(),qlengths)])
    assert all(n<=cap for n in qlengths)
    csv('data/query_selection.csv',[dict(query_id=int(q),eligible_targets=int(eh.query_id.eq(q).sum()),selected=q in qids,
        reason='selected' if q in qids else 'hash_query_cap' if q in set(eh.query_id) else 'no_eligible_Exact_target') for q in eligible_q])
    pairkeys={(r['query_id'],r['product_id']) for r in pairs}
    csv('data/eligible_pair_disposition.csv',[dict(query_id=int(r.query_id),product_id=r.product_id,
        selected=(r.query_id,r.product_id) in pairkeys,reason='selected' if (r.query_id,r.product_id) in pairkeys else
        'hash_target_cap' if r.query_id in qids else 'hash_query_cap') for r in eh.itertuples()])
    smoke=sorted(pairs,key=lambda r:pick('smoke_pair',r['query_id'],r['product_id']))[:5]
    csv('data/smoke_pairs.csv',smoke)
    batches=batching(lengths_A);qbatches=batching(qlengths)
    dump('data/baseline_batch_plan.json',dict(catalog_batches=batches,query_batches=qbatches))
    token_summary=[]
    for label,vals in [('catalog_A',lengths_A),('preselected_A',[s['tokens_A'] for s in chosen]),('preselected_C',list(lengths_C.values())),
                       ('sampled_A',[s['tokens_A'] for s in actual]),('sampled_C',[s['tokens_C'] for s in actual])]:
        token_summary.append(dict(population=label,n=len(vals),minimum=min(vals),maximum=max(vals),**quantile(vals,'tokens'),
            above_cap_n=sum(x>cap for x in vals),above_cap_den=len(vals),above_cap_rate=sum(x>cap for x in vals)/len(vals)))
    csv('data/length_distribution.csv',token_summary)
    assert np.isclose(np.mean(lengths_A),cache['product']['metadata']['mean_wordpieces_with_special'],atol=1e-10)
    assert np.mean(np.array(lengths_A)>cap)==cache['product']['metadata']['fraction_truncated']
    counts=dict(catalog_products=len(products),source_queries=len(queries),split_queries=len(split),eligible_evaluation_queries=len(eligible_q),
        exact_pairs=len(high),parse_eligible_products=len(chosen),length_eligible_products=len(eligible),
        excluded_parse_products=len(products)-len(chosen),excluded_length_products=len(chosen)-len(eligible),
        eligible_exact_pairs=len(eh),queries_with_eligible_targets=eh.query_id.nunique(),sampled_queries=len(qids),
        sampled_pairs=len(pairs),sampled_distinct_products=len(actual),contradictory_qrel_pairs_removed=len(bad),
        baseline_native_truncated_products=int(np.sum(np.array(lengths_A)>cap)))
    dump('data/sample_counts.json',counts)
    snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+SPEC['name'].replace('/','--'))/'snapshots'/SPEC['revision']
    import torch,transformers,tokenizers,platform
    runtime=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,transformers=transformers.__version__,
        tokenizers=tokenizers.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),blas_threads=4)
    dump('config.json',dict(seed=SEED,model=SPEC,model_profile=NATIVE,query_prefix=CFG['encoding']['query_prefix'],document_prefix='',
        precision='FP16 model, FP32 CLS pooling/L2 normalization/scores',attention='sdpa',max_length=cap,runtime=runtime,
        cache=cache,model_snapshot=str(snapshot),model_snapshot_hashes={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()},
        parsing='first colon; key must contain an alphanumeric, no newline/tab; raw key/value substrings preserved',
        key_identity='casefold + whitespace collapse for uniqueness only; exclude selected normalized keys occurring more than once',
        missing_placeholders=sorted(MISSING),value_equality='casefold + whitespace collapse; no semantic/type whitelist',
        swap_selection='uniform random choice over valid entry-index pairs using Python Random seeded by SHA256([seed,value_swap_visibility:attribute_pair,product_id])',
        sampling='hash queries then hash targets; max 100 queries and 5 Exact targets/query, after actual A/C length checks',
        query_cap=100,target_cap=5,pair_cap=500,counts=counts,source_checksums_match=True,
        inherited_encoder='phase2/src/encoding.py:DenseEncoder',
        inference_batching='reconstruct exact source catalog batches; replay A in source companions; encode each C by replacing only its row in its source batch',
        scoring='numpy FP32 matmul, one unchanged query against full catalog; compute C score with identical full-matrix shape after replacing one embedding; verify all competitor scores bitwise unchanged',
        tie_break='descending exact FP32 score then ascending stable source catalog index (= ascending numeric product_id)',
        replay_gate='all selected A vectors bitwise equal to cache, before C inference; mismatch stops with audit',
        cases='all observed entry/exit; min/max rank_delta labeled extremes, hash breaks ties; 5 smallest independent ordinary-case hashes',
        cancelled_direction='round2_controlled cancelled by user; its files are not inputs to this experiment',
        historical_library_metadata='per-array sidecars omit library version; use existing phase2 venv 4.55.4 and fresh replay gate'))
    sources=['phase2/config/phase2.json','phase2/src/encoding.py','phase2/src/representations.py','phase2/scripts/prepare_data.py',
        'phase2/data/processed/wands_products.jsonl.gz','phase2/data/processed/wands_queries.csv','phase2/data/processed/wands_judgments.csv',
        'phase2/data/representations/wands/C0.jsonl.gz','phase4/config/splits.json','revision_graded_20260922/PROTOCOL.json',
        'revision_section6_20260924/data/historical_execution_matrix.csv','phase3/scripts/analyze_target_only_permutations.py']
    for c in cache.values():sources.extend([c['path'],c['metadata_path']])
    files=['common.py','prepare.py','config.json','data/selected_pairs.csv','data/smoke_pairs.csv','data/full_inputs.parquet',
        'data/preselected_product_swaps.csv','data/eligible_products.csv','data/catalog_token_counts.csv','data/baseline_batch_plan.json','data/sample_counts.json']
    dump('selection_manifest.json',dict(frozen_utc=now(),selection_uses_scores=False,new_scores_computed=False,
        files={p:sha(HERE/p) for p in files},source_files={p:sha(ROOT/p) for p in sources}))
    replay_batches={bi for bi,b in enumerate(batches) if any(ids[i] in selected_ids for i in b)}
    replay_rows=sum(len(batches[b]) for b in replay_batches)
    replacement_rows=sum(len(batches[b]) for b in replay_batches for i in batches[b] if ids[i] in selected_ids)
    dump('qa/cost_estimate.json',dict(reference_catalog_seconds=cache['product']['metadata']['seconds'],
        need_baseline_rebuild=False,existing_cache_sha_verified=True,
        matched_replay_forward_rows=replay_rows,replacement_forward_rows=replacement_rows,
        rough_seconds=(replay_rows+replacement_rows)/len(products)*cache['product']['metadata']['seconds'],
        caveat='Proportional to historical rows; actual length/padding/launch costs differ. If replay fails, no cache mixing; full source baseline rebuild estimate is historical source-only time, not a matrix.'))
    print('FROZEN',canonical(counts),flush=True)

if __name__=='__main__':main()
