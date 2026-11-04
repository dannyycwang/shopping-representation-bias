"""Select exchange sets, then audit actual lengths with each model, then freeze."""
from core import *
import argparse, subprocess

def select():
    assert not (HERE/'sample_manifest.json').exists(),'Already frozen'
    t=Timer();products,queries,j=load();dev=frame(PILOT/'data/selected_pairs.csv',dtype={'product_id':str})
    eligible=sorted(set(read(ROOT/'phase4/config/splits.json')['wands_heldout'])&set(j.loc[j.label.eq('Exact'),'query_id']))
    assert eligible==frame(PILOT/'data/query_selection.csv').query_id.tolist()
    confirm=sorted(set(eligible)-set(dev.query_id));assert set(confirm).isdisjoint(set(dev.query_id))
    exact=j[j.query_id.isin(confirm)&j.label.eq('Exact')];needed=set(exact.product_id)
    swaps=[];audit=[];exclusions=[]
    for ix,p in enumerate(products):
        if str(p['product_id']) not in needed:continue
        s,a=choose_five(p);audit.append(dict(product_id=str(p['product_id']),catalog_index=ix,**a))
        if s:swaps.extend(dict(**r,catalog_index=ix) for r in s)
        else:exclusions.append(dict(product_id=str(p['product_id']),reason=a['reason'],valid_pairs=a['valid_pairs']))
    csv('data/development_pairs.csv',dev);csv('data/query_partition.csv',[dict(query_id=q,partition='development' if q in set(dev.query_id) else 'confirmation') for q in eligible])
    csv('data/confirmation_exact_pool.csv',exact);csv('data/preselected_swaps.csv',swaps)
    csv('data/pair_eligibility_audit.csv',audit);csv('data/excluded_pair_count.csv',exclusions)
    prior=read(PILOT/'delivery_manifest.json')['files']
    preserved={str((PILOT/n).relative_to(ROOT)).replace('\\','/'):h for n,h in prior.items()}
    preserved.update({str((PILOT/'delivery_manifest.json').relative_to(ROOT)).replace('\\','/'):sha(PILOT/'delivery_manifest.json')})
    dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT).decode().splitlines()
    preserved.update({n:sha(ROOT/n) for n in dirty if (ROOT/n).is_file()})
    dump('qa/preservation.json',preserved)
    oldeligible=set(frame(PILOT/'data/eligible_products.csv',dtype={'product_id':str}).product_id)
    counts=dict(catalog_products=len(products),eligible_queries=len(eligible),development_queries=dev.query_id.nunique(),
        confirmation_queries=len(confirm),confirmation_exact_pool=len(exact),confirmation_pool_products=len(needed),
        previous_generic_pair_upper_bound=int(exact[exact.product_id.isin(oldeligible)].groupby('query_id').size().clip(upper=5).sum()),
        five_pair_eligible_products=len(swaps)//5,five_pair_excluded_products=len(exclusions))
    dump('data/preparation_counts.json',counts);dump('model_profiles.json',profiles());timed('qa/select_timing.json',t)
    print('SELECTED EXCHANGE SETS',canonical(counts),flush=True)

def lengths(key):
    assert not (HERE/'sample_manifest.json').exists();t=Timer();products,queries,_=load()
    prof=profiles()[key];s=frame(HERE/'data/preselected_swaps.csv',dtype={'product_id':str});audit=[]
    if key=='gte':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(prof['model']['name'],revision=prof['model']['revision'],local_files_only=True)
        def tokens(texts,role):return tokenizer([prof['prefixes'][role]+x for x in texts],truncation=False,verbose=False)['input_ids']
        adapter=None
    else:
        adapter=harrier_adapter()
        def tokens(texts,role):
            out=adapter.inputs(texts,role,False)
            return [i[m.bool()].tolist() for i,m in zip(out['input_ids'],out['attention_mask'])]
    try:
        for n,(pid,g) in enumerate(s.groupby('product_id',sort=False)):
            rows=variants(products[int(g.iloc[0].catalog_index)],g.to_dict('records'));ids=tokens([r['text'] for r in rows],'document')
            for r,ii in zip(rows,ids):audit.append(dict(product_id=pid,variant=r['variant'],tokens=len(ii),token_ids_sha256=digest(canonical(ii)),fits=len(ii)<=prof['max_length'],text_sha256=r['text_sha256']))
            if n%500==0:print(key,'LENGTHS',n,'/',s.product_id.nunique(),flush=True)
        qs=queries[queries.query_id.isin(frame(HERE/'data/query_partition.csv').query_id)]
        qids=tokens(qs['query'].tolist(),'query')
        csv(f'data/query_inputs_{key}.csv',[dict(query_id=int(r.query_id),query=r.query,model_input=prof['prefixes']['query']+r.query,tokens=len(ii),token_ids_sha256=digest(canonical(ii))) for r,ii in zip(qs.itertuples(),qids)])
        csv(f'data/token_audit_{key}.csv',audit)
        import importlib.metadata as md
        versions={name:md.version(name) for name in ['torch','transformers','numpy','pandas','tokenizers']}
        if key=='harrier':versions['sentence-transformers']=md.version('sentence-transformers')
        dump(f'qa/runtime_{key}.json',dict(versions=versions,python=sys.version,executable=sys.executable))
        timed(f'qa/tokenization_{key}_timing.json',t)
        print('TOKEN AUDIT COMPLETE',key,len(audit),flush=True)
    finally:
        if adapter:adapter.close()

def freeze():
    assert not (HERE/'sample_manifest.json').exists();t=Timer();products,queries,_=load()
    swaps=frame(HERE/'data/preselected_swaps.csv',dtype={'product_id':str});eligible=set(swaps.product_id);reject=[]
    for key in ['gte','harrier']:
        ta=frame(HERE/f'data/token_audit_{key}.csv',dtype={'product_id':str})
        assert len(ta)==swaps.product_id.nunique()*11
        bad=ta.loc[~ta.fits,'product_id'].unique();eligible-=set(bad)
        reject.extend(dict(product_id=p,model=key,reason='at_least_one_A_C_O_exceeds_cap') for p in bad)
    csv('data/excluded_length.csv',pd.DataFrame(reject,columns=['product_id','model','reason']))
    exact=frame(HERE/'data/confirmation_exact_pool.csv',dtype={'product_id':str});pairs=[];qmap=queries.set_index('query_id')['query'].to_dict()
    for qid,g in exact[exact.product_id.isin(eligible)].groupby('query_id'):
        for pid in sorted(g.product_id,key=lambda p:pick('target',qid,p))[:5]:
            ix=int(swaps.loc[swaps.product_id.eq(pid),'catalog_index'].iloc[0])
            pairs.append(dict(query_id=int(qid),query=qmap[qid],product_id=pid,catalog_index=ix,label='Exact',target_hash=pick('target',qid,pid)))
    selected={r['product_id'] for r in pairs};chosen=swaps[swaps.product_id.isin(selected)].copy()
    csv('data/selected_pairs.csv',pairs);csv('data/selected_swaps.csv',chosen)
    full=[]
    for pid,g in chosen.groupby('product_id',sort=False):full.extend(variants(products[int(g.iloc[0].catalog_index)],g.to_dict('records')))
    csv('data/full_inputs.csv',full);pd.DataFrame(full).to_parquet(HERE/'data/full_inputs.parquet',index=False)
    dev=frame(HERE/'data/development_pairs.csv',dtype={'product_id':str});f=pd.DataFrame(pairs)
    mapping=f.groupby('product_id').agg(query_count=('query_id','nunique'),query_ids=('query_id',lambda x:canonical(sorted(x.tolist())))).reset_index()
    mapping['in_development_products']=mapping.product_id.isin(set(dev.product_id));csv('data/product_mapping.csv',mapping)
    counts=read(HERE/'data/preparation_counts.json');counts.update(sampled_pairs=len(f),sampled_queries=f.query_id.nunique(),sampled_products=len(selected),
        products_repeated_across_confirm_queries=int(mapping.query_count.gt(1).sum()),pairs_on_repeated_products=int(mapping.loc[mapping.query_count.gt(1),'query_count'].sum()),
        development_product_overlap=int(mapping.in_development_products.sum()),pairs_with_development_product_overlap=int(mapping.loc[mapping.in_development_products,'query_count'].sum()),
        length_excluded_products=len(set(r['product_id'] for r in reject)),post_length_eligible_pool_products=len(eligible))
    dump('data/sample_counts.json',counts)
    csv('data/smoke_pairs.csv',sorted(pairs,key=lambda r:pick('smoke',r['query_id'],r['product_id']))[:5])
    cfg=dict(seed=SEED,models=profiles(),model_choice='User explicitly confirmed GTE + Harrier',variants=['A']+[f'{c}{j}' for c in ['C','O'] for j in range(1,6)],
        query_split='existing held-out eligible IDs minus all 100 development query IDs',
        sampling='5 distinct valid entry pairs per product via Python Random seeded by purpose hash; ordered sample without replacement; then common complete-input support, then up to 5 Exact targets/query by independent target hash',
        hash_scheme='SHA256 canonical JSON [str(seed), confirmatory_visibility:purpose, *str(ids)]',
        parser_source=str((PILOT/'common.py').relative_to(ROOT)),missing_placeholders=sorted(MISSING),
        yes_no='normalized entire raw value equals yes or no in either selected entry; pre-specified secondary analysis',
        primary_weighting='equal query-product pair weight; RQ1 average the 5 modifications within each pair before pooling; rates conditional on A outside/inside; secondary yes/no average within available subtype for each pair and report eligible pair counts',
        rq2=dict(budgets=[1,3,5],selection='source score maximum among A and first b variants; exact ties prefer A then smaller j',all_pairs_included=True,
            interpretation='same known query, source scores available; transfer across encoders, not unknown queries or disjoint products'),
        bootstrap=dict(draws=10000,unit='query',scheme='paired cluster percentile bootstrap; re-sample all confirmation queries with replacement, preserve all targets/modifications/conditions',
            pairing='one common draw-index matrix for all models, C/O, budgets and strata',confidence=.95,purpose='query_bootstrap',exploratory='budget and yes/no stratum comparisons'),
        tie_break='descending exact float32 score then ascending numeric catalog index',
        effect_interpretation='C-O is a contrast of two interventions, not a pure internal binding effect',counts=counts)
    dump('config.json',cfg)
    files=[p for p in (HERE/'data').glob('*') if p.is_file()]+[HERE/'config.json',HERE/'model_profiles.json',HERE/'core.py',HERE/'prepare.py']
    sources=[PILOT/'common.py',PILOT/'config.json',PILOT/'baseline_manifest.json',PILOT/'data/baseline_batch_plan.json',PILOT/'data/selected_pairs.csv',HAR/'PROTOCOL.json',HAR/'scripts/adapters.py',HAR/'scripts/encode.py',HAR/'scripts/common.py',ROOT/'phase2/src/encoding.py',ROOT/'phase2/src/representations.py',ROOT/'phase2/data/processed/wands_products.jsonl.gz',ROOT/'phase2/data/processed/wands_queries.csv',ROOT/'phase2/data/processed/wands_judgments.csv',ROOT/'phase4/config/splits.json']
    dump('sample_manifest.json',dict(frozen_utc=now(),new_scores_computed=False,selection_depends_on_scores=False,
        files={p.relative_to(HERE).as_posix():sha(p) for p in files},source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    timed('qa/freeze_timing.json',t);print('FROZEN',canonical(counts),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','lengths','freeze']);ap.add_argument('--model',choices=['gte','harrier']);a=ap.parse_args()
    if a.stage=='select':select()
    elif a.stage=='lengths':lengths(a.model)
    else:freeze()
