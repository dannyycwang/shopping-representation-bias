"""Select at most three cases, audit real templates/tokens and rescore cached catalogs.

No model forward pass, training, download, query rewrite or new comparison occurs.
Historical lists and newly rescored lists are always saved separately.
"""
import collections, hashlib, os, platform, sys
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from transformers import AutoTokenizer
from common import *
from core import metric
sys.path.insert(0,str(ROOT/'phase2'))
from src.representations import build

def main():
    protocol=read('query_constraint_cases_20261007/protocol.json')
    candidates=lines('data/all_candidates.jsonl')
    ranked=pd.read_csv(HERE/'candidate_ranking.csv',keep_default_na=False)
    selected=[]
    selection_path=HERE/'selection.json'
    if selection_path.exists():
        # Source-audit corrections must never reopen the case search.
        for x in json.loads(selection_path.read_text())['selected']:
            selected.append(next(c for c in candidates if c['summary']['constraint_id']==x['constraint_id'] and c['summary']['comparison']==x['comparison']))
    else:
        for qid in [367,292]:
            selected.append(next(c for c in candidates if c['summary']['query_id']==qid and c['summary']['primary']))
        row=next(r for r in ranked.to_dict('records') if r['query_id'] not in [367,292])
        selected.append(next(c for c in candidates if c['summary']['constraint_id']==row['constraint_id'] and c['summary']['comparison']==row['comparison']))
    choice=[dict(query_id=c['summary']['query_id'],comparison=c['summary']['comparison'],constraint_id=c['summary']['constraint_id']) for c in selected]
    selection_path=HERE/'selection.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())['selected']==choice,'Selection must not adapt after detailed audits'
    else:dump('selection.json',dict(selected=choice,reason='Two required seed audits plus first distinct additional query in fixed preliminary ranking; execution audit can downgrade but cannot trigger a fourth case.',all_comparisons_completed=True,candidates=len(candidates)))
    decisions=[]
    for r in ranked.to_dict('records'):
        chosen=any(r['query_id']==x['query_id'] and r['comparison']==x['comparison'] and r['constraint_id']==x['constraint_id'] for x in choice)
        r.update(selected_for_full_audit=chosen,selection_disposition='selected before full-text audit; retained even if downgraded' if chosen else 'additional clause/contrast of selected query; bounded audit uses selected contrast only' if r['query_id'] in [x['query_id'] for x in choice] else 'not selected in original three-case allocation; no replacement after audit corrections; not manually validated')
        decisions.append(r)
    pd.DataFrame(decisions).to_csv(HERE/'candidate_selection.csv',index=False)
    products,queries,j=load();pids=[str(p['product_id']) for p in products];pmap=dict(zip(pids,products));pindex={p:i for i,p in enumerate(pids)}
    qindex={int(q):i for i,q in enumerate(queries.query_id)}
    labels={int(q):dict(zip(g.product_id,g.label)) for q,g in j.groupby('query_id')}
    all_ev=lines('candidate_evidence.jsonl');summaries=lines('data/query_comparisons.jsonl')
    chosen_ids={str(p['product_id']) for c in selected for p in c['products']}
    representations={};source_hash={};rep_paths=[]
    for schedule,condition in [('C0','C0_original'),('C1','C1_reverse_attribute_order')]:
        path=src(f'phase2/data/representations/wands/{schedule}.jsonl.gz');rep_paths.append(path)
        digest=hashlib.sha256();ordered=[];representations[schedule]={}
        with gzip.open(path,'rt',encoding='utf8') as f:
            for line in f:
                digest.update(line.encode());r=json.loads(line);pid=str(r['product_id']);ordered.append(pid)
                if pid in chosen_ids:
                    generated=build(pmap[pid],condition,[20260911,20260912,20260913,20260914,20260915])
                    assert r['text']==generated,(schedule,pid,'saved template mismatch')
                    representations[schedule][pid]=r['text']
        assert ordered==pids
        source_hash[schedule]=digest.hexdigest()
    query_hash=hashlib.sha256('\0'.join(queries['query'].fillna('').astype(str)).encode()).hexdigest()
    runs={(r['model'],r['schedule']):r for r in read('query_constraint_cases_20261007/data/runs.json')}
    profiles={(r['model'],r['schedule']):r for r in protocol['execution_profiles']}
    runtime=dict(platform=platform.platform(),device='CPU',dtype='float32',operation='one-query by full-catalog matrix product',
        query_batch=1,catalog_batch=len(products),tie_break='numpy.argsort(-scores,kind=stable); original catalog index',
        numpy=np.__version__,python=sys.version,model_forward_pass=False,
        scope='Matched cached-vector scoring; not a fresh matched encoder run. Historical per-batch padding/OOM partitions and GPU model are unknown.')
    try:
        from threadpoolctl import threadpool_info
        runtime['blas']=threadpool_info()
    except ImportError:runtime['blas']='unknown; optional threadpoolctl unavailable'
    file_manifest=[];seen=set();case_index=[]
    def track(path,role):
        path=Path(path)
        if path not in seen:
            seen.add(path);file_manifest.append(dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),role=role,bytes=path.stat().st_size,sha256=sha(path)))
    for path in rep_paths:track(path,'saved full-catalog serialized representation')
    for model in sorted({c['summary']['model'] for c in selected}):
        prof=profiles[model,'C0'];spec=prof['product_metadata']['model'];limit=spec['max_tokens']
        tokenizer=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True)
        snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+spec['name'].replace('/','--'))/'snapshots'/spec['revision']
        assert snapshot.exists()
        for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','config.json']:
            if (snapshot/name).exists():track(snapshot/name,'pinned local tokenizer/config')
        qpath=src(runs[model,'C0']['query_embedding']);track(qpath,'query embedding payload')
        qemb=np.load(qpath,mmap_mode='r');assert qemb.dtype==np.float32 and qemb.shape==tuple(prof['query_metadata']['shape'])
        assert query_hash==prof['query_metadata']['source_sha256']
        model_cases=[c for c in selected if c['summary']['model']==model]
        for c in model_cases:
            qid=c['summary']['query_id'];case_name=f'q{qid}_{model}_C0_C1';folder='top_cases/'+case_name
            c['case_name']=case_name;c['source_products']=[pmap[p['product_id']] for p in c['products']]
            c['all_clauses']=[r for r in summaries if r['query_id']==qid and r['comparison']==c['summary']['comparison'] and r['scanned']]
            c['all_clause_evidence']=[r for r in all_ev if r['query_id']==qid and r['product_id'] in set(c['top_before'])|set(c['top_after'])]
            tokens=[];texts=[];qtext=c['summary']['query'];qtoken=tokenizer(qtext,add_special_tokens=True,truncation=False)['input_ids']
            for p in c['products']:
                pid=p['product_id'];record=pmap[pid]
                assert collections.Counter(record['attributes'])==collections.Counter(reversed(record['attributes']))
                assert collections.Counter(representations['C0'][pid])==collections.Counter(representations['C1'][pid])
                for schedule in ['C0','C1']:
                    text=representations[schedule][pid]
                    ids=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
                    actual=tokenizer(text,add_special_tokens=True,truncation=True,max_length=limit)['input_ids']
                    tokens.append(dict(query_id=qid,product_id=pid,membership=p['membership'],schedule=schedule,
                        tokens_including_special=len(ids),tokens_without_special=len(tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']),
                        encoder_input_tokens=len(actual),max_length=limit,fully_fitting=len(ids)<=limit,
                        special_tokens=tokenizer.num_special_tokens_to_add(pair=False),text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        raw_attribute_multiplicity_preserved=True,title_class_category_description_unchanged=True))
                    texts.append(dict(product_id=pid,schedule=schedule,text=text,actual_input_ids=actual))
            dump(folder+'/historical.json',c)
            pd.DataFrame(tokens).to_csv(HERE/folder/'token_audit.csv',index=False)
            save(folder+'/input_sequences.jsonl',texts)
            changed=[t for t in tokens if t['membership']!='retained']
            c['token_summary']=dict(query_tokens_with_special=len(qtoken),query_fully_fitting=len(qtoken)<=limit,
                union_products=len(c['products']),union_schedule_sequences=len(tokens),union_fitting_sequences=sum(t['fully_fitting'] for t in tokens),
                changed_products=len(changed)//2,changed_schedule_sequences=len(changed),changed_fitting_sequences=sum(t['fully_fitting'] for t in changed),
                changed_min_tokens=min(t['tokens_including_special'] for t in changed),changed_max_tokens=max(t['tokens_including_special'] for t in changed),
                union_min_tokens=min(t['tokens_including_special'] for t in tokens),union_max_tokens=max(t['tokens_including_special'] for t in tokens),
                tokenizer_class=tokenizer.__class__.__name__,revision=spec['revision'],max_length=limit,padding_side=tokenizer.padding_side,truncation_side=tokenizer.truncation_side)
            c['matched_scoring']={}
        for schedule in ['C0','C1']:
            ep=profiles[model,schedule];pm=ep['product_metadata'];qm=ep['query_metadata']
            assert pm['source_sha256']==source_hash[schedule] and qm['source_sha256']==query_hash
            for field in ['model','profile','implementation','precision','device']:
                assert pm[field]==prof['product_metadata'][field],(model,schedule,field)
                assert qm[field]==prof['query_metadata'][field]
            assert runs[model,schedule]['query_embedding']==runs[model,'C0']['query_embedding']
            path=src(runs[model,schedule]['product_embedding']);track(path,'product embedding payload')
            emb=np.load(path,mmap_mode='r');assert emb.dtype==np.float32 and emb.shape==tuple(pm['shape']) and len(emb)==len(products)
            assert np.isfinite(emb).all() and np.isfinite(qemb).all()
            for c in model_cases:
                qid=c['summary']['query_id'];query_vec=np.asarray(qemb[qindex[qid]:qindex[qid]+1],dtype=np.float32)
                scores=(query_vec@np.asarray(emb,dtype=np.float32).T)[0]
                order=np.argsort(-scores,kind='stable');ranks=np.empty(len(order),dtype=np.int32);ranks[order]=np.arange(1,len(order)+1)
                top=[pids[i] for i in order[:20]];h={p for p,l in labels[qid].items() if l=='Exact'}
                historical=c['top_before' if schedule=='C0' else 'top_after']
                s20,s21=float(scores[order[19]]),float(scores[order[20]])
                details=[]
                for p in c['products']:
                    pid=p['product_id'];idx=pindex[pid];old=p['before_rank' if schedule=='C0' else 'after_rank']
                    details.append(dict(product_id=pid,historical_rank=old,rescored_rank=int(ranks[idx]),score=float(scores[idx]),
                        score_minus_rank20=float(scores[idx])-s20,score_minus_rank21=float(scores[idx])-s21,
                        historical_rank_agrees=None if old=='>20' else int(old)==int(ranks[idx])))
                c['matched_scoring'][schedule]=dict(top20=top,metrics=metric(top,labels[qid],h),top20_order_matches_history=top==historical,
                    catalog_size=len(products),score20=s20,score21=s21,cutoff_margin=s20-s21,products=details,
                    exact_saved_union_ranks_all_agree=all(d['historical_rank_agrees'] is not False for d in details),
                    norm_query=float(np.linalg.norm(query_vec)),norm_product_min=float(np.linalg.norm(emb,axis=1).min()),norm_product_max=float(np.linalg.norm(emb,axis=1).max()))
        for c in model_cases:
            folder='top_cases/'+c['case_name'];s=c['summary']
            c['reproduced']=all(x['top20_order_matches_history'] for x in c['matched_scoring'].values())
            c['final_grade']=s['grade'] if c['reproduced'] else 'C'
            c['grade_scope']='Recorded attribute clause; cache-comparable scoring reproduced, not a claim of complete-query satisfaction or fresh encoder equivalence.'
            dump(folder+'/audit.json',c)
            dump(folder+'/matched_scoring.json',dict(runtime=runtime,results=c['matched_scoring']))
            product_rows=[]
            for p in c['products']:
                product_rows.append(dict(product_id=p['product_id'],title=p['title'],qrel=p['qrel'],membership=p['membership'],historical_C0=p['before_rank'],historical_C1=p['after_rank'],
                    condition_status=p['constraint_assessment']['status'],raw_mapped_fields=json.dumps(p['constraint_assessment']['raw_fields'],ensure_ascii=False),
                    type_status=p['type_assessment']['status'],use_status=json.dumps({u:x['status'] for u,x in p['use_assessments'].items()}),joint_status=p['joint_status']))
            pd.DataFrame(product_rows).to_csv(HERE/folder/'all_products.csv',index=False)
            pd.DataFrame([r for r in product_rows if r['membership']!='retained']).to_csv(HERE/folder/'all_changes.csv',index=False)
            pd.DataFrame([r for r in product_rows if r['membership']!='retained' and r['qrel']=='Exact'],columns=list(product_rows[0])).to_csv(HERE/folder/'relevant_changes.csv',index=False)
            for side in ['before','after']:
                lookup={r['product_id']:r for r in product_rows}
                pd.DataFrame([dict(rank=i+1,**lookup[p]) for i,p in enumerate(c['top_'+side])]).to_csv(HERE/folder/f'top20_{side}.csv',index=False)
            case_index.append(dict(case_name=c['case_name'],query_id=s['query_id'],comparison=s['comparison'],constraint_id=s['constraint_id'],
                final_grade=c['final_grade'],reproduced=c['reproduced'],token_summary=c['token_summary']))
            print(json.dumps(case_index[-1],ensure_ascii=False))
    dump('case_index.json',case_index);dump('qa/reproduction_runtime.json',runtime)
    dump('qa/cache_tokenizer_manifest.json',file_manifest)
    for r in decisions:
        audited=next((c for c in case_index if c['comparison']==r['comparison'] and c['constraint_id']==r['constraint_id']),None)
        r['cache_reproduction']='full-catalog Top20 order reproduced' if audited and audited['reproduced'] else 'not performed; outside three selected cases' if not audited else 'failed; downgraded'
        r['fully_audited_grade']=audited['final_grade'] if audited else 'not manually validated'
    pd.DataFrame(decisions).to_csv(HERE/'candidate_selection.csv',index=False)

if __name__=='__main__':main()
