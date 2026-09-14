"""Derive secondary tables from frozen rankings, never select a strategy here."""
from common import *
from threadpoolctl import threadpool_limits
threadpool_limits(4)

def diagnostics():
    rows=[]
    for ds in ['wands','esci']:
        f=pd.read_parquet(OUT/f'{ds}_diagnostic_pairs.parquet')
        # These display bins are descriptive, not model-selection parameters.
        f['margin_band']=pd.cut(f.C0_margin20,[-np.inf,-.1,-.05,-.01,0,.01,.05,.1,np.inf]).astype(str)
        f['query_length_band']=pd.cut(f.query_length,[0,1,3,np.inf],labels=['1','2-3','4+']).astype(str)
        length_col='max_tokens_seven'
        f['length_band']=pd.cut(f[length_col],[-1,128,256,512,1024,np.inf]).astype(str)
        dup_col='duplicate_atoms'
        f['has_duplicates']=f[dup_col]>0
        for factor in ['margin_band','query_length_band','length_band','has_duplicates']:
            for (level,label),g in f.groupby([factor,'label'],observed=True):
                mu,lo,hi=boot(g.groupby('query_id')['VI@20'].mean())
                rows.append(dict(dataset=ds,factor=factor,level=str(level),label=label,pairs=len(g),queries=g.query_id.nunique(),macro_vi20=mu,ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(OUT/'additional_diagnostic_strata.csv',index=False)

def completed():
    assert (OUT/'queue_completed.json').exists()
    selected=json.loads((P4/'config/selection.json').read_text())['method']
    saturation=[];equivalent=[];contrasts=[];unique=[];counts=[];budget_contrasts=[]
    for ds in ['wands','esci']:
        p,q,j=load(ds);ids=q.loc[~dev_mask(q),'query_id'] if ds=='wands' else q.query_id
        source=OUT/ds
        original=pd.read_csv(source/'bge_base_Original_per_query.csv').set_index('query_id').loc[ids]
        for method in ['centroid_2','max_2','centroid_4','max_4','centroid_7','max_7']:
            path=source/f'bge_base_{method}_per_query.csv'
            # m=2/4 vectors are already cached; derive an omitted fixed saturation cell without encoding.
            if not path.exists():
                m=int(method.split('_')[-1]);views=[]
                for i in range(1,m+1):
                    files=list((OUT/'embeddings').glob(f'{ds}_bge_base_view{i}_*.npy'));assert len(files)==1;views.append(np.load(files[0]))
                qv,_=original_cache(ds,'bge_base');s=qv@norm(np.mean(views,axis=0)).T if method.startswith('centroid') else np.maximum.reduce([qv@v.T for v in views]);evaluate(s,p,q,j,ds,f'bge_base_{method}')
            pq=pd.read_csv(path).set_index('query_id').loc[ids]
            for metric in ['Recall@100','cNDCG@10']:
                mu,lo,hi=boot(pq[metric]-original[metric]);saturation.append(dict(dataset=ds,method=method,metric=metric,mean=pq[metric].mean(),delta_vs_Original=mu,ci_low=lo,ci_high=hi,selection='fixed saturation; cannot replace selected strategy'))
        summary=pd.read_csv(OUT/f'{ds}_bge_base_retrieval_summary.csv')
        for method in ['Original','hybrid',selected]:
            for k in KS:
                r=float(summary[(summary.method==method)&(summary.metric==f'Recall@{k}')]['mean'].iloc[0])
                candidates=[(kk,float(summary[(summary.method=='Original')&(summary.metric==f'Recall@{kk}')]['mean'].iloc[0])) for kk in KS]
                matched=next(((kk,v) for kk,v in candidates if v>=r),None)
                equivalent.append(dict(dataset=ds,method=method,K=k,recall=r,Original_min_grid_K=matched[0] if matched else np.nan,Original_grid_recall=matched[1] if matched else np.nan,scope='discrete recall match; not exact cost match'))
        pq=pd.read_csv(OUT/f'{ds}_canonical_reranking_per_query.csv');pq=pq[pq.query_id.isin(ids)]
        for k in sorted(pq.candidate_K.unique()):
            for a,b in [(selected,'Original'),(selected,'hybrid'),('hybrid','Original')]:
                for metric in ['Recall@20','returned_condensed_NDCG@10']:
                    av=pq[(pq.method==a)&(pq.candidate_K==k)].set_index('query_id')[metric];bv=pq[(pq.method==b)&(pq.candidate_K==k)].set_index('query_id')[metric];mu,lo,hi=boot(av-bv)
                    contrasts.append(dict(dataset=ds,K=int(k),method_a=a,method_b=b,metric=metric,delta=mu,ci_low=lo,ci_high=hi))
        for method in ['Original','hybrid',selected]:
            for k in [200,500]:
                if k not in pq.candidate_K.unique():continue
                for metric in ['Recall@20','returned_condensed_NDCG@10']:
                    av=pq[(pq.method==method)&(pq.candidate_K==k)].set_index('query_id')[metric];bv=pq[(pq.method==method)&(pq.candidate_K==100)].set_index('query_id')[metric];mu,lo,hi=boot(av-bv)
                    budget_contrasts.append(dict(dataset=ds,method=method,K_a=k,K_b=100,metric=metric,delta=mu,ci_low=lo,ci_high=hi,scope='descriptive paired budget contrast; not strategy selection'))
        for m in [2,4,7]:
            u=pd.read_parquet(OUT/f'{ds}_unique_views_m{m}.parquet')
            unique.append(dict(dataset=ds,m=m,products=len(u),mean_unique_text=u.unique_views.mean(),mean_unique_vectors=u.unique_encoded_vectors.mean(),products_one_vector=int(u.unique_encoded_vectors.eq(1).sum())))
        counts.append(dict(dataset=ds,queries=len(ids),recall_eligible=int(original['Recall@100'].notna().sum()),cndcg_eligible=int(original['cNDCG@10'].notna().sum()),products=len(p),judgments=int(j.query_id.isin(ids).sum())))
    pd.DataFrame(saturation).to_csv(OUT/'saturation_summary.csv',index=False)
    pd.DataFrame(equivalent).to_csv(OUT/'recall_equivalent_candidate_budget.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'canonical_reranking_paired_contrasts.csv',index=False)
    pd.DataFrame(budget_contrasts).to_csv(OUT/'reranking_budget_contrasts.csv',index=False)
    pd.DataFrame(unique).to_csv(OUT/'unique_view_summary.csv',index=False)
    encoding=[]
    for path in sorted((OUT/'embeddings').glob('*.json')):
        meta=json.loads(path.read_text())
        if 'seconds' in meta:encoding.append(dict(cache=path.name,seconds=meta['seconds'],items=meta['shape'][0],dimension=meta['shape'][1],source_sha256=meta['source_sha256'],scope='measured cache construction, not matched cold-build baseline'))
    pd.DataFrame(encoding).to_csv(OUT/'encoding_costs.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'evaluation_denominators.csv',index=False)
    # Raw rescued/missed counts and query-macro net changes are kept side by side.
    transitions=[];descriptive=[];rule_coverage=[]
    for ds,model in [('wands','bge_base'),('esci','bge_base'),('wands','minilm'),('esci','minilm'),('esci_new','bge_base')]:
        _,q,_=load(ds);ids=q.loc[~dev_mask(q),'query_id'] if ds=='wands' else q.query_id
        allmetrics={m:pd.read_csv(OUT/ds/f'{model}_{m}_per_query.csv').set_index('query_id').loc[ids] for m in ['Original','hybrid','canonical_raw','canonical','set_mean',selected]}
        for a,b in [('hybrid','Original'),('canonical_raw','Original'),('canonical','Original'),('set_mean','Original'),(selected,'canonical_raw')]:
            for metric in ['Recall@100','cNDCG@10']:
                mu,lo,hi=boot(allmetrics[a][metric]-allmetrics[b][metric]);descriptive.append(dict(dataset=ds,encoder=model,method_a=a,method_b=b,metric=metric,delta=mu,ci_low=lo,ci_high=hi,scope='descriptive additional contrasts; not used for selection'))
        f=pd.read_csv(OUT/f'{ds}_{model}_transitions.csv')
        for (method,k),g in f.groupby(['method','K']):
            mu,lo,hi=boot(g.net_fraction)
            transitions.append(dict(dataset=ds,encoder=model,method=method,K=k,highest_pairs=g.n_highest.sum(),rescued_pairs=g.rescued.sum(),newly_missed_pairs=g.newly_missed.sum(),macro_net=mu,ci_low=lo,ci_high=hi))
    pd.DataFrame(transitions).to_csv(OUT/'transition_summary.csv',index=False)
    pd.DataFrame(descriptive).to_csv(OUT/'descriptive_baseline_contrasts.csv',index=False)
    for ds in ['wands','esci','esci_new']:
        p,_,_=load(ds);raw=texts(p,'canonical_raw');ordered=texts(p,selected)
        from common import _plain
        matched=0
        for x in p:
            fields=[a.split(':',1)[0].lower() if ':' in a else a[:40].lower() for a in x['attributes']]
            matched+=any(any(word in field for word in RULES[selected]) for field in fields)
        rule_coverage.append(dict(dataset=ds,products=len(p),products_with_priority_match=matched,products_different_from_raw_sort=sum(a!=b for a,b in zip(raw,ordered)),products_different_from_Original=sum(a!=_plain(x) for a,x in zip(ordered,p)),scope='literal field/prefix matching; no semantic field inference on unkeyed atoms'))
    pd.DataFrame(rule_coverage).to_csv(OUT/'selected_rule_coverage.csv',index=False)
    oldp,oldq,_=load('esci');_,newq,newj=load('esci_new');oldids={str(x['product_id']) for x in oldp};judged=set(newj.product_id.astype(str));highest=set(newj.loc[newj.label.eq('E'),'product_id'].astype(str))
    _,wq,wj=load('wands');dev_ids=set(wq.loc[dev_mask(wq),'query_id']);devp=set(wj.loc[wj.query_id.isin(dev_ids),'product_id']);testp=set(wj.loc[~wj.query_id.isin(dev_ids),'product_id'])
    scope={'new_esci_query_overlap_with_historical':len(set(newq.query_id)&set(oldq.query_id)),'new_esci_unique_judged_products':len(judged),'new_esci_judged_products_in_historical_catalog':len(judged&oldids),'new_esci_highest_products_in_historical_catalog':len(highest&oldids),'new_esci_unique_highest_products':len(highest),'wands_dev_heldout_judged_product_overlap':len(devp&testp),'design':'query holdout and unchanged strategy transfer; not a designed product-disjoint evaluation'}
    (OUT/'generalization_scope.json').write_text(json.dumps(scope,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics-only',action='store_true');args=parser.parse_args()
    diagnostics()
    if not args.diagnostics_only:completed()
