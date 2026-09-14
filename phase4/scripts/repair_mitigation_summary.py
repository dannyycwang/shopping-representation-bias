from common import *
source=pd.read_parquet(OUT/'esci_gain_repair_per_query.parquet');source['source']=source.source.str.replace('\\','/',regex=False)
rows=[]
for ds in ['wands','esci']:
    p,q,j=load(ds);evalq=q.loc[~dev_mask(q),'query_id'] if ds=='wands' else q.query_id
    methods={}
    paths={'Original':P2/f'results/phase2_pair_ranks/{ds}_bge_base_native_C0.parquet','canonical':P2/f'results/phase2_pair_ranks/{ds}_bge_base_native_M2C0.parquet','factual':P2/f'results/phase2_pair_ranks/{ds}_bge_base_native_M1C0.parquet'}
    paths.update({m:ROOT/f'phase3/results/phase3_invariant_method/{ds}_{m}_pairs.parquet' for m in ['set_mean','late_mean','late_max','late_top3']})
    for method,path in paths.items():
        f=pd.read_parquet(path);qq=q[q.query_id.isin(f.query_id)];pq=pair_metrics(f,qq,ds);methods[method]=pq[pq.query_id.isin(evalq)].set_index('query_id')
    for method,pq in methods.items():
        delta=pq['cNDCG@10']-methods['Original']['cNDCG@10'];mu,lo,hi=boot(delta)
        rows.append({'dataset':ds,'method':method,'queries':pq['cNDCG@10'].notna().sum(),'cNDCG@10':pq['cNDCG@10'].mean(),'delta_vs_Original':mu,'ci_low':lo,'ci_high':hi,'gains':'benchmark_standard' if ds=='esci' else 'study_3_1_0'})
    difference=methods['set_mean']['cNDCG@10']-methods['canonical']['cNDCG@10'];mu,lo,hi=boot(difference);rows.append({'dataset':ds,'method':'DIRECT set_mean minus canonical','queries':difference.notna().sum(),'cNDCG@10':np.nan,'delta_vs_Original':mu,'ci_low':lo,'ci_high':hi,'gains':'benchmark_standard' if ds=='esci' else 'study_3_1_0'})
pd.DataFrame(rows).to_csv(OUT/'repaired_legacy_mitigation.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
