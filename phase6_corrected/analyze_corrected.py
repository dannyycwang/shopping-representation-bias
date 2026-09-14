"""Report all frozen configurations, seeds, supports and paired uncertainty without reselection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_corrected import HERE,ROOT,CONFIG,METHODS,STEMS,measures,exact_queries,graded_queries,prior,sha,dump,now,validate
from baselines import BASELINES,historical_exact

METRICS=['Recall@20','Recall@100','cNDCG@10','VI@20','VI@100','Robust@100','Never@100','worst_schedule@100']


def main():
    cfg=validate();lock=json.loads((HERE/'selection.json').read_text())
    assert json.loads((HERE/'evaluation_complete.json').read_text())['selection_sha256']==sha(HERE/'selection.json')
    features=pd.read_csv(ROOT/'phase4/results/wands_product_features.csv',dtype={'product_id':str}).set_index('product_id')
    fitids=set(features.index[features.fully_fits])
    longids=set(features.index[features.attribute_count>=cfg['long_attribute_threshold']])
    query_meta=pd.read_csv(ROOT/'phase2/data/processed/wands_queries.csv').set_index('query_id')
    seed_rows,rows,frames,seed_frames,raw_frames=[],[],{},{},{}
    for split in ['dev','test']:
        config_split='validation' if split=='dev' else 'test'
        modes=['catalog'] if split=='dev' else ['catalog','catalog_fully_fitting','regenerated_catalog','regenerated_catalog_fully_fitting','target','target_fully_fitting','regenerated_target','regenerated_target_fully_fitting']
        for mode in modes:
            target='target' in mode
            for method in BASELINES+list(METHODS):
                if target and method not in ['Original','Canonical','Set-Attention',lock['best_method']]:continue
                is_learned=method in METHODS
                members=cfg['seeds'] if is_learned else [42 if method=='Set-Attention' else None]
                member_rows,member_frames=[],[]
                for seed in members:
                    if is_learned:
                        selected=lock['selected'][method]
                        if split=='dev':
                            folder=(HERE/selected['checkpoints'][str(seed)]['path']).parent
                            f=pd.read_parquet(folder/'dev_pairs.parquet')
                        else:
                            folder=HERE/'results'/f'{method}_s{seed}'
                            filename=('regenerated_' if mode.startswith('regenerated') else '')+('target_pairs.parquet' if target else ('graded_pairs.parquet' if mode.startswith('regenerated') else 'catalog_graded_pairs.parquet'))
                            f=pd.read_parquet(folder/filename)
                        ef=f[f.label.eq('Exact')].copy()
                        epoch,lr,params=selected['epoch'],selected['lr'],selected['parameters']
                    else:
                        f=pd.read_parquet(HERE/'baselines'/method/f'{split}_graded_pairs.parquet') if not target else None
                        ef=historical_exact(method,config_split,cfg,target=target)
                        epoch,lr,params=(1,None,98561) if method=='Set-Attention' else (None,None,0)
                    if 'fully_fitting' in mode:
                        ef=ef[ef.product_id.astype(str).isin(fitids)]
                        if f is not None:f=f[f.product_id.astype(str).isin(fitids)]
                    eq=exact_queries(ef)
                    gq=graded_queries(f) if not target else None
                    point={m:float(eq[m].mean()) for m in METRICS if m!='cNDCG@10'}
                    point['cNDCG@10']=float(gq['cNDCG@10'].mean()) if gq is not None else np.nan
                    if not target:
                        point['minimum_macro_schedule_R100']=float(eq[[s+'_Recall@100' for s in STEMS]].mean().min())
                    pq=eq.copy()
                    if gq is not None:pq=pq.join(gq[['cNDCG@10']],how='outer')
                    if target:
                        point={m.replace('Recall','Inclusion'):v for m,v in point.items()}
                        pq=pq.rename(columns={m:m.replace('Recall','Inclusion') for m in pq if 'Recall' in m})
                    row=dict(Method=method,Label=METHODS.get(method,method),Split=split,Mode=mode,Seed=seed,Epoch=epoch,LR=lr,Params=params,
                             Exact_pairs=len(ef),Exact_queries=len(eq),cNDCG_queries=int(gq.positive_idcg.sum()) if gq is not None else 0,
                             **{m:100*v for m,v in point.items()})
                    member_rows.append(row);member_frames.append(pq)
                    seed_rows.append(row);seed_frames[split,mode,method,seed]=pq
                    if mode=='catalog':raw_frames[split,method,seed]=(ef,f)
                mean_frame=sum(member_frames)/len(member_frames)
                frames[split,mode,method]=mean_frame
                mf=pd.DataFrame(member_rows)
                summary={k:member_rows[0][k] for k in ['Method','Label','Split','Mode','Epoch','LR','Params','Exact_pairs','Exact_queries','cNDCG_queries']}
                summary['Seeds']=len(members) if is_learned else 1
                for metric in point:
                    summary[metric]=float(mf[metric].mean())
                    summary[metric+'_seed_sd']=float(mf[metric].std(ddof=1)) if is_learned else np.nan
                summary['units']='percent; seed SD and deltas in percentage points'
                rows.append(summary)
    results=pd.DataFrame(rows)
    for mode in results.Mode.unique():
        for split in results.Split.unique():
            mask=(results.Mode==mode)&(results.Split==split)
            section=results[mask].set_index('Method')
            if section.empty or 'target' in mode:continue
            for reference in ['Set-Attention','Canonical']:
                results.loc[mask,'Delta_R100_vs_'+reference]=results.loc[mask,'Recall@100']-section.loc[reference,'Recall@100']
    results.to_csv(HERE/'PHASE6_RESULTS.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(HERE/'PHASE6_SEEDS.csv',index=False)
    comparisons=[];best=lock['best_method']
    contrasts=list(dict.fromkeys([(best,b) for b in ['Set-Attention','Canonical','Original','Set-Mean']]+[('C1','Set-Attention'),('C2','Set-Attention'),('C3','C2'),('C4','C3')]))
    for split,mode,_ in sorted(frames):
        if _!=BASELINES[0]:continue
        targets=['Inclusion@20','Inclusion@100','VI@20','VI@100','Robust@100','Never@100'] if 'target' in mode else METRICS
        for method,reference in contrasts:
            if (split,mode,method) not in frames or (split,mode,reference) not in frames:continue
            for metric in targets:
                append_ci(comparisons,frames[split,mode,method],frames[split,mode,reference],method,reference,split,mode,metric,'seed_mean')
        # Individual best-model seeds show that query uncertainty and seed variation differ.
        for seed in cfg['seeds']:
            if (split,mode,best,seed) not in seed_frames:continue
            for reference in ['Set-Attention','Canonical','Original']:
                for metric in targets:
                    append_ci(comparisons,seed_frames[split,mode,best,seed],frames[split,mode,reference],best,reference,split,mode,metric,seed)
    pd.DataFrame(comparisons).to_csv(HERE/'PHASE6_BOOTSTRAP.csv',index=False)
    curves=[]
    for group in lock['groups']:
        c=pd.read_csv(HERE/'checkpoints'/group['name']/'epochs.csv')
        for metric in METRICS:c[metric]*=100
        c['selected_configuration']=group['name']==lock['selected'][group['method']]['name']
        c['selected_epoch']=c.epoch.eq(group['epoch'])
        curves.append(c)
    curves=pd.concat(curves,ignore_index=True)
    curves.to_csv(HERE/'PHASE6_TRAINING_CURVES.csv',index=False)
    plot_curves(curves,lock,results)
    field_rows,key_rows=[],[]
    for method in METHODS:
        for seed in cfg['seeds']:
            folder=HERE/'results'/f'{method}_s{seed}'
            summary=json.loads((folder/'field_weight_summary.json').read_text())
            for metric,values in summary.items():
                if metric=='concentration':continue
                field_rows.append(dict(Method=method,Seed=seed,Metric=metric,Products=42994,**values,
                                       Fraction_max_weight_gt_0_5=summary['concentration']['0.5']['all_products'],
                                       Fraction_max_weight_gt_0_9=summary['concentration']['0.9']['all_products'],
                                       Multiple_field_fraction_max_gt_0_9=summary['concentration']['0.9']['multiple_fields']))
            if method in ['C3','C4']:
                k=pd.read_csv(folder/'field_keys.csv');k.insert(0,'Seed',seed);k.insert(0,'Method',method);key_rows.append(k)
    pd.DataFrame(field_rows).to_csv(HERE/'PHASE6_FIELD_WEIGHTS.csv',index=False)
    keys=pd.concat(key_rows,ignore_index=True)
    keys.to_csv(HERE/'all_field_key_weights.csv',index=False)
    frequent=keys[keys.occurrences>=100].groupby(['Method','field_key']).agg(Occurrences=('occurrences','first'),Mean_weight=('mean_weight','mean'),Seed_SD=('mean_weight','std')).reset_index()
    frequent.to_csv(HERE/'frequent_field_weights.csv',index=False)
    category_rows,query_rows,length_rows=[],[],[]
    for split in ['dev','test']:
        for method in METHODS:
            a=frames[split,'catalog',method]['Recall@100'].dropna()
            b=frames[split,'catalog','Set-Attention']['Recall@100'].dropna()
            z=pd.DataFrame({'Delta_R100_pp':100*(a-b),'Recall@100':100*a}).join(query_meta[['query','query_type']])
            z['Method']=method;z['Split']=split;query_rows.append(z.reset_index())
            for category,g in z.groupby('query_type'):
                category_rows.append(dict(Method=method,Split=split,Query_type=category,Queries=len(g),Delta_R100_pp=g.Delta_R100_pp.mean(),Recall100=g['Recall@100'].mean()))
            for subset,ids in [('fully_fitting',fitids),('not_fully_fitting',set(features.index)-fitids),('long_attributes',longids),('shorter_attributes',set(features.index)-longids)]:
                baseline_ef,_=raw_frames[split,'Set-Attention',42]
                be=baseline_ef[baseline_ef.product_id.astype(str).isin(ids)]
                bq=exact_queries(be)
                for seed in cfg['seeds']:
                    ef,gf=raw_frames[split,method,seed]
                    ef=ef[ef.product_id.astype(str).isin(ids)]
                    eq=exact_queries(ef)
                    length_rows.append(dict(Method=method,Seed=seed,Split=split,Subset=subset,Pairs=len(ef),Queries=len(eq),
                                           Recall100=100*eq['Recall@100'].mean(),Delta_R100_vs_SetAttention=100*(eq['Recall@100']-bq['Recall@100']).mean(),
                                           Robust100=100*eq['Robust@100'].mean(),Never100=100*eq['Never@100'].mean()))
    pd.DataFrame(category_rows).to_csv(HERE/'category_analysis.csv',index=False)
    pd.concat(query_rows,ignore_index=True).to_csv(HERE/'per_query_gains_losses.csv',index=False)
    pd.DataFrame(length_rows).to_csv(HERE/'length_analysis.csv',index=False)
    cost_frame=pd.DataFrame([json.loads(x) for x in (HERE/'cost.jsonl').read_text().splitlines()])
    cost_frame.groupby(['name','stage']).seconds.sum().reset_index().to_csv(HERE/'cost_summary.csv',index=False)
    exposures=[]
    for group in lock['groups']:
        x=pd.read_csv(HERE/'checkpoints'/group['name']/'query_presentations.csv');x.insert(0,'Group',group['name']);exposures.append(x)
    pd.concat(exposures,ignore_index=True).to_csv(HERE/'actual_query_presentations.csv',index=False)
    boundary_rows=[]
    for method in METHODS:
        for seed in cfg['seeds']:
            f=pd.read_parquet(HERE/'results'/f'{method}_s{seed}'/'regenerated_graded_pairs.parquet')
            f=f[f.label.eq('Exact')]
            for schedule in STEMS[1:]:
                for cutoff in [20,100]:
                    changed=f[(f[schedule]<=cutoff)!=(f.C0<=cutoff)]
                    for row in changed.itertuples(index=False):
                        boundary_rows.append(dict(Method=method,Seed=seed,Schedule=schedule,Cutoff=cutoff,
                                                  Query_id=row.query_id,Product_id=row.product_id,C0_rank=row.C0,
                                                  Regenerated_rank=getattr(row,schedule)))
    pd.DataFrame(boundary_rows,columns=['Method','Seed','Schedule','Cutoff','Query_id','Product_id','C0_rank','Regenerated_rank']).to_csv(HERE/'numerical_boundary_changes.csv',index=False)
    seeds=pd.DataFrame(seed_rows)
    dv=seeds[(seeds.Method==best)&(seeds.Split=='dev')&(seeds.Mode=='catalog')]['Recall@100']
    ts=seeds[(seeds.Method==best)&(seeds.Split=='test')&(seeds.Mode=='catalog')]['Recall@100']
    baseline_dev=results[(results.Method=='Set-Attention')&(results.Split=='dev')&(results.Mode=='catalog')]['Recall@100'].iloc[0]
    refs=results[(results.Split=='test')&(results.Mode=='catalog')].set_index('Method')['Recall@100']
    ci=pd.DataFrame(comparisons)
    test_ci=ci[(ci.Method==best)&(ci.Reference=='Set-Attention')&(ci.Split=='test')&(ci.Mode=='catalog')&(ci.Metric=='Recall@100')&ci.Seed.eq('seed_mean')].iloc[0]
    benefit=bool((dv>baseline_dev+1e-10).all() and test_ci.CI_low_pp>0)
    code='D' if benefit and (ts>=refs['Original']-1e-10).all() else 'C' if benefit and (ts>=refs['Canonical']-1e-10).all() else 'B' if benefit else 'A'
    titles={'A':'KEEP DIAGNOSIS PAPER','B':'ADD LEARNED INVARIANT BASELINE','C':'COMPETITIVE STRUCTURAL SOLUTION','D':'STRONG SOLUTION — CONFIRM EXTERNALLY'}
    dump(HERE/'decision.json',dict(code=code,title=titles[code],best_method=best,all_seeds_improve_dev_vs_SetAttention=bool((dv>baseline_dev+1e-10).all()),
                                 heldout_R100_CI_lower_vs_SetAttention=float(test_ci.CI_low_pp),qualified_learned_benefit=benefit,
                                 all_test_seeds_at_least_Canonical=bool((ts>=refs['Canonical']-1e-10).all()),all_test_seeds_at_least_Original=bool((ts>=refs['Original']-1e-10).all()),
                                 selection_sha256=sha(HERE/'selection.json'),generated_utc=now()))
    print(code,titles[code],best,flush=True)
    print(results[(results.Split=='test')&(results.Mode=='catalog')][['Method','Epoch','Recall@100','cNDCG@10','Recall@100_seed_sd']].to_string(index=False))


def append_ci(rows,a,b,method,reference,split,mode,metric,seed):
    x=a[metric].dropna();y=b[metric].dropna()
    assert set(x.index)==set(y.index),(split,mode,metric)
    mean,lo,hi=prior.bootstrap(x-y.loc[x.index],seed=20260963,samples=10000)
    rows.append(dict(Method=method,Reference=reference,Seed=seed,Split=split,Mode=mode,Metric=metric,Queries=len(x),
                     Delta_pp=100*mean,CI_low_pp=100*lo,CI_high_pp=100*hi,Draws=10000,Bootstrap_seed=20260963,Multiplicity_correction='none'))


def plot_curves(curves,lock,results):
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    baseline=results[(results.Split=='dev')&(results.Mode=='catalog')].set_index('Method')
    for ax,method,color in zip(axes.flat,METHODS,['#1266a4','#c05a36','#41846a','#8d619e']):
        g=curves[(curves.method==method)&curves.selected_configuration].groupby('epoch')['Recall@100'].agg(['mean','std','count'])
        assert g['count'].eq(3).all()
        ax.plot(g.index,g['mean'],color=color,marker='o',ms=3,label='Mean across three seeds')
        ax.fill_between(g.index,(g['mean']-g['std']).to_numpy(),(g['mean']+g['std']).to_numpy(),color=color,alpha=.16,label='±1 seed SD')
        for ref,ls,c in [('Set-Mean','--','#555555'),('Set-Attention','-.','#358375'),('Canonical',':','#8a7350')]:
            ax.axhline(baseline.loc[ref,'Recall@100'],ls=ls,color=c,lw=1.1,label=ref)
        ax.axvline(lock['selected'][method]['epoch'],ls=':',color=color,alpha=.7)
        ax.set_title(METHODS[method]+'\n'+f"Shared LR {lock['selected'][method]['lr']:g}; selected epoch {lock['selected'][method]['epoch']}",fontsize=10)
        ax.set_xlabel('Epoch (0 = neutral check only)');ax.set_ylabel('Development Recall@100 (%)');ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.suptitle('Corrected Phase VI: learned weights over frozen BGE values',fontsize=14)
    fig.tight_layout(rect=[0,.055,1,.95])
    fig.savefig(HERE/'PHASE6_TRAINING.png',dpi=180);fig.savefig(HERE/'PHASE6_TRAINING.pdf');plt.close(fig)


if __name__=='__main__':main()
