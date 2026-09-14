"""Only consumes completed, real per-query results. No synthetic result placeholders."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation import paired_stats, mean_ci

def md(df,precision=4):
    def fmt(x):
        if pd.isna(x): return '—'
        if isinstance(x,(float,np.floating)): return f'{x:.{precision}f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    cfg=json.loads((ROOT/'configs/phase1.json').read_text()); seed=cfg['seed']; n=cfg['bootstrap_samples']
    metrics=['cNDCG@10','cNDCG@20','Recall@10','Recall@20','Recall@50','MRR','ExactHit@20','JudgedCoverage@10','ExplicitIrrelevant@10','Unjudged@10','Top20BoundaryTied']
    data={}; aggregates=[]; cis=[]; comparisons=[]; hidden=[]; changes=[]; sensitivity=[]
    for file in sorted((ROOT/'results/per_query').glob('*.csv')):
        condition,retriever=file.stem.rsplit('_',1)
        if retriever not in ['BM25','Dense','Hybrid']: continue
        frame=pd.read_csv(file).sort_values('query_id'); data[(condition,retriever)]=frame
        aggregates.append({'representation':condition,'retriever':retriever,**{m:frame[m].mean() for m in metrics}})
        for metric in metrics:
            lo,hi=mean_ci(frame[metric],seed,n)
            cis.append({'representation':condition,'retriever':retriever,'metric':metric,'mean':frame[metric].mean(),'ci_low':lo,'ci_high':hi,'n_queries':int(frame[metric].notna().sum())})
    for (condition,retriever),frame in data.items():
        if condition=='R0': continue
        base=data[('R0',retriever)]
        assert np.array_equal(base.query_id,frame.query_id)
        for metric in metrics:
            effect,lo,hi,p=paired_stats(base[metric],frame[metric],seed,n)
            comparisons.append({'representation':condition,'retriever':retriever,'metric':metric,'delta':effect,'ci_low':lo,'ci_high':hi,'permutation_p':p,'role':'primary' if condition=='R8' and metric=='cNDCG@10' else 'exploratory'})
        a=pd.read_csv(ROOT/f'results/raw/R0_{retriever}_exact.csv')
        b=pd.read_csv(ROOT/f'results/raw/{condition}_{retriever}_exact.csv')
        merged=a.merge(b,on=['query_id','product_id'],suffixes=('_original','_alternative'),validate='one_to_one')
        merged['representation']=condition; merged['retriever']=retriever
        merged['rank_gain']=merged.rank_original-merged.rank_alternative
        merged['log2_rank_ratio']=np.log2(merged.rank_original/merged.rank_alternative)
        changes.append(merged)
        for k in [10,20,50]:
            merged['rescued']=(merged.rank_original>k)&(merged.rank_alternative<=k)
            merged['harmed']=(merged.rank_original<=k)&(merged.rank_alternative>k)
            counts=merged.groupby('query_id').agg(rescued=('rescued','sum'),harmed=('harmed','sum'),exact=('product_id','size'))
            macro=(counts.rescued-counts.harmed)/counts.exact
            _,lo,hi,pval=paired_stats(np.zeros(len(macro)),macro,seed,n)
            churn=(counts.rescued+counts.harmed)/counts.exact
            churn_lo,churn_hi=mean_ci(churn,seed,n)
            hidden.append({'representation':condition,'retriever':retriever,'k':k,'exact_pairs':len(merged),
                'rescued':int(merged.rescued.sum()),'harmed':int(merged.harmed.sum()),
                'RelevantHidden_micro':merged.rescued.mean(),'ReverseHidden_micro':merged.harmed.mean(),
                'NetHidden_micro':(merged.rescued.sum()-merged.harmed.sum())/len(merged),
                'net_macro':macro.mean(),'net_macro_ci_low':lo,'net_macro_ci_high':hi,'net_permutation_p':pval,
                'crossing_macro':churn.mean(),'crossing_macro_ci_low':churn_lo,'crossing_macro_ci_high':churn_hi,
                'rescued_from_zero_score':int((merged.rescued & merged.zero_score_original).sum())})
    table=pd.DataFrame(aggregates); ci=pd.DataFrame(cis); compare=pd.DataFrame(comparisons); hid=pd.DataFrame(hidden)
    # Holm adjustment across the three primary retrievers, which are correlated.
    compare['holm_primary_p']=np.nan
    primary=compare[compare.role=='primary'].sort_values('permutation_p')
    adjusted=np.maximum.accumulate([min(1.,row.permutation_p*(len(primary)-i)) for i,row in enumerate(primary.itertuples())])
    compare.loc[primary.index,'holm_primary_p']=adjusted
    table.to_csv(ROOT/'results/tables/aggregate.csv',index=False); ci.to_csv(ROOT/'results/tables/confidence_intervals.csv',index=False)
    compare.to_csv(ROOT/'results/tables/paired_comparisons.csv',index=False); hid.to_csv(ROOT/'results/tables/hidden.csv',index=False)
    all_changes=pd.concat(changes,ignore_index=True); all_changes.to_csv(ROOT/'results/raw/rank_changes.csv',index=False)
    fidelity=pd.read_csv(ROOT/'results/tables/fidelity.csv')
    original_fidelity=fidelity[fidelity.representation=='R0']
    clean_ids=set(original_fidelity[(original_fidelity.conflicting_keys==0)&(original_fidelity.malformed_features==0)].product_id)
    clean_rows=[]
    for (condition,retriever),group in all_changes[all_changes.product_id.isin(clean_ids)].groupby(['representation','retriever']):
        rescued=(group.rank_original>20)&(group.rank_alternative<=20)
        harmed=(group.rank_original<=20)&(group.rank_alternative>20)
        clean_rows.append({'representation':condition,'retriever':retriever,'eligible_products':len(clean_ids),'exact_pairs':len(group),
            'queries':group.query_id.nunique(),'rescued':int(rescued.sum()),'harmed':int(harmed.sum()),
            'crossing_rate':(rescued.sum()+harmed.sum())/len(group)})
    pd.DataFrame(clean_rows).to_csv(ROOT/'results/tables/clean_source_subgroup.csv',index=False)
    for retriever in ['BM25','Dense','Hybrid']:
        files=[ROOT/f'results/raw/{c}_{retriever}_exact.csv' for c in cfg['representations'] if c!='C_repeat']
        if not all(f.exists() for f in files): continue
        wide=None
        for f in files:
            c=f.stem.rsplit('_',2)[0]; current=pd.read_csv(f)[['query_id','product_id','rank']].rename(columns={'rank':c})
            wide=current if wide is None else wide.merge(current,on=['query_id','product_id'],validate='one_to_one')
        ranks=wide.drop(columns=['query_id','product_id']).to_numpy()
        wide['rank_min']=ranks.min(1); wide['rank_max']=ranks.max(1)
        wide['rank_span']=ranks.max(1)-ranks.min(1); wide['rank_ratio']=ranks.max(1)/ranks.min(1)
        for k in [10,20,50]: wide[f'HitInstability@{k}']=(ranks.min(1)<=k)&(ranks.max(1)>k)
        wide.to_csv(ROOT/f'results/raw/sensitivity_{retriever}.csv',index=False)
        instability_macro=wide.groupby('query_id')['HitInstability@20'].mean()
        instability_lo,instability_hi=mean_ci(instability_macro,seed,n)
        sensitivity.append({'retriever':retriever,'exact_pairs':len(wide),'median_rank_span':wide.rank_span.median(),'median_rank_ratio':wide.rank_ratio.median(),
            'HitInstability@20':wide['HitInstability@20'].mean(), 'instability_query_macro':instability_macro.mean(),
            'instability_macro_ci_low':instability_lo,'instability_macro_ci_high':instability_hi})
    pd.DataFrame(sensitivity).to_csv(ROOT/'results/tables/sensitivity.csv',index=False)
    # Controls compare directly against Dual View: positive delta favors Dual.
    controls=[]
    for retriever in ['BM25','Dense','Hybrid']:
        if ('R8',retriever) not in data or ('C_repeat',retriever) not in data: continue
        for m in ['cNDCG@10','Recall@20']:
            effect,lo,hi,p=paired_stats(data[('C_repeat',retriever)][m],data[('R8',retriever)][m],seed,n)
            controls.append({'retriever':retriever,'metric':m,'Dual_minus_repeat':effect,'ci_low':lo,'ci_high':hi,'p':p})
    pd.DataFrame(controls).to_csv(ROOT/'results/tables/length_control.csv',index=False)
    # Deterministic top-20 crossing examples; one occurrence per query-product pair.
    products=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').fillna('').set_index('product_id')
    queries=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').set_index('query_id')['query']
    selected=all_changes[all_changes.representation.str.startswith('R')]
    example_frames=[]
    for direction in ['positive','negative']:
        if direction=='positive':
            eligible=selected[(selected.rank_original>20)&(selected.rank_alternative<=20)].sort_values(['rank_gain','query_id','product_id'],ascending=[False,True,True])
        else:
            eligible=selected[(selected.rank_original<=20)&(selected.rank_alternative>20)].sort_values(['rank_gain','query_id','product_id'],ascending=[True,True,True])
        examples=eligible.drop_duplicates(['query_id','product_id']).head(20).copy(); examples['direction']=direction
        example_frames.append(examples)
        lines=[f'# Twenty strongest {direction} Top-20 crossings','',
               'Selected automatically by absolute rank movement, then inspected separately. Exact means the cleaned WANDS human label; it does not certify every physical product claim. Extremes are illustrative, not representative. Full source fields follow; zero-score ranks can reflect tie breaking.','']
        for i,row in enumerate(examples.itertuples(),1):
            product=products.loc[row.product_id]
            lines += [f'## {i}. Query {row.query_id}: {queries.loc[row.query_id]}','',
                f'Product {row.product_id}: **{product.product_name}**. {row.retriever}/{row.representation}: **{row.rank_original} → {row.rank_alternative}**. Original zero score: {row.zero_score_original}.',
                '',f'Class: {product.product_class}',f'Category: {product["category hierarchy"]}','',
                '**Source description**','',str(product.product_description),'','**Source attributes (including conflicts)**','',str(product.product_features),'']
        (ROOT/f'results/qualitative/{direction}_20.md').write_text('\n'.join(lines),encoding='utf-8')
    pd.concat(example_frames).to_csv(ROOT/'results/qualitative/selected_examples.csv',index=False)
    # Research figures are exported PNG and SVG, independent of UI dependencies.
    plt.rcParams.update({'font.size':10,'figure.dpi':160})
    display_conditions=cfg['representations']+(['C_order'] if 'C_order' in set(table.representation) else [])
    pivot=table.pivot(index='representation',columns='retriever',values='cNDCG@10').reindex(display_conditions)
    fig,ax=plt.subplots(figsize=(6,4)); im=ax.imshow(pivot.to_numpy(),vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns); ax.set_yticks(range(len(pivot.index)),pivot.index)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)): ax.text(j,i,f'{pivot.iloc[i,j]:.3f}',ha='center',va='center',color='white' if pivot.iloc[i,j]>.6 else 'black')
    ax.set_title('cNDCG@10 by representation'); fig.colorbar(im,ax=ax); fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(ROOT/f'results/figures/representation_heatmap.{ext}')
    plt.close(fig)
    h=hid[(hid.k==20)&(hid.representation=='R8')]; x=np.arange(len(h))
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(x-.18,h.rescued,width=.36,label='Hidden → visible'); ax.bar(x+.18,-h.harmed,width=.36,label='Visible → hidden')
    ax.set_xticks(x,h.retriever); ax.axhline(0,color='black',linewidth=.6); ax.set_ylabel('Human-labeled Exact query–product pairs'); ax.set_title('Dual View: both directions of Top-20 crossing'); ax.legend(); fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(ROOT/f'results/figures/hidden_products.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,r in enumerate(['BM25','Dense','Hybrid']):
        sub=compare[(compare.retriever==r)&(compare.metric=='cNDCG@10')]
        xx=np.arange(len(sub))+i*.2
        ax.errorbar(xx,sub.delta,yerr=[sub.delta-sub.ci_low,sub.ci_high-sub.delta],fmt='o',capsize=3,label=r)
    ax.axhline(0,color='black',linewidth=.6); ax.set_xticks(np.arange(len(sub))+.2,sub.representation); ax.set_ylabel('Paired change in cNDCG@10 (95% bootstrap CI)'); ax.legend(); fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(ROOT/f'results/figures/paired_effects.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    main_conditions=[x for x in cfg['representations'] if x.startswith('R')]
    for ax,example_frame,direction in zip(axes,example_frames,['Recovered example','Hidden example']):
        case=example_frame.iloc[0]
        for retriever in ['BM25','Dense','Hybrid']:
            rr=[]
            for condition in main_conditions:
                pairs=pd.read_csv(ROOT/f'results/raw/{condition}_{retriever}_exact.csv')
                rr.append(int(pairs[(pairs.query_id==case.query_id)&(pairs.product_id==case.product_id)]['rank'].iloc[0]))
            ax.plot(main_conditions,rr,marker='o',label=retriever)
        ax.set_yscale('log'); ax.axhline(20,color='grey',linestyle='--',linewidth=1)
        ax.set_title(f'{direction}: product {int(case.product_id)}\nQuery: {str(queries.loc[case.query_id])[:55]}')
        ax.set_xlabel('Whole-catalog representation'); ax.grid(axis='y',alpha=.2)
    axes[0].invert_yaxis(); axes[0].set_ylabel('Full-catalog rank (log scale; lower rank is better)'); axes[1].legend()
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(ROOT/f'results/figures/real_rank_examples.{ext}')
    plt.close(fig)
    # A machine-generated appendix is kept separate from the authored interpretation.
    sections=['# Phase 1 numerical appendix','', 'All values are computed from saved per-query CSV files.','',
      '## Aggregate results','',md(table),'','## Primary paired comparisons: Dual minus Original','',
      md(compare[compare.role=='primary']),'','## Top-20 rescue and harm','',md(hid[hid.k==20]),'',
      '## Rank sensitivity across six representations','',md(pd.DataFrame(sensitivity)),'',
      '## Repetition control','',md(pd.DataFrame(controls)),'','## Field inspection','',md(pd.read_csv(ROOT/'results/tables/field_statistics.csv'))]
    (ROOT/'results/NUMERICAL_APPENDIX.md').write_text('\n'.join(sections),encoding='utf-8')
    print(table.to_string(index=False)); print('\nPRIMARY\n'+compare[compare.role=='primary'].to_string(index=False))

if __name__=='__main__': main()
