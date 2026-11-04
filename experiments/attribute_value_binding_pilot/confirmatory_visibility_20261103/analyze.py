"""RQ1 pair-averaged random changes and RQ2 source-selected cross-encoder transfer."""
from bootstrap_stats import QueryBootstrap,load_scores,concentration
from core import *

def source_choices(source_scores,condition,budget):
    candidates=source_scores[source_scores.variant.eq('A')|(source_scores.condition.eq(condition)&source_scores.j.le(budget))]
    return candidates.sort_values(['query_id','product_id','score','j'],ascending=[True,True,False,True],kind='stable').drop_duplicates(['query_id','product_id'])

def rq1(scores,boot):
    summary=[];pairrows=[];dist=[];concentrations=[];byquery=[];paired={}
    swaps=frame(HERE/'data/selected_swaps.csv',dtype={'product_id':str})[['product_id','j','involves_yes_no']]
    for model,s in scores.groupby('model'):
        baseline=s[s.variant.eq('A')]
        bins=pd.cut(baseline['rank'],[0,20,100,1000,np.inf],labels=['1-20','21-100','101-1000','1001+'])
        for label in bins.cat.categories:dist.append(dict(model=model,rank_bin=label,numerator=int(bins.eq(label).sum()),denominator=len(baseline)))
        for cond in ['C','O']:
            raw=s[s.condition.eq(cond)].drop(columns='involves_yes_no').merge(swaps,on=['product_id','j'],validate='many_to_one')
            raw['entry']=(~raw.in_top20_A)&raw.in_top20;raw['exit']=raw.in_top20_A&(~raw.in_top20)
            raw['crossing']=raw.in_top20!=raw.in_top20_A;raw['net']=raw.in_top20.astype(int)-raw.in_top20_A.astype(int)
            for stratum,part in [('all',raw),('yes_no',raw[raw.involves_yes_no]),('not_yes_no',raw[~raw.involves_yes_no])]:
                pairs=part.groupby(['query_id','product_id']).agg(n_modifications=('j','size'),
                    in_top20_A=('in_top20_A','first'),entry=('entry','mean'),exit=('exit','mean'),crossing=('crossing','mean'),
                    net=('net','mean'),raw_crossings=('crossing','sum'),score_delta=('score_delta','mean'),rank_delta=('rank_delta','mean')).reset_index()
                if stratum=='all':assert pairs.n_modifications.eq(5).all()
                pairrows.extend(dict(model=model,condition=cond,stratum=stratum,**r) for r in pairs.to_dict('records'))
                for metric in ['entry','exit','crossing','net']:
                    den=(~pairs.in_top20_A) if metric=='entry' else pairs.in_top20_A if metric=='exit' else np.ones(len(pairs))
                    stats,draws=boot.estimate(pairs.query_id,pairs[metric],den)
                    raw_den=int((~part.in_top20_A).sum()) if metric=='entry' else int(part.in_top20_A.sum()) if metric=='exit' else len(part)
                    summary.append(dict(model=model,condition=cond,stratum=stratum,metric=metric,pairs=len(pairs),queries=pairs.query_id.nunique(),
                        raw_events=int(part[metric].sum()),raw_trial_denominator=raw_den,**stats))
                    if stratum=='all':paired[(model,cond,metric)]=(stats,draws)
                if len(pairs):
                    c,q=concentration(pairs,'raw_crossings','net',dict(rq='RQ1',model=model,condition=cond,stratum=stratum))
                    concentrations.append(c);byquery.extend(q)
    contrasts=[]
    for model in ['gte','harrier']:
        for metric in ['entry','exit','crossing','net']:
            c,cb=paired[(model,'C',metric)];o,ob=paired[(model,'O',metric)];d=cb-ob;valid=np.isfinite(d)
            low,high=np.quantile(d[valid],[.025,.975]);contrasts.append(dict(model=model,metric=metric,contrast='C_minus_O',
                estimate=c['estimate']-o['estimate'],ci_low=low,ci_high=high,valid_bootstrap_draws=int(valid.sum()),interpretation='intervention contrast, not pure binding effect'))
    csv('results/rq1_summary.csv',summary);csv('results/rq1_pair_averages.csv',pairrows);csv('results/rq1_C_minus_O.csv',contrasts)
    csv('results/baseline_rank_distribution.csv',dist);csv('results/rq1_query_concentration.csv',concentrations);csv('results/rq1_by_query.csv',byquery)
    return pd.DataFrame(summary)

def rq2(scores,boot):
    allchoices=[];summaries=[];concentrations=[];byquery=[];paired={}
    for source,target in [('gte','harrier'),('harrier','gte')]:
        a=scores[scores.model.eq(source)];b=scores[scores.model.eq(target)]
        target_lookup=b.set_index(['query_id','product_id','variant'])
        for cond in ['C','O']:
            for budget in [1,3,5]:
                # Sort by numeric score, then j: A has j=0 and wins all exact ties.
                winners=source_choices(a,cond,budget)
                choices=[]
                for r in winners.itertuples():
                    tr=target_lookup.loc[(r.query_id,r.product_id,r.variant)]
                    source_net=int(r.in_top20)-int(r.in_top20_A);target_net=int(tr.in_top20)-int(tr.in_top20_A)
                    assert r.score>=r.score_A and r.rank<=r.rank_A and source_net>=0
                    choices.append(dict(query_id=r.query_id,product_id=r.product_id,query=r.query,source=source,target=target,condition=cond,budget=budget,
                        chosen_variant=r.variant,modified=r.variant!='A',source_score=r.score,source_score_A=r.score_A,source_rank=r.rank,source_rank_A=r.rank_A,
                        source_inclusion=r.in_top20,source_inclusion_A=r.in_top20_A,source_net=source_net,
                        target_score=float(tr.score),target_score_A=float(tr.score_A),target_rank=int(tr['rank']),target_rank_A=int(tr.rank_A),
                        target_inclusion=bool(tr.in_top20),target_inclusion_A=bool(tr.in_top20_A),target_net=target_net))
                f=pd.DataFrame(choices);assert len(f)==int(a.variant.eq('A').sum())
                allchoices.extend(choices)
                for evaluation,encoder in [('source',source),('target',target)]:
                    inside=f[evaluation+'_inclusion_A'];after=f[evaluation+'_inclusion'];net=f[evaluation+'_net']
                    entry=(~inside)&after;exit_=inside&(~after);cross=entry|exit_
                    for metric,num,den in [('delta_inclusion',net,np.ones(len(f))),('entry',entry,~inside),('exit',exit_,inside),
                                           ('crossing',cross,np.ones(len(f))),('modified',f.modified,np.ones(len(f)))]:
                        stats,draws=boot.estimate(f.query_id,num,den)
                        row=dict(source=source,target=target,evaluation=evaluation,encoder=encoder,condition=cond,budget=budget,metric=metric,
                            pairs=len(f),queries=f.query_id.nunique(),raw_events=int(num.sum()),**stats,
                            estimate_pp=100*stats['estimate'],ci_low_pp=100*stats['ci_low'],ci_high_pp=100*stats['ci_high'])
                        summaries.append(row);paired[(source,target,evaluation,cond,budget,metric)]=(stats,draws)
                    if evaluation=='target' and budget==5:
                        c,q=concentration(f.assign(events=cross),'events','target_net',dict(rq='RQ2',source=source,target=target,condition=cond,budget=budget))
                        concentrations.append(c);byquery.extend(q)
    contrast=[]
    for source,target in [('gte','harrier'),('harrier','gte')]:
        for b in [1,3,5]:
            for ev in ['source','target']:
                c,cb=paired[(source,target,ev,'C',b,'delta_inclusion')];o,ob=paired[(source,target,ev,'O',b,'delta_inclusion')]
                difference=cb-ob;valid=np.isfinite(difference);lo,hi=np.quantile(difference[valid],[.025,.975])
                contrast.append(dict(source=source,target=target,evaluation=ev,budget=b,
                    contrast='C_minus_O',estimate_pp=100*(c['estimate']-o['estimate']),ci_low_pp=100*lo,ci_high_pp=100*hi,valid_bootstrap_draws=int(valid.sum())))
    csv('results/rq2_choices.csv',allchoices);csv('results/rq2_summary.csv',summaries);csv('results/rq2_C_minus_O.csv',contrast)
    csv('results/rq2_query_concentration.csv',concentrations);csv('results/rq2_by_query.csv',byquery)
    return pd.DataFrame(summaries)

def main():
    freeze_check()
    for model in ['gte','harrier']:assert read(HERE/f'qa/formal/{model}_complete.json')['passed']
    scores=load_scores();boot=QueryBootstrap();t=Timer();rq1(scores,boot);rq2(scores,boot)
    timed('qa/analysis_timing.json',t);print('RQ1/RQ2 COMPLETE with 10000 paired query bootstrap draws',flush=True)

if __name__=='__main__':main()
