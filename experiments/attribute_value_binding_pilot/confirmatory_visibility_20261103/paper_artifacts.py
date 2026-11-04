"""Standalone two tables and budget-transfer figure; no manuscript edits."""
from core import *
NAMES={'gte':'GTE','harrier':'Harrier'}
def pct(x):
    try:return f'{100*float(x):.2f}' if np.isfinite(float(x)) else 'NA'
    except (ValueError,TypeError):return 'NA'
def cell(row):return f"{pct(row.estimate)} [{pct(row.ci_low)}, {pct(row.ci_high)}]"
def mdtable(columns,rows):return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def latex(columns,rows,caption,label):
    def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('→',r'$\to$').replace('Δ',r'$\Delta$')
    return '\n'.join([r'\begin{table*}[t]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular}{'+'l'*len(columns)+'}',r'\toprule',' & '.join(map(esc,columns))+r' \\',r'\midrule',
        *[' & '.join(map(esc,r))+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',r'\end{table*}'])+'\n'
def main():
    t=Timer();(HERE/'paper').mkdir(exist_ok=True)
    s=frame(HERE/'results/rq1_summary.csv');s=s[s.stratum.eq('all')];table1=[];flat1=[]
    for model in ['gte','harrier']:
        for c in ['C','O']:
            g=s[s.model.eq(model)&s.condition.eq(c)].set_index('metric');cells=[]
            flat=dict(model=model,condition=c)
            for metric in ['entry','exit','crossing']:
                r=g.loc[metric];cells.append(f'{int(r.raw_events)}/{int(r.raw_trial_denominator)}; '+cell(r))
                flat.update({metric+'_'+name:r[name] for name in ['raw_events','raw_trial_denominator','denominator_pairs','estimate','ci_low','ci_high']})
            table1.append([NAMES[model],c,*cells]);flat1.append(flat)
    cols1=['Encoder','Edit','Entry: events/trials; % [95% CI]','Exit: events/trials; % [95% CI]','Crossing: events/trials; % [95% CI]']
    (HERE/'paper/table1.md').write_text(mdtable(cols1,table1)+'\n\nTrials = 5 × baseline-eligible pairs. Estimates average five edits within each pair; CIs resample queries, not edits.\n',encoding='utf8')
    tex1=[[r[0],r[1],*[r'\shortstack[l]{'+v.split('; ')[1]+r' \\ ('+v.split('; ')[0]+')}' for v in r[2:]]] for r in table1]
    (HERE/'paper/table1.tex').write_text(latex(['Encoder','Edit','Entry (%)','Exit (%)','Crossing (%)'],tex1,'Random changes: pair-averaged exposure rates with paired query-bootstrap 95\% intervals. Parentheses show raw events over five-change trials; trials are not independent observations.','tab:random_visibility'),encoding='utf8')
    csv('paper/table1.csv',flat1)
    r=frame(HERE/'results/rq2_summary.csv');table2=[];flat2=[]
    for source,target in [('gte','harrier'),('harrier','gte')]:
        for c in ['C','O']:
            g=r[r.source.eq(source)&r.target.eq(target)&r.condition.eq(c)&r.budget.eq(5)].set_index(['evaluation','metric'])
            src=g.loc[('source','delta_inclusion')];dst=g.loc[('target','delta_inclusion')];mod=g.loc[('source','modified')]
            ent=g.loc[('target','entry')];ext=g.loc[('target','exit')]
            table2.append([NAMES[source]+' → '+NAMES[target],c,f'{int(mod.raw_events)}/{int(mod.denominator_pairs)}',cell(src),cell(dst),
                f'{int(ent.raw_events)}/{int(ent.denominator_pairs)}',f'{int(ext.raw_events)}/{int(ext.denominator_pairs)}'])
            flat2.append(dict(source=source,target=target,condition=c,budget=5,modified_n=mod.raw_events,pairs=mod.denominator_pairs,
                source_delta_pp=src.estimate_pp,source_ci_low_pp=src.ci_low_pp,source_ci_high_pp=src.ci_high_pp,
                target_delta_pp=dst.estimate_pp,target_ci_low_pp=dst.ci_low_pp,target_ci_high_pp=dst.ci_high_pp,
                target_entry_n=ent.raw_events,target_entry_den=ent.denominator_pairs,target_exit_n=ext.raw_events,target_exit_den=ext.denominator_pairs))
    cols2=['Source → target','Edit','Changed / all','Source Δ pp [95% CI]','Target Δ pp [95% CI]','Target entry','Target exit']
    (HERE/'paper/table2.md').write_text(mdtable(cols2,table2)+'\n\nBudget b=5 excludes the common A baseline. All pairs, including retained A, are included. Source gains are selected in-sample.\n',encoding='utf8')
    (HERE/'paper/table2.tex').write_text(latex(['Source → target','Edit','Changed / all','Source Δ (pp)','Target Δ (pp)','Entry','Exit'],table2,'Cross-encoder transfer at budget five, with paired query-bootstrap 95\% intervals. Inclusion changes are percentage points. A is always eligible; all pairs are retained. Source gains reflect score-based selection. Entry/exit counts use the target baseline outside/inside denominators.','tab:transfer_visibility'),encoding='utf8')
    csv('paper/table2.csv',flat2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),sharey=True)
    colors={'C':'#0072B2','O':'#D55E00'};labels={'C':'Value swap (C)','O':'Entry reorder (O)'}
    plot=r[r.evaluation.eq('target')&r.metric.eq('delta_inclusion')]
    for ax,(source,target) in zip(axes,[('gte','harrier'),('harrier','gte')]):
        for c in ['C','O']:
            g=plot[plot.source.eq(source)&plot.condition.eq(c)].sort_values('budget')
            x=g.budget.to_numpy(dtype=float)+(-.055 if c=='C' else .055);y=g.estimate_pp.to_numpy(dtype=float)
            lo=g.ci_low_pp.to_numpy(dtype=float);hi=g.ci_high_pp.to_numpy(dtype=float)
            ax.plot(x,y,marker='o' if c=='C' else 's',color=colors[c],lw=1.4,ms=4,label=labels[c])
            ax.vlines(x,lo,hi,color=colors[c],lw=1.2)
            ax.hlines(lo,x-.035,x+.035,color=colors[c],lw=1.2);ax.hlines(hi,x-.035,x+.035,color=colors[c],lw=1.2)
        ax.axhline(0,color='.4',lw=.8,ls='--');ax.set_xticks([1,3,5]);ax.set_xlabel('Additional variants tried (b)')
        ax.set_title(NAMES[source]+' → '+NAMES[target]);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Target ΔInclusion@20 (pp)')
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.09,1,1))
    for suffix in ['png','pdf','svg']:fig.savefig(HERE/f'paper/figure1.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig);csv('paper/figure1_data.csv',plot)
    pair=frame(HERE/'results/rq1_pair_averages.csv');secondary=[]
    for keys,g in pair.groupby(['model','condition','stratum']):
        for metric in ['score_delta','rank_delta']:
            q=np.quantile(g[metric],[.25,.5,.75]);secondary.append(dict(model=keys[0],condition=keys[1],stratum=keys[2],metric=metric,
                pair_mean_q25=q[0],pair_mean_median=q[1],pair_mean_q75=q[2]))
    csv('results/secondary_deltas.csv',secondary);timed('qa/paper_artifact_timing.json',t)
    print('TABLES 1/2 AND FIGURE 1 READY',flush=True)

if __name__=='__main__':main()
