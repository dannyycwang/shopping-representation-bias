"""Compact vector publication assets, 7-inch two-column width, source CSVs."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import textwrap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
NAMES={'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE'}
LABELS={'raw':'Raw dense (7 mean)','lexical_ascending':'Canonical: ascending','lexical_descending':'Canonical: descending','field_priority_type':'Canonical: type priority','BM25':'BM25','raw_hybrid':'Raw hybrid (7 mean)','canonical_hybrid_lexical_ascending':'Canonical hybrid: ascending','canonical_hybrid_lexical_descending':'Canonical hybrid: descending','canonical_hybrid_field_priority_type':'Canonical hybrid: type priority','set_mean':'Set-Mean'}
def save(fig,name):
    fig.savefig(HERE/f'figures/{name}.pdf',metadata={'Creator':'Section 6 audited evidence package'})
    fig.savefig(HERE/f'figures/{name}.png',dpi=220)
    plt.close(fig)

def rq1():
    f=pd.read_csv(HERE/'tables/inclusion_complete.csv');f=f[f.intervention.eq('target_only')&f.K.eq(20)&f.metric.eq('VI')].copy()
    # GTE is one estimate on exactly the same support, not two observations.
    plotted=f[~(f.model.eq('gte_modernbert')&f.target_support.eq('fully_fitting'))].copy();csv(plotted,HERE/'data/figure3_plotted.csv')
    fig=plt.figure(figsize=(7,2.6));gs=fig.add_gridspec(1,5,width_ratios=[1.7,1.15,.35,1.7,1.15],left=.08,right=.99,bottom=.24,top=.80,wspace=.08)
    for plotcol,ds,limit in zip([0,3],['wands','esci'],[25,8]):
        ax=fig.add_subplot(gs[0,plotcol]);tx=fig.add_subplot(gs[0,plotcol+1]);tx.set(xlim=(0,1),ylim=(-.1,5.4));tx.axis('off')
        ff=plotted[plotted.dataset.eq(ds)];positions={('minilm','full'):4.5,('minilm','fully_fitting'):3.8,('bge_base','full'):2.7,('bge_base','fully_fitting'):2.,('gte_modernbert','full'):.5}
        for r in ff.itertuples():
            y=positions[(r.model,r.target_support)];full=r.target_support=='full';marker='s' if r.model=='gte_modernbert' else ('o' if full else '^')
            ax.errorbar(r.estimate*100,y,xerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],fmt=marker,color='#174c62' if full else '#555555',mfc='#174c62' if full else 'white',markersize=4.5,capsize=2,lw=.9)
            tx.text(.02,y,f'{r.estimate*100:.2f} | {r.queries:,}/{r.pairs:,}',va='center',ha='left',fontsize=7)
        ax.set(xlim=(0,limit),ylim=(-.1,5.4),yticks=[4.15,2.35,.5],yticklabels=['MiniLM','BGE','GTE*'],xlabel=f'VI@20 (%)  |  axis: 0-{limit}%')
        ax.set_title(ds.upper(),loc='left',fontweight='bold');ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        tx.text(.02,5.1,'VI (%) | queries/pairs',ha='left',fontsize=6.7)
    fig.legend(handles=[Line2D([],[],marker='o',color='#174c62',linestyle='',label='Full'),Line2D([],[],marker='^',mfc='white',color='#555555',linestyle='',label='Fully fitting')],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.text(.5,.015,'* GTE: Full = fully fitting. Whiskers: 95% query-cluster intervals. Competitors stay at source order.',ha='center',fontsize=7)
    save(fig,'figure3_target_fitting')

def rq2():
    f=pd.read_csv(HERE/'tables/membership_all_contrasts.csv');f=f[f.method.eq('raw')&f.K.eq(20)&f.schedule.eq('C1')];csv(f,HERE/'data/figure4_plotted.csv')
    fig=plt.figure(figsize=(7,2.75));gs=fig.add_gridspec(1,5,width_ratios=[1.75,1,.45,1.75,1],left=.08,right=.985,bottom=.29,top=.75,wspace=.08)
    for plotcol,ds,limit in zip([0,3],['wands','esci'],[7.,2.5]):
        ax=fig.add_subplot(gs[0,plotcol]);tx=fig.add_subplot(gs[0,plotcol+1]);tx.set(xlim=(0,1),ylim=(-.6,2.8));tx.axis('off')
        ff=f[f.dataset.eq(ds)].set_index('model')
        for i,m in enumerate(MODELS):
            r=ff.loc[m];y=2-i
            ax.barh(y,-100*r.loss_share,height=.44,color='#d0d0d0',hatch='///',edgecolor='#555',lw=.5)
            ax.barh(y,100*r.gain_share,height=.44,color='#39738b',edgecolor='#222',lw=.5)
            ax.errorbar(100*r.delta_recall,y,xerr=[[100*(r.delta_recall-r.delta_recall_ci_low)],[100*(r.delta_recall_ci_high-r.delta_recall)]],fmt='D',markersize=3.5,color='black',capsize=2,lw=.8)
            tx.text(.03,y,f'{int(r.equal_positive_queries)}/{int(r.queries)}\n({r.equal_positive_share_all*100:.1f}%)',ha='left',va='center',fontsize=7)
        ax.set(xlim=(-limit,limit),ylim=(-.6,2.8),yticks=[2,1,0],yticklabels=['MiniLM','BGE','GTE'],xticks=[-limit,0,limit],xlabel='Share (%) / ΔRecall (pp)')
        tx.text(.03,2.6,'Equal gains/losses\nQueries (%)',ha='left',va='center',fontsize=6.7)
        ax.axvline(0,lw=.5,color='black');ax.tick_params(axis='y',length=0);ax.set_title(ds.upper(),loc='left',fontweight='bold')
    fig.legend(handles=[Patch(facecolor='#d0d0d0',hatch='///',edgecolor='#555',label='Lost (negative for display)'),Patch(facecolor='#39738b',label='Gained'),Line2D([],[],marker='D',color='black',label='Net Recall, 95% CI',lw=.8)],loc='upper center',ncol=3,frameon=False,fontsize=7.2)
    fig.text(.5,.065,'Right column: gain count = loss count > 0; denominator is all eligible queries.',ha='center',fontsize=7)
    fig.text(.5,.015,'Catalog-wide Reverse vs Source; shares use |Hq|. All six contrasts are in companions. Panel scales differ.',ha='center',fontsize=7)
    save(fig,'figure4_membership_replacement')

def table1():
    f=pd.read_csv(HERE/'tables/controls_complete.csv');rows=[]
    for ds in ['wands','esci']:
        for method,label in LABELS.items():
            row={'dataset':ds,'method':method,'label':label}
            for m in ['minilm','bge_base']:
                for metric,k in [('Recall',100),('VI',100),('nDCG',20)]:
                    s='seven_mean' if method in ['raw','raw_hybrid'] else 'fixed'
                    r=f[f.dataset.eq(ds)&f.model.eq(m)&f.method.eq(method)&f.schedule.eq(s)&f.K.eq(k)&f.metric.eq(metric)].iloc[0]
                    row[m+'_'+metric]=r.estimate
            rows.append(row)
    df=pd.DataFrame(rows);csv(df,HERE/'tables/main_table_candidate.csv')
    fig,ax=plt.subplots(figsize=(7,4.55));ax.axis('off');table=[]
    for ds in ['wands','esci']:
        table.append([ds.upper(),'MiniLM','','','BGE','',''])
        table.append(['Method','R@100','VI@100','nDCG@20','R@100','VI@100','nDCG@20'])
        for r in df[df.dataset.eq(ds)].itertuples():table.append([r.label]+[f'{getattr(r,m+"_"+metric)*(100 if metric!="nDCG" else 1):.{2 if metric!="nDCG" else 4}f}' for m in ['minilm','bge_base'] for metric in ['Recall','VI','nDCG']])
    t=ax.table(cellText=table,colWidths=[.38,.103,.103,.11,.103,.103,.11],cellLoc='right',bbox=[0,.16,1,.84]);t.auto_set_font_size(False);t.set_fontsize(7)
    for (r,c),cell in t.get_celld().items():
        cell.set_linewidth(0);cell.PAD=.07
        if c==0:cell.get_text().set_ha('left')
        if r in [0,1,12,13]:cell.set_facecolor('#e8eef0');cell.get_text().set_fontweight('bold')
    fig.text(.03,.10,'R and VI: %. nDCG: 0-1. R@100 and VI@100 assess candidate inclusion; nDCG@20 assesses ranking near the top.',fontsize=7)
    fig.text(.03,.065,'Raw rows average seven schedules. All fixed controls have structural VI = 0. Paired intervals and R@20 are in the companion.',fontsize=7)
    fig.text(.03,.03,'Source-order raw results, GTE, centroids, multi-vector retrieval and all view budgets remain in complete outputs.',fontsize=7)
    fig.subplots_adjust(left=.025,right=.975,bottom=0,top=.99);save(fig,'table1_controls_candidate')
    lines=['% Audited candidate. Requires booktabs. Two-column table; R/VI@100 for candidates, nDCG@20 for top ranks.','\\begin{tabular}{lrrrrrr}','\\toprule','Method & \\multicolumn{3}{c}{MiniLM} & \\multicolumn{3}{c}{BGE} \\\\',' & R@100 & VI@100 & nDCG@20 & R@100 & VI@100 & nDCG@20 \\\\']
    for ds in ['wands','esci']:
        lines += ['\\midrule',f'\\multicolumn{{7}}{{l}}{{{ds.upper()}}} \\\\']
        for row in table[(2 if ds=='wands' else 14):(12 if ds=='wands' else 24)]:lines.append(' & '.join(row)+' \\\\')
    lines += ['\\bottomrule','\\end{tabular}'];(HERE/'tables/main_table_candidate.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')

def transition():
    f=pd.read_csv(HERE/'tables/transitions_complete.csv');f=f[f.dataset.eq('wands')&f.model.eq('minilm')&f.rule.eq('lexical_ascending')&f.K.eq(20)];csv(f,HERE/'data/table2_plotted.csv')
    labels=['Persistent inclusion','Crossing','Persistent omission'];rows=[]
    for state,label in zip(['persistent_inclusion','crossing','persistent_omission'],labels):
        rows.append([label]+[f'{f[f.family.eq(fam)&f.reference_state.eq(state)&f.control_state.eq(out)].estimate.iloc[0]*100:.2f}' for fam in ['dense','hybrid'] for out in ['included','omitted']])
    fig,ax=plt.subplots(figsize=(7,1.6));ax.axis('off');t=ax.table(cellText=rows,colLabels=['Raw reference state','Dense: included','Dense: omitted','Hybrid: included','Hybrid: omitted'],colWidths=[.27,.1825,.1825,.1825,.1825],cellLoc='center',bbox=[0,.32,1,.62]);t.auto_set_font_size(False);t.set_fontsize(8)
    for (r,c),cell in t.get_celld().items():
        cell.set_linewidth(.3);cell.set_edgecolor('#aaa')
        if r==0:cell.set_facecolor('#e8eef0')
    fig.text(.02,.19,'WANDS / MiniLM, ascending canonicalization, K = 20. Each value is a query-macro % of all highest-label support.',fontsize=7)
    fig.text(.02,.06,'Each six-cell block sums to 100% before rounding. Dense and hybrid have separate raw reference families.',fontsize=7)
    fig.subplots_adjust(left=.025,right=.975,top=1,bottom=0);save(fig,'table2_transition_single')
    (HERE/'tables/transition_single.tex').write_text('\\begin{tabular}{lrrrr}\n\\toprule\nRaw reference state & \\multicolumn{2}{c}{Dense} & \\multicolumn{2}{c}{Hybrid} \\\\\n & Included & Omitted & Included & Omitted \\\\\n\\midrule\n'+'\n'.join(' & '.join(r)+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n',encoding='utf8')

def case():
    f=pd.read_csv(HERE/'data/illustrative_replacement_case.csv',dtype={'product_id':str});m=read(HERE/'data/illustrative_case_selection.json')
    rows=[]
    for r in f.itertuples():rows.append([r.change.title(),r.product_id,'\n'.join(textwrap.wrap(r.product_name,48)),r.label,str(r.rank_C0),str(r.rank_C1)])
    height=max(2.0,.22*len(rows)+1.05);fig,ax=plt.subplots(figsize=(7,height));ax.axis('off')
    fig.text(.025,.93,f'Illustrative query: “{m["query"]}”  (ID {m["query_id"]})',fontweight='bold',fontsize=9)
    t=ax.table(cellText=rows,colLabels=['Change','Product ID','Product name','Label','Source rank','Reverse rank'],colWidths=[.09,.105,.475,.08,.125,.125],cellLoc='left',bbox=[0,.22,1,.63]);t.auto_set_font_size(False);t.set_fontsize(7.1)
    for (r,c),cell in t.get_celld().items():
        cell.set_linewidth(.3);cell.set_edgecolor('#aaa');cell.PAD=.045
        if r==0:cell.set_facecolor('#e8eef0')
    fig.text(.025,.15,'WANDS / BGE; same 42,994-product catalog, reordered catalog-wide.',fontsize=8)
    fig.text(.025,.105,f'Source and Reverse Recall@20 = {m["source_numerator"]}/{m["denominator"]} = {m["recall"]:.6f}.',fontsize=8)
    fig.text(.025,.045,'Hash-selected after eligibility was defined; illustrative, with no all-fitting requirement. Full orders and attributes are in the CSV.',fontsize=7)
    fig.subplots_adjust(left=.025,right=.975,top=1,bottom=0);save(fig,'illustrative_replacement_case')

if __name__=='__main__':rq1();rq2();table1();transition();case()
