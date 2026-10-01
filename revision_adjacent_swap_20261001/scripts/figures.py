"""Vector publication figures and independent Poppler renders for visual QA."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

ORDER=[('wands','minilm'),('wands','bge_base'),('esci','minilm'),('esci','bge_base')]
LABELS=['WANDS / MiniLM','WANDS / BGE','ESCI / MiniLM','ESCI / BGE']
COLORS=['#21618c','#148f77','#21618c','#148f77']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.spines.left':False,'axes.edgecolor':'#999999','xtick.color':'#333333','ytick.color':'#222222'})

def save(fig,name):
    fig.savefig(HERE/f'figures/{name}.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(HERE/f'figures/{name}.png',dpi=220,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    render=HERE/'qa/rendered';render.mkdir(parents=True,exist_ok=True)
    subprocess.run(['pdftoppm','-r','150','-png','-singlefile',str(HERE/f'figures/{name}.pdf'),str(render/name)],check=True)

def aggregate(k):
    estimates=pd.read_csv(HERE/'data/aggregate_estimates.csv');counts=pd.read_csv(HERE/'data/event_counts.csv')
    source=estimates[estimates.K.eq(k)&estimates.metric.isin(['A','F','A_confirmed','F_confirmed'])]
    csv(source,HERE/f'figures/aggregate_K{k}_source.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.35),sharey=True,gridspec_kw={'wspace':.12})
    fig.subplots_adjust(bottom=.23,top=.86)
    y=np.arange(4)[::-1]
    for ax,metric,title in zip(axes,['A','F'],['A  Targets affected by any swap','B  Mean one-swap flip frequency']):
        vals=[]
        for i,(ds,m) in enumerate(ORDER):
            r=source[source.dataset.eq(ds)&source.model.eq(m)&source.metric.eq(metric)].iloc[0]
            lower=source[source.dataset.eq(ds)&source.model.eq(m)&source.metric.eq(metric+'_confirmed')].iloc[0]
            x=100*r.estimate;l=100*r.ci_low;h=100*r.ci_high;vals.append(h)
            ax.errorbar(x,y[i],xerr=np.array([[x-l],[h-x]]),fmt='o',color=COLORS[i],ecolor=COLORS[i],capsize=3,markersize=5,lw=1.4)
            if lower.estimate<r.estimate:ax.plot(100*lower.estimate,y[i],marker='D',ms=4,color='#b03a2e')
            ax.annotate(f'{x:.2f}%',(x,y[i]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(title,loc='left',pad=16,fontweight='bold');ax.set_xlabel('Query-macro share (%)')
        ax.set_xlim(0,max(vals+[.1])*1.23);ax.set_ylim(-.6,3.65);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_yticks(y);ax.tick_params(axis='y',length=0)
    axes[0].set_yticklabels(LABELS)
    unresolved=int(counts[counts.K.eq(k)].unresolved_flips.sum())
    note=f'Top-{k} | Fixed C0 competitors | Fully fitting sampled targets | 95% query-cluster bootstrap intervals'
    if unresolved:note+='\nCircles: observed; red diamonds: confirmed lower bounds.'
    fig.text(.5,.035,note,ha='center',fontsize=8,color='#444444')
    save(fig,f'aggregate_K{k}')

def case_figure():
    c=read(HERE/'data/case_selection.json')
    if c['status']!='selected':return
    r=c['selected'];b=c['original_entries'];v=c['swapped_entries'];position=json.loads(r['positions'])[0]
    source={k:value for k,value in c.items() if k!='all_serializations'}
    dump(HERE/'figures/case_source.json',source)
    fig=plt.figure(figsize=(8.2,5.8));fig.patch.set_facecolor('white')
    fig.text(.035,.95,'One adjacent swap changes Top-20 inclusion',fontsize=14,fontweight='bold')
    fig.text(.035,.905,f"{r['dataset'].upper()} / {'MiniLM' if r['model']=='minilm' else 'BGE'}    Query ID: {r['query_id']}    Product ID: {r['product_id']}",fontsize=9,color='#444444')
    query='\n'.join(textwrap.wrap('Query: '+c['query'],width=107))
    fig.text(.035,.866,query,fontsize=10,va='top')
    title='\n'.join(textwrap.wrap('Product: '+c['product_title'],width=116))
    fig.text(.035,.79,title,fontsize=8.5,va='top',color='#444444')
    # Dynamic row height accommodates the hash-selected entries without choosing a shorter case.
    for x,head,entries,color in [(.04,'C0: original order',b,'#21618c'),(.535,'After one adjacent swap',v,'#a93226')]:
        ax=fig.add_axes([x,.405,.425,.30]);ax.set_axis_off()
        ax.text(0,1,head,va='top',fontweight='bold',fontsize=10,color=color)
        first='\n'.join(textwrap.wrap(entries[0],width=48));second='\n'.join(textwrap.wrap(entries[1],width=48))
        ax.text(0,.77,f'{position}  '+first,va='top',fontsize=8.7,linespacing=1.2)
        y2=.77-max(2,len(first.splitlines())+1)*.12
        ax.text(0,y2,f'{position+1}  '+second,va='top',fontsize=8.7,linespacing=1.2)
    fig.text(.035,.391,'All remaining content, entry multiplicities, and separators are unchanged.',fontsize=9,fontweight='bold')
    cols=['','C0','Adjacent swap']
    vals=[['Input tokens / limit',f"{r['c0_tokens']} / {r['context_limit']}",f"{r['tokens']} / {r['context_limit']}"],
          ['Native score',f"{r['c0_score']:.9f}",f"{r['swap_score']:.9f}"],
          ['Rank',str(r['c0_rank']),str(r['swap_rank'])],
          ['Margin to fixed Top-20 boundary',f"{r['c0_margin_20']:+.9f}",f"{r['swap_margin_20']:+.9f}"],
          ['Top-20 inclusion','Included' if r['c0_included_20'] else 'Omitted','Included' if r['swap_included_20'] else 'Omitted']]
    ax=fig.add_axes([.03,.13,.94,.23]);ax.set_axis_off()
    table=ax.table(cellText=vals,colLabels=cols,colWidths=[.48,.25,.27],cellLoc='left',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.25)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('#f0f3f4' if row%2 else 'white')
        if row==0:cell.set_text_props(weight='bold')
    fig.text(.035,.067,'Competitors remain fixed at C0. Both decisions reproduced in all three repeated forward checks.',fontsize=8.3)
    fig.text(.035,.033,f"Hash-selected conditional on a confirmed K=20 {c['direction']}; this case is not a prevalence sample.",fontsize=8.3,color='#555555')
    save(fig,'illustrative_case')

def main():
    verify_freeze();aggregate(20);aggregate(100);case_figure()
    print('FIGURES AND PDF RENDERS COMPLETE',flush=True)

if __name__=='__main__':main()
