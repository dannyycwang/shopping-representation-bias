"""Vector manuscript figures. Every empirical value comes from saved plotting CSVs."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import xml.etree.ElementTree as ET

FONT=HERE/'assets/fonts'
for f in FONT.glob('*.ttf'): fm.fontManager.addfont(str(f))
FAMILY=fm.FontProperties(fname=str(FONT/'ManuscriptLibertine-R.ttf')).get_name()
plt.rcParams.update({'font.family':FAMILY,'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':10,'pdf.fonttype':42,'svg.fonttype':'path',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.facecolor':'white'})
INK='#202832';GAIN='#187C80';LOSS='#B24C36';MUTED='#D8DFE5';BLUE='#3C6589'
MODELS=['minilm','bge_base','gte_modernbert'];NAMES={'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE'}

def svg(fig,path,title,desc):
    fig.savefig(path,format='svg')
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree=ET.parse(path);root=tree.getroot();ns='{http://www.w3.org/2000/svg}'
    definitions=ET.Element(ns+'defs')
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag==ns+'defs':
                for entry in list(child): definitions.append(entry)
                parent.remove(child)
    root.insert(0,definitions)
    t=ET.Element(ns+'title');t.text=title;d=ET.Element(ns+'desc');d.text=desc
    root.insert(0,t);root.insert(1,d);root.set('role','img');tree.write(path,encoding='utf-8',xml_declaration=True)

def schematic():
    dump(DATA/'figure2_schematic_data.json',dict(kind='Schematic illustration; not empirical evidence',
      block_semantics='Each letter denotes one intact complete attribute entry. Letters are not shuffled characters within an entry.',
      raw_C0=dict(target_p=list('ABC'),competitor_u=list('DEF'),competitor_v=list('GHJ')),
      catalog_wide_alternative=dict(target_p=list('CAB'),competitor_u=list('FDE'),competitor_v=list('JGH')),
      target_only_alternative=dict(target_p=list('CAB'),competitor_u=list('DEF'),competitor_v=list('GHJ')),
      fixed=['query','encoder','catalog membership','entry contents','non-attribute text','section placement'],
      target_index='Old target entry is replaced; each target defines a separate index.',
      inclusion_states={'persistent inclusion':[1,1,1],'cutoff crossing':[1,0,1],'persistent omission':[0,0,0]}))
    fig=plt.figure(figsize=(7.1,4.8));ax=fig.add_axes([.02,.02,.96,.96]);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    ax.text(0,98,'Schematic illustration',weight='bold',fontsize=12,color=INK)
    ax.text(0,91,'A  What changes in the catalog?',weight='bold',fontsize=11)
    colors={'A':'#BBDDE0','B':'#F0D2BA','C':'#D4CCEA','D':'#DCE4E8','E':'#DCE4E8','F':'#DCE4E8','G':'#DCE4E8','H':'#DCE4E8','J':'#DCE4E8'}
    def blocks(x,y,letters):
        for i,letter in enumerate(letters):
            ax.add_patch(Rectangle((x+4.25*i,y-2.7),3.6,5.4,facecolor=colors[letter],edgecolor=INK,lw=.45))
            ax.text(x+4.25*i+1.8,y,letter,ha='center',va='center',fontsize=10)
    for left,targetonly in [(0,False),(52,True)]:
        ax.text(left,84,'Single-target change' if targetonly else 'Catalog-wide change',fontsize=11,weight='bold')
        ax.text(left+13,77,'Raw C0',ha='center',fontsize=10)
        ax.text(left+37,77,'Alternative order',ha='center',fontsize=10)
        before=['ABC','DEF','GHJ'];after=['CAB','DEF','GHJ'] if targetonly else ['CAB','FDE','JGH']
        for i,(b,a) in enumerate(zip(before,after)):
            y=69-i*8
            ax.text(left,y,'Target p' if i==0 else ['','Other u','Other v'][i],va='center',fontsize=9.5)
            blocks(left+8,y,b);blocks(left+31,y,a)
            ax.annotate('',xy=(left+29.5,y),xytext=(left+22,y),arrowprops=dict(arrowstyle='->',lw=.8,color=INK))
        ax.text(left+27,45,'Competitors fixed at raw C0' if targetonly else 'Targets and competitors change',ha='center',fontsize=9.5,color=INK)
    ax.text(0,37,'Intact attribute entries move; other text, the query and the encoder stay fixed.',fontsize=10)
    ax.text(0,31,'A single-target change replaces its old entry in a separate index.',fontsize=10)
    ax.plot([0,100],[27,27],color='#CBD1D5',lw=.7)
    ax.text(0,22,'B  Inclusion states over a finite order family',weight='bold',fontsize=11)
    for i,s in enumerate(['Order 1','Order 2','Order 3']):ax.text(40+13*i,16.5,s,ha='center',fontsize=9.5)
    labels=['Persistent inclusion','Cutoff crossing','Persistent omission'];values=[[1,1,1],[1,0,1],[0,0,0]]
    for row,(label,v) in enumerate(zip(labels,values)):
        y=12-row*4.7;ax.text(0,y,label,va='center',fontsize=10)
        for i,value in enumerate(v):
            ax.add_patch(Rectangle((37.5+13*i,y-2.05),5,4.1,facecolor=GAIN if value else 'white',edgecolor=GAIN if value else '#8C969E',lw=.65))
            ax.text(40+13*i,y,str(value),ha='center',va='center',color='white' if value else INK,fontsize=10)
    ax.text(77,6,'1: rank <= K\n0: rank > K',va='center',fontsize=10,linespacing=1.6)
    fig.savefig(FIG/'figure2_intervention_and_states.pdf')
    svg(fig,FIG/'figure2_intervention_and_states.svg','Interventions and inclusion states',
        'Schematic illustration, not empirical evidence. Catalog-wide changes reorder intact attribute blocks for every product. Single-target changes replace one target while competitors retain raw C0 representations. Three rows show 1,1,1 persistent inclusion; 1,0,1 cutoff crossing; and 0,0,0 persistent omission.')
    plt.close(fig)

def membership():
    data=pd.read_csv(DATA/'membership_changes_summary.csv')
    data=data[(data.population=='primary_existing_evaluation')&(data.aggregation=='query_macro')&(data.reference=='C0')].copy()
    data.to_csv(DATA/'rq2_membership_plot_data.csv',index=False)
    cols=['dataset','model','K','alternative','eligible_queries','queries_with_any_change','queries_with_exact_zero_net_and_changes',
          'within_query_cancellation_fraction','within_query_cancellation_conditional','cancellation_all_denominator','cancellation_conditional_denominator',
          'positive_query_net_mass','negative_query_net_mass','across_query_cancellation_mass']
    data[cols].to_csv(DATA/'rq2_cancellation_companion.csv',index=False)
    limit=7.0
    def draw(fig,axes,ks):
      for ik,k in enumerate(ks):
        for idata,ds in enumerate(['wands','esci']):
          for im,model in enumerate(MODELS):
            ax=axes[2*ik+idata,im];d=data[(data.dataset==ds)&(data.model==model)&(data.K==k)].set_index('alternative').loc[STEMS[1:]]
            y=np.arange(6)
            ax.barh(y,-100*d.loss_fraction,height=.62,color=LOSS,lw=0)
            ax.barh(y,100*d.gain_fraction,height=.62,color=GAIN,lw=0)
            ax.errorbar(100*d.net_delta,y,xerr=np.stack([100*(d.net_delta-d.net_delta_ci_low),100*(d.net_delta_ci_high-d.net_delta)]),
                        fmt='D',ms=3.8,color=INK,lw=.8,capsize=2,zorder=4)
            ax.axvline(0,color=INK,lw=.65);ax.set_xlim(-limit,limit);ax.set_yticks(y,STEMS[1:]);ax.invert_yaxis()
            ax.set_xticks([-6,-3,0,3,6]);ax.grid(axis='x',color='#E5E9ED',lw=.55);ax.set_axisbelow(True)
            ax.set_title(f'{ds.upper()} / {NAMES[model]} / K = {k}',loc='left',pad=8)
            ax.text(.0,-.19,f'{int(d.eligible_queries.iloc[0])} queries; {int(d.highest_relevance_pairs.iloc[0]):,} pairs',transform=ax.transAxes,fontsize=9)
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
      handles=[Patch(color=GAIN,label='Newly included'),Patch(color=LOSS,label='Newly omitted (negative)'),Line2D([0],[0],marker='D',color=INK,lw=.8,ms=4,label='Net Recall change + 95% CI')]
      fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,.995))
      fig.supxlabel('Query-macro fraction of highest-relevance products (percentage points)',y=.042,fontsize=10)
      fig.text(.5,.008,'Six raw schedules vs C0. Paired query-cluster intervals are descriptive and uncorrected.',ha='center',fontsize=9)
    with PdfPages(FIG/'rq2_membership_changes.pdf') as pdf:
      for k in [20,100]:
        fig,axes=plt.subplots(2,3,figsize=(7.1,5.55))
        fig.subplots_adjust(left=.065,right=.985,top=.86,bottom=.17,wspace=.29,hspace=.62)
        draw(fig,axes,[k]);pdf.savefig(fig)
        svg(fig,FIG/f'rq2_membership_changes_K{k}.svg',f'Membership changes at K={k}',
            'Matched newly included and newly omitted query-macro fractions for all six C0-relative raw attribute schedules. Black diamonds show net Recall change and descriptive 95% paired query-bootstrap intervals. WANDS: 308 eligible held-out queries, 21,299 highest-label pairs. ESCI: 499 eligible queries, 4,434 E pairs. All six panels use the same horizontal scale. Exact within-query cancellation denominators are in rq2_cancellation_companion.csv.')
        plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(7.1,10.6));fig.subplots_adjust(left=.065,right=.985,top=.93,bottom=.10,wspace=.29,hspace=.64)
    draw(fig,axes,[20,100]);svg(fig,FIG/'rq2_membership_changes.svg','Membership changes at K=20 and K=100',
       'Twelve panels show both cutoffs for WANDS and ESCI, with MiniLM, BGE and GTE encoders. Teal positive bars indicate newly included products, red negative bars newly omitted products, and black diamonds net Recall change. All measurements use identical query-macro support within a dataset. See accompanying CSVs for exact counts and per-query cancellation.')
    plt.close(fig)
    lines=['# Exact within-query cancellation','',
      'A query counts only when integer newly_included = newly_omitted > 0. The first denominator is all eligible queries; the second is queries with any membership change. Values below are query counts, not pair-micro rates. WANDS uses its existing held-out split; ESCI uses its existing evaluation sample.','',
      '| Dataset | Encoder | K | Alternative | Zero net, changed / all | Zero net, changed / changed |','|---|---|---:|---|---:|---:|']
    for r in data.itertuples():
        c=r.queries_with_exact_zero_net_and_changes;n=r.eligible_queries;m=r.queries_with_any_change
        lines.append(f'| {r.dataset.upper()} | {NAMES[r.model]} | {r.K} | {r.alternative} | {c}/{n} ({100*c/n:.2f}%) | {c}/{m} ({100*c/m:.2f}%) |')
    (HERE/'CANCELLATION_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def fitting():
    path=DATA/'inclusion_states_summary.csv'
    if not path.exists():return
    data=pd.read_csv(path);data=data[(data.population=='primary_existing_evaluation')&(data.intervention=='target_only')&(data.aggregation=='query_macro')]
    data.to_csv(DATA/'rq1_target_fully_fitting_plot_data.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.7));fig.subplots_adjust(left=.14,right=.985,top=.90,bottom=.17,hspace=.47,wspace=.50)
    for idata,ds in enumerate(['wands','esci']):
      for ik,k in enumerate([20,100]):
        ax=axes[idata,ik];ax.set_title(f'{ds.upper()} / K = {k}',loc='left',pad=9)
        labels=[];yy=[]
        for im,model in enumerate(MODELS):
          for iscope,scope in enumerate(['full','fully_fitting']):
            d=data[(data.dataset==ds)&(data.model==model)&(data.K==k)&(data.target_support==scope)].iloc[0]
            y=im*2.6+iscope;left=0
            for state,col in [('Always',GAIN),('VI',LOSS),('Never',MUTED)]:
              width=100*d[state];ax.barh(y,width,left=left,height=.72,color=col)
              if width>=10:ax.text(left+width/2,y,f'{width:.1f}',ha='center',va='center',color='white' if col!=MUTED else INK,fontsize=9)
              left+=width
            ax.text(101.5,y,f'{int(d.eligible_queries)}/{int(d.highest_relevance_pairs):,}',va='center',fontsize=8.5)
            labels.append(NAMES[model]+(' full' if scope=='full' else ' fitting'));yy.append(y)
        ax.set_yticks(yy,labels,fontsize=9);ax.set_xlim(0,131);ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.text(101.5,-1.1,'q / pairs',fontsize=9)
    fig.legend(handles=[Patch(color=GAIN,label='Persistent inclusion (Always)'),Patch(color=LOSS,label='Cutoff crossing (VI)'),Patch(color=MUTED,label='Persistent omission (Never)')],loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.supxlabel('Query-macro state share (%)',y=.065,fontsize=10)
    fig.text(.5,.015,'Targets filtered by all-seven-order token lengths; full raw C0 competitor catalogs remain fixed.',ha='center',fontsize=9)
    fig.savefig(FIG/'rq1_target_fully_fitting.pdf');svg(fig,FIG/'rq1_target_fully_fitting.svg','Target-only states on full and fully-fitting support',
      'Stacked bars show query-macro Always, VI and Never for full and encoder-specific fully-fitting target populations. The full C0 competitor catalog is retained. Query and highest-relevance pair denominators appear beside each bar. WANDS uses the existing held-out split and ESCI the existing evaluation sample. Full-minus-fitting differences are descriptive, not causal truncation effects.')
    plt.close(fig)

if __name__=='__main__':schematic();membership();fitting()
