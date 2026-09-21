"""Vector figures at 7.1-inch ACM width, using the inherited Libertine font."""
from common import *
import shutil, xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text

FIG=HERE/'figures';SOURCE=OLD/'figure_revision';GEOMETRY=[];CAPTIONS={}
for path in (SOURCE/'assets/fonts').glob('*.ttf'):fm.fontManager.addfont(str(path))
FAMILY=fm.FontProperties(fname=str(SOURCE/'assets/fonts/ManuscriptLibertine-R.ttf')).get_name()
plt.rcParams.update({'font.family':FAMILY,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'completion-20260921','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'savefig.facecolor':'white'})
INK,GAIN,LOSS='#202832','#187C80','#B24C36'
NAMES={'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE'}

def export(fig,name,caption,description):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    assert not outside,(name,outside)
    minimum=min(t.get_fontsize() for t in texts);assert minimum>=8
    fig.savefig(FIG/f'{name}.pdf',metadata={'Title':name,'Subject':description,'CreationDate':None,'ModDate':None})
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(HERE/f'qa/{name}.png',dpi=130)
    ns='{http://www.w3.org/2000/svg}';ET.register_namespace('',ns[1:-1]);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree=ET.parse(FIG/f'{name}.svg');root=tree.getroot()
    title=ET.Element(ns+'title');title.text=name;desc=ET.Element(ns+'desc');desc.text=description
    root.insert(0,desc);root.insert(0,title);root.set('role','img');tree.write(FIG/f'{name}.svg',encoding='utf8',xml_declaration=True)
    CAPTIONS[name]=dict(caption=caption,ACM_Description=description,command='python revision_completion_20260921/scripts/figures.py',width_inches=7.1)
    GEOMETRY.append(dict(name=name,width_inches=float(fig.get_figwidth()),height_inches=float(fig.get_figheight()),minimum_font_pt=minimum,text_outside_page=outside))
    plt.close(fig)

def rq2(k):
    path=SOURCE/f'data/rq2_top{k}.csv';data=pd.read_csv(path);shutil.copyfile(path,FIG/f'rq2_membership_top{k}_plot.csv')
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.4));fig.subplots_adjust(left=.07,right=.985,bottom=.12,top=.845,wspace=.32,hspace=.48)
    for i,ds in enumerate(['wands','esci']):
      for j,model in enumerate(MODELS):
        ax=axes[i,j];d=data[(data.dataset==ds)&(data.model==model)].set_index('alternative').loc[STEMS[1:]];y=np.arange(6)
        ax.barh(y,-100*d.loss_fraction,height=.63,color=LOSS,lw=0);ax.barh(y,100*d.gain_fraction,height=.63,color=GAIN,lw=0)
        ax.errorbar(100*d.net_delta,y,xerr=np.stack([100*(d.net_delta-d.net_delta_ci_low),100*(d.net_delta_ci_high-d.net_delta)]),fmt='D',ms=3.5,color='black',lw=.85,capsize=2,zorder=4)
        ax.axvline(0,color=INK,lw=.65);ax.set(xlim=(-7,7),ylim=(5.65,-.65));ax.set_yticks(y,STEMS[1:]);ax.set_xticks([-6,-3,0,3,6]);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
        ax.set_title(f'{ds.upper()} / {NAMES[model]}',loc='left',weight='bold',pad=6);ax.grid(axis='x',color='#E3E7EA',lw=.5);ax.set_axisbelow(True)
    fig.text(.07,.97,f'Top-{k}',weight='bold',fontsize=10)
    handles=[Patch(color=LOSS,label='Newly omitted'),Patch(color=GAIN,label='Newly included'),Line2D([],[],color='black',marker='D',ms=3.5,lw=.8,label='ΔRecall (pp), 95% CI')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,1),ncol=3,frameon=False,fontsize=8.8,handlelength=1.5,columnspacing=1.2)
    fig.text(.5,.025,'Relevant-product share (%) and ΔRecall (percentage points)',ha='center',fontsize=9)
    name=f'rq2_membership_top{k}_{"main" if k==20 else "appendix"}'
    export(fig,name,f'Catalog-wide membership changes relative to source order at K={k}. Left/right bars are newly omitted/included fractions of each complete Hq, averaged across queries. Diamonds and 95% paired intervals are net Recall. All six comparisons and six panels are retained; values and intervals are unchanged from the historical figure.',f'Six panels show WANDS and ESCI with MiniLM, BGE and GTE. Newly omitted shares extend left and newly included shares extend right of zero. Black diamonds and intervals represent net Recall. C1 and five random schedules are each compared with C0. Interval bars do not describe the colored bars.')

def cutoff(intervention):
    data=pd.read_csv(DATA/'cutoff_summary.csv');data=data[(data.aggregation=='query_macro')&(data.intervention==intervention)]
    name=f'cutoff_{intervention}_appendix';data.to_csv(FIG/f'{name}_plot.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.6));fig.subplots_adjust(left=.085,right=.985,top=.80,bottom=.13,wspace=.28,hspace=.47)
    for i,ds in enumerate(['wands','esci']):
      for j,model in enumerate(MODELS):
        ax=axes[i,j]
        for support,linestyle,marker in [('full','-','o'),('fully_fitting','--','^')]:
          d=data[(data.dataset==ds)&(data.model==model)&(data.target_support==support)].sort_values('K')
          for col,color in [('persistent_inclusion',GAIN),('VI',LOSS)]:
            ax.plot(d.K,100*d[col],ls=linestyle,marker=marker,ms=3,color=color,lw=1)
            ax.errorbar(d.K,100*d[col],yerr=np.stack([100*(d[col]-d[col+'_ci_low']),100*(d[col+'_ci_high']-d[col])]),ls='none',color=color,lw=.5,capsize=1,alpha=.65)
        ax.set(xscale='log',ylim=(0,102));ax.set_xticks([20,100,1000],['20','100','1000']);ax.set_yticks([0,25,50,75,100]);ax.set_title(f'{ds.upper()} / {NAMES[model]}',loc='left',weight='bold');ax.grid(alpha=.2)
        if j==0:ax.set_ylabel('Target share (%)')
        if i==1:ax.set_xlabel('Cutoff K (log scale)')
    handles=[Line2D([],[],color=GAIN,label='Persistent inclusion'),Line2D([],[],color=LOSS,label='VI'),Line2D([],[],color=INK,ls='-',marker='o',ms=3,label='Full'),Line2D([],[],color=INK,ls='--',marker='^',ms=3,label='Fully-fitting')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,frameon=False,fontsize=8.5,columnspacing=1)
    fig.text(.5,.905,('Catalog-wide: all products reordered' if intervention=='catalog_wide' else 'Target-only: every competitor fixed at raw C0'),ha='center',fontsize=10)
    fig.text(.5,.025,'Fixed support across K within each curve; GTE full and fitting curves coincide.',ha='center',fontsize=8.5)
    export(fig,name,f'{intervention.replace("_","-")} cutoff sensitivity at K=20,50,100,200,500,1000. Query-macro persistent inclusion and VI use fixed full or encoder-specific fully-fitting targets and complete unchanged competitor catalogs. Error bars are 95% query-cluster intervals with common paired resamples. Persistent omission and paired cutoff changes are in the data tables. VI need not decrease with K.',f'Six panels compare full solid curves and fully-fitting dashed curves for WANDS and ESCI and three encoders. Green curves show persistent inclusion and rust curves show VI. Cutoffs are on a logarithmic horizontal axis. Full and fitting GTE curves overlap. The target-only curves describe separate counterfactual indexes, not Recall from one shared index.')

def severity():
    all_data=pd.read_csv(DATA/'severity_summary.csv');d=all_data[(all_data.aggregation=='query_macro')&(all_data.event=='crossing')]
    d.to_csv(FIG/'crossing_severity_appendix_plot.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.9));fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.12,wspace=.27,hspace=.55)
    colors={2:GAIN,5:LOSS}
    for i,ds in enumerate(['wands','esci']):
      for j,k in enumerate([20,100]):
        ax=axes[i,j]
        for m,model in enumerate(MODELS):
          for s,support in enumerate(['full','fully_fitting']):
            for t,mult in enumerate([2,5]):
                r=d[(d.dataset==ds)&(d.model==model)&(d.K==k)&(d.target_support==support)&(d.threshold_multiple==mult)].iloc[0]
                x=m+(s-.5)*.28+(t-.5)*.10
                ax.errorbar(x,100*r.severe_share,yerr=[[100*(r.severe_share-r.severe_share_ci_low)],[100*(r.severe_share_ci_high-r.severe_share)]],fmt='o' if s==0 else '^',color=colors[mult],mfc=colors[mult] if s==0 else 'white',ms=4,capsize=2,lw=.7)
        ax.set_xticks(range(3),[NAMES[x] for x in MODELS]);ax.set_xlim(-.45,2.45);ax.set_ylim(bottom=-.035*ax.get_ylim()[1]);ax.set_yticks([v for v in ax.get_yticks() if 0<=v<=ax.get_ylim()[1]]);ax.set_title(f'{ds.upper()} / K={k}',loc='left',weight='bold');ax.grid(axis='y',alpha=.2)
        if j==0:ax.set_ylabel('All eligible target share (%)')
    handles=[Line2D([],[],color=GAIN,marker='o',ls='none',label='Crossing, worst > 2K'),Line2D([],[],color=LOSS,marker='o',ls='none',label='Crossing, worst > 5K'),Line2D([],[],color=INK,marker='o',ls='none',label='Full'),Line2D([],[],color=INK,marker='^',mfc='white',ls='none',label='Fully-fitting')]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=8.5)
    fig.text(.5,.025,'Target-only; query-macro fractions of complete Hq on each stated support.',ha='center',fontsize=8.5)
    export(fig,'crossing_severity_appendix','Target-only cutoff crossings with worst rank beyond 2K or 5K. Fractions first divide by all eligible targets within each query, then average queries. Full and fully-fitting supports are separate; competitors are never filtered. Error bars are 95% paired query-cluster intervals. Conditional crossing-query proportions, pair counts and source-order-directed losses are supplied separately.','Four panels cross WANDS and ESCI with K=20 and K=100. Teal points are crossing targets with worst rank above twice K, rust points above five times K. Filled circles denote full targets; open triangles denote fully-fitting targets. Vertical scales differ between panels. Some ESCI estimates are exactly zero.')

def ecdf():
    d=pd.read_csv(DATA/'crossing_ecdf.csv');d=d[(d.target_support=='full')&(d.variable=='worst_rank')];d.to_csv(FIG/'crossing_worst_rank_ecdf_appendix_plot.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.4));fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.14,wspace=.3,hspace=.48)
    for i,ds in enumerate(['wands','esci']):
      for j,model in enumerate(MODELS):
        ax=axes[i,j]
        for k,color in [(20,GAIN),(100,LOSS)]:
            g=d[(d.dataset==ds)&(d.model==model)&(d.K==k)]
            ax.step(g['rank'],g.query_balanced_ecdf,where='post',color=color,label=f'K={k}')
        ax.set(xscale='log',xlim=(20,50000 if ds=='wands' else 12000),ylim=(0,1.02));ax.set_xticks([100,1000,10000],['100','1k','10k']);ax.set_title(f'{ds.upper()} / {NAMES[model]}',loc='left',weight='bold');ax.grid(alpha=.2)
        if j==0:ax.set_ylabel('Conditional cumulative share')
        if i==1:ax.set_xlabel('Worst rank (log scale)')
    fig.legend([Line2D([],[],color=GAIN),Line2D([],[],color=LOSS)],['K=20','K=100'],loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.025,'Target-only crossings; equal weight per crossing query, then per crossing product.',ha='center',fontsize=8.5)
    export(fig,'crossing_worst_rank_ecdf_appendix','Worst-rank ECDF among target-only crossing pairs at K=20 and 100, full primary populations. Crossing means best <= K < worst. Each query with at least one crossing has equal weight and each crossing product within that query has equal weight. This conditional weighting differs from fractions of all Hq and from pair-micro ECDFs. Fully-fitting and rank-range distributions are in the machine-readable data.','Six panels show conditional worst-rank ECDFs for each dataset and encoder. K=20 is teal and K=100 is rust. The horizontal rank scale is logarithmic and panel ranges differ. Curves represent full target supports only, with crossing-query weighting.')

def main():
    oldcap=read(SOURCE/'data/captions.json')
    for ext in ['pdf','svg']:shutil.copyfile(SOURCE/f'figures/rq1_vi_top20_main.{ext}',FIG/f'rq1_vi_top20_main.{ext}')
    shutil.copyfile(SOURCE/'data/rq1_main_top20.csv',FIG/'rq1_vi_top20_main_plot.csv')
    CAPTIONS['rq1_vi_top20_main']=dict(caption=oldcap['rq1_vi_top20_main']['full'],ACM_Description=oldcap['rq1_vi_top20_main']['description'],command='byte-for-byte reuse; python revision_completion_20260921/scripts/figures.py',width_inches=7.1)
    GEOMETRY.append(dict(name='rq1_vi_top20_main',width_inches=7.1,height_inches=3.25,minimum_font_pt=8.5,text_outside_page=[],reused_byte_identical=True))
    rq2(20);rq2(100);cutoff('catalog_wide');cutoff('target_only');severity();ecdf()
    dump(FIG/'captions.json',CAPTIONS);dump(HERE/'qa/figure_geometry.json',GEOMETRY)
    (FIG/'CAPTIONS_AND_ACCESSIBILITY.md').write_text('\n\n'.join(f'## {name}\n\n{r["caption"]}\n\nACM Description: {r["ACM_Description"]}\n\nCommand: `{r["command"]}`. Width: 7.1 inches.' for name,r in CAPTIONS.items()),encoding='utf8')

if __name__=='__main__':main()
