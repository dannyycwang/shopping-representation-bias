"""Reproducible static figures. Selection is frozen in PROTOCOL, never by attribution magnitude."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import textwrap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
MODEL={'bge_base':'BGE','minilm':'MiniLM','gte_modernbert':'GTE'}

def save(fig,name):
 for ext in ['pdf','svg','png']:
  path=HERE/f'figures/{name}.{ext}';fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
  if ext=='svg':path.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in path.read_bytes().splitlines())+b'\n')
 plt.close(fig)

def case_figure(cases,name):
 from xai import spans
 p,q=load('wands');products={str(x['product_id']):x for x in p};queries=q.set_index('query_id')['query'].to_dict()
 entries=pd.read_parquet(HERE/'data/xai_entry_attributions.parquet');qa=pd.read_csv(HERE/'qa/xai_forward_completeness.csv',dtype={'product_id':str});panel=pd.read_csv(HERE/'XAI_CASE_MANIFEST.csv',dtype={'product_id':str})
 source=[];specs=[];allvalues=[]
 for role in cases:
  row=panel[panel.display.eq(role)].iloc[0];model,qid,pid=row.model,int(row.query_id),row.product_id
  native=pd.read_parquet(HERE/f'data/wands_{model}_boundary_records.parquet');native=native[(native.query_id==qid)&(native.product_id==pid)&(native.K==20)].set_index('schedule').loc[STEMS]
  z=qa[(qa.model==model)&(qa.query_id==qid)&(qa.product_id==pid)];ee=entries[(entries.model==model)&(entries.query_id==qid)&(entries.product_id==pid)&entries.baseline.eq('zero')]
  product=products[pid];ids=[f'attr_{i:03d}' for i in range(min(8,len(product['attributes'])))];labels=[f'{i+1}. '+textwrap.shorten(product['attributes'][i],width=36,placeholder='…') for i in range(len(ids))]+['Other content (complete sum)']
  mat=[];orders=[]
  for variant in ['minimum','maximum']:
   zz=z[z.variant.eq(variant)&z.baseline.eq('zero')].iloc[0];gg=ee[ee.variant.eq(variant)].groupby('entry_id').attribution.sum()
   values=[float(gg.get(i,0)) for i in ids]+[float(gg[~gg.index.isin(ids)].sum())];mat.append(values)
   text,parts=spans(product,zz.schedule);order=[int(a[2][5:])+1 for a in parts if a[2] in ids];orders.append(' → '.join(map(str,order)))
   source.append(dict(display=role,model=model,query_id=qid,product_id=pid,variant=variant,schedule=zz.schedule,query=queries[qid],full_text=text,serializer_spans=parts,display_entry_ids=ids+['other_content'],display_values=values,baseline='zero',complete_original_attributes=product['attributes'],native_margin=float(zz.native_margin),native_rank=int(zz.native_rank),completeness_passed=bool(zz.completeness_passed),boundary_guard=bool(zz.variant_boundary_guard)))
  mat=np.array(mat).T;allvalues.extend(mat.ravel());specs.append((row,queries[qid],native,z,mat,labels,orders))
 vmax=max(abs(np.array(allvalues)));fig=plt.figure(figsize=(11.8,4.9*len(cases)));gs=GridSpec(len(cases),3,figure=fig,width_ratios=[1.45,.65,1.45],hspace=.85,wspace=.3)
 for n,(row,query,native,z,mat,labels,orders) in enumerate(specs):
  left=fig.add_subplot(gs[n,0]);left.axis('off')
  nativefull=bool(native.fully_fitting.all());title=('Crossing case' if row.display=='main_crossing' else 'Stable included control' if row.display=='main_stable_control' else 'Historical illustration')
  left.set_title(f'{chr(65+n)}  {title} · {MODEL[row.model]}\nq{int(row.query_id)} “{query}”\nproduct {row.product_id}',loc='left',fontsize=12,fontweight='bold',pad=25)
  for i,label in enumerate(labels):left.text(0,1-(i+.5)/len(labels),label,va='center',fontsize=11)
  left.text(0,-.1,'Shown IDs in the two input orders:\nmin: '+orders[0]+'\nmax: '+orders[1],va='top',fontsize=10.5)
  heat=fig.add_subplot(gs[n,1]);im=heat.imshow(mat,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
  minrow=z[z.variant.eq('minimum')&z.baseline.eq('zero')].iloc[0];maxrow=z[z.variant.eq('maximum')&z.baseline.eq('zero')].iloc[0]
  heat.set_xticks([0,1],['min\n'+minrow.schedule,'max\n'+maxrow.schedule]);heat.set_yticks([]);heat.tick_params(length=0)
  for i in range(len(mat)):
   for j in range(2):heat.text(j,i,f'{mat[i,j]:+.3f}',ha='center',va='center',fontsize=11,color='white' if abs(mat[i,j])>.65*vmax else '#17222c')
  heat.set_title('Entry IG · zero baseline\nshared signed scale',fontsize=11,pad=25)
  if not z.variant_boundary_guard.all():warning='Precision guard failed;\ncrossing explanation withheld.'
  elif not z.completeness_passed.all():warning='Completeness failed at 256;\nprimary IG is inconclusive.'
  else:warning='Forward/completeness pass.\nBaseline-dependent diagnostic.'
  heat.text(.5,-.30,warning,transform=heat.transAxes,ha='center',va='top',fontsize=10.5,color='#923923' if 'failed' in warning else '#414a55')
  ax=fig.add_subplot(gs[n,2]);x=np.arange(7);colors=np.where(native.inclusion,'#187f75','#a64f3d')
  ax.axhline(0,color='#777',lw=.8);ax.vlines(x,0,native.signed_margin,colors=colors,alpha=.45);ax.scatter(x,native.signed_margin,c=colors,s=30,zorder=3)
  for j,(s,rr) in enumerate(native.iterrows()):ax.annotate(str(int(rr.saved_rank)),(j,rr.signed_margin),xytext=(0,8 if rr.signed_margin>=0 else -16),textcoords='offset points',ha='center',fontsize=11)
  for variant,mark in [('minimum','v'),('maximum','s')]:
   zz=z[z.variant.eq(variant)].iloc[0];j=STEMS.index(zz.schedule);ax.scatter(j,native.loc[zz.schedule,'signed_margin'],s=90,marker=mark,facecolors='none',edgecolors='#151c24',zorder=4)
  ax.set_xticks(x,['C0','C1','s1','s2','s3','s4','s5']);ax.set_ylabel('Native score − competitor threshold');ax.set_title('7 native orders · labels = ranks',fontsize=11,pad=25);ax.margins(y=.29)
  ax.yaxis.set_label_position('right')
  tok_label=str(int(native.token_length.max())) if native.token_length.nunique()==1 else f'{int(native.token_length.min())}–{int(native.token_length.max())}'
  ax.text(0,-.22,f'{tok_label} tokens / {int(native.tokenizer_cap.iloc[0])} cap; all 7 fit.\nK=20; 42,993 competitors fixed at C0.\nGreen: in; brown: out. s1–s5 = C2s1–C2s5.',transform=ax.transAxes,va='top',fontsize=10.5)
  source.extend(native.reset_index().assign(record_type='native_margin',display=row.display).to_dict('records'))
 fig.subplots_adjust(bottom=.19,top=.9)
 dump(HERE/f'figures/{name}_source.json',source);save(fig,name)

def coverage():
 f=pd.read_csv(HERE/'tables/coverage_summary.csv');plt.rcParams.update({'font.size':8});fig,axes=plt.subplots(4,2,figsize=(7.2,9.2),sharey='row')
 for row,(ds,model) in enumerate([(ds,m) for ds in ['wands','esci'] for m in ['minilm','bge_base']]):
  for col,k in enumerate([20,100]):
   ax=axes[row,col]
   for scope,color,marker in [('full','#226f98','o'),('common_all_fitting','#ae5d30','s')]:
    z=f[(f.dataset==ds)&f.model.eq(model)&f.K.eq(k)&f.support.eq(scope)&f.metric.eq('VI')&f.aggregation.eq('query_macro')].copy()
    names=['7','16','32'] if ds=='wands' else ['7','exhaustive_target'];z=z.set_index('family').loc[names];xx=np.array([7,16,32]) if ds=='wands' else np.arange(len(z))
    ax.errorbar(xx,z['mean']*100,yerr=[(z['mean']-z.ci_low)*100,(z.ci_high-z['mean'])*100],color=color,marker=marker,lw=1.2,capsize=3,label=f'{scope.replace("common_all_fitting","common fit")} ({int(z.queries.iloc[0])}q; {int(z.pairs.iloc[0])} pairs)')
   ax.set_xticks(xx,['7','16','32'] if ds=='wands' else ['7','all target\norders']);ax.set_title(f'{ds.upper()} · {MODEL[model]} · K={k}');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7,loc='best')
   if col==0:ax.set_ylabel('Crossing mass (query macro, %)')
 fig.suptitle('Fixed-competitor target-only coverage\n95% uncorrected query-cluster intervals',fontsize=11);fig.tight_layout(rect=[0,0,1,.94]);save(fig,'supplement_permutation_coverage')
 f.to_csv(HERE/'figures/supplement_permutation_coverage_source.csv',index=False)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--coverage-only',action='store_true');ap.add_argument('--cases-only',action='store_true');args=ap.parse_args()
 if not args.coverage_only:
  case_figure(['main_crossing','main_stable_control'],'main_query_conditioned_cases')
  case_figure(['supplement_historical'],'supplement_french_molding')
 if not args.cases_only:coverage()
 print('FIGURES COMPLETE',flush=True)
