from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'paper_www2027/figures'
d=pd.read_parquet(ROOT/'phase3/results/target_only_permutations/wands_minilm_pairs.parquet'); r=d[(d.query_id==162)&(d.product_id==34536)].iloc[0]
assert r.C2s4==5 and r.C2s1==1527
pd.DataFrame([{'query':'turquoise chair','query_id':162,'product_id':34536,'label':'Exact','model':'minilm','competitors':'all C0','representation':k,'rank':int(r[k])} for k in ['C0','C1','C2s1','C2s2','C2s3','C2s4','C2s5']]).to_csv(OUT/'figure1_target_only_source.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.35,2.05));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.5,.95,'Query: "turquoise chair"',ha='center',fontsize=10,weight='bold')
ax.text(.5,.82,'Same Exact-relevant product and facts',ha='center',fontsize=8)
ax.text(.5,.72,'Same model • Every competitor fixed in C0',ha='center',fontsize=7.5)
for x,col,label,rank,status in [(.03,'#0072B2','Order A (C2s4)',5,'VISIBLE at Top-20'),(.53,'#C65D00','Order B (C2s1)',1527,'HIDDEN at Top-20')]:
 ax.add_patch(FancyBboxPatch((x,.20),.44,.42,boxstyle='round,pad=0.01',facecolor='#f5f5f5',edgecolor=col,lw=1))
 ax.text(x+.22,.54,label,ha='center',fontsize=8)
 ax.text(x+.22,.39,f'Rank {rank:,}',ha='center',fontsize=13,weight='bold',color=col)
 ax.text(x+.22,.26,status,ha='center',fontsize=7,color=col)
ax.text(.5,.10,'Only the target attribute order changes',ha='center',fontsize=7.5)
ax.text(.5,.01,'Should serialization determine visibility?',ha='center',fontsize=8,weight='bold')
fig.savefig(OUT/'figure1_target_only.pdf',bbox_inches='tight');fig.savefig(OUT/'figure1_target_only.png',dpi=300,bbox_inches='tight');plt.close(fig)
