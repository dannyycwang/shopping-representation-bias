"""Improve tick and label legibility without changing plotted data."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
ROOT=Path(__file__).resolve().parents[2]
d=pd.read_csv(ROOT/'phase3/results/phase3_tables/robustness_relevance_pareto.csv')
plt.rcParams.update({'font.size':6,'axes.titlesize':7,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5})
fig,axes=plt.subplots(1,2,figsize=(3.5,1.55))
for ax,ds,title in zip(axes,['wands','esci'],['WANDS held-out','ESCI union']):
    for method,label,marker in [('Original','Original','o'),('M1 factual','Factual','s'),('M2 canonical','Canonical','^'),('set_mean','Set mean','*')]:
        r=d[(d.dataset==ds)&(d.method==method)].iloc[0]
        x,y=r['cNDCG@10'],100*r['query_macro_VI@20']
        ax.scatter(x,y,s=20 if method=='set_mean' else 10,marker=marker,color='#0072B2' if method=='set_mean' else '#555555',zorder=3)
        offset=(-2,5) if method=='Original' else (2,5)
        align='right' if method=='Original' else 'left'
        if method=='M2 canonical' and ds=='esci': offset=(-2,5);align='right'
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=5.5,ha=align)
    ax.set_title(title);ax.set_xlabel('cNDCG@10');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylim(-1,15 if ds=='wands' else 5);ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.margins(x=.13)
axes[0].set_ylabel('Macro VI@20 (%)')
fig.tight_layout(pad=.5,w_pad=.8)
fig.savefig(ROOT/'paper_www2027/figures/figure4_pareto.pdf')
