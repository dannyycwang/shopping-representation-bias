from common5 import *
import matplotlib.pyplot as plt

paper=ROOT/'paper_www2027';gen=paper/'generated';fig=paper/'figures';gen.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
s=pd.read_csv(OUT/'optimization_robustness_summary.csv');fact=pd.read_csv(OUT/'factuality_cost.csv');contr=pd.read_csv(OUT/'primary_contrasts.csv')
labels={'Raw':'Raw','Canonical':'Canonical','G1':'Heuristic','CG1':r'C$\to$Heuristic','G2':'Optimized','CG2':r'C$\to$Optimized'};order=list(labels)
def pct(x):return f'{100*x:.1f}'
rows=[]
for m in order:
 cells=[labels[m]]
 for ds in ['wands','esci']:
  r=s[(s.dataset==ds)&(s.model=='bge_base')&(s.method==m)].iloc[0];cells += [pct(r['inclusion@100_macro']),pct(r['VI@100_macro']),pct(r['robust_coverage@100_macro']),pct(r['never@100']/r['pairs'])]
 rows.append(cells)
tex='''\\begin{table*}[t]
\\caption{Primary target-only BGE results at $K=100$. Mean inclusion, VI, and robust coverage (included under every tested order) are query-macro percentages; Never is pair-micro. This is a family of single-target counterfactual indexes, not one catalog Recall.}
\\label{tab:optimization}\\centering\\small
\\begin{tabular}{lrrrrrrrr}\\toprule
&\\multicolumn{4}{c}{WANDS}&\\multicolumn{4}{c}{ESCI}\\\\
Method&Mean&VI&Robust&Never&Mean&VI&Robust&Never\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}
\\end{table*}
''';(gen/'optimization_main.tex').write_text(tex,encoding='utf8')

# BGE quality-versus-sensitivity plot. Coordinates come directly from summary CSV.
figr,axes=plt.subplots(1,2,figsize=(7.05,2.55),sharex=False)
markers=['o','s','^','v','D','P']
for ax,ds,title in zip(axes,['wands','esci'],['WANDS','ESCI']):
 for m,mk in zip(order,markers):
  r=s[(s.dataset==ds)&(s.model=='bge_base')&(s.method==m)].iloc[0];ax.scatter(100*r['VI@100_macro'],100*r['inclusion@100_macro'],marker=mk,s=38,label=labels[m].replace('$\\to$','→'))
  ax.annotate(labels[m].replace('$\\to$','→'),(100*r['VI@100_macro'],100*r['inclusion@100_macro']),xytext=(3,3),textcoords='offset points',fontsize=6.5)
 ax.set_title(title,fontsize=9);ax.set_xlabel('Order sensitivity: macro VI@100 (%)',fontsize=8);ax.set_ylabel('Mean target inclusion@100 (%)',fontsize=8);ax.grid(alpha=.25);ax.tick_params(labelsize=7)
figr.tight_layout();figr.savefig(fig/'optimization_quality_robustness.pdf',bbox_inches='tight');plt.close(figr)

def interval(r):return f"{100*r.delta:+.1f} [{100*r.ci_low:+.1f},{100*r.ci_high:+.1f}]"
rows=[]
for ds in ['wands','esci']:
 cells=[ds.upper()]
 for a,b in [('G1','Raw'),('CG1','G1'),('CG1','Canonical'),('G2','Raw'),('CG2','G2'),('CG2','Canonical')]:cells.append(interval(contr[(contr.dataset==ds)&(contr.model=='bge_base')&(contr.method_a==a)&(contr.method_b==b)].iloc[0]))
 rows.append(cells)
(gen/'optimization_contrasts.tex').write_text('''\\begin{table*}[t]
\\caption{Prespecified BGE contrasts in query-macro target inclusion@100 (percentage points; paired query-bootstrap 95\\% CI). Holm-adjusted randomization results are in the artifact. Intervals crossing zero do not establish equivalence.}
\\label{tab:optcontrasts}\\centering\\scriptsize
\\begin{tabular}{lrrrrrr}\\toprule
&$G_1-$Raw&$C\\!\\to\\!G_1-G_1$&$C\\!\\to\\!G_1-C$&$G_2-$Raw&$C\\!\\to\\!G_2-G_2$&$C\\!\\to\\!G_2-C$\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table*}
''',encoding='utf8')

cost=pd.read_csv(OUT/'factuality_cost.csv');common=pd.read_csv(OUT/'common_valid_support.csv');rows=[]
for r in cost.itertuples():rows.append([r.dataset.upper(),r.method,str(int(r.outputs)),pct(r.fact_pass_rate),pct(r.execution_fallback_rate),pct(r.guarded_fallback_rate),f'{r.mean_output_tokens:.0f}',f'{r.generation_seconds/60:.1f}'])
(gen/'optimization_factuality.tex').write_text('''\\begin{table}[t]\\caption{Adapted-generator factuality and local cost. Strict pass uses the frozen rule checker; Exec. and Guard are fallback rates. Minutes are unique cached batched GPU generation time; monetary API cost is zero.}\\label{tab:optfacts}\\centering\\scriptsize
\\begin{tabular}{llrrrrrr}\\toprule Data&Method&$n$&Pass\\%&Exec.\\%&Guard\\%&Tok.&min\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table}
''',encoding='utf8')

noise=pd.read_csv(OUT/'generation_noise_order_effect.csv');same=pd.read_csv(OUT/'generation_noise_same_input.csv');rows=[]
if len(noise):
 for (ds,m),g in noise.groupby(['dataset','method']):
  c0=same[(same.dataset==ds)&(same.method==m)&(same.input=='C0')];can=same[(same.dataset==ds)&(same.method==m)&(same.input=='canonical')]
  c0_cross=c0['replicate_crossing@100'].iloc[0];can_cross=can['replicate_crossing@100'].iloc[0]
  rows.append([ds.upper(),m,'/'.join(f'{100*x:.1f}' for x in g['VI@100_micro']),f'{100*c0_cross:.1f}',f'{100*can_cross:.1f}'])
(gen/'optimization_noise.tex').write_text('''\\begin{table}[t]\\caption{Stochastic generation control. Order VI is computed separately within each of three replicates; same-input columns vary only generation. Percentages are pair-micro on the frozen eight-product control.}\\label{tab:optnoise}\\centering\\scriptsize
\\begin{tabular}{llccc}\\toprule Data&Method&Order VI reps.&C0 repl. VI&Canonical repl. VI\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table}
''',encoding='utf8')

o=pd.read_csv(OUT/'per_order_inclusion.csv');rows=[]
for (ds,model,method),g in o[(o.K==100)&o.method.isin(order)].groupby(['dataset','model','method']):rows.append([ds.upper(),model.replace('_',r'\_'),labels[method],*[pct(g.set_index('stem').loc[x,'query_macro']) for x in STEMS]])
(gen/'optimization_orders.tex').write_text('''\\begin{table*}[t]\\caption{Query-macro target inclusion@100 for every tested order. Canonical and canonical-first rows repeat by construction.}\\label{tab:optorders}\\centering\\scriptsize
\\begin{tabular}{lllrrrrrrr}\\toprule Data&Encoder&Method&C0&C1&C2-1&C2-2&C2-3&C2-4&C2-5\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table*}
''',encoding='utf8')

catalog=pd.read_csv(OUT/'canonical_catalog_contrasts.csv');rows=[]
for r in catalog.itertuples():rows.append([r.dataset.upper(),r.model.replace('_',r'\_'), r.metric, interval(r)])
(gen/'canonical_catalog.tex').write_text('''\\begin{table}[t]\\caption{Pure raw-atom canonical sorting versus Original in full-catalog query-macro metrics (percentage points; paired query-bootstrap 95\\% CI). This is deployable catalog Recall, unlike target-intervention inclusion.}\\label{tab:cancatalog}\\centering\\scriptsize
\\begin{tabular}{lllr}\\toprule Data&Encoder&Metric&$\\Delta$ [95\\% CI]\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table}
''',encoding='utf8')

rows=[]
for ds in ['wands','esci']:
 for m in order:
  r=s[(s.dataset==ds)&(s.model=='minilm')&(s.method==m)].iloc[0]
  rows.append([ds.upper(),labels[m],pct(r['inclusion@100_macro']),pct(r['VI@100_macro']),pct(r['robust_coverage@100_macro'])])
(gen/'optimization_transfer.tex').write_text('''\\begin{table}[t]\\caption{Unchanged generated texts transferred to MiniLM. Values are query-macro target inclusion, VI, and robust coverage at 100 (\\%).}\\label{tab:opttransfer}\\centering\\scriptsize
\\begin{tabular}{llrrr}\\toprule Data&Method&Mean&VI&Robust\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table}
''',encoding='utf8')

support=pd.read_csv(OUT/'common_valid_support.csv');valid=pd.read_csv(OUT/'common_valid_metrics.csv');rows=[]
for r in support.itertuples():
 vals=valid[(valid.dataset==r.dataset)&(valid.optimizer==r.method)&(valid.K==100)&(valid.model=='bge_base')]
 def val(m):
  z=vals[vals.method==m];return '--' if z.empty or pd.isna(z.macro_inclusion.iloc[0]) else pct(z.macro_inclusion.iloc[0])
 rows.append([r.dataset.upper(),r.method,str(int(r.all_seven_fact_valid_products)),pct(r.all_seven_fact_valid_rate),val('Raw'),val(r.method),val('C'+r.method)])
(gen/'optimization_valid.tex').write_text('''\\begin{table}[t]\\caption{Common fact-valid support for all seven generated orders. Inclusion values are BGE query-macro at 100 on that common support; empty support is shown explicitly. Automated validity is conservative and is not human ground truth.}\\label{tab:optvalid}\\centering\\scriptsize
\\begin{tabular}{llrrrrr}\\toprule Data&Opt.&Products&Rate&Raw&G&C$\\to$G\\\\\\midrule
'''+''.join(' & '.join(x)+r'\\'+'\n' for x in rows)+'''\\bottomrule\\end{tabular}\\end{table}
''',encoding='utf8')
print('presentation artifacts written')
