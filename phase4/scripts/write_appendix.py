from common import *
PAPER=ROOT/'paper_www2027';GEN=PAPER/'generated';S=PAPER/'sections'
assert (OUT/'queue_completed.json').exists()
selected=json.loads((P4/'config/selection.json').read_text())['method']
labels={'rule_type':'Type-first','rule_appearance':'Appearance-first','rule_use':'Use-first','rule_dimension':'Dimension-first','Original':'Original','hybrid':'Hybrid','canonical':'Canonical','factual':'Factual','set_mean':'Set mean','late_mean':'Late mean','late_max':'Late max','late_top3':'Late top3'}
def name(m):return labels.get(m,m.replace('_',' '))
def table(key,caption,cols,headers,rows,wide=False):
    env='table*' if wide else 'table';position='tp' if wide else 'htbp';s=f'\\begin{{{env}}}[{position}]\n\\caption{{{caption}}}\\label{{tab:{key}}}\n\\centering\\small\n\\begin{{tabular}}{{{cols}}}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'
    s+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+f'\n\\bottomrule\\end{{tabular}}\n\\end{{{env}}}\n';(GEN/f'{key}.tex').write_text(s,encoding='utf8')

f=pd.read_csv(OUT/'development_selection.csv');rows=[[name(r.method),f'{r.recall:.4f}',f'{r.cndcg:.4f}',int(r.vectors)] for r in f.itertuples()]
table('screening','All eight added configurations on the frozen WANDS development queries. Sorting reflects the prespecified selection rule, not test performance.','lrrr',['Strategy','R100','cN@10','Vectors'],rows)
f=pd.read_csv(P2/'results/phase2_tables/table1_representation_sensitivity_native.csv');target=pd.read_csv(ROOT/'phase3/results/target_only_permutations/summary.csv');rows=[]
for r in f.to_dict('records'):
    t=target[(target.dataset==r['dataset'])&(target.retriever==r['retriever'])].iloc[0]
    rows.append([r['dataset'].upper(),{'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE'}.get(r['retriever'],r['retriever']),f"{100*r['VI@20_micro']:.2f}",f"{100*r['VI@20_query_macro']:.2f}",f"{100*t['VI@20_micro']:.2f}",f"{100*t['VI@20_query_macro']:.2f}"])
table('sensitivitydetail','Primary seven-order VI@20 percentages. Catalog-wide and target-only interventions are distinct. Micro weights pairs; macro weights eligible queries. Query-bootstrap intervals remain in the corresponding source tables and main visualization.','llrrrr',['Data','Encoder','Catalog micro','Catalog macro','Target micro','Target macro'],rows,True)
f=pd.read_csv(OUT/'saturation_summary.csv');rows=[]
for ds in ['wands','esci']:
    for m in [2,4,7]:
        for family in ['centroid','max']:
            r=f[(f.dataset==ds)&(f.method==f'{family}_{m}')&(f.metric=='Recall@100')].iloc[0]
            rows.append([ds.upper(),f'{family} {m}',f'{r["mean"]:.3f}',f'{100*r.delta_vs_Original:+.2f}',f'[{100*r.ci_low:+.2f},{100*r.ci_high:+.2f}]'])
table('saturation','Fixed BGE view-count checks on held-out WANDS and historical ESCI. Differences are Recall@100 percentage points from Original. These checks do not change the selected strategy.','llrrl',['Data','Method','R100','$\\Delta$','95\\% CI'],rows)
f=pd.read_csv(OUT/'repaired_legacy_mitigation.csv');rows=[]
for r in f.itertuples(index=False):
    d=r._asdict()
    if r.method.startswith('DIRECT'):continue
    rows.append([r.dataset.upper(),name(r.method),f'{r[3]:.4f}',f'{r.delta_vs_Original:+.4f}',f'[{r.ci_low:+.4f},{r.ci_high:+.4f}]'])
table('legacyrepair','Reconciled historical mitigation cNDCG@10 and paired differences from Original. ESCI uses E/S/C/I=1/.1/.01/0. WANDS uses 383 positive-IDCG held-out queries; ESCI uses 500.','llrrl',['Data','Method','cN@10','$\\Delta$','95\\% CI'],rows)
rows=[]
for ds in ['wands','esci']:
    f=pd.read_csv(OUT/f'{ds}_visibility_states.csv')
    for k in KS:
        v=[f[(f.K==k)&(f.state==s)].iloc[0] for s in ['always','sometimes','never']]
        rows.append([ds.upper(),k,*[int(r.pairs) for r in v],*[f'{100*r["macro"]:.2f}' for r in v]])
table('allcoverage','BGE seven-order coverage. Counts are highest-label query--product pairs; macro percentages average within eligible queries. All historical queries are used here, unlike strategy selection/held-out tables. Sometimes means crossing; never is persistent omission.','lrrrrrrr',['Data','$K$','Always','Some','Never','Always \\%','Some \\%','Never \\%'],rows,True)
rows=[]
for ds in ['wands','esci']:
    f=pd.read_csv(OUT/f'{ds}_canonical_reranking_summary.csv');timing=json.loads((OUT/f'{ds}_canonical_reranker_timing.json').read_text())
    for m in ['Original','hybrid',selected]:
        cells=[]
        for k in [50,100,200,500]:
            if k not in f.candidate_K.unique():cells.append('--');continue
            r=f[(f.method==m)&(f.candidate_K==k)&(f.metric=='Recall@20')].iloc[0];n=f[(f.method==m)&(f.candidate_K==k)&(f.metric=='returned_condensed_NDCG@10')].iloc[0]
            cells.append(f'{r["mean"]:.3f} / {n["mean"]:.3f}')
        rows.append([ds.upper(),name(m),*cells,f'{1000*timing["seconds_per_pair_amortized"]:.3f}'])
table('rerankdetail','Fixed canonical-input reranking. Each K cell gives Recall@20 / returned-list condensed NDCG@10; the latter is not full-catalog cNDCG. Multiply measured amortized ms/pair by K to estimate candidate inference time, excluding loading and retrieval. This is batch throughput, not per-request latency. Paired intervals are in the artifact.','llllllr',['Data','Method','K=50','K=100','K=200','K=500','ms/pair'],rows,True)
f=pd.read_csv(OUT/'transition_summary.csv');rows=[]
for ds in ['wands','esci']:
    for m in ['hybrid',selected]:
        r=f[(f.dataset==ds)&(f.encoder=='bge_base')&(f.method==m)&(f.K==100)].iloc[0]
        rows.append([ds.upper(),name(m),int(r.rescued_pairs),int(r.newly_missed_pairs),f'{100*r.macro_net:+.2f}'])
table('transitions','BGE Top-100 pair transitions relative to Original. Raw pair counts and query-macro net percentage points have different weighting.','llrrr',['Data','Method','Rescued','Missed','Net pp'],rows)
f=pd.read_csv(OUT/'unique_view_summary.csv');rows=[[r.dataset.upper(),int(r.m),f'{r.mean_unique_text:.2f}',f'{r.mean_unique_vectors:.2f}',int(r.products_one_vector)] for r in f.itertuples()]
table('uniqueviews','Actual unique views per product. Exact vector equality can merge different complete texts after truncation; it is not a tolerance-based semantic equivalence test.','lrrrr',['Data','$m$','Text mean','Vec. mean','One vec.'],rows)
case=pd.read_csv(PAPER/'figures/figure1_target_only_source.csv')
# Preserve the source case table verbatim from the audited prior draft.
old=(P4/'archive/paper_before_retrieval_extension/sections/results.tex').read_text(encoding='utf8')
start=old.index('\\begin{table}',old.index('The motivating example across all tested orders'))
end=old.index('\\end{table}',start)+len('\\end{table}')
(GEN/'caseorders.tex').write_text(old[start:end].replace('[t]','[htbp]')+'\n',encoding='utf8')
coverage=(GEN/'allcoverage.tex').read_text(encoding='utf8')
rerank=(GEN/'rerankdetail.tex').read_text(encoding='utf8')
inner=rerank[rerank.index('\\caption'):rerank.rindex('\\end{table*}')]
(GEN/'appendix_pipeline_tables.tex').write_text(coverage.replace('\\end{table*}','\\par\\medskip\n'+inner+'\\end{table*}'),encoding='utf8')
text=r'''
\appendix
\section{Audit, Selection, and Additional Results}
The frozen protocol and explicit query IDs precede this extension's strategy evaluation, but do not preregister the historical study. The archived prior manuscript and result artifacts remain unchanged. Gain reconciliation updates effectiveness from fixed ranks; it does not silently replace the old ESCI gain convention. The new exact rankings use stable catalog-index ties. Legacy candidate reconciliation finds small numerical/order differences but no changed highest-label Top-100 membership in any original serialization condition.

\input{generated/screening}
\input{generated/sensitivitydetail}
\input{generated/saturation}
\input{generated/legacyrepair}
\input{generated/appendix_pipeline_tables}
\input{generated/uniqueviews}

\paragraph{Negative results and strategy boundaries.}
Table~\ref{tab:screening} retains every screening outcome. Table~\ref{tab:saturation} reports both aggregation families at every fixed view count without reopening selection. Canonical-anchored view generation is a deterministic input-order-independent construction, not proof that arbitrary sampled averaging is invariant. All methods preserve complete raw atoms; text and encoded-vector multiplicities differ when small attribute sets or encoder truncation produce duplicate views. Set mean encodes atoms independently and therefore differs from full-record centroids and max retrieval.

\paragraph{Meaning of candidate coverage.}
Table~\ref{tab:allcoverage} explicitly includes persistently absent pairs. At Top-100 the historical common-pool reranker analysis excludes those pairs as well as sometimes-retrieved pairs. Its apparent stability must be interpreted conditional on survival. Table~\ref{tab:transitions} separately exposes recovered and newly lost pairs; the full artifact provides these quantities per query at every budget, normalized by the same highest-label population.

\section{Reranking and Reproduction}
The first-stage order arrays are reduced to unique product IDs before candidate selection. All candidate sources use identical canonical product text, model weights, truncation, and tie handling for reranking. The larger-budget decision is based on development recall and measured compute, not on a favorable held-out final-rank result. Full per-query metrics, final Top-20 IDs, pairwise cross-encoder scores, and paired bootstrap contrasts accompany Table~\ref{tab:rerankdetail}. Candidate-union caching reuses a pair's inference across budgets and methods.

Model cache keys include input-content hashes. The reproduction scripts retain split IDs, field-priority keywords, product-dependent view seeds, encoder revisions, native pooling and normalization, official-gain repair, exact ranking, and local timing records. Nine integrity tests cover existing metric reconciliation, target insertion against explicit reranking, preserved atom multiplicity, independent metric eligibility, normalization, and product-level max reduction. Timings distinguish offline encoding, query encoding, score/sort operations, and amortized reranker inference. No ANN approximation is included.

Historical set mean encoded queries in a separate cached pass. The extension instead shares the frozen Original query matrix across all methods; the audit attributes tiny baseline score differences to these query embeddings. Historical repaired comparisons retain their original ranks, while new tables use the common query matrix and stable ranking. These two conditions are explicitly distinguished rather than silently overwriting a validated result.

\section{Illustrative Case and Diagnostic Scope}
The motivating query is \emph{turquoise chair} (WANDS query 162); the Exact-relevant item is product 34536, titled ``adjustable swivel office chair computer chair task chair mesh chair , turquoise.'' Its raw attributes include ``framecolor : green,'' ``weightcapacity:250,'' and ``framematerial : steel.'' The apparent title/attribute color discrepancy is present in the source; we do not repair or rewrite it. The full atom list and all unchanged free text are stored in the case artifact. This is an intentionally inspectable, post-hoc illustration, not a representative effect-size estimate.
\input{generated/caseorders}

The fitting-input diagnostic requires every tested serialization to fit the encoder budget. Coarse rank bands compare relevance strata only where both occur in the same query. Additional descriptive bins cover score margin, maximum input length, attribute multiplicity, query length, and literal value overlap. These bins characterize association and support; they do not establish an internal positional mechanism. The earlier official ESCI judged-pool check is retained in the artifact, but its median pool of 16 cannot substantiate large-catalog Recall@100.
'''
(S/'appendix.tex').write_text(text.strip()+'\n',encoding='utf8')
print('Generated supplementary tables and integrated appendix')
