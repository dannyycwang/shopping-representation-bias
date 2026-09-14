"""Render outcome prose from completed artifacts; inspect interpretation before release."""
from common import *
PAPER=ROOT/'paper_www2027';S=PAPER/'sections'
assert (OUT/'queue_completed.json').exists()
choice=json.loads((P4/'config/selection.json').read_text());selected=choice['method']
names={'rule_type':'type-first','rule_appearance':'appearance-first','rule_use':'use-first','rule_dimension':'dimension-first'}
def name(m):return names.get(m,m.replace('_',' '))
def read(f):return pd.read_csv(OUT/f)
def filt(df,**kw):
    for k,v in kw.items():df=df[df[k]==v]
    assert len(df)==1,(kw,len(df))
    return df.iloc[0]
def contrast(ds,a,b,metric='Recall@100',model='bge_base'):
    return filt(read(f'{ds}_{model}_primary_contrasts.csv'),method_a=a,method_b=b,metric=metric)
def ci(r,scale=100):return f"${scale*r.delta:+.2f}$ [{scale*r.ci_low:+.2f}, {scale*r.ci_high:+.2f}]"
def pct(x):return f'{100*x:.2f}\\%'
def put(f,t):(S/f).write_text(t.strip()+'\n',encoding='utf8')

text=r'''
\section{Results}
\subsection{RQ1: How Does Representation Affect Relevant-Product Inclusion?}
Fact-equivalent orders repeatedly move judged-relevant products across retrieval cutoffs. Across the three general encoders and two datasets, catalog-wide pair-micro VI@20 ranges from 4.42\% to 13.26\%. Figure~\ref{fig:crossmodel} presents query-macro estimates and uncertainty, rather than duplicating every statistic in a main-text table. The e-commerce-specific encoder also remains sensitive: its pair-micro VI@20 is 6.07\% on WANDS and 7.22\% on ESCI. These are seven-representation crossing rates, not expected losses from one arbitrary update.

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{figures/figure2_cross_model_vi.pdf}
\caption{Catalog-wide query-macro VI@20 with query-bootstrap 95\% CIs. General and product-specific retrievers remain sensitive. The dashed 1\% reference is a historical study threshold, not a preregistration of this extension.}
\Description{Cross-model visibility instability on two product-search benchmarks with confidence intervals.}
\label{fig:crossmodel}
\end{figure}

The primary-family target-only intervention fixes every competitor in C0. WANDS pair-micro VI@20 is 12.72\%, 8.84\%, and 11.39\% for MiniLM, BGE, and GTE; ESCI values are 3.11\%, 3.16\%, and 6.68\%. The difference from catalog-wide results reflects the intervention, not a contradiction. Figure~\ref{fig:example} uses this stricter condition. Its ranks 5 and 1,527 must not be confused with the old catalog-wide ranks 3 and 1,303. A merchant changing one record and a platform changing all records face different competition.

\paragraph{Instability omits persistent failure.}
Across all historical queries at Top-100, BGE retrieves 8,384 WANDS highest-label pairs under every order and 3,856 under only some orders; 13,230 never enter any tested pool. ESCI has 3,995 always retrieved, 44 sometimes retrieved, and 395 never retrieved pairs. The sometimes-retrieved population is representation-dependent candidate loss; the never-retrieved population has VI=0 despite complete failure at this budget. This is why the strategy evaluation below prioritizes recall. Appendix tables retain all cutoffs and denominators.

\subsection{RQ2: Where Does Instability Concentrate?}
Distance to the retrieval boundary is a strong interpretive constraint. An irrelevant product far below Top-20 has little opportunity to cross, so an unadjusted relevance-stratum difference cannot establish preferential harm to relevant products. Figure~\ref{fig:rankcontrol} compares highest and irrelevant labels within the same original-rank band and query, reporting only their common support.

\begin{figure*}[t]
\centering\includegraphics[width=.92\textwidth]{figures/rank_controlled.pdf}
\caption{Rank-conditioned highest-minus-irrelevant query-macro VI@20, in percentage points. Intervals use query bootstrap; $n$ is common-query support in each band. These coarse strata do not eliminate residual score-margin confounding.}
\Description{Rank-conditioned contrasts vary in direction, with limited common support and different patterns across the datasets.}
\label{fig:rankcontrol}
\end{figure*}

On WANDS the contrast is negative for original ranks 1--10 ($-12.68$ points, CI $[-23.83,-2.30]$) but positive for ranks 21--50 ($+19.74$, CI $[11.78,27.79]$). ESCI does not show a consistent positive contrast across strata. We therefore weaken the earlier unadjusted interpretation: relevant products clearly suffer instability, but the data do not establish a universal relevance-specific gradient after accounting for rank. These are descriptive associations, not fairness or disparate-impact results.

Truncation does not explain every crossing. Among WANDS Exact pairs whose complete inputs fit BGE's token budget in all seven orders, query-macro VI@20 is 8.69\% (CI [7.15,10.39]; 8,536 pairs, 292 queries). The over-budget subset is 12.77\% ([10.90,14.75]; 16,934 pairs, 343 queries). The two populations differ in more than truncation, and queries may appear in both. Their contrast is not a causal truncation effect; the fitting subset establishes sensitivity even with complete input coverage.

The artifact also stratifies by score margin, attribute count, maximum token length, duplicate atoms, query length, and literal query--attribute-value overlap. Lexical overlap is an auditable surface feature, not a validated query-intent taxonomy. Historical chunking and pooling controls likewise retain nonzero instability. Together these checks narrow explanations without isolating attention, position, or truncation as a unique mechanism.

\subsection{RQ3: Do Fixed Strategies Improve Relevant-Product Recall?}
@SELECTION@
\input{generated/recall_main}
\input{generated/transfer}

@TRANSFER@

The selected field rule is evaluated separately with one target changed at a time. @TARGET@ A direct target gain does not imply the same catalog-wide gain, since competitors also change under platform deployment. No future query or test label is used to construct the rule; relevance labels only define the diagnostic population.

\paragraph{Order invariance is not a recall guarantee.}
Set mean and canonical constructions remove input-order dependence, but their matching behavior differs. In the repaired historical comparison, set mean changes WANDS cNDCG@10 by $-0.0096$ (CI $[-0.0194,0.0003]$) and ESCI by $-0.0038$ ($[-0.0147,0.0071]$), using the corrected ESCI gains. These are small, statistically inconclusive differences, not evidence of equivalence. Direct set-mean--canonical cNDCG contrasts are $+0.0121$ ($[-0.0015,0.0259]$) on WANDS and $-0.0096$ ($[-0.0210,0.0020]$) on ESCI. Separate significance against Original would not establish superiority between these methods.

All three historical late-aggregation variants remain order invariant yet reduce WANDS effectiveness, with paired intervals below zero. Their corrected ESCI intervals include zero. The negative variants and the fixed $m=7$ saturation results remain in the appendix; extra views cannot retroactively change the selected strategy. Invariance supplies a design property, while recall and relevance determine whether that property is useful.

\subsection{RQ4: What Are the Candidate and Computational Costs?}
\begin{figure*}[t]
\centering\includegraphics[width=.92\textwidth]{figures/recall_budget.pdf}
\caption{First-stage recall versus unique candidate budget (top) and estimated local processing cost (bottom). Cost adds median query encoding and exact retrieval to $K$ times amortized cross-encoder pair time; it is a batch-throughput estimate, not measured online latency or final reranked recall. All curves reuse the same ranks.}
\Description{First-stage recall for Original, hybrid and the frozen strategy at K from 20 to 1000, against candidate count and estimated processing cost.}
\label{fig:budget}
\end{figure*}
@BUDGET@
\input{generated/cost}
\input{generated/multiview_cost}
\input{generated/transitions}

The timing comparison separates index arrays, query encoding, and exact scoring/sorting. A centroid retains one vector per product after additional offline encoding; max aggregation retains all view vectors and repeats scoring. Every method returns $K$ unique product IDs. Duplicate views are measured, not silently counted as independent evidence. Exact full-matrix scoring is largely insensitive to the requested $K$; downstream reranking cost increases with the number of candidates. Thus equal $K$, equal index memory, and equal end-to-end cost are different comparisons.

\paragraph{Fixed-text reranking.}
\input{generated/reranking}
@RERANK@

Reranking can correct ordering among candidates, but it cannot recover content it never receives. The legacy varying-input experiment reduced common-pool VI@20 from 25.94\% to 13.03\% on WANDS and from 3.95\% to 1.37\% on ESCI. That condition both varies candidate generation and exposes the reranker to different text; it is kept separate from the new canonical-input comparison. Its VI denominator includes only products surviving every pool, so it cannot be compared directly with all-highest VI in Figure~\ref{fig:crossmodel}. Reporting persistent omissions, candidate recall, and final recall prevents conditional reranker stability from being mistaken for recovery.
'''
dev=read('development_selection.csv');winner=dev.iloc[0];screen=[]
for method in ['Original','hybrid']:
    pq=read(f'wands_dev/bge_base_{method}_per_query.csv');screen.append(float(pq['Recall@100'].mean()))
text=text.replace('@SELECTION@',f"Development selects {name(selected)} from the eight frozen added configurations, with Recall@100 {winner.recall:.3f}; Original is {screen[0]:.3f} and hybrid {screen[1]:.3f}. The independently selected global rule is {name(choice['rule'])}. Being the best added configuration does not imply beating either baseline. The full screening table is retained, and the choice is frozen before transfer evaluation.")
parts=[]
for ds in ['wands','esci']:
    r=contrast(ds,selected,'Original');h=contrast(ds,selected,'hybrid')
    parts.append(f"On {'held-out WANDS' if ds=='wands' else 'historical ESCI'}, the selected strategy changes Recall@100 by {ci(r)} percentage points relative to Original and {ci(h)} relative to hybrid.")
parts.append('Table~\\ref{tab:transfer} additionally tests the unchanged choice with MiniLM and newly sampled ESCI queries. These cells test transfer without encoder-specific retuning; historical test inspection and the new common-catalog construction limit confirmatory claims.')
r=contrast('wands',selected,'Original',model='minilm')
parts.append(f'The WANDS MiniLM transfer changes recall by {ci(r)} points. Thus the development gain does not establish a transferable improvement; a simple fixed rule can remove incoming-order dependence and still reduce relevant-product inclusion.')
text=text.replace('@TRANSFER@',' '.join(parts))
saturation=read('saturation_summary.csv');c4=filt(saturation,dataset='wands',method='centroid_4',metric='Recall@100');c7=filt(saturation,dataset='wands',method='centroid_7',metric='Recall@100')
extra=f"The fixed saturation check is more nuanced than a uniform failure: WANDS centroid recall is {c4['mean']:.4f} at four views and {c7['mean']:.4f} at seven, so adding views is not monotonically beneficial. The four-view change from Original is ${100*c4.delta_vs_Original:+.2f}$ points (CI [{100*c4.ci_low:+.2f},{100*c4.ci_high:+.2f}]). These follow-up cells retain their uncertainty and cost; they do not reopen selection or establish cross-encoder generalization for a newly chosen winner."
text=text.replace('All three historical late-aggregation variants remain order invariant',extra+'\n\nAll three historical late-aggregation variants remain order invariant')
parts=[]
for ds in ['wands','esci']:
    r=filt(read(f'{ds}_bge_base_target_only_summary.csv'),K=100)
    parts.append(f"On {ds.upper()}, its target-only macro Recall@100 change is ${100*r.net_macro_recall:+.2f}$ points (CI [{100*r.ci_low:+.2f},{100*r.ci_high:+.2f}]), with {pct(r.crossing_micro)} of highest-label pairs crossing this boundary.")
text=text.replace('@TARGET@',' '.join(parts))
coverage=read('selected_rule_coverage.csv')
wr=filt(coverage,dataset='wands');er=filt(coverage,dataset='esci')
coverage_sentence=f"The rule changes the raw-sort order of {int(wr.products_different_from_raw_sort):,}/{int(wr.products):,} WANDS products but only {int(er.products_different_from_raw_sort):,}/{int(er.products):,} historical ESCI products. ESCI's largely unkeyed atoms make this literal transfer close to raw sorting; it is not evidence that semantic field priorities transfer across schemas."
text=text.replace('No future query or test label is used to construct the rule; relevance labels only define the diagnostic population.','No future query or test label is used to construct the rule; relevance labels only define the diagnostic population. '+coverage_sentence)
equiv=read('recall_equivalent_candidate_budget.csv');parts=[]
for ds in ['wands','esci']:
    r=filt(equiv,dataset=ds,method=selected,K=100)
    match='none of the tested budgets' if pd.isna(r.Original_min_grid_K) else f"$K={int(r.Original_min_grid_K)}$"
    parts.append(f"On {ds.upper()}, selected-method Recall@100 is {r.recall:.3f}; Original first meets or exceeds that value at {match} on the tested grid.")
parts.append('This is a discrete recall match, not an exact cost match. Rescued and newly missed pair counts are reported in Table~\\ref{tab:transitions} because similar net recall can conceal replacements of individual relevant products.')
text=text.replace('@BUDGET@',' '.join(parts))
decision=json.loads((OUT/'reranking_extension_decision.json').read_text());parts=[]
parts.append('The development recall gain from 200 to 500 candidates passes the frozen threshold; measured throughput also permits a Top-500 extension.' if decision['triggered'] else 'The larger reranking extension is not activated under the documented development-and-cost rule; the standard budgets are retained.')
parts.append('Each method supplies the same fixed canonical product text to the same cross-encoder. The main table reports final highest-label Recall@20; returned-list condensed NDCG and paired intervals are retained in the appendix. Amortized batched inference timing excludes model loading and is not an online request latency claim.')
parts.append('Unlike first-stage Recall@K, final Top-20 recall need not grow monotonically with the candidate budget: additional candidates also compete for the fixed final slots. The WANDS point estimates illustrate this distinction; a dip between individual budgets is not automatically a significant loss.')
rerank=read('canonical_reranking_paired_contrasts.csv');budgets=read('reranking_budget_contrasts.csv')
for ds in ['wands','esci']:
    r=filt(rerank,dataset=ds,K=100,method_a=selected,method_b='Original',metric='Recall@20')
    b=filt(budgets,dataset=ds,method='Original',K_a=500 if decision['triggered'] else 200,K_b=100,metric='Recall@20')
    parts.append(f"For {ds.upper()} at $K=100$, the selected first stage changes final Recall@20 by {ci(r)} points relative to Original. Increasing Original's budget from 100 to {int(b.K_a)} changes final recall by {ci(b)} points. These paired contrasts separate the representation decision from access to more candidates.")
text=text.replace('@RERANK@',' '.join(parts))
setrec=filt(read('descriptive_baseline_contrasts.csv'),dataset='esci',encoder='bge_base',method_a='set_mean',method_b='Original',metric='Recall@100')
text=text.replace('These are small, statistically inconclusive differences, not evidence of equivalence.',f'These are small, statistically inconclusive effectiveness differences, not evidence of equivalence. Crucially, the new common-query ESCI comparison changes set-mean Recall@100 by {ci(setrec)} percentage points: the cNDCG result must not be mistaken for preserved recall.')
put('results.tex',text)
put('discussion.tex',r'''
\section{Discussion and Limitations}
\subsection{Robustness, Recall, and Deployment Scope}
Representation-induced instability is a failure mode, not a complete optimization objective. Stabilizing an index can preserve persistent omission; maximizing over more views can raise irrelevant scores as well as relevant ones. A credible improvement must therefore be judged against simple canonical and hybrid baselines, unchanged strategy transfer, candidate budgets, and computational costs. Our experiments separate merchant-editable text from platform-only aggregation. Single-target effects do not establish a collective benefit when every competitor adopts the same rule.

These results suggest a concrete evaluation practice for structured Web indexing. First verify that an intervention preserves complete source facts, then report both cutoff crossings and persistently missing relevant items. Evaluate a local record replacement separately from changing the entire catalog, and retain query-level gains and losses rather than only their average. Finally, compare candidate and final-stage outcomes at declared resource budgets. A deterministic serializer supplies a reproducible interface, but choosing one ordering does not establish that it retrieves the right content. This distinction matters when a downstream agent only receives a small candidate set and cannot inspect the omitted catalog.

\subsection{Scope of the Visibility Claim}
Visibility means eligibility at an offline retrieval cutoff. It is not observed exposure, clicks, satisfaction, purchase, or welfare, and human relevance labels are not objective product quality. The shopping-agent motivation explains why candidate inclusion matters but does not replace an agent evaluation. The mechanism may arise wherever unordered records are serialized for neural retrieval, including jobs, hotels, restaurants, and events; neither its existence nor its magnitude in those domains is established here.

\subsection{Limitations and Evaluation History}
The evidence covers two English public datasets, synthetic fact-preserving permutations, fixed catalogs, and offline exact retrieval. ESCI judgments are incomplete and the historical union differs from its official small candidate pools. New query IDs were frozen before evaluation, but the source dataset and historical test results had already been accessed. Query holdout is not unseen-product or unseen-category generalization. Empty query prefixes retain the historical model condition; optimized prompts, retraining, ANN indexing, commercial search, and live users remain untested. Rank strata have limited common support and residual margin confounding; no matched attribute-position experiment supports an internal attention mechanism. No non-inferiority margin or broad method-superiority claim follows from intervals that include zero.
''')
put('conclusion.tex',r'''
\section{Conclusion}
Fact-equivalent representations can change which judged-relevant products enter a retrieval pipeline, even when only one product changes and every competitor remains fixed. Persistent omission shows why eliminating order instability is insufficient. The development-selected ordering rule does not deliver a consistent recall gain and loses recall on WANDS when transferred to another encoder; simple hybrid retrieval remains a strong comparator. Fixed text rules and platform aggregation must therefore be evaluated against candidate-budget and cost controls, not credited for invariance alone. The resulting principle is to measure representation robustness together with relevant-product recall and the resources required to obtain it.
''')
print('Rendered results; inspect empirical interpretation before release.')

