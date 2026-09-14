"""Expand presentation from frozen artifacts; no retrieval or inference."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper_www2027'
S=P/'sections'
if 'What the Instability Estimator Measures' in (S/'problem.tex').read_text(encoding='utf8'):
    raise SystemExit('One-time expansion already applied. Edit the manuscript directly; do not append twice.')

def append(name, text):
    path=S/name
    path.write_text(path.read_text(encoding='utf8').rstrip()+'\n\n'+text.strip()+'\n',encoding='utf8')

append('problem.tex',r'''
\subsection{What the Instability Estimator Measures}

The equivalence class is finite and operational: it comprises the tested serializations, not every possible rendering of a product record. VI therefore measures whether the observed rank interval straddles a specified cutoff. It is neither the probability that a randomly chosen production serializer will fail nor an estimate over all possible attribute orders. Enlarging the tested family can reveal additional crossings even if the underlying retriever is unchanged. Comparisons consequently keep the same primary family across systems and interventions.

Two features of VI are useful for interpretation. First, the measure is symmetric. A product that becomes visible under a permutation and a product that becomes hidden both exhibit representation dependence. Calling a crossing a loss relative to C0 would incorrectly treat the source serialization as a privileged semantic reference. Second, VI is local to a boundary. A large rank range entirely below the retrieval cutoff produces no crossing there, whereas a small movement across the cutoff does. Rank range and VI answer complementary questions: how far an item moves, and whether that movement changes its eligibility for retrieval.

Changing the cutoff also changes the question. VI@50 need not exceed VI@20: an item can cross the smaller boundary while remaining visible under every order at the larger one. Conversely, an item invisible under every order at Top-20 may cross Top-50. Thus a cutoff sweep should be read as a profile of operational sensitivity, not as a cumulative failure curve. Reporting several boundaries helps distinguish a result tied to one evaluation convention from a broader pattern of rank instability.

Pair-micro and query-macro aggregation express different populations. The former weights every judged relevant pair equally, so queries with many relevant products contribute more observations. The latter gives each eligible query equal weight after computing its within-query crossing fraction. A difference between the two is not a contradiction or a model-performance gap. It indicates that instability is distributed unevenly across queries with different numbers of judged relevant products. We retain both rather than select the larger estimate as the headline.

Finally, stability alone is insufficient. A retriever that consistently hides every relevant product could have zero VI. Likewise, a constant scoring rule with deterministic ties could be stable but ineffective. Relevance--visibility alignment therefore treats relevance as an essential companion criterion rather than building a single score that could conceal a tradeoff. The empirical question is whether a representation can reduce the nuisance dependence while retaining useful query--product discrimination.
''')

append('methodology.tex',r'''
\subsection{Intervention Logic and Controls}

The catalog-wide and target-only designs identify different effects. In the catalog-wide design, both a product score and the scores of its competitors may change. This is the appropriate intervention for a shared ingestion template or a catalog reserialization. In the target-only design, every competing score is held fixed. Any rank change must then follow from the target representation and its score relative to that fixed comparison set. This removes competition changes as an explanation for the target's movement, without assuming that competitors are themselves representation robust.

For target-only evaluation, every relevant pair constitutes a separate counterfactual ranking. We do not simultaneously substitute all relevant products, which would turn the test back into a catalog-level intervention. For each target, we retain its C0 rank and insert each alternative score into the C0 ranking after removing that target's former entry. Removing the old entry matters: leaving both copies in the list would create artificial self-competition. Stable tie handling matters for the same reason, particularly in smaller candidate pools. The implementation is checked against explicit stable reranking, including unchanged scores and ties.

Holding competitors in C0 is a controlled reference environment, not a claim that C0 is optimal or universal. The experiment establishes the target effect conditional on that environment. Different competitor representations can change the size or direction of the resulting rank movement. Accordingly, the difference between catalog-wide and target-only VI is not an additive estimate of a separate ``competition effect.'' Crossing indicators are nonlinear functions of the whole ranked list, and the two interventions can make different pairs cross a boundary.

The representation audit supports this causal interpretation at the input level. Reordering complete attribute atoms leaves their contents and multiplicities intact, and keeps the non-attribute fields fixed. It does not guarantee that a sequence encoder will internally process the variants identically; that is the behavior being tested. Nor does it certify that every arbitrary permutation is equally natural to a human reader. Our intervention is deliberately narrower than a paraphrase or generated rewrite, for which preserving all relevant facts is much harder to verify.

The invariant method changes the architectural treatment of the nuisance variable. Because attributes are encoded independently, an attribute's vector does not depend on which other attribute appears immediately before it. Commutative pooling then removes dependence on the ordering of those vectors. The non-attribute view remains a sequence, so the method does not discard meaningful word order in titles and descriptions. It is invariant to the specified attribute permutations, not to changes in attribute wording, schema names, values, or the grouping of facts into atoms.

This structure also explains a possible effectiveness cost. Independent encoding removes cross-attribute contextual interaction before aggregation. A useful relation between material, intended use, and dimensions may be easier for a joint encoder to represent than for a simple average. Canonical sorting retains joint encoding but commits to one particular sequence. The empirical comparison therefore concerns two ways of controlling a nuisance variable, each with a different inductive bias; there is no reason to expect either to dominate every dataset.
''')

append('experiments.tex',r'''
\subsection{Reading the Evaluation Populations}

The experiments use three distinct evaluation populations. Primary sensitivity uses all highest-relevance pairs in each complete evaluation catalog. The reranking comparison conditions on membership in every first-stage candidate pool, because ranks after reranking are otherwise undefined for omitted products. Mitigation effectiveness uses the held-out WANDS queries and the ESCI evaluation queries. Values from these populations should not be directly subtracted to infer a treatment effect: even the Original baseline can differ when its query population changes.

Judgment coverage also determines what each metric can support. In the ESCI union, a product judged for one query can compete for another query without having a label there. For visibility analysis it remains a real competitor in the constructed catalog; removing it would change the rank boundary. For condensed effectiveness, unjudged products are removed before computing discounts. This avoids treating unknown relevance as a known negative, but also means that cNDCG does not fully measure the burden of unjudged products occupying early complete-catalog positions. We therefore report visibility and effectiveness as complementary views.

The official ESCI pool provides a useful diagnostic of this construction. Every candidate there has an official query-specific judgment, but the pool is much smaller than the union catalog. It tests whether instability survives under that alternative ranking population. It does not create a second independent benchmark or an independent effectiveness replication. In particular, when condensing the union removes all additional unjudged competitors, the judged order and resulting NDCG coincide with those of the official pool.

The target-only extension was performed after the confirmatory study. Its role is to strengthen causal isolation using the frozen primary representations, not to redefine the original gate or select new winning configurations. The aggregate estimates cover the same three general retrievers and both datasets rather than only the motivating example. We retain negative and small effects, and label illustrations separately from population summaries. No new model training, relevance annotation, or encoder inference was needed for this extension.
''')

# New figures expose different questions, rather than duplicating Figure 2.
df=pd.read_csv(ROOT/'phase3/results/target_only_permutations/summary.csv')
fig,axes=plt.subplots(1,2,figsize=(7,2.45),sharey=True)
colors=['#0072B2','#D55E00','#009E73']
for ax,ds in zip(axes,['wands','esci']):
    for color,model,label in zip(colors,['minilm','bge_base','gte_modernbert'],['MiniLM','BGE-base','GTE']):
        row=df[(df.dataset==ds)&(df.retriever==model)].iloc[0]
        ax.plot([10,20,50],[100*row[f'VI@{k}_micro'] for k in [10,20,50]],'o-',label=label,color=color,lw=1.6,ms=4)
    ax.set_title('WANDS' if ds=='wands' else 'ESCI union',fontsize=10)
    ax.set_xticks([10,20,50]);ax.set_xlabel('Visibility cutoff K',fontsize=9);ax.grid(axis='y',alpha=.25)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
axes[0].set_ylabel('Target-only pair-micro VI@K (%)',fontsize=9)
axes[1].legend(fontsize=8,frameon=False)
fig.tight_layout();fig.savefig(P/'figures/figure5_target_cutoffs.pdf');plt.close(fig)
df.to_csv(P/'figures/figure5_target_cutoffs_source.csv',index=False)

pareto=pd.read_csv(ROOT/'phase3/results/phase3_tables/robustness_relevance_pareto.csv')
fig,axes=plt.subplots(1,2,figsize=(7,2.7),sharex=True,sharey=True)
methods=['M1 factual','M2 canonical','set_mean','late_mean','late_max','late_top3']
for ax,ds in zip(axes,['wands','esci']):
    for y,m in enumerate(methods):
        r=pareto[(pareto.dataset==ds)&(pareto.method==m)].iloc[0]
        v=r['delta_cNDCG@10'];lo=r.cNDCG_ci_low;hi=r.cNDCG_ci_high
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#0072B2' if m=='set_mean' else '#666666',ms=4)
    ax.axvline(0,color='black',ls='--',lw=.8);ax.set_title('WANDS held-out' if ds=='wands' else 'ESCI union',fontsize=10)
    ax.set_xlabel('Paired change in cNDCG@10 (95% CI)',fontsize=8)
    ax.set_yticks(range(6),['Factual','Canonical','Set mean','Late mean','Late max','Late top-3']);ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis();fig.tight_layout();fig.savefig(P/'figures/figure6_effectiveness_intervals.pdf');plt.close(fig)
pareto.to_csv(P/'figures/figure6_effectiveness_intervals_source.csv',index=False)

results=(S/'results.tex').read_text(encoding='utf8')
target=r'''
\paragraph{Boundary sensitivity with fixed competitors.}
Figure~\ref{fig:targetcutoffs} reports the same target-only intervention at additional cutoffs. On WANDS, the crossing fraction increases as the evaluated boundary moves deeper, whereas the ESCI rates peak at the smaller reported cutoff and decline thereafter. This difference is compatible with different rank distributions and candidate environments. It should not be read as evidence that a larger result set intrinsically improves or worsens robustness. A boundary can move past the entire rank interval of one product while entering the interval of another.

\begin{figure*}[t]
\centering\includegraphics[width=.92\textwidth]{figures/figure5_target_cutoffs.pdf}
\caption{Target-only sensitivity at three visibility boundaries. Each target changes independently; every competitor remains in C0. Points are pair-micro rates, not query-bootstrap estimates. Connecting lines guide the eye and do not interpolate a failure probability.}
\Description{Target-only Top-K crossing fractions at K ten, twenty, and fifty, with separate panels for WANDS and ESCI and curves for MiniLM, BGE, and GTE.}
\label{fig:targetcutoffs}
\end{figure*}

The target-only median rank range is 210, 83, and 105 on WANDS for MiniLM, BGE, and GTE, respectively; on ESCI it is 0, 1, and 2. The zero ESCI median for MiniLM is compatible with nonzero Top-20 instability: most judged relevant pairs can have stable ranks while a smaller subset crosses the boundary. The aggregate result therefore does not depend on asserting that every product moves substantially. Conversely, the much larger WANDS ranges establish movement beyond adjacent swaps without implying that every such movement reaches a user-visible cutoff.

\paragraph{The motivating example across all tested orders.}
Table~\ref{tab:caseorders} exposes every primary serialization for Figure~\ref{fig:example}, including those not selected for the opening illustration. The source-order target is outside Top-20, two tested orders put it inside, and another gets close while remaining outside. The direction is not uniformly favorable. This makes the example a test of nuisance dependence rather than a recommended rewriting policy. The table also shows why fixing competitors matters: the same target score can receive a different rank when the surrounding catalog is reserialized.

\begin{table}[t]
\caption{All orders for the Exact-relevant turquoise chair (MiniLM). Target-only competitors stay in C0. Positive $\Delta$ means a worse rank than target C0.}
\label{tab:caseorders}\centering\small
\begin{tabular}{lrrr}\toprule
Order & Target-only rank & $\Delta$ vs C0 & Catalog-wide rank\\\midrule
C0 & 486 & 0 & 486\\
C1 & 476 & $-10$ & 295\\
C2s1 & 1,527 & $+1,041$ & 1,303\\
C2s2 & 28 & $-458$ & 21\\
C2s3 & 582 & $+96$ & 484\\
C2s4 & 5 & $-481$ & 3\\
C2s5 & 5 & $-481$ & 4\\\bottomrule
\end{tabular}
\end{table}

This case was selected after inspecting the outputs for an intuitive query and a large crossing. It is not an unbiased sample of effect magnitude and cannot substitute for the six aggregate cells. Its scientific role is to make the controlled comparison inspectable: identical facts and fixed competitors can still yield an operationally different result. The population-level inference rests on the full highest-relevance evaluation, not on the visual extremity of this example.
'''
results=results.replace('Earlier single-product template interventions',target+'\nEarlier single-product template interventions')

pipeline=r'''
\paragraph{How many pairs are excluded from the common-pool analysis?}
Table~\ref{tab:coverage} makes the conditioning explicit. On WANDS, 12,240 highest-relevance pairs appear in at least one top-100 pool, but only 8,384 occur in every pool. The difference, 3,856 pairs, is the representation-dependent candidate-loss population. On ESCI, the corresponding counts are 4,039 and 3,995, a difference of 44. These are query--product pairs rather than unique products: the same catalog item can be relevant to multiple queries.

\begin{table}[t]
\caption{Candidate coverage under catalog-wide BGE permutations. ``Some only'' is ever retrieved minus common to all pools. The conditional loss divides that count by ever retrieved; it is distinct from the all-highest and query-macro rates.}
\label{tab:coverage}\centering\small
\begin{tabular}{lrr}\toprule
Population & WANDS & ESCI\\\midrule
All highest-relevance pairs & 25,470 & 4,434\\
Ever in Top-100 & 12,240 & 4,039\\
In every Top-100 & 8,384 & 3,995\\
Some pools only & 3,856 & 44\\
Loss conditional on ever retrieved & 31.50\% & 1.09\%\\\bottomrule
\end{tabular}
\end{table}

The conditional rate is useful because it asks about relevant content that the retriever demonstrates it can retrieve under at least one equivalent input. For WANDS this is 31.50\%; for ESCI it is 1.09\%. These values must not replace the primary all-highest rates without changing the claim. Pairs absent under every serialization are outside this conditional denominator and contribute zero to representation-induced crossing even though their persistent omission can still be a retrieval-effectiveness problem.

There is also a selection effect in evaluating only common candidates. Membership in every pool removes the very products whose availability is unstable at the first-stage boundary. The reduced VI after reranking is therefore evidence about ordering conditional on survival, not a complete end-to-end robustness guarantee. Reporting both candidate coverage and conditional reranker stability prevents this survivorship restriction from being mistaken for recovery of missing content.

For additional context, C0 highest-label Recall@20 rises from 12.58\% to 13.61\% after reranking on WANDS and from 68.04\% to 70.37\% on ESCI. These stored C0 point estimates show that improved relevance retrieval and residual representation instability can coexist. They are not paired significance claims, and their cross-dataset difference reflects distinct catalogs and relevance populations. The key pipeline conclusion concerns availability: a later ranker can improve a candidate's position only when that candidate has reached it.
'''
results=results.replace('\subsection{RQ4:',pipeline+'\n\subsection{RQ4:')

mitigation=r'''
\paragraph{Inspecting the effectiveness uncertainty.}
Figure~\ref{fig:effectivenessci} adds paired confidence intervals for all evaluated mitigation variants. It complements the robustness--relevance scatter plot by exposing the uncertainty that a point-only frontier cannot show. Set mean's interval crosses zero on both datasets, but much of the WANDS interval lies on the loss side. Calling this ``no loss'' would conflate failure to establish a difference with evidence that the difference is negligible. A deployment decision would require a prespecified acceptable-loss margin and a study designed to assess that margin.

\begin{figure*}[t]
\centering\includegraphics[width=.92\textwidth]{figures/figure6_effectiveness_intervals.pdf}
\caption{Paired effectiveness changes relative to Original, with stored 95\% confidence intervals. The dashed line denotes zero change, not an equivalence margin. Set mean is highlighted; the late-aggregation negative results remain visible.}
\Description{Confidence intervals for six representation methods on held-out WANDS and ESCI. Set-mean intervals span zero on both; the late variants have entirely negative WANDS intervals.}
\label{fig:effectivenessci}
\end{figure*}

\paragraph{What the negative variants teach us.}
All late-aggregation variants satisfy the numerical order-invariance criterion, yet on held-out WANDS their cNDCG@10 values are 0.8058, 0.8093, and 0.8106 for late mean, maximum, and top-3 aggregation. Each paired interval establishes a decrease relative to Original. On ESCI their values are 0.7026, 0.7041, and 0.7026, with intervals spanning zero. Thus invariance does not determine effectiveness: the aggregation rule still controls which aspects of a structured record dominate matching.

Canonical sorting makes the same point from another direction. It obtains zero measured instability but loses effectiveness on WANDS, whereas its ESCI point estimate is higher than Original without a confirmed gain. The available evidence does not establish a universal ranking of the invariant approaches. In particular, a significant Original-versus-canonical contrast and an inconclusive Original-versus-set-mean contrast do not establish that set mean significantly outperforms canonical sorting. Such a claim would require their direct paired comparison.

The useful outcome is therefore a design constraint and a proof of feasibility within the tested family. A representation can remove dependence on attribute order while retaining substantial relevance effectiveness, but the relevance cost must be measured rather than assumed away. Further optimization should preserve this two-objective evaluation, including unsuccessful variants, so that apparent robustness gains cannot be obtained simply by erasing information needed for matching.
'''
results+='\n'+mitigation
(S/'results.tex').write_text(results,encoding='utf8')

append('discussion.tex',r'''
\subsection{An Evaluation Practice for Structured Web Retrieval}

The results suggest treating a serializer change as a retrieval-system change, even when it preserves the source record. A useful regression suite would retain query--item relevance judgments together with a small, explicitly defined family of equivalent inputs. Relevance metrics would assess whether the updated pipeline retrieves useful content overall; item-level crossing metrics would assess whether irrelevant formatting decisions determine which particular relevant content survives. Neither assessment substitutes for the other.

The intervention should match the engineering decision. A shared indexing change calls for a catalog-wide test because competitors change together. A record-specific ingestion path calls for a target-only test against a fixed catalog. A multistage service also needs candidate-coverage accounting before any final-ranking comparison. This is a proposed evaluation practice supported by the structure of our findings, not an empirically validated deployment protocol. The appropriate cutoff and tolerance remain application-dependent.

The controlled equivalence definition is equally important. A field list can often be permuted safely, whereas a procedural instruction or a chronological event history may have semantically necessary order. Applying invariance indiscriminately would destroy useful meaning. Broader Web replication should therefore begin by identifying which components are genuinely unordered for the task and which must remain sequential. Our product setting demonstrates the approach for the audited attribute family; it does not license treating all Web text as a bag of facts.

Finally, representation robustness does not require identical ranking of every relevant item in every context. The expectation is narrower: when the query, facts, and comparison environment are fixed, a nuisance transformation should not arbitrarily determine retrieval eligibility. Personalization, fresh information, changed inventory, and meaningful textual edits are different interventions. Keeping that distinction explicit makes the visibility claim both useful and falsifiable without implying a fairness entitlement or a measured downstream business effect.
''')

# Keep detailed tables in a separate supplement, so the requested main-paper budget is real.
main=(P/'main.tex').read_text(encoding='utf8')
main=main.replace('\\bibliographystyle{ACM-Reference-Format}', '\\clearpage\n\\bibliographystyle{ACM-Reference-Format}')
main=main.replace('\\clearpage\n\\appendix\n\\input{appendix}', '')
(P/'main.tex').write_text(main,encoding='utf8')
supp=r'''\documentclass[sigconf,anonymous,review]{acmart}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\usepackage{booktabs}\usepackage{multirow}
\begin{document}
\appendix
\input{appendix}
\end{document}
'''
(P/'supplementary.tex').write_text(supp,encoding='utf8')
# External references to supplement must not become undefined in the main PDF.
for f in ['results.tex']:
    path=S/f;t=path.read_text(encoding='utf8')
    t=t.replace('Appendix Table~\\ref{tab:sensitivity}','Supplementary Table S1').replace('Appendix Table~\\ref{tab:targetonly}','Supplementary Table S2')
    t=t.replace('Late-aggregation ablations appear in Appendix~\\ref{app:mitigations}.','Effectiveness intervals for all variants appear in Figure~\\ref{fig:effectivenessci}.')
    path.write_text(t,encoding='utf8')
print('Expanded manuscript and generated two figures from existing result CSVs.')
