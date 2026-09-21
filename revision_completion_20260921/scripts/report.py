"""Generate tables and revise supplied chapter copies only from computed evidence."""
from common import *
import shutil
T=HERE/'tables';M=HERE/'manuscript';INPUT=HERE/'inputs/chapter56_revision'
NAMES={'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE'}
LABEL={'raw_seven':'Raw (seven schedules)','raw_C0':'Raw C0 reference','lexical_ascending':'Canonical ascending','lexical_descending':'Canonical descending','field_priority_type':'Canonical type/style','set_mean':'Set-Mean','centroid_2':'Centroid, $M=2$','max_2':'Multi-vector max, $M=2$','BM25':'BM25','hybrid_seven':'Hybrid (seven schedules)'}
def pct(x,n=2):return f'{100*x:.{n}f}'
def effect(r,key,n=2):return f'{100*r[key]:+.{n}f} [{100*r[key+"_ci_low"]:+.{n}f}, {100*r[key+"_ci_high"]:+.{n}f}]'
def save(name,text): (T/name).write_text(text,encoding='utf8')
def table(body,caption,label,cols,wide=True,notes=''):
    env='table*' if wide else 'table'
    return '\\begin{'+env+'}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+body+'\n\\bottomrule\n\\end{tabular}\n'+('\\par\\smallskip\\begin{minipage}{\\linewidth}\\small '+notes+'\\end{minipage}\n' if notes else '')+'\\end{'+env+'}\n'

def main():
    s=pd.read_csv(DATA/'strategy_summary.csv');sm=s[s.aggregation=='query_macro'];ms=pd.read_csv(DATA/'membership_summary.csv');cost=pd.read_csv(DATA/'strategy_costs.csv')
    joint=sm[(sm.model!='gte_modernbert')&sm.method.isin(LABEL)]
    joint.merge(cost,on=['dataset','model','method','catalog_size'],how='left',suffixes=('','_cost')).to_csv(T/'strategy_joint_complete.csv',index=False)
    for name in ['strategy_costs','membership_summary','matched_hybrid_raw_contrasts','cutoff_summary','cutoff_paired_changes','severity_summary']:
        shutil.copyfile(DATA/(name+'.csv'),T/(name+'.csv'))
    can=sm[sm.method.isin(RULES)].merge(ms[(ms.aggregation=='query_macro')&ms.method.isin(RULES)],on=['dataset','model','method','K'],suffixes=('','_membership'))
    can.to_csv(T/'canonical_complete.csv',index=False)
    cells=[('wands','minilm'),('wands','bge_base'),('esci','minilm'),('esci','bge_base')]
    body=' & \\multicolumn{2}{c}{WANDS} & \\multicolumn{2}{c}{ESCI} \\\\\nMethod & MiniLM & BGE & MiniLM & BGE \\\\\n\\midrule\n'
    for method in LABEL:
        if method=='raw_C0':continue
        vals=[]
        for ds,model in cells:
            r=joint[(joint.dataset==ds)&(joint.model==model)&(joint.method==method)&(joint.K==100)].iloc[0]
            vals.append(pct(r.seven_schedule_mean_recall)+' / '+pct(r.VI))
        body+=LABEL[method]+' & '+' & '.join(vals)+' \\\\\n'
    save('strategy_joint_main.tex',table(body.rstrip(),'Joint strategy evaluation: mean Recall@100 / VI@100 (\\%).','tab:strategy_joint','lrrrr',notes='Raw and hybrid average all seven schedules. Canonical, Set-Mean, canonical-seeded centroid/max and BM25 use a fixed input-order independent index: their per-schedule Recall is identical and structural VI is zero. All three canonical rules are retained. WANDS: 308 queries, 21,299 Exact pairs; ESCI: 499 queries, 4,434 E pairs. Source-order Recall, persistent states, paired intervals, membership counts and cost are reported in the artifact tables.'))
    body='Dataset / encoder & C0 & Asc. & Desc. & Type \\\\\n\\midrule\n'
    for ds in ['wands','esci']:
      for model in MODELS:
        base=sm[(sm.dataset==ds)&(sm.model==model)&(sm.K==20)&(sm.method=='raw_C0')].iloc[0]
        vals=[pct(base.source_order_recall)]
        for rule in RULES:
            r=can[(can.dataset==ds)&(can.model==model)&(can.K==20)&(can.method==rule)].iloc[0]
            vals.append(f'{100*r.delta_source_recall_vs_raw_C0:+.2f}')
        body+=ds.upper()+' / '+NAMES[model]+' & '+' & '.join(vals)+' \\\\\n'
    save('canonical_top20.tex',table(body.rstrip(),'Raw C0 Recall@20 (\\%) and paired canonical differences (pp).','tab:canonical_top20','lrrrr',False,notes='Asc., Desc., and Type are the three fixed pure rules. Each has structural VI=0. Full paired 95\\% intervals are supplied in Table~\\ref{tab:canonical_intervals20} and the artifact; no held-out rule is selected.'))
    appendix=''
    for k in [20,100]:
        body='Dataset / encoder & Rule & Recall & $\\Delta$ C0 [95\\% CI] & $\\Delta$ raw mean [95\\% CI] & Gain / loss \\\\\n\\midrule\n'
        for _,r in can[can.K==k].iterrows():
            vals=[r.dataset.upper()+' / '+NAMES[r.model],{'lexical_ascending':'Ascending','lexical_descending':'Descending','field_priority_type':'Type/style'}[r.method],pct(r.source_order_recall),effect(r,'delta_source_recall_vs_raw_C0'),effect(r,'delta_seven_schedule_mean_recall_vs_raw_seven'),pct(r.newly_included_fraction)+' / '+pct(r.newly_omitted_fraction)]
            body+=' & '.join(vals)+' \\\\\n'
        appendix+=table(body.rstrip(),f'Complete pure-canonical results at $K={k}$. Recall and shares are percentages; differences are percentage points.',f'tab:canonical_intervals{k}','llrrrr',notes='Query-macro estimates, 10,000 paired query-cluster draws, seed 2026091701. Intervals are descriptive and uncorrected; zero in an interval is not equivalence. Gains/losses compare each actual canonical list to raw C0; no membership list is assigned to the raw seven-order mean.')+'\n'
    save('canonical_complete_intervals.tex',appendix)
    cancel=pd.read_csv(DATA/'historical_rq2_cancellation_companion.csv');cancel.to_csv(T/'cancellation_all_conditions.csv',index=False)
    body='\\begin{longtable}{llrrrrrr}\n\\caption{Exact within-query cancellation, all six C0 contrasts at both cutoffs.}\\\\\n\\toprule\nDataset / encoder & Order & $K$ & Equal $>0$ & All q & Changed q & All (\\%) & Changed (\\%) \\\\\n\\midrule\\endfirsthead\n\\toprule\nDataset / encoder & Order & $K$ & Equal $>0$ & All q & Changed q & All (\\%) & Changed (\\%) \\\\\n\\midrule\\endhead\n'
    for _,r in cancel.iterrows():body+=' & '.join([r.dataset.upper()+' / '+NAMES[r.model],r.alternative,str(r.K),str(r.queries_with_exact_zero_net_and_changes),str(r.cancellation_all_denominator),str(r.cancellation_conditional_denominator),pct(r.within_query_cancellation_fraction),pct(r.within_query_cancellation_conditional)])+' \\\\\n'
    save('cancellation_all_conditions.tex',body+'\\bottomrule\n\\end{longtable}\n')
    # Machine-readable long-form intervals for every main strategy metric/contrast.
    intervals=[]
    for _,r in s.iterrows():
        for col in s.columns:
            if col.endswith('_ci_low') and pd.notna(r[col]):
                key=col[:-7];intervals.append(dict(dataset=r.dataset,model=r.model,method=r.method,K=r.K,aggregation=r.aggregation,metric=key,estimate=r[key],ci_low=r[col],ci_high=r[key+'_ci_high'],bootstrap_seed=SEED,bootstrap_draws=DRAWS))
    pd.DataFrame(intervals).to_csv(DATA/'paired_intervals_long.csv',index=False)
    # Tables intended as artifact appendices, avoiding an overloaded eight-page main text.
    for k in [20,100]:
        body='Cell & Method & Mean R & VI & Always & Never & $\\Delta R$ C0 [95\\% CI] \\\\\n\\midrule\n'
        for _,r in joint[(joint.K==k)&~joint.method.eq('raw_C0')].iterrows():
            body+=' & '.join([r.dataset.upper()+'/'+NAMES[r.model],LABEL[r.method],pct(r.seven_schedule_mean_recall),pct(r.VI),pct(r.persistent_inclusion),pct(r.persistent_omission),effect(r,'delta_mean_recall_vs_raw_C0')])+' \\\\\n'
        save(f'strategy_states_K{k}.tex',table(body.rstrip(),f'Joint strategy states and paired mean-Recall differences at $K={k}$. Percentages and percentage points.',f'tab:joint_states{k}','llrrrrr'))
    sweep=sm[(sm.model=='bge_base')&sm.method.str.startswith(('centroid_','max_'))];sweep.to_csv(T/'bge_budget_sweep.csv',index=False)
    body='Dataset & Method & $M$ & R@20 & R@100 & $\\Delta$R@100 vs C0 [95\\% CI] \\\\\n\\midrule\n'
    for (ds,method),g in sweep.groupby(['dataset','method'],sort=False):
        a=g[g.K==20].iloc[0];b=g[g.K==100].iloc[0]
        body+=' & '.join([ds.upper(),method.split('_')[0],method.split('_')[1],pct(a.source_order_recall),pct(b.source_order_recall),effect(b,'delta_source_recall_vs_raw_C0')])+' \\\\\n'
    save('bge_budget_sweep.tex',table(body.rstrip(),'BGE fixed view-budget sweep. Each method has structural VI=0.','tab:bge_budget','llrrrr',notes='M=2 remains the common setting. M=4 and M=7 do not replace it. All intervals are recomputed using seed 2026091701. Centroid stores one vector per product; max stores M, retaining repeated views.'))
    for name in ['fitting_support.tex','cancellation_top20.tex']:shutil.copyfile(INPUT/'tables'/name,T/name)
    # Supplied chapters are updated only in new copies.
    ch5=(INPUT/'chapter5_experimental_setup.tex').read_text(encoding='utf8')
    ch5=ch5.replace('Every comparison within a dataset retains the same catalog and eligible query population.','Every primary strategy comparison within a dataset retains the same catalog and eligible query population; fully-fitting diagnostics restrict targets only.')
    ch5=ch5.replace('Order and canonicalization analyses use','All common strategy, order, canonicalization, and cutoff analyses use')
    ch5=ch5.replace('The common aggregation table reports point estimates from saved query-level results.','The common strategy table reports seven-schedule mean Recall and VI. Paired intervals, source-order comparisons, membership counts, and persistent states are supplied separately. Hybrid uses the complete 1-based dense and BM25 ranks for each schedule; its C0 results exactly reproduce the saved source-order hybrid. Historical embeddings are reused without re-encoding, and reconstructed highest-label raw ranks agree with the authoritative saved ranks in all 28 hybrid conditions.')
    ch5=ch5.replace('Cost distinguishes offline encoding from stored retrieval vectors and query-time scoring.','Cost distinguishes historical offline cache construction, query encoding, exact scoring/sorting, and dense/sparse array bytes. Cache-hit elapsed times are not cold-build costs; local exact retrieval timings are not ANN deployment measurements. Additional diagnostics use K=20,50,100,200,500,1000 on fixed target support within each curve.')
    for name in ['all-MiniLM-L6-v2','bge-base-en-v1.5','gte-modernbert-base']:
        ch5=ch5.replace(name,name.replace('-',r'-\allowbreak '))
    (M/'chapter5_experimental_setup.tex').write_text(ch5,encoding='utf8')
    ch6=(INPUT/'chapter6_results.tex').read_text(encoding='utf8')
    ch6=ch6.replace('% IMPORTANT: P1/P2 in CODEX_EXECUTION_PROMPT.md complete the joint RQ3 comparison.\n% Do not describe the current source-order hybrid rows as seven-schedule stability results.','% Updated from revision_completion_20260921; P1/P2/P3 computed from audited saved inputs.')
    ch6=ch6.replace('Table~\\ref{tab:platform_recall}','Table~\\ref{tab:strategy_joint}').replace('\\input{tables/platform_recall100.tex}','\\input{tables/strategy_joint_main.tex}')
    start=ch6.index('Hybrid retrieval has higher source-order')
    hybrid=r'''Hybrid increases seven-schedule mean Recall@100 from 56.613\% to 65.548\% for WANDS/MiniLM, 63.620\% to 67.773\% for WANDS/BGE, 86.193\% to 89.567\% for ESCI/MiniLM, and 91.041\% to 91.610\% for ESCI/BGE. Paired changes are $+8.93$ [7.21, 10.82], $+4.15$ [2.55, 5.82], $+3.37$ [2.18, 4.69], and $+0.57$ [$-0.21$, 1.37] percentage points, respectively. VI@100 decreases by 10.81, 5.26, and 0.85 points in the first three settings, while ESCI/BGE increases by 0.025 points with an interval of [$-0.409$, 0.572]. The last setting provides descriptive co-occurrence of higher mean Recall and VI, but neither interval excludes zero. It does not establish a general effectiveness--instability trade-off.

Persistent inclusion also increases at $K=100$, by 15.01, 7.08, 3.79, and 0.52 points in the same order. These joint outcomes differ from a single source-order Recall comparison. Deterministic controls remove incoming-order variation but still retrieve different products; hybrid retains order dependence while often reducing its measured extent. The evidence supports evaluating effectiveness, consistent inclusion, product membership, and cost together.
'''
    ch6=ch6[:start]+hybrid
    cutoff=r'''
Increasing the cutoff improves persistent inclusion but need not reduce VI monotonically. For example, WANDS/MiniLM target-only VI rises from 19.45\% at $K=20$ to 22.45\% at $K=100$, then falls to 10.45\% at $K=1000$ on the same support. Crossings are not uniformly confined to the cutoff neighborhood: at $K=20$, 12.77\% of all WANDS/MiniLM targets cross and have worst rank above 40, and 5.25\% cross and exceed 100, using query-macro shares. Corresponding ESCI shares are smaller. The supplementary cutoff and severity analyses distinguish all-target shares from conditional crossing proportions and from source-order losses; rank movement alone does not identify an attention mechanism.

'''
    ch6=ch6.replace('\\subsection{RQ2:',cutoff+'\\subsection{RQ2:')
    for ds in ['WANDS','ESCI']:
        for model in NAMES.values():ch6=ch6.replace(ds+'/'+model,ds+r'/\allowbreak '+model)
    (M/'chapter6_results.tex').write_text(ch6,encoding='utf8')
    app=(INPUT/'appendix_evaluation_details.tex').read_text(encoding='utf8').replace('tables/canonical_top20_intervals.tex','tables/canonical_complete_intervals.tex')
    app=app.replace('The new order and canonicalization intervals use seed 2026091701. Historical\nBGE view-budget intervals, when retained in supplementary tables, used seed\n20260907; they must be identified separately or recomputed with the common\npaired-query protocol.','All completion-package intervals, including BGE view-budget contrasts, use\n10,000 paired query-cluster draws with seed 2026091701. Historical seed-20260907\nintervals are not used in the common main or supplementary strategy tables.')
    for revision in ['1110a243fdf4','a5beb1e3e68b','e7f32e3c00f9']:app=app.replace(revision,revision[:6]+r'\allowbreak '+revision[6:])
    app=app.replace('10,000 paired query-cluster draws with seed 2026091701.','10,000 paired query-cluster draws (seed 2026091701).')
    app+=r'''
Target-only severity uses best and worst ranks over the same seven schedules.
A crossing satisfies $\mathrm{best}\leq K<\mathrm{worst}$.
All-target severe-crossing shares divide by every product in $H_q$ before query
averaging. Conditional query-macro shares first divide by a query's crossing
count and then average only queries with crossings. Pair-micro conditional
fractions weight crossing pairs equally. The artifact also reports the ratio
of query-normalized severe and crossing mass, which is a different weighting.
Source-order-directed losses additionally require C0 inclusion and omission in
at least one alternative schedule. ECDFs and rank-range distributions retain
both full and fully-fitting supports; the complete competitor catalog remains.
'''
    (M/'appendix_evaluation_details.tex').write_text(app,encoding='utf8')
    (HERE/'MANUSCRIPT_INSERTIONS.tex').write_text('% Insert after the author-controlled Chapters 1--4. Paths assume this package is the working root.\n\\input{manuscript/chapter5_experimental_setup.tex}\n\\input{manuscript/chapter6_results.tex}\n% Appendix material and complete artifact tables are optional and need full-paper layout review.\n',encoding='utf8')
    # Claim decisions separate point estimates from inferential evidence.
    claims='''# Claim verdicts\n\nAll primary effects use complete Hq, query macro, WANDS 308/21,299 and ESCI 499/4,434. CIs use 10,000 paired query clusters, seed 2026091701, uncorrected 95% percentile. No test-set winner is selected.\n\n| Claim | Verdict | Exact evidence and scope |\n|---|---|---|\n| Fully-fitting target-only order sensitivity persists | Supported | `data/cutoff_summary.csv`: intervention=target_only, target_support=fully_fitting, K=20, aggregation=query_macro. VI is nonzero in all six cells; Fig. 3 retains encoder-specific q/pair support and CIs. |\n| Small net Recall can accompany larger membership changes | Supported | `data/historical_membership_changes_summary.csv`, primary population, query_macro, six C0 contrasts, K=20/100. WANDS/BGE C1 at K20: gain 2.357%, loss 2.263%, sum 4.621%, net +0.094 pp. Aggregate Recall is not uniformly stable across cells. |\n| Substitutions occur with exactly unchanged within-query Recall | Supported | `tables/cancellation_all_conditions.csv`: all 72 dataset/encoder/cutoff/alternative cells, exact integer gains=losses>0, both all-query and changed-query denominators. Some individual ESCI K100 contrasts have zero cancellation; no universal per-contrast claim is made. All 21-pair historical contrasts also remain in the copied membership sources. |\n| Canonicalization always reduces effectiveness | Not supported | `tables/canonical_complete.csv`: all 18 rules/cells, both K; some fixed rules increase Recall versus C0, and the raw-seven reference can change the sign. Structural VI=0 does not imply a Recall penalty. |\n| A matched raw-seven/hybrid-seven condition has higher mean Recall and higher VI | Descriptive only | `data/matched_hybrid_raw_contrasts.csv`: ESCI/BGE K100, query_macro, Delta mean Recall +0.569 pp [-0.205,+1.367]; Delta VI +0.025 pp [-0.409,+0.572]. The only such positive-positive point estimate among eight cells/cutoffs. Reconstruction matches every authoritative highest-label rank; this is a matched comparison. |\n| Statistical evidence establishes gains in average retrieval performance coexisting with heightened inclusion instability | Not supported | In that ESCI/BGE condition both paired intervals include zero; no matched condition has both intervals wholly positive. This is not equivalence. Prefer the neutral contribution “joint evaluation of effectiveness, consistent inclusion, product identity, and cost.” |\n| Full-record aggregation generally improves retrieval | Not supported | `tables/strategy_joint_complete.csv` and `tables/bge_budget_sweep.csv`: M2 centroid/max decrease WANDS/MiniLM Recall versus raw C0 and increase point estimates in the other three common cells; larger BGE budgets are nonmonotonic. Effects and CIs are retained, not selected. |\n| Crossing is primarily limited to the immediate cutoff neighborhood across settings | Not supported | `data/severity_summary.csv`, event=crossing, full/query_macro: WANDS/MiniLM K20 has 12.767% of all Hq crossing with worst>2K and 5.253% with worst>5K; among crossing queries, corresponding conditional query-macro shares are 65.319% and 30.577%. ESCI generally has much smaller severe-crossing shares. “Near” must have an explicit threshold and weighting. |\n| Increasing K monotonically reduces VI | Not supported | `data/cutoff_summary.csv` and `data/cutoff_paired_changes.csv`: WANDS/MiniLM target-only full VI is 19.447%, 22.455%, 10.449% at K20/100/1000. Supports are fixed within each curve. Persistent inclusion is nondecreasing and omission nonincreasing at every checked step. |\n| Larger K improves persistent coverage on this finite family | Supported | Same cutoff files, both interventions, full and fitting separately, all six cells; all monotonicity checks pass. Competitors are unchanged. It does not prove all-permutation invariance or a neural mechanism. |\n| Complete latest manuscript satisfies 8/12 page limits | Incomplete | PDF (6) is available, but its complete matching LaTeX source is absent. A local ACM chapter/table proof cannot certify final placement or page count. |\n\n## Recommendations to the author for Chapters 1--4 (not applied)\n\nReplace the Section 4.3 opening with: “We include lexical and hybrid retrieval controls to examine the contribution of order-invariant lexical evidence.” Hybrid is not structurally invariant. Narrow the Introduction's stable-aggregate/significant-turnover wording to “Net Recall changes can understate relevant-product membership changes.” Replace the third contribution's asserted performance/instability trade-off with the neutral joint-evaluation contribution above. Explain casefold and exact fallback field rules as supplied in the appendix. No Chapters 1--4 source was edited.\n'''
    (HERE/'CLAIM_VERDICTS.md').write_text(claims,encoding='utf8')
    dictionary='''# Data definitions\n\nEvery fraction is in [0,1]; LaTeX tables convert to percentages and differences to percentage points. `strategy_summary.csv` and `paired_intervals_long.csv` contain common-seed uncertainty. `strategy_per_query.csv` contains every schedule's Recall, state fractions, support and paired contrasts. `strategy_per_pair.parquet` includes all highest-label pairs, per-schedule ranks and inclusion bits. For structural controls, identical bits represent reuse of one fixed index, not seven encoder executions. Raw C0 is a single-order reference: its seven-schedule mean and VI are undefined (blank), not structural zeros.\n\n`membership_*` compares explicit lists to Raw C0; no gains/losses are assigned to a mean list. Counts retained+gain+loss+absent=Hq; gain-loss equals the integer net count. Exact cancellation uses positive integer equality. Mean-Recall contrasts are separate columns. `matched_hybrid_raw_contrasts.csv` compares the common reconstruction to hybrid; the reconstruction has zero highest-label rank or K-membership mismatches with authoritative ranks.\n\n`cutoff_*` separates intervention and support. Target-only source/mean inclusion comes from separate counterfactual indexes, not one index Recall. Support hashes and counts remain fixed across K; fitting never removes competitors. `target_rank_movements.parquet` saves best, worst, range and all crossing/severity/source-loss flags. `crossing_ecdf.csv` has both query-balanced and pair-micro ECDFs of worst rank and range.\n\n`severity_summary.csv`: base_share and severe_share divide by full Hq before query-macro aggregation (or pair micro, explicitly). `conditional_query_macro` weights each query with crossings equally and each crossing within it equally; `conditional_pair_micro` pools crossing pairs; `query_normalized_mass_ratio` divides the summed per-query severe shares by summed crossing shares. Their denominators and valid bootstrap draws are explicit. No-crossing queries are excluded only from the conditional query mean, never from the all-target rate.\n\n`strategy_costs.csv` blank means unmeasured/not applicable. Dense bytes equal catalog x dimension x 4 x vectors_per_product; sparse bytes are actual CSC arrays, excluding vocabulary/metadata. Raw/hybrid bytes refer to one schedule's index, not simultaneous storage of seven experiments. Raw/hybrid historical offline time is C0 cache construction, explicitly a reference per schedule, not all seven work. Centroid/max offline times sum M construction metadata records without deleting repeated views. Query encoding and exact CPU scoring/sorting are distinct. BGE timings use the historical first 50 queries and five repeats, not a held-out latency estimate.\n'''
    (DATA/'DATA_DICTIONARY.md').write_text(dictionary,encoding='utf8')
    print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
