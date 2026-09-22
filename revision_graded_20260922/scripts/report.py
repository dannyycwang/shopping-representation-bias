"""Produce evidence and bounded claims only after scores and intervals exist."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'minilm':'MiniLM','bge_base':'BGE','gte_modernbert':'GTE','lexical_ascending':'Ascending','lexical_descending':'Descending','field_priority_type':'Type/style'}

def md(frame):
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def main():
    verify_protocol();s=pd.read_csv(HERE/'tables/effectiveness_wide.csv');c=pd.read_csv(HERE/'tables/paired_contrasts.csv')
    r=pd.read_csv(HERE/'tables/rq2_distribution_and_thresholds.csv');cor=pd.read_csv(HERE/'tables/gain_correction.csv')
    rq=pd.read_parquet(HERE/'data/rq2_membership_ndcg_per_query.parquet');pop=read(HERE/'PROTOCOL.json')['populations']
    def get(ds,model,rule,ref,metric):
        f=c[(c.dataset==ds)&(c.model==model)&(c.method=='canonical_hybrid_'+rule)&(c.contrast=='canonical_hybrid_comparator')&(c.reference==ref)&(c.metric==metric)]
        assert len(f)==1;return f.iloc[0]
    def cell(row):return f"{row.delta:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
    evidence=[];verdicts=[]
    for ds in ['wands','esci']:
        for model in ['minilm','bge_base']:
            for rule in RULES:
                evidence.append({'Dataset':ds.upper(),'Encoder':NAMES[model],'Rule':NAMES[rule],
                    'nDCG20 vs dense':cell(get(ds,model,rule,rule,'nDCG@20')),
                    'nDCG20 vs BM25':cell(get(ds,model,rule,'BM25','nDCG@20')),
                    'nDCG20 vs raw hybrid mean':cell(get(ds,model,rule,'raw_hybrid_seven_mean','nDCG@20')),
                    'Recall20 vs raw hybrid mean':cell(get(ds,model,rule,'raw_hybrid_seven_mean','Recall@20'))})
                for ref in [rule,'BM25','raw_hybrid_seven_mean']:
                    for metric in ['nDCG@10','nDCG@20','cNDCG@10','cNDCG@20','Recall@20','Recall@100']:
                        a=get(ds,model,rule,ref,metric)
                        verdicts.append(dict(dataset=ds,model=model,rule=rule,reference=ref,metric=metric,delta=a.delta,ci_low=a.ci_low,ci_high=a.ci_high,effectiveness_verdict=a.verdict,claim_of_improvement='supported' if a.ci_low>0 else 'unsupported' if a.ci_high<0 else 'inconclusive'))
    evidence=pd.DataFrame(evidence);evidence.to_csv(HERE/'tables/compact_evidence.csv',index=False)
    evidence.to_latex(HERE/'tables/compact_evidence.tex',index=False,escape=True,longtable=True)
    pd.DataFrame(verdicts).to_csv(HERE/'tables/canonical_hybrid_claim_verdicts.csv',index=False)
    compact=[]
    raw=r[(r.method=='raw')&(r.ndcg_cutoff==20)]
    for (ds,model),g in raw.groupby(['dataset','model'],sort=False):
        compact.append({'Dataset':ds.upper(),'Encoder':NAMES[model],'All queries':int(g.all_queries.iloc[0]),
         'Changed queries':f'{int(g.changed_queries.min())}–{int(g.changed_queries.max())}',
         'Median |delta nDCG20|':f'{g.abs_delta_q50.min():.4f}–{g.abs_delta_q50.max():.4f}',
         '95th percentile':f'{g.abs_delta_q95.min():.4f}–{g.abs_delta_q95.max():.4f}',
         'Changed with |delta| > .01':f'{100*(1-g["fraction_changed_at_or_below_0.01"].max()):.1f}–{100*(1-g["fraction_changed_at_or_below_0.01"].min()):.1f}%',
         'Numerical zero, changed':f'{int(g.numerical_zero_changed.min())}–{int(g.numerical_zero_changed.max())}'})
    compact=pd.DataFrame(compact);compact.to_csv(HERE/'tables/rq2_compact_evidence.csv',index=False)
    compact.to_latex(HERE/'tables/rq2_compact_evidence.tex',index=False,escape=True)
    zero=rq[rq.membership_changed&rq.numerical_zero_nDCG20]
    zero.to_csv(HERE/'data/observed_numerical_zero_membership_records.csv',index=False)
    # All six contrasts appear in each ECDF panel; no selected encoder or threshold.
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#206095','#d1495b','#2f855a','#8a55a2','#bf7300','#53585f']
    fig,axes=plt.subplots(2,3,figsize=(11.4,6.5),sharex=True,sharey=True)
    for i,ds in enumerate(['wands','esci']):
        for j,model in enumerate(MODELS):
            ax=axes[i,j]
            for col,stem in zip(colors,STEMS[1:]):
                values=rq[(rq.dataset==ds)&(rq.model==model)&(rq.method=='raw')&(rq.schedule==stem)&rq.membership_changed].abs_delta_nDCG20.sort_values().to_numpy()
                ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=stem+' − C0',color=col,lw=1.4)
            for threshold in [.001,.005,.01]:ax.axvline(threshold,color='#a1a1a1',lw=.6,ls='--',zorder=0)
            ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(0,1);ax.set_ylim(0,1.015)
            ax.set_xticks([0,.001,.01,.1,1],['0','.001','.01','.1','1'])
            ax.set_title(ds.upper()+' · '+NAMES[model],loc='left');ax.grid(axis='y',color='#eeeeee')
            if j==0:ax.set_ylabel('Fraction of changed queries')
            if i==1:ax.set_xlabel('|Δ nDCG@20| (0–1 scale)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,.01))
    fig.suptitle('Highest-label Top-20 membership change often accompanies graded score change',x=.07,ha='left',fontsize=13)
    fig.text(.07,.935,'Each curve retains its own changed-query denominator; dashed lines are descriptive thresholds .001, .005 and .01.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.99,bottom=.15,top=.86,hspace=.32,wspace=.12)
    for ext in ['svg','pdf','png']:fig.savefig(figdir/f'rq2_raw_changed_query_ecdf.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.4),sharex=True)
    for i,ds in enumerate(['wands','esci']):
        for j,model in enumerate(['minilm','bge_base']):
            ax=axes[i,j]
            for shift,color,kind in [(-.2,'#206095','dense'),(0,'#bf7300','BM25'),(.2,'#2f855a','raw hybrid mean')]:
                for y,rule in enumerate(RULES):
                    ref=rule if kind=='dense' else 'BM25' if kind=='BM25' else 'raw_hybrid_seven_mean';a=get(ds,model,rule,ref,'nDCG@20')
                    ax.errorbar(a.delta,y+shift,xerr=[[a.delta-a.ci_low],[a.ci_high-a.delta]],fmt='o',color=color,markersize=4,capsize=2,label=kind if y==0 else None)
            ax.axvline(0,color='#555555',lw=.8);ax.set_yticks(range(3),[NAMES[r] for r in RULES]);ax.invert_yaxis();ax.grid(axis='x',color='#eeeeee')
            ax.set_title(ds.upper()+' · '+NAMES[model],loc='left')
            if i==1:ax.set_xlabel('Canonical hybrid minus comparator: nDCG@20')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Canonical–lexical hybrid: all three rules and all comparators',x=.12,ha='left',fontsize=13)
    fig.text(.12,.92,'Paired 95% query-cluster intervals, 10,000 draws; uncorrected. Intervals crossing zero do not establish equivalence.',fontsize=9)
    fig.subplots_adjust(left=.12,right=.99,bottom=.14,top=.84,hspace=.4,wspace=.32)
    for ext in ['svg','pdf','png']:fig.savefig(figdir/f'canonical_hybrid_ndcg20_contrasts.{ext}',dpi=180)
    plt.close(fig)
    correction=cor[(cor.dataset=='esci')&(cor.method=='raw')&(cor.schedule=='C0')&(cor.metric.isin(['nDCG@20','cNDCG@20']))].copy()
    correction=correction[['model','metric','old_mean','corrected_mean','delta']].round(6)
    audit=f'''# Graded metric audit — 22 September 2026

P0 and P1 are complete. The package contains 122 conditions and 49,227 per-query records. No encoding, model update, training, or manuscript integration was performed. P2's optional 16/32-schedule expansion was deferred under a budget of zero new encodings.

## Repository and provenance

Current branch: `codex/chapter56-completion-20260921`, HEAD `edf4b3258677f5bc58c1957508fda0c1313285e3`. It is the direct parent of reference merge `32b24160afcaa229a8b7057c04ecf638c7fae488`; their tracked trees are identical. The reference was fetched without switching branches. No applicable AGENTS.md was found in the repository or workspace ancestors. Existing working changes and historical outputs were preserved and hashed. All new work is under this dated directory.

The frozen protocol SHA-256 is `{sha(HERE/'PROTOCOL.json')}`. It was written before new score evaluation. `data/input_manifest.json` inventories 445 available files, including ignored local payloads and Git tracking status; `data/supplemental_audit_inputs.json` hashes 24 read-only label/cost/consumer/score audit inputs. The four authoritative ascending-canonical full-score hashes also match their expected hashes in the frozen historical manifest. No required payload is missing. Git alone does not distribute every embedding or full-score payload. Restore the exact listed artifacts when reproducing elsewhere; never regenerate them under a different model revision. Model revisions, profiles and source paths remain in the hashed historical metadata and `encoder_profiles_and_rank_sources.csv`.

## Actual label and grade trace

The [ESCI paper, Section 3.1](https://arxiv.org/pdf/2206.06588) assigns direct gains E=1, S=.1, C=.01, I=0. Phase II's `phase2.json` instead declares E=1, C=.1, S=.01, I=0. `prepare_data.py` maps the original `esci_label` into the stored `grade` column, and `evaluation.py` uses that numeric column directly. This is an actual data-path issue, not only a documentation error.

`qa/raw_label_trace.json` verifies original ESCI example labels against every processed query/product judgment. On the frozen eligible support there are 4,434 E, 3,530 S, 505 C and 1,650 I judgments. All stored S grades are .01 and C grades .1. WANDS has 21,299 Exact, 96,566 Partial and 44,353 Irrelevant judgments; its 3/1/0 gains remain unchanged. The WANDS raw-label audit also reproduces the historical conflict-removal and deduplication rule.

The correction was already implemented in Phase IV's `pair_metrics`, and a historical audit recomputed 79 ESCI rank artifacts. The new follow-up aligns full-ranking nDCG and condensed cNDCG on the exact 308/499 query populations. It does not claim that all historical outputs were uncorrected. `tables/historical_metric_consumers.csv` and `data/historical_gain_repair_inventory.csv` identify consumers and earlier repairs.

Affected historical outputs include Phase II cNDCG, gain/rank Spearman, SAR, paired graded mitigation contrasts and their derived success/frontier classifications; Phase III stored-grade audits, official judged-pool NDCG, frozen Marqo and invariant-method graded reports, and downstream Pareto summaries. Grade>0 support is unchanged by swapping two positive gains. Phase III alpha selection used WANDS development cNDCG; Phase IV rule/view selection used WANDS Recall@100 with corrected cNDCG as a tiebreaker. Those selections are unaffected by the ESCI gain swap. The reused pretrained encoders, query vectors, canonical construction and BM25/RRF scores do not train on these gains. Separate later WANDS training studies use WANDS labels/gains and are outside this follow-up; there is no evidence that this ESCI reporting correction changes the frozen retrieval training pipeline.

`GAIN_MAPPING.json` preserves old and corrected mappings. New gains are always computed from labels. The `old_*` fields are counterfactual evaluation of the same rankings under the old mapping; for newly constructed hybrids they are not historical published measurements.

## Evaluation scope and limitations

WANDS: 308 eligible queries, 21,299 Exact pairs, all 162,218 judgments for those queries, full 42,994-product catalog. ESCI: 499 eligible queries, 4,434 E pairs, all 10,119 judgments, fixed 10,076-product pooled catalog. Exact IDs and ordered catalog/query axes are exported under `data/`. The remaining historically requested queries are excluded only to hold the stated highest-label-eligible population fixed. This differs from older positive-IDCG-only population summaries and must not be compared as though the denominators matched.

nDCG@10/@20 discounts at original full catalog ranks. Unjudged items have zero gain for this computation only; their true relevance is unknown. IDCG sorts the same query's available judged direct gains, without an exponential transform. cNDCG@10/@20 removes unjudged items before assigning positions and is always named as condensed. JudgedCoverage uses original top-K IDs, counting every judged label including zero-gain ones. Target-only counterfactual ranks are never combined into a query ranking. Full-ranking nDCG is a secondary measure under incomplete judgments, not a claim of complete relevance assessment; coverage and condensed results are reported alongside it.

Raw retrieval includes all 42 dataset/encoder/schedule configurations. All 18 pure canonical conditions are reused (GTE is secondary). MiniLM/BGE controls include Set-Mean, centroid M=2, multi-vector M=2, BM25 and all seven raw hybrids. BGE M=4/7 centroid/multi-vector conditions remain secondary. C0 and the seven-schedule query mean are distinct. Every raw hybrid is paired with its corresponding raw schedule before query-level averaging. The common historical query embeddings, frozen tokenization/model metadata and fixed catalog-index tie rule are preserved.

All 18 reconstructed canonical judged rankings match their saved ranks exactly on the frozen support. Reconstructed raw scores differ at six non-highest judged ranks: WANDS BGE at 442/6,384/4,340/5,737 and ESCI BGE at 7,545/7,548, each by one rank. The exact records are in `qa/rank_precision_differences.csv`; saved raw ranks remain authoritative. Highest-label ranks match exactly. Every reconstructed raw hybrid matches its authoritative highest-label ranks and top-1000 IDs; all C0 hybrid judged ranks and full scores also match. No truncated candidate fusion was used.

The evaluator explicitly verifies identical highest-label sets and inclusion IDs under old/new gains for every query, including Recall@20/@100. Validation additionally reproduces 42 historical old-gain raw cNDCG conditions and 80 historical Recall summary cells. Eight small tests ran before reevaluation; an independent dense gain-vector implementation checks 366 query/condition records. All 469 source-hash records and pre-existing edits are rechecked.

## Correction magnitude on raw C0

{md(correction)}

All WANDS gain-correction differences are exactly zero. ESCI graded values change while rank and highest-label inclusion do not. `tables/gain_correction.csv` retains all conditions and paired intervals for the correction.

## Membership change and nDCG

Across the six source-order contrasts per encoder, the following ranges retain every contrast and use each contrast's changed-query denominator. Full records and separate nDCG@10 diagnostics are provided; K10 conditioned on K20 membership is explicitly a different-cutoff diagnostic.

{md(compact)}

Among changed queries, |delta nDCG@20| exceeds .01 in 64.1–79.2% of WANDS cases and 81.1–96.2% of ESCI cases across dataset/encoder/schedule cells. Thus small mean shifts cannot be generalized into negligible query-level graded change. Raw mean nDCG@20 deltas span -0.01101 to +0.00151 on WANDS and -0.00393 to +0.00092 on ESCI. Five of the 36 raw schedule contrasts have uncorrected intervals wholly below zero; the other 31 include zero. This does not establish equivalence.

Numerical zero with changed membership is observed: WANDS has 18–29 such query records per raw schedule contrast; ESCI has 0–2. All observed examples are retained in `data/observed_numerical_zero_membership_records.csv`, including gains/losses and integer Recall cancellation. Numerical equality uses absolute tolerance 1e-12, relative tolerance 0; it is distinct from exact integer gains=losses. These examples support a limited non-identification statement, not a stable-aggregate claim. Thresholds .001, .005 and .01 were frozen and are descriptive sensitivity values, not tests of equivalence.

## Canonical–lexical hybrid

Twelve fixed indexes combine BM25 and one of the three pure canonical dense rankings. Each branch ranks the entire catalog; fusion is float32 1/(60+r_BM25)+1/(60+r_dense), with stable original catalog-index ties. No new weight or rule is selected. All complete entries and duplicate multiplicities are retained. Canonical input equality is verified over all seven incoming schedules, and intrinsic sorting plus deterministic branch ranking makes VI=0 structural. One fixed index suffices; seven encodings were not performed.

{md(evidence)}

Values are signed differences on the 0–1 metric scale, with uncorrected paired 95% query-cluster bootstrap intervals (10,000 draws, seed 2026091701). Canonical hybrids improve nDCG@20 over BM25 in all 12 cells and over their corresponding dense branch in nine; the three ESCI/BGE dense-branch comparisons are inconclusive. Against the mean of seven raw-hybrid references, ten nDCG@20 contrasts are inconclusive, while ESCI/BGE ascending and type/style show small positive differences under uncorrected intervals. All 12 nDCG@10 and Recall@100 comparisons against raw-hybrid means are inconclusive. WANDS/BGE ascending reduces Recall@20 by 0.00624 (CI -0.01370 to -0.00084); ESCI/BGE type/style increases it by 0.00195 (CI +0.00001 to +0.00402), also uncorrected. These metric-specific results prevent a claim of universal preservation or dominance.

Persistent inclusion/omission, mean/source Recall and VI are in `tables/persistent_states.csv` with query-macro and pair-micro results separated. `tables/canonical_hybrid_state_contrasts.csv` supplies paired intervals for each persistent-state difference against the own dense branch, BM25 and the seven-schedule raw hybrid. Every P1 membership contrast against its dense branch, BM25 and all seven raw-hybrid references is retained, plus matched query-level means. The graded nDCG analysis is query-macro; it does not silently substitute pair weighting.

Each canonical hybrid stores one dense vector per product plus the fixed sparse BM25 index. Dense+sparse array bytes are 118,698,552/184,737,336 for WANDS MiniLM/BGE and 24,049,044/39,525,780 for ESCI MiniLM/BGE. Exact inference scores one dense catalog and one lexical branch and performs two branch sorts plus a fusion sort. `tables/canonical_hybrid_costs.csv` separates historical dense encoding time, zero new encoding, measured local batch work and unmeasured online latency. It excludes model weights, vocabulary/metadata overhead and research result files. Inherited costs retain M=2/4/7 construction budgets, actual vector counts and duplicate views; centroid scoring uses one vector after construction, while multi-vector scoring retains M vectors. No ANN or cold deployment latency is claimed.

## Deliverables

`PROTOCOL.json`, `GAIN_MAPPING.json`, manifests, exact supports, all-query metrics, paired differences, membership/ECDF records, CSV/LaTeX tables, figure sources, `MANUSCRIPT_INSERTIONS.md`, and `CLAIM_VERDICTS.md` are included. `REPRODUCE.md` gives commands. Historical manuscripts are untouched; the English insertion is for review. Optional finite-permutation expansion and universal robustness claims are not part of this execution.
'''
    (HERE/'GRADED_METRIC_AUDIT.md').write_text(audit,encoding='utf8')
    claims='''# Claim verdicts

| Claim | Verdict | Evidence and scope |
| --- | --- | --- |
| Phase II stored ESCI S/C gains are reversed | Supported | Original labels, processed grade columns and 42 raw artifacts traced; paper Section 3.1 confirms E/S/C/I=1/.1/.01/0. |
| Every historical graded result still used the error | Unsupported | Phase IV already corrected label-derived cNDCG and reconciled 79 historical rank artifacts. |
| The correction alters frozen training or retrieval ranks | Unsupported | Reused frozen encoders/query embeddings/RRF do not consume graded training targets; correction is evaluation only. |
| Highest-label Hq, inclusion, Recall20/100 and VI survive the gain swap | Supported | All reused supports and query records verified; 80 prior Recall summary cells reproduced. |
| Membership changes can occur with numerically identical nDCG20 | Supported, bounded | Observed records retained; atol=1e-12, rtol=0; distinct from integer Recall cancellation. |
| Membership changes usually have negligible graded effect | Unsupported at the frozen .01 descriptive threshold | Most changed queries exceed .01 in every raw dataset/encoder/schedule cell; complete distributions retained. |
| Mean nDCG is invariant/equivalent across raw schedules | Unsupported as a blanket claim | Five uncorrected raw nDCG20 intervals establish reductions; zero-spanning intervals cannot establish equivalence. |
| All canonical–BM25 indexes are independent of incoming attribute order | Supported structurally | Complete-input equality, fixed branch inputs, full ranks and deterministic ties; VI=0. |
| Canonical–BM25 improves nDCG20 over BM25 | Supported in all 12 tested cells | Paired uncorrected intervals above zero, all rules retained. |
| Canonical–BM25 improves nDCG20 over its own dense branch | Supported in 9 cells; inconclusive in 3 ESCI/BGE cells | Full condition-specific verdict table included. |
| Canonical–BM25 is equivalent or uniformly superior to raw-seven hybrid | Unsupported as a universal claim | Mostly inconclusive nDCG contrasts; WANDS/BGE ascending Recall20 is lower. |
| There is a general effectiveness–instability trade-off | Unsupported | Fixed canonical hybrids retain lexical benefit in several settings; no universal Pareto law follows. |
| This establishes all-permutation coverage for raw models | Unsupported / not evaluated | Optional P2 expansion deferred; seven-schedule observations are finite. |
| Judged gains are complete relevance ground truth for the catalog | Unsupported | Incomplete judgments; unjudged-zero is an evaluation convention, with coverage and condensed measures reported. |

Intervals are paired query-cluster percentile intervals, 10,000 draws, seed 2026091701, without multiplicity correction. They quantify query sampling conditional on fixed catalogs/encoders, not model, training-seed or annotation uncertainty. Descriptive thresholds are not equivalence margins. See `tables/canonical_hybrid_claim_verdicts.csv` for each setting, comparator and metric.
'''
    (HERE/'CLAIM_VERDICTS.md').write_text(claims,encoding='utf8')
    insert='''# Proposed English insertions — review copy only

## Section 5.3: Secondary graded effectiveness evaluation

We evaluate full-ranking nDCG@10 and nDCG@20 on the same 308 WANDS and 499 ESCI highest-label-eligible queries used for inclusion analysis, retaining all their judged labels. We use direct gains Exact/Partial/Irrelevant=3/1/0 for WANDS and E/S/C/I=1/.1/.01/0 for ESCI, correcting the Substitute–Complement swap in the historical Phase II grade column. No additional exponential gain transform is applied. The correction leaves the frozen rankings, highest-label supports and Recall unchanged. Full-ranking nDCG discounts at original catalog positions, assigns unjudged items zero gain only for computation, and normalizes by the available judged ideal ranking. We separately report condensed cNDCG, which removes unjudged items before discounting, and JudgedCoverage on the original top lists. Incomplete judgments limit both interpretations. All method and schedule comparisons use paired query-cluster bootstrap intervals with 10,000 draws and seed 2026091701; intervals are uncorrected and do not establish equivalence when they include zero.

## Section 6.2: Membership changes beyond Recall

The graded analysis supports a bounded distinction between aggregate effectiveness and membership, rather than a general claim of negligible score change. Across the six raw schedule contrasts against C0, median absolute nDCG@20 changes among queries with Top-20 highest-label membership change range from .0226 to .0399 on WANDS and .0389 to .0482 on ESCI. The fraction exceeding the prespecified descriptive threshold .01 ranges from 64.1% to 79.2% on WANDS and 81.1% to 96.2% on ESCI. Query-mean deltas are smaller but not uniformly negligible: they range from -.01101 to +.00151 on WANDS and -.00393 to +.00092 on ESCI, with five of the 36 uncorrected paired intervals wholly below zero. Membership changes with numerically identical nDCG@20 also occur (absolute tolerance 1e-12), but these examples do not characterize the full distribution. Integer Recall cancellation and floating-point nDCG equality are recorded separately. Complete ECDFs, all-query and changed-query denominators, and sensitivity summaries at .001/.005/.01 are retained; these thresholds are descriptive, not equivalence tests. nDCG@10 is reported separately because its cutoff differs from the Top-20 membership event.

## Section 6.3: A fixed canonical–lexical control

We add full-catalog equal-weight RRF (constant 60) between BM25 and each of three pure canonical dense indexes—ascending, descending and type/style priority—for MiniLM and BGE on both datasets. These 12 configurations reuse frozen embeddings and common query vectors without training or weight tuning. Input-order independence and deterministic ranking imply structural VI=0. All 12 improve nDCG@20 over BM25, and nine improve it over the corresponding dense branch; the three ESCI/BGE dense-branch comparisons remain inconclusive. Relative to the per-query mean of the existing seven raw-hybrid references, ten nDCG@20 contrasts are inconclusive and two ESCI/BGE contrasts have small positive uncorrected intervals. Effectiveness is metric-dependent: WANDS/BGE ascending lowers Recall@20 by .00624 (95% CI [-.01370,-.00084]), while all Recall@100 comparisons with the raw-hybrid mean remain inconclusive. Thus lexical benefit can coexist with a fixed representation, refining the practical interpretation without negating raw order sensitivity or establishing universal preservation. All rules, persistent states, membership gains/losses, graded metrics, coverage and one-dense-vector-plus-BM25 costs are reported.
'''
    (HERE/'MANUSCRIPT_INSERTIONS.md').write_text(insert,encoding='utf8')
    (HERE/'REPRODUCE.md').write_text('''# Reproduction

Run from the repository root in the recorded Python 3.10 environment (NumPy 1.26.4, pandas 2.3.3, SciPy, PyArrow 24.0.0, threadpoolctl, Matplotlib and Jinja2). Exact runtime is in `qa/runtime.json`. Four BLAS threads are used. No model download, inference or training is needed.

The dated protocol is already frozen. Do not rerun `prepare.py` over it: the script deliberately refuses to overwrite the design. That script records the initial preparation and may be used in a separately named copy of the package before scores are viewed.

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s revision_graded_20260922/scripts -p test_metrics.py -v
python revision_graded_20260922/scripts/audit.py
python revision_graded_20260922/scripts/evaluate.py
python revision_graded_20260922/scripts/analyze.py
python revision_graded_20260922/scripts/state_contrasts.py
python revision_graded_20260922/scripts/report.py
python revision_graded_20260922/scripts/validate.py
python revision_graded_20260922/scripts/finalize.py
```

`data/input_manifest.json` identifies exact required payloads, sizes, hashes, array shapes and Git tracking state. All 445 main inputs were available locally. Git omits some large NPY payloads; restore the author's exact saved artifacts and verify SHA-256 before executing. `data/supplemental_audit_inputs.json` adds raw-label and historical-consumer/cost inputs, without changing the frozen design. Missing inputs cause a failure; there is no fallback to re-encoding.

`evaluate.py` reuses historical all-judged ranks and top lists. It reconstructs full-catalog hybrid rankings from frozen vectors/scores only where those all-judged hybrid ranks were not saved. It also constructs the 12 new canonical hybrids, requiring exact saved canonical judged-rank agreement. Every raw-hybrid highest rank and top-1000 list must match the previous completion. All outputs remain in this dated package.

`analyze.py` creates the shared query bootstrap weight matrices once per dataset and keeps each query's entire method/schedule cluster together. It averages matched schedule contrasts within a query before bootstrapping seven-schedule comparisons. Output tables use decimal 0–1 units (multiply by 100 for percentage points). No primary results use target-only ranks.

`validate.py` checks all supports and source hashes, reproduces 80 historical Recall cells and 42 old-gain raw cNDCG conditions, independently recalculates graded metrics on 366 query/condition records and checks bootstrap/membership logic. Tests and validation outcomes are under `qa/`.
''',encoding='utf8')
    (HERE/'data/DATA_DICTIONARY.md').write_text('''# Data dictionary

- `per_query.parquet` and equivalent compressed CSV: one dataset/model/method/schedule/query record; all 49,227 records. `nDCG@K` uses full absolute ranks; `cNDCG@K` uses condensed judged order. `old_*` is the same ranking with the old mapping, including counterfactual old-mapping values for newly built methods. `highest_support_ids` and `included_ids_K20/K100` are JSON product-ID lists. `rank_source` points to all-judged ranks, not target-only ranks. Fixed controls have one `fixed` row per query. Seven-schedule means are separate aggregate/contrast outputs.
- `*_highest_support.csv`, `*_judged_support.parquet`, `*_catalog_ordering.csv`, `*_queries_ordering.csv`: exact Hq, all-judgment support and full original axis orders. Eligible query IDs are in `PROTOCOL.json`.
- New `*_pairs.parquet`: all judged query/product rows with labels, full-catalog absolute rank and score. Raw-hybrid rows beyond C0 fill the previously missing all-judged evaluation payload. `*_top1000.npz` contains original catalog indices and explicit query IDs; existing raw-hybrid top lists remain referenced from the prior package.
- `paired_contrasts_per_query.parquet`: signed alternative-minus-reference metric differences; each schedule retained, with separate per-query seven-schedule means. Bootstrap weight columns follow the explicit eligible query IDs in the adjacent JSON file.
- `rq2_membership_ndcg_per_query.parquet` and compressed CSV: each of 60 raw/raw-hybrid non-C0 schedule contrasts, all queries, K20 highest-label gained/lost IDs, integer Recall cancellation, signed/absolute graded deltas and numerical-zero flags. `rq2_changed_query_ecdf.csv.gz` preserves sorted observations for K20 and separate K10 diagnostics. Tied observations produce repeated x values; the empirical CDF at a tied value is the maximum recorded cumulative fraction for that value.
- `observed_numerical_zero_membership_records.csv`: all observed membership-changed records within absolute nDCG20 tolerance 1e-12 (rtol 0); includes raw and hybrid, each identified. It is an observed subset, not an equivalence analysis.
- `persistent_states_per_query.parquet`: original Hq denominators, persistent inclusion, omission, VI, source Recall and mean Recall; fixed methods have structural VI=0. Summary macro/micro weighting is explicit.
- `canonical_hybrid_membership_per_query.parquet`: all gained/lost IDs at K20 and K100 versus the own dense branch, BM25 and each of seven raw-hybrid references. The numeric file also contains matched per-query means of the seven references. Mean membership fractions are not presented as a single set of product IDs.
- All metrics and deltas are 0–1 fractions. CIs are uncorrected 95% paired query-cluster percentile intervals. Costs separate array bytes, construction budget and measurement scope.
''',encoding='utf8')
    (HERE/'README.md').write_text('''# Graded evaluation and canonical–lexical follow-up

P0/P1 complete: **122 conditions, 49,227 per-query records**, fixed 308 WANDS / 499 ESCI query populations, all judged labels, corrected direct gains. No new encoding or training. Optional P2 permutation expansion deferred.

Start with [GRADED_METRIC_AUDIT.md](GRADED_METRIC_AUDIT.md), [CLAIM_VERDICTS.md](CLAIM_VERDICTS.md), and the [proposed English insertion](MANUSCRIPT_INSERTIONS.md). The current manuscript is unchanged.

- The ESCI grade swap is confirmed in actual stored grades; Phase IV's earlier correction is acknowledged. Frozen ranks, Hq and Recall remain unchanged.
- Changed Top-20 membership often accompanies substantial per-query nDCG change. A blanket stable-effectiveness conclusion is unsupported.
- All 12 canonical–BM25 hybrids improve nDCG20 over BM25; 9 improve over their own dense branch. Comparisons with raw-seven hybrid means are mostly inconclusive, with metric-specific exceptions retained.
- Protocol, exact input hashes/IDs, per-query data, all schedule/control tables, paired intervals, ECDF source data, CSV/LaTeX evidence, figure sources and reproducible scripts are included.

[Compact hybrid evidence](tables/compact_evidence.csv) · [All effectiveness values](tables/effectiveness_wide.csv) · [All paired intervals](tables/paired_contrasts.csv) · [RQ2 distributions/thresholds](tables/rq2_distribution_and_thresholds.csv) · [Validation](qa/validation.json) · [Reproduction](REPRODUCE.md)

![Changed-query nDCG distributions](figures/rq2_raw_changed_query_ecdf.png)

![Canonical hybrid comparisons](figures/canonical_hybrid_ndcg20_contrasts.png)
''',encoding='utf8')
    print('REPORT AND FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
