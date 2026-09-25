"""One row per numeric assertion/cell/mark in v11 Section 6, with exact selectors."""
from common import *
import re

def main():
    inc=pd.read_csv(HERE/'tables/inclusion_complete.csv');mem=pd.read_csv(HERE/'tables/membership_all_contrasts.csv');ctl=pd.read_csv(HERE/'tables/controls_complete.csv');deltas=pd.read_csv(HERE/'tables/controls_paired_comparisons.csv');trans=pd.read_csv(HERE/'tables/transitions_complete.csv');rows=[]
    repository_revision=git('rev-parse','HEAD');serializer_revision=sha(ROOT/'phase2/src/representations.py')
    def add(location,claim,value,printed,ds='',model='',k='',metric='',source='',selector='',intervention='catalog_wide',schedule='original seven',comparator='',queries='',pairs='',decimals=2,scale=100,lo=None,hi=None,status=None,note=''):
        if ds and queries=='':queries={'wands':308,'esci':499}[ds]
        if ds and pairs=='':pairs={'wands':21299,'esci':4434}[ds]
        formatted=f'{float(value)*scale:.{decimals}f}'
        expected=str(printed)
        same=formatted==expected
        if status is None:status='PASS' if same else 'NEEDS_CORRECTION'
        modelrev=next((m['revision'] for m in CFG['models'] if m['key']==model),'not applicable / see per-encoder rows')
        rows.append(dict(claim_id=f'S6-{len(rows)+1:04d}',manuscript_location=location,claim=claim,dataset=ds,encoder=model,intervention=intervention,cutoff=k,schedule_family=schedule,queries=queries,pairs=pairs,query_support='308 WANDS eligible heldout / 499 ESCI eligible sampled; fitting queries restricted to nonempty target support',relevance_definition='WANDS Exact / ESCI E for inclusion; nDCG direct gains WANDS 3/1/0 and ESCI 1/.1/.01/0',weighting='query_macro unless explicitly count',comparator=comparator,source_file=source,source_columns=metric,source_selector=selector,run_id='revision_section6_20260924 (executed 2026-09-25)',model_revision=modelrev,serializer_revision=serializer_revision,repository_revision=repository_revision,full_precision_recomputation=float(value),manuscript_value=expected,formatted_recomputation=formatted,display_scale=scale,display_decimals=decimals,status=status,discrepancy='' if same else f'formatted {formatted} vs manuscript {expected}',bootstrap_population='eligible queries within the named support',bootstrap_pairing='same query sampled jointly across compared conditions; all its products retained',bootstrap_resamples=DRAWS,bootstrap_seed=SEED,ci_low=lo,ci_high=hi,interval='95% unadjusted paired query-cluster percentile' if lo is not None else 'not applicable to count or range; endpoint CIs in companion',notes=note))
        weighting='query_macro'
        if 'changed_abs_ndcg' in metric:weighting='query quantile conditional on changed relevant membership; range across named contrasts'
        elif metric=='vectors_per_product':weighting='structural dense vector count per product; no statistical weighting'
        elif 'direction' in claim:weighting='sign of a query-macro estimate or paired query-macro interval'
        elif scale==1 and decimals==0:weighting='integer count or structural audit predicate; no averaging unless explicitly stated'
        elif 'range' in claim:weighting='range across separate query-macro estimates; no pooling'
        rows[-1]['weighting']=weighting
        rows[-1]['ci_units']='raw metric units, before display_scale; a direction predicate retains its underlying delta interval' if lo is not None else 'not applicable'
        if claim.startswith('BM25 '):rows[-1]['model_revision']='not applicable to BM25; lexical reference repeated under each encoder comparison'
    for ds,n,p in [('wands',308,21299),('esci',499,4434)]:add('p6 opening','eligible query count',n,n,ds,source='data/'+ds+'_sampling_audit.csv',metric='eligible',queries=n,pairs=p,decimals=0,scale=1)
    # Ranges in RQ1 are across encoders, not pooled estimates.
    ranges=[('catalog_wide','full','wands','11.93','19.58'),('catalog_wide','full','esci','3.66','6.91'),('target_only','full','wands','10.83','19.45'),('target_only','full','esci','2.42','5.88'),('target_only','fully_fitting','wands','7.16','14.55'),('target_only','fully_fitting','esci','2.55','5.88')]
    for intervention,scope,ds,low,high in ranges:
        f=inc[inc.dataset.eq(ds)&inc.intervention.eq(intervention)&inc.target_support.eq(scope)&inc.K.eq(20)&inc.metric.eq('VI')]
        for kind,printed in [('min',low),('max',high)]:
            r=f.loc[f.estimate.idxmin() if kind=='min' else f.estimate.idxmax()]
            add('p6 RQ1 paragraphs 1–2',f'{intervention} {scope} VI range {kind}',r.estimate,printed,ds,r.model,20,'VI','tables/inclusion_complete.csv',f'{ds}/{intervention}/{scope}/K20/{r.model}',intervention,queries=r.queries,pairs=r.pairs,lo=r.ci_low,hi=r.ci_high,note='Range of encoder point estimates; not a confidence interval or matched causal contrast.')
    # Figure 3: each printed estimate and support count, including the duplicated GTE mark.
    f=inc[inc.intervention.eq('target_only')&inc.K.eq(20)&inc.metric.eq('VI')]
    anchors={('wands','minilm','full'):('19.45',308,21299),('wands','minilm','fully_fitting'):('10.59',118,925),('wands','bge_base','full'):('10.83',308,21299),('wands','bge_base','fully_fitting'):('7.16',239,6797),('wands','gte_modernbert','full'):('14.55',308,21299),('wands','gte_modernbert','fully_fitting'):('14.55',308,21299),('esci','minilm','full'):('2.91',499,4434),('esci','minilm','fully_fitting'):('3.20',460,2578),('esci','bge_base','full'):('2.42',499,4434),('esci','bge_base','fully_fitting'):('2.55',488,3587),('esci','gte_modernbert','full'):('5.88',499,4434),('esci','gte_modernbert','fully_fitting'):('5.88',499,4434)}
    for r in f.itertuples():
        val,n,p=anchors[(r.dataset,r.model,r.target_support)]
        for metric,v,pr,sc,dp in [('VI',r.estimate,val,100,2),('queries',r.queries,n,1,0),('pairs',r.pairs,p,1,0)]:
            add('p7 Figure 3',f'{r.target_support} {metric}',v,pr,r.dataset,r.model,20,metric,'tables/inclusion_complete.csv',f'{r.dataset}/{r.model}/target_only/{r.target_support}/K20',intervention='target_only',queries=r.queries,pairs=r.pairs,scale=sc,decimals=dp,lo=r.ci_low if metric=='VI' else None,hi=r.ci_high if metric=='VI' else None,note='GTE full and fitting are the same estimate, displayed once in revised asset.' if r.model=='gte_modernbert' else '')
    historic=pd.read_csv(FINAL/'tables/coverage_summary.csv')
    for m,fam,pr in [('minilm','7','20.85'),('minilm','32','28.21'),('bge_base','7','11.40'),('bge_base','32','17.55')]:
        r=historic[historic.dataset.eq('wands')&historic.model.eq(m)&historic.support.eq('full')&historic.family.eq(fam)&historic.K.eq(20)&historic.aggregation.eq('query_macro')&historic.metric.eq('VI')].iloc[0]
        add('p6 RQ1 paragraph 3','historical expanded-family VI',r['mean'],pr,'wands',m,20,'mean','../revision_final_strengthening_20260922/tables/coverage_summary.csv',f'wands/{m}/full/{fam}/20/query_macro/VI',intervention='target_only',schedule=fam+' schedules (historical mixed execution)',queries=128,pairs=7125,lo=r.ci_low,hi=r.ci_high,status='NEEDS_CORRECTION',note='Number reproduced, but mixed execution cannot isolate additional order effects. Use separately harmonized supplement; do not overwrite historical estimate.')
    add('p6 RQ1 paragraph 3','frozen supplement queries',128,128,'wands',source='../revision_final_strengthening_20260922/data/wands_coverage_support.csv',metric='query_id.nunique',queries=128,pairs=7125,scale=1,decimals=0)
    # RQ2 example and all Figure 4 marks retain six contrasts.
    raw=mem[mem.method.eq('raw')&mem.K.eq(20)]
    example=raw[raw.dataset.eq('wands')&raw.model.eq('bge_base')&raw.schedule.eq('C1')].iloc[0]
    for metric,pr in [('gain_share','2.357'),('loss_share','2.263'),('turnover','4.621'),('delta_recall','0.094')]:
        add('p6 RQ2 paragraph 1','BGE reversal '+metric,example[metric],pr,'wands','bge_base',20,metric,'tables/membership_all_contrasts.csv','wands/bge_base/raw/C1/K20',schedule='source vs Reverse',comparator='source C0',queries=308,pairs=21299,decimals=3,lo=example[metric+'_ci_low'],hi=example[metric+'_ci_high'],note='Gain+loss is summed at full precision. Displayed 2.357+2.263 vs rounded 4.621 is not an error.')
    for metric,pr in [('delta_recall_ci_low','-0.409'),('delta_recall_ci_high','0.563')]:add('p6 RQ2 paragraph 1','BGE reversal net interval endpoint',example[metric],pr,'wands','bge_base',20,metric,'tables/membership_all_contrasts.csv','wands/bge_base/raw/C1/K20',schedule='source vs Reverse',comparator='source C0',queries=308,pairs=21299,decimals=3)
    bg=raw[raw.dataset.eq('wands')&raw.model.eq('bge_base')]
    for kind,pr in [('min',67),('max',78)]:add('p6 RQ2 paragraph 1','equal positive gain/loss query count range '+kind,getattr(bg.equal_positive_queries,kind)(),pr,'wands','bge_base',20,'equal_positive_queries','tables/membership_all_contrasts.csv','wands/bge_base/raw/K20; six separate contrasts',queries=308,pairs=21299,scale=1,decimals=0,note='Per-contrast range, never a union or sum of query counts.')
    for r in raw.itertuples():
        for metric in ['gain_share','loss_share','delta_recall']:
            add('p7 Figure 4',f'{r.schedule} plotted {metric}',getattr(r,metric),f'{100*getattr(r,metric):.6f}',r.dataset,r.model,20,metric,'tables/membership_all_contrasts.csv',f'{r.dataset}/{r.model}/raw/{r.schedule}/K20',schedule='source vs '+r.schedule,comparator='source C0',queries=r.queries,pairs=r.pairs,decimals=6,lo=getattr(r,metric+'_ci_low'),hi=getattr(r,metric+'_ci_high'),note='Plot has no numeric label; full-precision plotted mark recovered from source.')
    for ds,medlo,medhi,dlo,dhi in [('wands','.0226','.0399','-.01101','.00151'),('esci','.0388','.0482','-.00393','.00092')]:
        f=raw[raw.dataset.eq(ds)]
        for metric,kind,pr,dp in [('changed_abs_ndcg_q0.5','min',medlo,4),('changed_abs_ndcg_q0.5','max',medhi,4),('delta_ndcg','min',dlo,5),('delta_ndcg','max',dhi,5)]:
            v=getattr(f[metric],kind)();add('p6 RQ2 paragraph 2',metric+' range '+kind,v,f'{float(pr):.{dp}f}',ds,k=20,metric=metric,source='tables/membership_all_contrasts.csv',selector=ds+'/raw/K20/all encoders and six contrasts',scale=1,decimals=dp,note='Absolute distributions conditional on changed queries; signed mean delta uses all eligible queries.')
    add('p6 RQ2 paragraph 2','uncorrected negative intervals count',int((raw.delta_ndcg_ci_high<0).sum()),5,k=20,metric='delta_ndcg_ci_high<0',source='tables/membership_all_contrasts.csv',selector='raw/K20; 2 datasets x 3 models x 6 contrasts',scale=1,decimals=0,note='Remove significant-count inference from replacement prose; no multiplicity-adjusted claim is made.')
    # Recover actual Table 1 numbers from the local v11 text; no forced old values.
    txt=(HERE/'inputs/section6_v11_raw.txt').read_text(encoding='utf8');body=txt.split('Table 1: Effectiveness')[1].split('omits 0.23%')[0];ds='wands'
    lookup={'Raw dense':'raw','Canonical: ascending':'lexical_ascending','Canonical: descending':'lexical_descending','Canonical: type priority':'field_priority_type','Set-Mean':'set_mean','Centroid (𝑀 = 2)':'centroid_2','Multi-vector (𝑀 = 2)':'max_2','BM25':'BM25','Raw hybrid':'raw_hybrid','Canonical hybrid: ascending':'canonical_hybrid_lexical_ascending','Canonical hybrid: descending':'canonical_hybrid_lexical_descending','Canonical hybrid: type priority':'canonical_hybrid_field_priority_type'}
    count=0
    for line in body.splitlines():
        if line.strip()=='ESCI':ds='esci'
        match=re.match(r'(.+?)\s+((?:\d+\.\d+\s*){6})$',line.strip())
        if not match or match[1] not in lookup:continue
        method=lookup[match[1]];values=match[2].split();s='seven_mean' if method in ['raw','raw_hybrid'] else 'fixed'
        for j,m in enumerate(['minilm','bge_base']):
            for i,(metric,k) in enumerate([('Recall',100),('VI',100),('nDCG',20)]):
                r=ctl[ctl.dataset.eq(ds)&ctl.model.eq(m)&ctl.method.eq(method)&ctl.schedule.eq(s)&ctl.K.eq(k)&ctl.metric.eq(metric)].iloc[0]
                pr=values[j*3+i];add('p8 Table 1',method+' '+metric,r.estimate,pr,ds,m,k,metric,'tables/controls_complete.csv',f'{ds}/{m}/{method}/{s}/{k}/{metric}',schedule=s,queries=r.queries,pairs=r.pairs,decimals=4 if metric=='nDCG' else 2,scale=1 if metric=='nDCG' else 100,lo=r.ci_low,hi=r.ci_high);count+=1
    assert count==144,('Table 1 parsed cells',count)
    # Tables 2 and 3 duplicate all twelve numeric cells.
    for table in [2,3]:
        for family,vals in [('dense',[20.72,.23,7.57,12.01,.43,59.04]),('hybrid',[29.27,.21,6.77,6.93,.42,56.40])]:
            for (rs,cs),pr in zip([(r,c) for r in ['persistent_inclusion','crossing','persistent_omission'] for c in ['included','omitted']],vals):
                r=trans[trans.dataset.eq('wands')&trans.model.eq('minilm')&trans.rule.eq('lexical_ascending')&trans.K.eq(20)&trans.family.eq(family)&trans.reference_state.eq(rs)&trans.control_state.eq(cs)].iloc[0]
                add(f'p9 Table {table}',family+' '+rs+' to '+cs,r.estimate,f'{pr:.2f}','wands','minilm',20,'estimate','tables/transitions_complete.csv',f'wands/minilm/{family}/ascending/20/{rs}/{cs}',schedule='fixed ascending versus own raw seven',comparator=family+' raw seven',queries=308,pairs=21299,lo=r.ci_low,hi=r.ci_high,note='Duplicate table: retain exactly one main transition table.' if table==3 else 'Six cells partition all Hq; not conditional failure rates.')
    for rs,cs,pr in [('crossing','included','7.57'),('crossing','omitted','12.01'),('persistent_inclusion','omitted','0.23')]:
        r=trans[trans.dataset.eq('wands')&trans.model.eq('minilm')&trans.family.eq('dense')&trans.rule.eq('lexical_ascending')&trans.K.eq(20)&trans.reference_state.eq(rs)&trans.control_state.eq(cs)].iloc[0]
        add('p6 RQ3 paragraph 2 / p8 continuation',rs+' to '+cs,r.estimate,pr,'wands','minilm',20,'estimate','tables/transitions_complete.csv',f'wands/minilm/dense/ascending/20/{rs}/{cs}',schedule='fixed ascending',comparator='raw dense seven',queries=308,pairs=21299,lo=r.ci_low,hi=r.ci_high,note='Revised text also includes 0.43% persistent omission to inclusion.')
    # Explicit canonical-hybrid comparator and visible R@20 loss.
    f=deltas[deltas.method.str.startswith('canonical_hybrid_')&deltas.metric.eq('nDCG')&deltas.K.eq(20)]
    for ref,printed in [('BM25:fixed',12),('own dense',9),('raw_hybrid:seven_mean',2)]:
        mask=f.comparator.eq(ref) if ref!='own dense' else f.apply(lambda r:r.comparator==r.method.replace('canonical_hybrid_','')+':fixed',axis=1)
        ff=f[mask];add('p8 RQ3 canonical hybrid paragraph','positive unadjusted nDCG intervals vs '+ref,int((ff.ci_low>0).sum()),printed,k=20,metric='ci_low>0',source='tables/controls_paired_comparisons.csv',selector=ref,comparator=ref,scale=1,decimals=0,note='Counts audited descriptively; revised prose uses named effect sizes and intervals without multiple-significance counting.')
    r=deltas[deltas.dataset.eq('wands')&deltas.model.eq('bge_base')&deltas.method.eq('canonical_hybrid_lexical_ascending')&deltas.comparator.eq('raw_hybrid:seven_mean')&deltas.K.eq(20)&deltas.metric.eq('Recall')].iloc[0]
    for col,pr in [('estimate','-0.624'),('ci_low','-1.370'),('ci_high','-0.084')]:add('p8 RQ3 canonical hybrid paragraph','canonical ascending HYBRID Recall@20 '+col,r[col],pr,'wands','bge_base',20,col,'tables/controls_paired_comparisons.csv','canonical_hybrid_lexical_ascending vs raw_hybrid:seven_mean',schedule='fixed canonical hybrid',comparator='raw hybrid seven-schedule mean',queries=308,pairs=21299,decimals=3,lo=r.ci_low,hi=r.ci_high,note='Comparator clarified. This is not pure ascending dense vs source order.')
    # Quantified directional prose is checked against its named comparator, too.
    for ds,m in [('wands','minilm'),('wands','bge_base'),('esci','minilm'),('esci','bge_base')]:
        r=deltas[deltas.dataset.eq(ds)&deltas.model.eq(m)&deltas.method.eq('set_mean')&deltas.K.eq(100)&deltas.metric.eq('Recall')&deltas.comparator.eq('raw:seven_mean')].iloc[0]
        intended=1 if (ds,m)==('wands','minilm') else -1
        add('p8 RQ3 effectiveness paragraph','Set-Mean minus raw-seven Recall@100 direction',np.sign(r.estimate),intended,ds,m,100,'sign(estimate)','tables/controls_paired_comparisons.csv',f'{ds}/{m}/set_mean/K100/Recall vs raw:seven_mean',schedule='fixed',comparator='raw dense seven-schedule mean',scale=1,decimals=0,lo=r.ci_low,hi=r.ci_high,note=f'Prose direction only; full delta = {r.estimate:.17g}. No significance assertion for this sentence.')
        for metric in ['Recall','VI']:
            r=deltas[deltas.dataset.eq(ds)&deltas.model.eq(m)&deltas.method.eq('raw_hybrid')&deltas.schedule.eq('seven_mean')&deltas.K.eq(100)&deltas.metric.eq(metric)&deltas.comparator.eq('raw:seven_mean')].iloc[0]
            intended=0 if (ds,m)==('esci','bge_base') else (1 if metric=='Recall' else -1)
            direction=1 if r.ci_low>0 else (-1 if r.ci_high<0 else 0)
            add('p8 RQ3 effectiveness paragraph','raw hybrid minus raw dense '+metric+'@100 interval direction',direction,intended,ds,m,100,'ci_low,ci_high','tables/controls_paired_comparisons.csv',f'{ds}/{m}/raw_hybrid/seven_mean/K100/{metric} vs raw:seven_mean',schedule='seven_mean',comparator='raw dense seven-schedule mean',scale=1,decimals=0,lo=r.ci_low,hi=r.ci_high,note=f'Direction coding: +1 entirely positive, -1 entirely negative, 0 contains zero; full delta = {r.estimate:.17g}. Unadjusted intervals.')
    fixed=ctl[ctl.schedule.eq('fixed')&ctl.metric.eq('VI')]
    add('p6 RQ3 opening / p8 Table 1 caption','fixed-control VI across all saved cells',fixed.estimate.abs().max(),0,metric='max(abs(VI))',source='tables/controls_complete.csv',selector='schedule=fixed; metric=VI; every available dataset/model/method/K',schedule='fixed',scale=1,decimals=0,note='Structural invariance by construction, not a learned success. Support varies by dataset; all method cells preserved.')
    f=ctl[ctl.model.isin(['minilm','bge_base'])&ctl.method.eq('raw_hybrid')&ctl.schedule.eq('seven_mean')&ctl.K.eq(100)&ctl.metric.eq('VI')]
    add('p8 RQ3 effectiveness paragraph','raw hybrid nonzero VI settings',int((f.estimate>0).sum()),4,k=100,metric='estimate>0',source='tables/controls_complete.csv',selector='MiniLM/BGE x WANDS/ESCI raw_hybrid/seven_mean/K100/VI',scale=1,decimals=0)
    f=deltas[deltas.method.str.startswith('canonical_hybrid_')&deltas.comparator.eq('raw_hybrid:seven_mean')]
    for metric,k,printed in [('nDCG',20,10),('Recall',100,12)]:
        g=f[f.metric.eq(metric)&f.K.eq(k)]
        add('p8 RQ3 canonical hybrid paragraph',f'{metric}@{k} intervals containing zero versus raw hybrid',int(((g.ci_low<=0)&(g.ci_high>=0)).sum()),printed,k=k,metric='ci_low<=0 and ci_high>=0',source='tables/controls_paired_comparisons.csv',selector=f'12 dataset/encoder/rule contrasts; {metric}/K{k}',comparator='raw hybrid seven-schedule mean',scale=1,decimals=0,note='Audited count; revised prose avoids inferential significance counting. Zero-containing intervals do not establish equivalence.')
    for method in RULES+['centroid_2','centroid_4','centroid_7','max_2','max_4','max_7']:
        g=ctl[ctl.method.eq(method)]
        expected=int(method.split('_')[-1]) if method.startswith('max_') else 1
        assert g.vectors_per_product.nunique()==1
        add('p8 RQ3 cost paragraph',method+' dense vectors stored/scored per product',g.vectors_per_product.iloc[0],expected,metric='vectors_per_product',source='tables/controls_complete.csv',selector=method,intervention='index construction and retrieval',schedule='fixed',scale=1,decimals=0,note='Construction encodings are separate: centroid and multi-vector encode M views; only multi-vector retains/scores M. Sparse branch cost is additional.')
    add('p6 RQ2 paragraph 2','remaining unadjusted nDCG intervals contain zero',int(((raw.delta_ndcg_ci_low<=0)&(raw.delta_ndcg_ci_high>=0)).sum()),31,k=20,metric='ci_low<=0 and ci_high>=0',source='tables/membership_all_contrasts.csv',selector='raw/K20; all 36 source-versus-alternative contrasts',scale=1,decimals=0,note='The original prose says the other intervals contain zero: 36 total minus five negative = 31. Audited as a descriptive predicate; not used as a multiple-comparison inference in the revision.')
    result=pd.DataFrame(rows)
    result['resolution']='verified against saved evidence'
    result.loc[result.status.eq('NEEDS_CORRECTION'),'resolution']='historical mixed execution; replace with separately harmonized supplement when complete'
    if (HERE/'qa/harmonized_validation.json').exists():
        result.loc[result.status.eq('NEEDS_CORRECTION'),'resolution']='resolved in proposed revision by tables/harmonized_summary.csv and harmonized_supplement.md; original v11 historical value intentionally unchanged'
    csv(result,HERE/'section6_claim_ledger.csv')
    dump(HERE/'qa/ledger_summary.json',dict(rows=len(rows),status_counts=pd.Series([r['status'] for r in rows]).value_counts().to_dict(),scope='v11 Section 6 prose, Figure 3/4 marks and support labels, Table 1 and duplicate Tables 2/3; structural/cost prose in editorial handoff'))
    print(pd.DataFrame(rows).query("status!='PASS'")[['claim_id','manuscript_location','claim','discrepancy','notes']].to_string(index=False))
if __name__=='__main__':main()
