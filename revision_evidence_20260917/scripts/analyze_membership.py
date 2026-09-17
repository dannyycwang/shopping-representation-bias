"""All membership results are newly derived from inherited exact catalog ranks."""
from common import *
from itertools import combinations

METRICS = ['reference_recall','alternative_recall','gain_fraction','loss_fraction','net_delta','membership_change_fraction']
COUNTS = ['retained','newly_included','newly_omitted','absent_in_both']

def per_query(frame, ref, alt, k):
    a,b = frame[ref].to_numpy()<=k, frame[alt].to_numpy()<=k
    p = frame[['query_id']].copy()
    p['retained']=a&b; p['newly_included']=~a&b; p['newly_omitted']=a&~b; p['absent_in_both']=~a&~b
    p['relevant_count']=1
    q = p.groupby('query_id',sort=True).sum().astype(int)
    n = q.relevant_count
    q['reference_recall']=(q.retained+q.newly_omitted)/n
    q['alternative_recall']=(q.retained+q.newly_included)/n
    q['gain_fraction']=q.newly_included/n
    q['loss_fraction']=q.newly_omitted/n
    q['net_delta']=(q.newly_included-q.newly_omitted)/n
    q['membership_change_fraction']=(q.newly_included+q.newly_omitted)/n
    q['any_membership_change']=(q.newly_included+q.newly_omitted)>0
    q['exact_zero_net_with_changes']=(q.newly_included==q.newly_omitted)&q.any_membership_change
    q['positive_net']=q.newly_included>q.newly_omitted
    q['negative_net']=q.newly_included<q.newly_omitted
    assert (q[COUNTS].sum(axis=1)==n).all()
    assert np.allclose(q.net_delta,q.gain_fraction-q.loss_fraction,atol=1e-14,rtol=0)
    assert np.allclose(q.net_delta,q.alternative_recall-q.reference_recall,atol=1e-14,rtol=0)
    assert np.allclose(q.membership_change_fraction,q.gain_fraction+q.loss_fraction,atol=1e-14,rtol=0)
    return q

def summarize(q,w,meta):
    n=len(q); pairs=int(q.relevant_count.sum()); den=w@q.relevant_count.to_numpy(float)
    changed=int(q.any_membership_change.sum()); cancel=int(q.exact_zero_net_with_changes.sum())
    qnet=q.net_delta.to_numpy()
    extra=dict(eligible_queries=n,highest_relevance_pairs=pairs,
      queries_with_any_change=changed,queries_with_exact_zero_net_and_changes=cancel,
      within_query_cancellation_fraction=cancel/n,
      within_query_cancellation_conditional=cancel/changed if changed else np.nan,
      cancellation_all_denominator=n,cancellation_conditional_denominator=changed,
      queries_positive_net=int(q.positive_net.sum()),queries_negative_net=int(q.negative_net.sum()),
      positive_query_net_mass=float(np.maximum(qnet,0).mean()),negative_query_net_mass=float(np.maximum(-qnet,0).mean()),
      across_query_cancellation_mass=float(2*min(np.maximum(qnet,0).mean(),np.maximum(-qnet,0).mean())),
      within_query_balanced_exchange_mass=float((2*np.minimum(q.newly_included,q.newly_omitted)/q.relevant_count).mean()),
      bootstrap_seed=SEED,bootstrap_draws=DRAWS,ci_type='descriptive uncorrected 95% query-cluster percentile')
    for col in ['any_membership_change','exact_zero_net_with_changes']:
        lo,hi=ci((w@q[col].to_numpy(float))/n)
        extra[col+'_fraction_ci_low']=lo; extra[col+'_fraction_ci_high']=hi
    bc=w@q.exact_zero_net_with_changes.to_numpy(float); bd=w@q.any_membership_change.to_numpy(float)
    conditional=np.divide(bc,bd,out=np.full_like(bc,np.nan),where=bd>0)
    extra['cancellation_conditional_ci_low'],extra['cancellation_conditional_ci_high']=ci(conditional)
    extra['conditional_bootstrap_valid_draws']=int(np.isfinite(conditional).sum())
    rows=[]
    for agg in ['query_macro','pair_micro']:
        row={**meta,**extra,'aggregation':agg}
        if agg=='query_macro':
            v=q[METRICS].to_numpy(float); value=v.mean(0); boot=w@v/n
        else:
            v=q[METRICS].to_numpy(float)*q.relevant_count.to_numpy()[:,None]
            value=v.sum(0)/pairs; boot=(w@v)/den[:,None]
        for j,col in enumerate(METRICS):
            row[col]=float(value[j]); row[col+'_ci_low'],row[col+'_ci_high']=ci(boot[:,j])
        for col in COUNTS: row[col+'_total']=int(q[col].sum())
        rows.append(row)
    return rows

def main():
    pqout=[]; summary=[]; sources=[]; populations_out=[]; reconciliation=[]
    manifest=json.loads((ROOT/'GITHUB_BACKUP_MANIFEST.json').read_text())
    backup={x['path'].replace('\\','/'):x for x in manifest['files']}
    for ds in ['wands','esci']:
      p=products(ds); qsource=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
      j=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv',dtype={'product_id':str})
      h=j[j.label.eq(highest_label(ds))].sort_values(['query_id','product_id']).reset_index(drop=True)
      for model in ['minilm','bge_base','gte_modernbert']:
        f=rawwide(ds,model)
        assert np.array_equal(f[['query_id','product_id']].to_numpy(),h[['query_id','product_id']].to_numpy()),(ds,model,'judgments')
        assert f[STEMS].max().max()<=len(p)
        files=[ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet' for s in STEMS]
        prov=next(x for x in PROVENANCE['sources'] if x['dataset']==ds and x['model']==model)
        for path in files:
            record=source(path); old=backup.get(record['relative_path'])
            record['backup_sha256_matches']=old is not None and old['sha256']==record['sha256']
            assert record['backup_sha256_matches'],path
            sources.append(record)
        for pop,ids in populations(ds):
            z=f[f.query_id.isin(ids)]; eligible=sorted(map(int,z.query_id.unique())); w=weights(eligible)
            population=dict(dataset=ds,model=model,population=pop,requested_query_ids=ids,
              eligible_query_ids=eligible,eligible_queries=len(eligible),highest_relevance_pairs=len(z),catalog_size=len(p),
              historical_queries=len(qsource),highest_label=highest_label(ds),
              primary_role='WANDS existing held-out' if ds=='wands' else 'ESCI existing evaluation sample, not a new held-out split')
            populations_out.append(population)
            for k in [20,100]:
              for ref,alt in combinations(STEMS,2):
                q=per_query(z,ref,alt,k)
                assert list(q.index)==eligible
                meta=dict(dataset=ds,model=model,encoder_profile='native',catalog_size=len(p),population=pop,K=k,
                  reference=ref,alternative=alt,comparison_role='primary_C0_relative' if ref=='C0' else 'supplement_all_pairs',
                  rank_origin='inherited complete-catalog exact ranks',intervention='catalog_wide')
                summary.extend(summarize(q,w,meta))
                pqout.append(q.reset_index().assign(**meta))
        reconciliation.append(dict(dataset=ds,model=model,all_seven_pair_alignment=True,highest_judgments_exact=True,
                                   catalog_size=len(p),highest_relevance_pairs=len(f),eligible_queries=f.query_id.nunique()))
        print('MEMBERSHIP',ds,model,len(f),flush=True)
    pd.concat(pqout,ignore_index=True).to_csv(DATA/'membership_changes_per_query.csv',index=False)
    pd.DataFrame(summary).to_csv(DATA/'membership_changes_summary.csv',index=False)
    pd.DataFrame(reconciliation).to_csv(DATA/'membership_integrity.csv',index=False)
    dump(DATA/'evaluation_populations.json',populations_out)
    metadata=dict(derivation='No retrieval was run for this membership analysis.',schedules=STEMS,
      schedule_seeds=CFG['attribute_permutation_seeds'],cutoffs=[20,100],models=CFG['models'],query_prefix='',product_prefix='',
      tie_rule='Descending fp32 score, then ascending original catalog index. Catalog order is retained exactly.',
      baseline_names={'C0 Recall':'Within-query fraction of Hq ranked <=K in C0, then macro mean.',
        'mean_over_seven_schedules Recall':'Mean of the seven shared-index schedule Recalls, including C0.',
        'target inclusion':'Average independent-target inclusion; targets do not share one counterfactual index.',
        'VI':'At least one included and at least one omitted order in the finite seven-schedule family.',
        'Always':'Included in all seven schedules.','Never':'Omitted in all seven schedules.'},
      count_definitions={'retained':'I_ref & I_alt','newly_included':'not I_ref & I_alt','newly_omitted':'I_ref & not I_alt','absent_in_both':'not I_ref & not I_alt'},
      fraction_denominator='All distinct Hq products for the query, including absent_in_both.',
      aggregation='query_macro first averages within query; pair_micro separately weights each highest-relevance pair equally.',
      cancellation='Exact integer gains == losses > 0 before rounding. All-query and changed-query denominators exported separately. Across-query cancellation uses positive and negative query net masses; it is not within-query substitution.',
      uncertainty=dict(seed=SEED,draws=DRAWS,paired_query_clusters=True,interval='95% percentile, descriptive, uncorrected',equivalence=False),
      query_support_file=source(DATA/'evaluation_populations.json'),source_files=sources,
      context_sources=[source(ROOT/p) for p in ['phase2/config/phase2.json','phase4/config/splits.json','phase2/src/representations.py','phase2/src/evaluation.py','phase3/results/target_only_permutations/provenance.json']],
      output_files=[source(DATA/p) for p in ['membership_changes_per_query.csv','membership_changes_summary.csv','membership_integrity.csv']])
    dump(DATA/'membership_changes_sources.json',metadata)

if __name__=='__main__': main()
