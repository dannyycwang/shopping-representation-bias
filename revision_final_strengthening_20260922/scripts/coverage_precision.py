from common import *

def main():
 results=[];allrows=[]
 states=pd.read_parquet(HERE/'data/coverage_pair_states.parquet');increments=pd.read_parquet(HERE/'data/coverage_increment_pair_identities.parquet')
 for ds in ['wands','esci']:
  for model in ['minilm','bge_base']:
   audit=pd.read_csv(HERE/f'qa/{model}_coverage_forward_audit.csv');eps=float(audit[audit.dataset==ds].max_abs_query_score_error.max())
   rr=pd.read_parquet(HERE/f'data/{ds}_{model}_coverage_ranks.parquet')
   for scope in ['full','common_all_fitting']:
    f=rr if scope=='full' else rr[rr.fully_fitting]
    for k in [20,100]:
     for contrast,g in increments[(increments.dataset==ds)&increments.model.eq(model)&increments.support.eq(scope)&increments.K.eq(k)].groupby('contrast'):
      fam=contrast.split(' minus ')[0];ff=f[f.schedule_index<int(fam)] if ds=='wands' else f
      extremes=ff.groupby(['query_id','product_id'])[f'margin_K{k}'].agg(['min','max'])
      gg=g.set_index(['query_id','product_id']).copy();gg['observed_audit_score_error']=eps
      gg['witness_exceeds_observed_audit_error']=((gg.from_original_always_in.eq(1)&extremes['min'].lt(-eps))|(gg.from_original_always_out.eq(1)&extremes['max'].gt(eps)))
      gg['incremental_crossing_near_audit_error']=gg.incremental_crossing.eq(1)&~gg.witness_exceeds_observed_audit_error
      vals=gg[['incremental_crossing','witness_exceeds_observed_audit_error','incremental_crossing_near_audit_error']].astype(float).groupby(level='query_id').mean()
      results.append(dict(dataset=ds,model=model,support=scope,K=k,contrast=contrast,observed_audit_score_error=eps,new_crossing_pairs=int(gg.incremental_crossing.sum()),pairs_with_witness_beyond_audit_error=int(gg.witness_exceeds_observed_audit_error.sum()),pairs_near_observed_error=int(gg.incremental_crossing_near_audit_error.sum()),query_macro_increment=float(vals.incremental_crossing.mean()),query_macro_witness_beyond_error=float(vals.witness_exceeds_observed_audit_error.mean()),query_macro_near_error=float(vals.incremental_crossing_near_audit_error.mean()),interpretation='Descriptive sensitivity to largest score discrepancy over 16 hash-fixed historical products and cached queries; not a proven error bound or changed ranking tolerance.'))
      allrows.append(gg.reset_index())
 pd.DataFrame(results).to_csv(HERE/'qa/coverage_precision_sensitivity.csv',index=False);pd.concat(allrows).to_parquet(HERE/'data/coverage_precision_witnesses.parquet',index=False)
 print('COVERAGE PRECISION COMPLETE',flush=True)
if __name__=='__main__':main()
