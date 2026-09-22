from common import *

def main():
 verify();pq=pd.read_parquet(GRADED/'data/per_query.parquet');records=[];summ=[];identities=[]
 states=['persistent_inclusion','crossing','persistent_omission']
 for ds in ['wands','esci']:
  ids=support(ds);w=weights(ids)
  for model in ['minilm','bge_base']:
   for family,raw,controlprefix in [('dense','raw',''),('hybrid','raw_hybrid','canonical_hybrid_')]:
    fs=[pq[(pq.dataset==ds)&(pq.model==model)&(pq.method==raw)&(pq.schedule==s)].set_index('query_id').loc[ids] for s in STEMS]
    for rule in RULES:
     control=pq[(pq.dataset==ds)&(pq.model==model)&(pq.method==controlprefix+rule)].set_index('query_id').loc[ids]
     for k in [20,100]:
      counts=[];N=[]
      for qid in ids:
       h=set(json.loads(control.loc[qid,'highest_support_ids']));sets=[set(json.loads(f.loc[qid,f'included_ids_K{k}'])) for f in fs];always=set.intersection(*sets);ever=set.union(*sets);fixed=set(json.loads(control.loc[qid,f'included_ids_K{k}']));stateids=[always,ever-always,h-ever];nn=[]
       assert set.union(*stateids)==h and sum(map(len,stateids))==len(h)
       for state,ss in zip(states,stateids):
        for column,cs in [('included',ss&fixed),('omitted',ss-fixed)]:
         nn.append(len(cs));records.append(dict(dataset=ds,model=model,family=family,rule=rule,K=k,query_id=qid,reference_state=state,control_state=column,cell_pairs=len(cs),Hq=len(h),state_pairs=len(ss),cell_fraction=len(cs)/len(h),conditional_fraction=len(cs)/len(ss) if ss else None))
         for pid in sorted(cs):identities.append(dict(dataset=ds,model=model,family=family,rule=rule,K=k,query_id=qid,product_id=pid,reference_state=state,control_state=column))
       assert sum(nn)==len(h) and sum(nn[::2])==len(fixed)
       counts.append(nn);N.append(len(h))
      cnt=np.asarray(counts,dtype=float);N=np.asarray(N,dtype=float)
      for agg in ['query_macro','pair_micro']:
       vals=cnt/N[:,None] if agg=='query_macro' else cnt;den=np.ones(len(ids)) if agg=='query_macro' else N
       mass=vals.sum(0)/den.sum();boots=(w@vals)/(w@den)[:,None];low,high=np.quantile(boots,[.025,.975],axis=0)
       for i,state in enumerate(states):
        st=vals[:,2*i]+vals[:,2*i+1];bd=w@st;bn=w@vals[:,2*i];ratio=np.divide(bn,bd,out=np.full(DRAWS,np.nan),where=bd>0);ql,qh=np.nanquantile(ratio,[.025,.975]) if np.any(bd>0) else [None,None]
        for j,column in enumerate(['included','omitted']):
         idx=2*i+j;summ.append(dict(dataset=ds,model=model,family=family,rule=rule,K=k,aggregation=agg,reference_state=state,control_state=column,cell_fraction=mass[idx],ci_low=low[idx],ci_high=high[idx],eligible_queries=len(ids),highest_pairs=int(N.sum()),cell_pairs=int(cnt[:,idx].sum()),reference_state_pairs=int((cnt[:,2*i]+cnt[:,2*i+1]).sum()),queries_with_reference_state=int(np.sum(st>0)),conditional_inclusion=float(vals[:,2*i].sum()/st.sum()) if st.sum()>0 else None,conditional_ci_low=ql,conditional_ci_high=qh,conditional_weighting='ratio of query-macro cell masses' if agg=='query_macro' else 'pair-micro conditional ratio',bootstrap_seed=SEED,bootstrap_draws=DRAWS,interval='95% paired query cluster; uncorrected'))
    print('TRANSITIONS',ds,model,family,flush=True)
 pd.DataFrame(records).to_parquet(HERE/'data/control_transitions_per_query.parquet',index=False);pd.DataFrame(identities).to_parquet(HERE/'data/control_transition_identities.parquet',index=False);pd.DataFrame(summ).to_csv(HERE/'tables/control_transitions.csv',index=False)
 dump(HERE/'qa/transitions.json',dict(configurations=48,summary_rows=len(summ),identity_records=len(identities),row_partition_verified=True,column_fixed_recall_verified=True,identities_not_aggregate_subtraction=True))

if __name__=='__main__':main()
