"""Paired query-cluster inference and complete membership/graded diagnostics."""
from common import *
METRICS=['nDCG@10','nDCG@20','cNDCG@10','cNDCG@20','JudgedCoverage@10','JudgedCoverage@20','Recall@20','Recall@100']

def main():
    verify_protocol();pq=pd.read_parquet(HERE/'data/per_query.parquet')
    summary=[];contrasts=[];delta_records=[];rq=[];rqsummary=[];ecdf=[];stateq=[];states=[];p1members=[];p1summary=[]
    frames={}
    def ci_rows(frame,meta):
        ds=meta['dataset'];assert frame.index.tolist()==support(ds)
        mean,lo,hi=interval(frame[METRICS].to_numpy(),ds)
        for metric,m,l,h in zip(METRICS,mean,lo,hi):
            summary.append(dict(**meta,metric=metric,mean=float(m),ci_low=float(l),ci_high=float(h),eligible_queries=len(frame),highest_pairs=int(frame.n_highest.sum()),bootstrap_seed=SEED,bootstrap_draws=DRAWS,interval='uncorrected 95% query-cluster percentile'))
    def contrast(alt,ref,meta):
        ds=meta['dataset'];assert alt.index.equals(ref.index) and alt.index.tolist()==support(ds)
        delta=alt[METRICS]-ref[METRICS];mean,lo,hi=interval(delta.to_numpy(),ds)
        for metric,m,l,h in zip(METRICS,mean,lo,hi):
            verdict='improves' if l>0 else 'reduces' if h<0 else 'inconclusive; equivalence not established'
            contrasts.append(dict(**meta,metric=metric,delta=float(m),ci_low=float(l),ci_high=float(h),verdict=verdict,eligible_queries=len(delta),bootstrap_seed=SEED,bootstrap_draws=DRAWS,interval='uncorrected 95% paired query-cluster percentile'))
        d=delta.reset_index()
        for key,val in meta.items():d[key]=val
        delta_records.append(d)
    for key,g in pq.groupby(['dataset','model','method','schedule'],sort=True):
        ds,model,method,schedule=key;f=g.set_index('query_id').sort_index();frames[key]=f
        ci_rows(f,dict(dataset=ds,model=model,method=method,schedule=schedule))
    for ds in ['wands','esci']:
        ids=support(ds)
        for model in MODELS:
            for method in ['raw']+(['raw_hybrid'] if model!='gte_modernbert' else []):
                fs=[frames[(ds,model,method,s)] for s in STEMS]
                avg=fs[0][['n_highest','n_judged']].copy();avg[METRICS]=sum(f[METRICS] for f in fs)/7
                frames[(ds,model,method,'seven_mean')]=avg
                ci_rows(avg,dict(dataset=ds,model=model,method=method,schedule='seven_mean'))
                base=fs[0]
                for s,alt in zip(STEMS[1:],fs[1:]):
                    contrast(alt,base,dict(dataset=ds,model=model,method=method,contrast='schedule_minus_C0',schedule=s,reference='C0'))
                    rows=[]
                    for qid in ids:
                        b=base.loc[qid];a=alt.loc[qid];bi=set(json.loads(b.included_ids_K20));ai=set(json.loads(a.included_ids_K20));gained=ai-bi;lost=bi-ai
                        changed=bool(gained or lost);d20=float(a['nDCG@20']-b['nDCG@20']);d10=float(a['nDCG@10']-b['nDCG@10'])
                        r=dict(dataset=ds,model=model,method=method,schedule=s,reference='C0',query_id=qid,n_highest=int(a.n_highest),membership_cutoff=20,gained_count=len(gained),lost_count=len(lost),gained_ids=json.dumps(sorted(gained),separators=(',',':')),lost_ids=json.dumps(sorted(lost),separators=(',',':')),membership_changed=changed,integer_recall_cancellation=bool(gained and len(gained)==len(lost)),delta_Recall20=(len(gained)-len(lost))/int(a.n_highest),delta_nDCG20=d20,abs_delta_nDCG20=abs(d20),delta_nDCG10=d10,abs_delta_nDCG10=abs(d10),numerical_zero_nDCG20=abs(d20)<=ZERO_TOL,numerical_zero_nDCG10=abs(d10)<=ZERO_TOL)
                        assert abs(r['delta_Recall20']-(a['Recall@20']-b['Recall@20']))<1e-14
                        rows.append(r)
                    rq.extend(rows);rf=pd.DataFrame(rows);ch=rf[rf.membership_changed]
                    for cutoff in [20,10]:
                        col=f'abs_delta_nDCG{cutoff}';v=ch[col].to_numpy()
                        r=dict(dataset=ds,model=model,method=method,schedule=s,ndcg_cutoff=cutoff,membership_cutoff=20,all_queries=len(rf),changed_queries=len(ch),changed_fraction=len(ch)/len(rf),integer_recall_cancellations_changed=int(ch.integer_recall_cancellation.sum()),numerical_zero_changed=int(ch[f'numerical_zero_nDCG{cutoff}'].sum()),numerical_atol=ZERO_TOL,numerical_rtol=0,cutoff_note='same K20' if cutoff==20 else 'K10 diagnostic conditioned on K20 membership; different cutoff')
                        for quantile in [0,.25,.5,.75,.9,.95,1.]:r[f'abs_delta_q{int(100*quantile):02d}']=float(np.quantile(v,quantile)) if len(v) else None
                        for threshold in [.001,.005,.01]:
                            n=int(np.sum(v<=threshold));tag=str(threshold)
                            r[f'changed_at_or_below_{tag}']=n;r[f'changed_above_{tag}']=len(v)-n
                            r[f'fraction_changed_at_or_below_{tag}']=n/len(v) if len(v) else None
                            r[f'fraction_all_changed_at_or_below_{tag}']=n/len(rf)
                        rqsummary.append(r)
                        for i,(qid,value) in enumerate(ch[['query_id',col]].sort_values([col,'query_id']).itertuples(index=False,name=None),1):
                            ecdf.append(dict(dataset=ds,model=model,method=method,schedule=s,ndcg_cutoff=cutoff,membership_cutoff=20,query_id=int(qid),absolute_delta=float(value),ecdf=i/len(ch),changed_queries=len(ch),all_queries=len(rf)))
                # Persistent states under the same Hq for the seven schedules.
                for k in [20,100]:
                    for qid in ids:
                        sets=[set(json.loads(f.loc[qid,f'included_ids_K{k}'])) for f in fs]
                        h=set(json.loads(fs[0].loc[qid,'highest_support_ids']));always=set.intersection(*sets);ever=set.union(*sets)
                        stateq.append(dict(dataset=ds,model=model,method=method,K=k,query_id=qid,n_highest=len(h),persistent_inclusion=len(always)/len(h),persistent_omission=len(h-ever)/len(h),VI=len(ever-always)/len(h),mean_recall=float(np.mean([len(s)/len(h) for s in sets])),source_recall=len(sets[0])/len(h),basis='observed seven catalog schedules'))
            if model!='gte_modernbert':
                for s in [*STEMS,'seven_mean']:
                    contrast(frames[(ds,model,'raw_hybrid',s)],frames[(ds,model,'raw',s)],dict(dataset=ds,model=model,method='raw_hybrid',contrast='matched_hybrid_minus_raw',schedule=s,reference='raw_'+s))
            # Every fixed control versus both explicitly identified raw references.
            controlkeys=[key for key in frames if key[0]==ds and key[1]==model and key[3]=='fixed']+[(ds,'lexical','BM25','fixed')]
            for key in controlkeys:
                f=frames[key];method=key[2]
                for s in ['C0','seven_mean']:
                    contrast(f,frames[(ds,model,'raw',s)],dict(dataset=ds,model=model,method=method,contrast='control_minus_raw',schedule='fixed',reference='raw_'+s))
                for k in [20,100]:
                    for qid,row in f.iterrows():
                        recall=row[f'Recall@{k}']
                        stateq.append(dict(dataset=ds,model=model,method=method,K=k,query_id=int(qid),n_highest=int(row.n_highest),persistent_inclusion=recall,persistent_omission=1-recall,VI=0.,mean_recall=recall,source_recall=recall,basis='structural: single deterministic fixed index'))
                if method.startswith('canonical_hybrid_'):
                    rule=method[len('canonical_hybrid_'):]
                    refs=[(rule,frames[(ds,model,rule,'fixed')]),('BM25',frames[(ds,'lexical','BM25','fixed')]),('raw_hybrid_seven_mean',frames[(ds,model,'raw_hybrid','seven_mean')])]
                    for name,ref in refs:contrast(f,ref,dict(dataset=ds,model=model,method=method,contrast='canonical_hybrid_comparator',schedule='fixed',reference=name))
                    # Membership records for every reference schedule and both cutoffs.
                    mrefs=refs[:2]+[('raw_hybrid_'+s,frames[(ds,model,'raw_hybrid',s)]) for s in STEMS]
                    for name,ref in mrefs:
                        for k in [20,100]:
                            for qid in ids:
                                aa=set(json.loads(f.loc[qid,f'included_ids_K{k}']));bb=set(json.loads(ref.loc[qid,f'included_ids_K{k}']));n=int(f.loc[qid,'n_highest'])
                                p1members.append(dict(dataset=ds,model=model,method=method,reference=name,K=k,query_id=qid,n_highest=n,retained=len(aa&bb)/n,newly_included=len(aa-bb)/n,newly_omitted=len(bb-aa)/n,absent_in_both=(n-len(aa|bb))/n,delta_recall=(len(aa)-len(bb))/n,gained_ids=json.dumps(sorted(aa-bb),separators=(',',':')),lost_ids=json.dumps(sorted(bb-aa),separators=(',',':'))))
        print('ANALYZED',ds,flush=True)
    sq=pd.DataFrame(stateq);statecols=['persistent_inclusion','persistent_omission','VI','mean_recall','source_recall']
    assert np.allclose(sq.persistent_inclusion+sq.persistent_omission+sq.VI,1,atol=1e-14,rtol=0)
    for key,g in sq.groupby(['dataset','model','method','K']):
        ds,model,method,k=key;g=g.sort_values('query_id');n=g.n_highest.to_numpy();v=g[statecols].to_numpy();w=weights(ds)
        for agg in ['query_macro','pair_micro']:
            point=v.mean(0) if agg=='query_macro' else n@v/n.sum()
            boot=w@v/len(v) if agg=='query_macro' else (w@(v*n[:,None]))/(w@n)[:,None]
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            r=dict(dataset=ds,model=model,method=method,K=k,aggregation=agg,eligible_queries=len(g),highest_pairs=int(n.sum()),basis=g.basis.iloc[0])
            for c,m,l,h in zip(statecols,point,lo,hi):r.update({c:m,c+'_ci_low':l,c+'_ci_high':h})
            states.append(r)
    mf=pd.DataFrame(p1members);mcols=['retained','newly_included','newly_omitted','absent_in_both','delta_recall']
    # First average matched raw references within each query, preserving the query cluster.
    mm=mf[mf.reference.str.startswith('raw_hybrid_')].groupby(['dataset','model','method','K','query_id'],as_index=False)[['n_highest',*mcols]].mean();mm['reference']='raw_hybrid_seven_mean'
    numeric=pd.concat([mf.drop(columns=['gained_ids','lost_ids']),mm],ignore_index=True)
    for key,g in numeric.groupby(['dataset','model','method','reference','K']):
        ds,model,method,ref,k=key;g=g.sort_values('query_id');m,lo,hi=interval(g[mcols].to_numpy(),ds)
        r=dict(dataset=ds,model=model,method=method,reference=ref,K=k,eligible_queries=len(g),highest_pairs=int(g.n_highest.sum()))
        for col,a,b,c in zip(mcols,m,lo,hi):r.update({col:a,col+'_ci_low':b,col+'_ci_high':c})
        p1summary.append(r)
    pd.DataFrame(summary).to_csv(HERE/'tables/effectiveness_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(HERE/'tables/paired_contrasts.csv',index=False)
    pd.concat(delta_records,ignore_index=True).to_parquet(HERE/'data/paired_contrasts_per_query.parquet',index=False)
    pd.DataFrame(rq).to_parquet(HERE/'data/rq2_membership_ndcg_per_query.parquet',index=False)
    pd.DataFrame(rq).to_csv(HERE/'data/rq2_membership_ndcg_per_query.csv.gz',index=False)
    pd.DataFrame(rqsummary).to_csv(HERE/'tables/rq2_distribution_and_thresholds.csv',index=False)
    pd.DataFrame(ecdf).to_csv(HERE/'data/rq2_changed_query_ecdf.csv.gz',index=False)
    sq.to_parquet(HERE/'data/persistent_states_per_query.parquet',index=False)
    pd.DataFrame(states).to_csv(HERE/'tables/persistent_states.csv',index=False)
    mf.to_parquet(HERE/'data/canonical_hybrid_membership_per_query.parquet',index=False)
    numeric.to_parquet(HERE/'data/canonical_hybrid_membership_numeric.parquet',index=False)
    pd.DataFrame(p1summary).to_csv(HERE/'tables/canonical_hybrid_membership.csv',index=False)
    # Quantify the correction on exactly the same rankings and Hq.
    correction=[]
    for key,g in pq.groupby(['dataset','model','method','schedule']):
        ds,model,method,s=key;g=g.sort_values('query_id')
        for metric in METRICS[:4]:
            m,l,h=interval(g[metric].to_numpy()-g['old_'+metric].to_numpy(),ds)
            correction.append(dict(dataset=ds,model=model,method=method,schedule=s,metric=metric,old_mean=g['old_'+metric].mean(),corrected_mean=g[metric].mean(),delta=float(m),ci_low=float(l),ci_high=float(h),eligible_queries=len(g)))
    pd.DataFrame(correction).to_csv(HERE/'tables/gain_correction.csv',index=False)
    # Compact wide tables, with all rules retained and C0 distinct from seven_mean.
    sm=pd.DataFrame(summary)
    wide=sm.pivot(index=['dataset','model','method','schedule'],columns='metric',values='mean').reset_index()
    wide.to_csv(HERE/'tables/effectiveness_wide.csv',index=False)
    for ds in ['wands','esci']:
        f=wide[(wide.dataset==ds)&(wide.schedule.isin(['C0','seven_mean','fixed']))]
        f.to_latex(HERE/f'tables/{ds}_effectiveness.tex',index=False,float_format='%.4f',escape=True,longtable=True)
    print('ANALYSIS COMPLETE',len(summary),'summary rows',len(contrasts),'contrasts',flush=True)

if __name__=='__main__':main()
