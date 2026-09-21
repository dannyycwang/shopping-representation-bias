"""Joint strategies and cutoff/severity analyses with shared query resamples."""
from common import *
import argparse, shutil

STATE_COLS=['source_order_recall','seven_schedule_mean_recall','persistent_inclusion','VI','persistent_omission']

def state_pairs(rank,k):
    inc=rank<=k
    return pd.DataFrame(dict(source_order_recall=inc[:,0].astype(float),seven_schedule_mean_recall=inc.mean(1),persistent_inclusion=inc.all(1).astype(float),VI=(inc.any(1)&~inc.all(1)).astype(float),persistent_omission=(~inc.any(1)).astype(float)))

def joint():
    summaries=[];queries=[];pairs=[];mq=[];ms=[];mp=[];checks=[];matched=[]
    aud=pd.read_csv(DATA/'hybrid_reconstruction_audit.csv')
    for ds in ['wands','esci']:
      ncat=42994 if ds=='wands' else 10076
      for model in MODELS:
        raw=rawwide(ds,model);r=raw[STEMS].to_numpy();methods={'raw_seven':(r,';'.join(f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_{s}.parquet' for s in STEMS),7,'empirical seven raw schedules')}
        methods['raw_C0']=(r[:,[0]],methods['raw_seven'][1].split(';')[0],1,'single-order reference; seven-order VI not defined')
        for rule in RULES:
            path,source=canonical_source(ds,model,rule);f=relevant(pairread(path),ds)
            assert f.index.equals(raw.index)
            methods[rule]=(np.repeat(f[['rank']].to_numpy(),7,axis=1),path.relative_to(ROOT).as_posix(),1,'structural; identical serialized input, one materialized index')
        if model!='gte_modernbert':
            for method in ['set_mean','centroid_2','max_2','BM25']+(['centroid_4','max_4','centroid_7','max_7'] if model=='bge_base' else []):
                stem=method if method=='BM25' else f'{model}_{method}'
                path=ROOT/f'phase4/results/{ds}/{stem}_pairs.parquet';f=relevant(pairread(path),ds)
                assert f.index.equals(raw.index)
                budget=int(method.rsplit('_',1)[1]) if method.startswith(('max_','centroid_')) else 1
                methods[method]=(np.repeat(f[['rank']].to_numpy(),7,axis=1),path.relative_to(ROOT).as_posix(),budget,'structural; input-order independent construction, one materialized index')
                oldq=pd.read_csv(path.with_name(stem+'_per_query.csv')).set_index('query_id').sort_index()
                for k in [20,100]:
                    now=f['rank'].le(k).groupby(level=0).mean()
                    assert np.allclose(now,oldq.loc[now.index,f'Recall@{k}'],rtol=0,atol=1e-14)
            for method in ['hybrid_seven','reconstructed_raw_seven']:
                path=DATA/f'{ds}_{model}_{method}_pairs.parquet';f=pairread(path).set_index(['query_id','product_id']).sort_index();assert f.index.equals(raw.index)
                methods[method]=(f[STEMS].to_numpy(),path.relative_to(ROOT).as_posix(),7,'empirical seven schedules; same dense scoring basis as fusion')
        for k in [20,100]:
            basestate=state_pairs(r,k);basestate.index=raw.index
            baseq=aggregate_pairs(basestate,STATE_COLS)
            cellqueries={}
            for method,(rank,source,budget,basis) in methods.items():
                meta=dict(dataset=ds,model=model,K=k,method=method,catalog_size=ncat,source_ids=source,view_budget=budget,invariance_basis=basis)
                f=state_pairs(rank,k);f.index=raw.index
                assert np.all(f[['persistent_inclusion','VI','persistent_omission']].sum(1)==1)
                if method in RULES or basis.startswith('structural'):assert f.VI.eq(0).all()
                for i,s in enumerate(STEMS[:rank.shape[1]]):f['rank_'+s]=rank[:,i];f['included_'+s]=rank[:,i]<=k
                q=aggregate_pairs(f,STATE_COLS)
                for i,s in enumerate(STEMS[:rank.shape[1]]):
                    q['recall_'+s]=f['included_'+s].groupby(level='query_id',sort=True).mean()
                for col in STATE_COLS:
                    if col=='source_order_recall':continue
                    q['delta_'+col+'_vs_raw_seven']=q[col]-baseq[col]
                q['delta_source_recall_vs_raw_C0']=q.source_order_recall-baseq.source_order_recall
                q['delta_mean_recall_vs_raw_C0']=q.seven_schedule_mean_recall-baseq.source_order_recall
                cols=[c for c in q.columns if c!='n_highest']
                if method=='raw_C0':
                    # A single list does not establish seven-order invariance.
                    cols=[c for c in cols if not any(s in c for s in ['persistent','VI','seven_schedule_mean','mean_recall'])]
                    f=f.drop(columns=['persistent_inclusion','VI','persistent_omission','seven_schedule_mean_recall'])
                summaries.extend(summarize(q,cols,meta))
                queries.append(q[cols+['n_highest']].reset_index().assign(**meta))
                pairs.append(f.reset_index().assign(dataset=ds,model=model,K=k,method=method))
                cellqueries[method]=q
                # Explicit list identities only: never compare membership to a mean list.
                schedules=STEMS if method in ['raw_seven','hybrid_seven','reconstructed_raw_seven'] else ['C0']
                for si,s in enumerate(schedules):
                    pf,m=membership(r[:,0]<=k,rank[:,si]<=k,raw.index)
                    mm={**meta,'reference':'raw_C0','alternative_schedule':s}
                    mcols=[c for c in m.columns if c.endswith('_fraction')]+['delta_recall']
                    sm=summarize(m,mcols,mm)
                    cancel=int(m.exact_cancellation.sum());change=int(m.changed.sum())
                    for row in sm:
                        row.update({c+'_count':int(m[c].sum()) for c in ['retained','newly_included','newly_omitted','absent_in_both']})
                        row.update(exact_cancellation_queries=cancel,all_query_denominator=len(m),changed_query_denominator=change,cancellation_all_fraction=cancel/len(m),cancellation_changed_fraction=cancel/change if change else np.nan)
                    ms.extend(sm);mq.append(m.reset_index().assign(**mm));mp.append(pf.reset_index().assign(dataset=ds,model=model,K=k,method=method,alternative_schedule=s,reference='raw_C0'))
                    direct_delta=pd.Series((rank[:,si]<=k).astype(float)-(r[:,0]<=k).astype(float),index=raw.index).groupby(level=0).mean()
                    assert np.allclose(m.delta_recall,direct_delta,atol=1e-15)
                    if model!='gte_modernbert' and method in ['set_mean','centroid_2','max_2','BM25']:
                        tr=ROOT/f'phase4/results/{ds}_{model}_transitions.csv'
                        old=pd.read_csv(tr);old=old[(old.method==method)&(old.K==k)].set_index('query_id').sort_index()
                        if len(old):
                            assert old.index.equals(m.index)
                            assert np.array_equal(old.rescued,m.newly_included) and np.array_equal(old.newly_missed,m.newly_omitted)
                checks.append(dict(dataset=ds,model=model,K=k,method=method,all_pairs_retained=True,states_sum_one=True,membership_partition=True,gain_minus_loss_exact=True,query_count=len(q),pair_count=len(f)))
            if model!='gte_modernbert':
                a=aud[(aud.dataset==ds)&(aud.model==model)]
                mismatch=int(a[f'raw_membership_mismatches_K{k}'].sum())
                reference='reconstructed_raw_seven'
                h=cellqueries['hybrid_seven'];bq=cellqueries[reference]
                delta=pd.DataFrame({c:h[c]-bq[c] for c in STATE_COLS},index=h.index);delta['n_highest']=h.n_highest
                matched.extend(summarize(delta,STATE_COLS,dict(dataset=ds,model=model,K=k,contrast='hybrid_seven minus reconstructed_raw_seven',authoritative_raw_membership_mismatches=mismatch,authoritative_raw_equivalent_at_K=mismatch==0)))
            print('JOINT',ds,model,k,flush=True)
    pd.DataFrame(summaries).to_csv(DATA/'strategy_summary.csv',index=False)
    pd.concat(queries).to_csv(DATA/'strategy_per_query.csv',index=False)
    pd.concat(pairs).to_parquet(DATA/'strategy_per_pair.parquet',index=False)
    pd.concat(mq).to_csv(DATA/'membership_per_query.csv',index=False)
    pd.DataFrame(ms).to_csv(DATA/'membership_summary.csv',index=False)
    pd.concat(mp).to_parquet(DATA/'membership_per_pair.parquet',index=False)
    pd.DataFrame(matched).to_csv(DATA/'matched_hybrid_raw_contrasts.csv',index=False)
    pd.DataFrame(checks).to_csv(DATA/'joint_integrity_checks.csv',index=False)
    # Retain every historical cancellation condition, with both denominator definitions.
    for name in ['rq2_cancellation_companion.csv','membership_changes_per_query.csv','membership_changes_summary.csv']:
        shutil.copyfile(OLD/'data'/name,DATA/('historical_'+name))
    pd.DataFrame(summaries)[['dataset','model','method','K','eligible_queries','highest_label_pairs','catalog_size']].drop_duplicates().to_csv(DATA/'joint_denominators.csv',index=False)

def conditional_summary(pair,baseflag,severeflag,meta):
    f=pd.DataFrame({'base':baseflag.astype(int),'severe':severeflag.astype(int)},index=pair.index)
    q=f.groupby(level=0,sort=True).sum();n=pair.groupby(level=0).size();q['n_highest']=n
    q['base_share']=q.base/n;q['severe_share']=q.severe/n
    rows=summarize(q,['base_share','severe_share'],meta)
    w=weights(q.index);valid=q.base.to_numpy()>0
    ratio=np.divide(q.severe,q.base,out=np.zeros(len(q)),where=valid)
    den=w@valid.astype(float);valid_draws=den>0
    boot=(w@ratio)[valid_draws]/den[valid_draws]
    cond=ratio[valid].mean() if valid.any() else np.nan
    for row in rows:
        row.update(base_pair_count=int(q.base.sum()),severe_pair_count=int(q.severe.sum()),base_queries=int(valid.sum()),conditional_query_macro=cond,conditional_query_denominator=int(valid.sum()),conditional_pair_micro=q.severe.sum()/q.base.sum() if q.base.sum() else np.nan,query_normalized_mass_ratio=q.severe_share.sum()/q.base_share.sum() if q.base_share.sum() else np.nan,conditional_bootstrap_valid_draws=int(valid_draws.sum()))
        if len(boot):row.update(conditional_query_macro_ci_low=float(np.quantile(boot,.025)),conditional_query_macro_ci_high=float(np.quantile(boot,.975)))
    return rows,q

def diagnostics():
    fits=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str})
    summary=[];pq=[];pp=[];severity=[];sq=[];ecdf=[];moves=[];kc=[];den=[]
    for ds in ['wands','esci']:
      for model in MODELS:
        raw=rawwide(ds,model);target=relevant(pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{model}_pairs.parquet'),ds)
        assert target.index.equals(raw.index) and np.array_equal(raw[STEMS],target[['whole_'+s for s in STEMS]])
        assert np.array_equal(raw.C0,target.C0)
        fit=fits[(fits.dataset==ds)&(fits.model==model)]
        assert np.array_equal(fit.fully_fits,fit[['tokens_'+s for s in STEMS]].max(axis=1)<=fit.tokenizer_cap)
        fitting=set(fit.loc[fit.fully_fits,'product_id'])
        for intervention,frame in [('catalog_wide',raw),('target_only',target)]:
          for support in ['full','fully_fitting']:
            f=frame if support=='full' else frame[frame.index.get_level_values('product_id').isin(fitting)]
            r=f[STEMS].to_numpy();ncat=42994 if ds=='wands' else 10076
            meta=dict(dataset=ds,model=model,intervention=intervention,target_support=support,catalog_size=ncat)
            allq={}
            for k in [20,50,100,200,500,1000]:
                z=state_pairs(r,k);z.index=f.index
                z=z.rename(columns={'source_order_recall':'source_order_inclusion','seven_schedule_mean_recall':'mean_schedule_inclusion'})
                cols=z.columns.tolist();q=aggregate_pairs(z,cols)
                summary.extend(summarize(q,cols,{**meta,'K':k}));pq.append(q.reset_index().assign(**meta,K=k));allq[k]=q
                pp.append(z.reset_index().assign(**meta,K=k))
                den.append(dict(**meta,K=k,eligible_queries=len(q),highest_label_pairs=len(f),support_ids_sha256=digest_ids([f'{a}:{b}' for a,b in f.index])))
                for ref in sorted(set([20]+([prev] if k!=20 else []))):
                    if ref==k:continue
                    d=q[cols]-allq[ref][cols];d['n_highest']=q.n_highest
                    kc.extend(summarize(d,cols,{**meta,'K':k,'reference_K':ref}))
                if k!=20:
                    assert (q.persistent_inclusion>=allq[prev].persistent_inclusion-1e-14).all()
                    assert (q.persistent_omission<=allq[prev].persistent_omission+1e-14).all()
                prev=k
            if intervention=='target_only':
                best=r.min(1);worst=r.max(1);ran=worst-best
                for k in [20,100]:
                    cross=(best<=k)&(worst>k);directed=(r[:,0]<=k)&(r[:,1:].max(1)>k)
                    move=pd.DataFrame(dict(best_rank=best,worst_rank=worst,rank_range=ran,source_rank=r[:,0],crossing=cross,source_order_loss=directed),index=f.index)
                    for mult in [2,5]:
                        for name,flag in [('crossing',cross),('source_order_loss',directed)]:
                            sev=flag&(worst>mult*k)
                            rows,qsev=conditional_summary(f,flag,sev,{**meta,'K':k,'event':name,'threshold_multiple':mult})
                            severity.extend(rows);sq.append(qsev.reset_index().assign(**meta,K=k,event=name,threshold_multiple=mult))
                            move[f'{name}_worst_gt_{mult}K']=sev
                    moves.append(move.reset_index().assign(**meta,K=k))
                    crossf=move[move.crossing]
                    for variable in ['worst_rank','rank_range']:
                        v=crossf[variable].to_numpy();counts=crossf.groupby(level=0).size();nq=len(counts)
                        mass=np.array([1/(nq*counts.loc[qid]) for qid in crossf.index.get_level_values(0)])
                        order=np.argsort(v,kind='stable'); vals=v[order];cs=np.cumsum(mass[order]);end=np.r_[vals[1:]!=vals[:-1],True]
                        for value,qcdf,pcdf in zip(vals[end],cs[end],(np.arange(len(vals))+1)[end]/len(vals)):
                            ecdf.append(dict(**meta,K=k,variable=variable,rank=int(value),query_balanced_ecdf=float(qcdf),pair_micro_ecdf=float(pcdf),crossing_pairs=len(vals),crossing_queries=nq,weighting='equal weight per crossing query, equal products within query'))
            print('DIAGNOSTIC',ds,model,intervention,support,flush=True)
    pd.DataFrame(summary).to_csv(DATA/'cutoff_summary.csv',index=False)
    pd.concat(pq).to_csv(DATA/'cutoff_per_query.csv',index=False)
    pd.concat(pp).to_parquet(DATA/'cutoff_per_pair.parquet',index=False)
    pd.DataFrame(kc).to_csv(DATA/'cutoff_paired_changes.csv',index=False)
    pd.DataFrame(den).to_csv(DATA/'cutoff_denominators.csv',index=False)
    pd.DataFrame(severity).to_csv(DATA/'severity_summary.csv',index=False)
    pd.concat(sq).to_csv(DATA/'severity_per_query.csv',index=False)
    pd.concat(moves).to_parquet(DATA/'target_rank_movements.parquet',index=False)
    pd.DataFrame(ecdf).to_csv(DATA/'crossing_ecdf.csv',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('part',choices=['joint','diagnostics']);args=a.parse_args()
    joint() if args.part=='joint' else diagnostics()
