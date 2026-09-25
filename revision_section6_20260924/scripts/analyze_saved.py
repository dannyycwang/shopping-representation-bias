"""Recompute Section 6 from saved ranks/labels; no model inference."""
from common import *
import itertools
GAINS={'wands':{'Exact':3.,'Partial':1.,'Irrelevant':0.},'esci':{'E':1.,'S':.1,'C':.01,'I':0.}}

def inclusion():
    fits=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str});queries=[];summary=[];checks=[]
    for ds,m in itertools.product(['wands','esci'],MODELS):
        full=pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{m}_pairs.parquet')
        f=full[full.query_id.isin(support(ds))].copy()
        valid=set(fits[(fits.dataset==ds)&fits.model.eq(m)&fits.fully_fits].product_id)
        assert len(f)==(21299 if ds=='wands' else 4434)
        for intervention,cols in [('target_only',STEMS),('catalog_wide',['whole_'+s for s in STEMS])]:
            for scope in ['full','fully_fitting']:
                part=f if scope=='full' else f[f.product_id.isin(valid)]
                for k in [20,100]:
                    inc=part[cols].to_numpy()<=k
                    z=part[['query_id','product_id']].copy()
                    z['persistent_inclusion']=inc.all(1).astype(float);z['VI']=(inc.any(1)&~inc.all(1)).astype(float);z['persistent_omission']=(~inc.any(1)).astype(float)
                    z['source_order_inclusion']=inc[:,0].astype(float);z['seven_schedule_mean_inclusion']=inc.mean(1)
                    assert (z[['persistent_inclusion','VI','persistent_omission']].sum(axis=1)==1).all()
                    names=list(z.columns[2:]);q=z.groupby('query_id')[names].mean();q['n_highest']=z.groupby('query_id').size();q=q.reset_index()
                    meta=dict(dataset=ds,model=m,intervention=intervention,target_support=scope,K=k,status='available',schedule_family='original seven',catalog_size=42994 if ds=='wands' else 10076)
                    queries.append(q.assign(**meta));summary.extend(summarize(q,names,meta))
                    checks.append(dict(dataset=ds,model=m,intervention=intervention,support=scope,K=k,queries=len(q),pairs=len(part),partition=True))
        print('INCLUSION',ds,m,flush=True)
    q=pd.concat(queries,ignore_index=True);csv(q,HERE/'data/inclusion_per_query.csv');q.to_parquet(HERE/'data/inclusion_per_query.parquet',index=False)
    csv(pd.DataFrame(summary),HERE/'tables/inclusion_complete.csv');dump(HERE/'qa/inclusion_checks.json',checks)

def graded():
    original=pd.read_parquet(GRADED/'data/per_query.parquet');frames=[];checks=[]
    for keys,g in original.groupby(['dataset','model','method','schedule','rank_source'],sort=False):
        ds,m,method,s,path=keys;f=pairread(ROOT/path);f=f[f.query_id.isin(support(ds))];ranks={int(q):v for q,v in f.groupby('query_id',sort=True)}
        h=g.sort_values('query_id').copy();new=[];errors=[]
        assert h.query_id.tolist()==support(ds)
        for row in h.itertuples(index=False):
            p=ranks[int(row.query_id)];labels=p.label.to_numpy();r=p['rank'].to_numpy();gain=p.label.map(GAINS[ds]).to_numpy();order=np.argsort(r,kind='stable')
            assert not np.isnan(gain).any() and len(set(r))==len(r)
            values={}
            for k in [20,100]:
                ideal=np.sort(gain)[::-1][:k];den=np.sum(ideal/np.log2(np.arange(2,len(ideal)+2)))
                v=np.sum(gain[r<=k]/np.log2(r[r<=k]+1))/den
                c=gain[order][:k];cv=np.sum(c/np.log2(np.arange(2,len(c)+2)))/den
                highest=labels==('Exact' if ds=='wands' else 'E')
                values.update({f'nDCG@{k}':v,f'cNDCG@{k}':cv,f'JudgedCoverage@{k}':float((r<=k).sum()/k),f'Recall@{k}':float((highest&(r<=k)).sum()/highest.sum())})
                assert int(highest.sum())==int(row.n_highest)
                assert set(p.product_id[highest&(r<=k)])==set(json.loads(h.loc[h.query_id.eq(row.query_id),f'included_ids_K{k}'].iloc[0]))
            new.append(values)
        n=pd.DataFrame(new,index=h.index)
        for col in n:
            if col in h:errors.append(float(np.max(abs(n[col]-h[col]))));assert np.allclose(n[col],h[col],atol=1e-14,rtol=0),(keys,col)
            else:h[col]=n[col]
        frames.append(h);checks.append(dict(dataset=ds,model=m,method=method,schedule=s,source=path,rows=len(h),max_existing_error=max(errors),nDCG100='recomputed from saved all-judged ranks'))
    pq=pd.concat(frames,ignore_index=True);pq.to_parquet(HERE/'data/effectiveness_per_query.parquet',index=False);csv(pq,HERE/'data/effectiveness_per_query.csv.gz');dump(HERE/'qa/graded_recomputation.json',checks)
    print('GRADED',len(pq),flush=True);return pq

def membership(pq):
    rows=[];summ=[]
    for (ds,m,method),g in pq[pq.method.isin(['raw','raw_hybrid'])].groupby(['dataset','model','method'],sort=False):
        base=g[g.schedule.eq('C0')].set_index('query_id').sort_index()
        for s in STEMS[1:]:
            alt=g[g.schedule.eq(s)].set_index('query_id').sort_index();assert base.index.equals(alt.index)
            for k in [20,100]:
                block=[]
                for qid,b in base.iterrows():
                    a=alt.loc[qid];old=set(json.loads(b[f'included_ids_K{k}']));new=set(json.loads(a[f'included_ids_K{k}']));gain=new-old;loss=old-new;assert gain.isdisjoint(loss)
                    delta=(len(gain)-len(loss))/int(b.n_highest)
                    assert abs(delta-(a[f'Recall@{k}']-b[f'Recall@{k}']))<1e-14
                    block.append(dict(dataset=ds,model=m,method=method,schedule=s,reference='source C0',K=k,query_id=int(qid),n_highest=int(b.n_highest),source_count=len(old),alternative_count=len(new),gain_count=len(gain),loss_count=len(loss),gained_ids=json.dumps(sorted(gain)),lost_ids=json.dumps(sorted(loss)),gain_share=len(gain)/b.n_highest,loss_share=len(loss)/b.n_highest,turnover=(len(gain)+len(loss))/b.n_highest,delta_recall=delta,changed=bool(gain or loss),exact_replacement=len(gain)==len(loss)>0,delta_ndcg=float(a[f'nDCG@{k}']-b[f'nDCG@{k}'])))
                q=pd.DataFrame(block);rows.append(q);changed=q[q.changed]
                row=dict(dataset=ds,model=m,method=method,schedule=s,reference='source C0',K=k,queries=len(q),pairs=int(q.n_highest.sum()),changed_queries=int(q.changed.sum()),equal_positive_queries=int(q.exact_replacement.sum()),equal_positive_share_all=float(q.exact_replacement.mean()),equal_positive_share_changed=float(changed.exact_replacement.mean()) if len(changed) else 0.,bootstrap_seed=SEED,bootstrap_resamples=DRAWS,interval='95% unadjusted paired query-cluster percentile')
                for col in ['gain_share','loss_share','turnover','delta_recall','delta_ndcg']:
                    e,l,h=ci(q[col],q.query_id);row.update({col:float(e),col+'_ci_low':float(l),col+'_ci_high':float(h)})
                for quant in [.0,.25,.5,.75,.9,.95,1.]:row['changed_abs_ndcg_q'+str(quant)]=float(np.quantile(abs(changed.delta_ndcg),quant)) if len(changed) else np.nan
                row['changed_abs_ndcg_mean']=float(abs(changed.delta_ndcg).mean()) if len(changed) else np.nan
                summ.append(row)
    q=pd.concat(rows,ignore_index=True);csv(q,HERE/'data/membership_per_query.csv.gz');q.to_parquet(HERE/'data/membership_per_query.parquet',index=False)
    csv(pd.DataFrame(summ),HERE/'tables/membership_all_contrasts.csv');print('MEMBERSHIP',len(q),flush=True)
    # Deterministic illustration, convention-based setting: WANDS/BGE source vs Reverse @20.
    # No all-fitting condition is imposed. This new post-hoc rule is explicit.
    eligible=q[q.dataset.eq('wands')&q.model.eq('bge_base')&q.method.eq('raw')&q.schedule.eq('C1')&q.K.eq(20)&q.exact_replacement].copy()
    eligible['selection_hash']=eligible.query_id.map(lambda x:digest(f'section6-replacement-v1|wands|bge_base|C1|20|{x}'))
    selected=eligible.sort_values(['selection_hash','query_id']).iloc[0];qid=int(selected.query_id)
    meta=dict(selection_rule='Minimum SHA256(section6-replacement-v1|wands|bge_base|C1|20|query_id) among exact positive integer replacement cases; WANDS/BGE/reversal chosen by manuscript example, not effect size.',selection_time_utc=now(),eligible_cases=len(eligible),query_id=qid,selection_hash=selected.selection_hash,illustrative=True,all_fitting_required=False,source_numerator=int(selected.source_count),alternative_numerator=int(selected.alternative_count),denominator=int(selected.n_highest),recall=float(selected.source_count/selected.n_highest),gain_count=int(selected.gain_count),loss_count=int(selected.loss_count))
    p,queries,_=load('wands');products={str(x['product_id']):x for x in p};meta['query']=queries.set_index('query_id').loc[qid,'query']
    meta.update(dataset='wands',model='bge_base',intervention='catalog_wide',catalog_size=len(p),source_schedule='C0',alternative_schedule='C1')
    pr={s:pairread(ROOT/f'phase2/results/phase2_pair_ranks/wands_bge_base_native_{s}.parquet') for s in ['C0','C1']};records=[]
    fit=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str});fit=fit[fit.dataset.eq('wands')&fit.model.eq('bge_base')].set_index('product_id')
    for change,column in [('gained','gained_ids'),('lost','lost_ids')]:
        for pid in json.loads(selected[column]):
            product=products[pid];row=dict(query_id=qid,query=meta['query'],change=change,product_id=pid,product_name=product['title'],label='Exact',source_order=json.dumps(attrs(product,'C0'),ensure_ascii=False),alternative_order=json.dumps(attrs(product,'C1'),ensure_ascii=False),source_text=text_for(product,'C0'),alternative_text=text_for(product,'C1'),fully_fitting_seven=bool(fit.loc[pid,'fully_fits']),numerator_source=meta['source_numerator'],numerator_alternative=meta['alternative_numerator'],denominator=meta['denominator'])
            row.update({key:meta[key] for key in ['dataset','model','intervention','catalog_size','source_schedule','alternative_schedule']})
            for s in ['C0','C1']:row['rank_'+s]=int(pr[s][pr[s].query_id.eq(qid)&pr[s].product_id.eq(pid)]['rank'].iloc[0])
            records.append(row)
    csv(pd.DataFrame(records),HERE/'data/illustrative_replacement_case.csv');csv(eligible[['query_id','selection_hash']],HERE/'data/illustrative_case_candidates.csv');dump(HERE/'data/illustrative_case_selection.json',meta)

def controls(pq):
    states=pd.read_parquet(GRADED/'data/persistent_states_per_query.parquet');rows=[];perquery=[];contrasts=[]
    methods=states.method.unique().tolist();available=[]
    for (ds,m,method,k),g in states.groupby(['dataset','model','method','K'],sort=False):
        g=g.sort_values('query_id');assert g.query_id.tolist()==support(ds)
        assert np.allclose(g.persistent_inclusion+g.VI+g.persistent_omission,1,atol=1e-15,rtol=0)
        f=pq[pq.dataset.eq(ds)&pq.model.eq('lexical' if method=='BM25' else m)&pq.method.eq(method)].copy()
        # Preserve source and seven-mean references as separate rows for raw families.
        schedules=['C0','seven_mean'] if method in ['raw','raw_hybrid'] else ['fixed']
        for schedule in schedules:
            metrics=[f'Recall@{k}',f'nDCG@{k}',f'cNDCG@{k}',f'JudgedCoverage@{k}']
            z=f.groupby('query_id')[metrics].mean() if schedule=='seven_mean' else f[f.schedule.eq(schedule)].set_index('query_id')[metrics]
            z=z.sort_index();z['n_highest']=g.n_highest.to_numpy()
            for col in ['persistent_inclusion','VI','persistent_omission']:z[col]=g[col].to_numpy()
            z=z.rename(columns={f'Recall@{k}':'Recall',f'nDCG@{k}':'nDCG',f'cNDCG@{k}':'cNDCG',f'JudgedCoverage@{k}':'judged_coverage'}).reset_index()
            if schedule=='fixed':assert np.allclose(z.persistent_inclusion,z.Recall,atol=1e-15,rtol=0) and (z.VI==0).all(),(ds,m,method,k,float(abs(z.persistent_inclusion-z.Recall).max()),z.head().to_dict('records'))
            meta=dict(dataset=ds,model=m,method=method,schedule=schedule,K=int(k),invariant_by_construction=schedule=='fixed',vectors_per_product=int(method.split('_')[-1]) if method.startswith('max_') else (0 if method=='BM25' else 1),construction_encodings_per_product=(method.split('_')[-1] if method.startswith(('max_','centroid_')) else ('one per attribute plus fixed-text atom' if method=='set_mean' else '1')),status='available',basis=g.basis.iloc[0])
            if method=='BM25':meta['construction_encodings_per_product']='0'
            meta['sparse_branch']=method=='BM25' or 'hybrid' in method
            cols=['Recall','nDCG','cNDCG','judged_coverage','persistent_inclusion','VI','persistent_omission'];rows.extend(summarize(z,cols,meta));perquery.append(z.assign(**meta));available.append((ds,m,method,k,schedule))
    q=pd.concat(perquery,ignore_index=True);q.to_parquet(HERE/'data/controls_per_query.parquet',index=False);csv(q,HERE/'data/controls_per_query.csv.gz')
    for (ds,m,method,s,k),g in q.groupby(['dataset','model','method','schedule','K'],sort=False):
        refs=[('raw','C0'),('raw','seven_mean')]
        if method.startswith('canonical_hybrid_'):refs += [(method.replace('canonical_hybrid_',''),'fixed'),('BM25','fixed'),('raw_hybrid','seven_mean')]
        for rm,rs in refs:
            base=q[q.dataset.eq(ds)&q.model.eq(m)&q.method.eq(rm)&q.schedule.eq(rs)&q.K.eq(k)].sort_values('query_id');g=g.sort_values('query_id');assert base.query_id.tolist()==g.query_id.tolist()
            cols=['Recall','nDCG','persistent_inclusion','VI','persistent_omission'];delta=g[['query_id','n_highest']].copy();delta[cols]=g[cols].to_numpy()-base[cols].to_numpy()
            contrasts.extend(summarize(delta,cols,dict(dataset=ds,model=m,method=method,schedule=s,K=int(k),comparator=rm+':'+rs)))
    summary=pd.DataFrame(rows);ct=pd.DataFrame(contrasts);csv(summary,HERE/'tables/controls_complete.csv');csv(ct,HERE/'tables/controls_paired_comparisons.csv')
    missing=[]
    for ds,m,method,k in itertools.product(['wands','esci'],MODELS,methods,[20,100]):
        if not any(t[:4]==(ds,m,method,k) for t in available):missing.append(dict(dataset=ds,model=m,method=method,K=k,status='unavailable',reason='This encoder/method was not evaluated in the saved evidence; no imputation.'))
    csv(pd.DataFrame(missing),HERE/'tables/unavailable_method_cells.csv')
    # One complete long companion, with explicit baseline/paired rows and missing cells.
    csv(pd.concat([summary.assign(record_type='estimate'),ct.rename(columns={'estimate':'delta'}).assign(record_type='paired_comparison'),pd.DataFrame(missing).assign(record_type='unavailable')],ignore_index=True),HERE/'tables/method_complete_companion.csv')
    print('CONTROLS',len(summary),len(ct),flush=True)

def transitions():
    f=pd.read_parquet(FINAL/'data/control_transitions_per_query.parquet');states=pd.read_parquet(GRADED/'data/persistent_states_per_query.parquet');summary=[];checks=[]
    for (ds,m,family,rule,k),g in f.groupby(['dataset','model','family','rule','K'],sort=False):
        raw='raw' if family=='dense' else 'raw_hybrid';control=rule if family=='dense' else 'canonical_hybrid_'+rule
        ref=states[states.dataset.eq(ds)&states.model.eq(m)&states.method.eq(raw)&states.K.eq(k)].set_index('query_id').sort_index()
        fixed=states[states.dataset.eq(ds)&states.model.eq(m)&states.method.eq(control)&states.K.eq(k)].set_index('query_id').sort_index()
        wide=g.pivot(index='query_id',columns=['reference_state','control_state'],values='cell_fraction').sort_index();assert wide.shape[1]==6
        assert np.allclose(wide.sum(1),1,atol=1e-15,rtol=0)
        for state,col in [('persistent_inclusion','persistent_inclusion'),('crossing','VI'),('persistent_omission','persistent_omission')]:assert np.allclose(wide[state].sum(1),ref[col],atol=1e-15,rtol=0)
        assert np.allclose(wide.xs('included',axis=1,level=1).sum(1),fixed.source_recall,atol=1e-15,rtol=0)
        assert np.allclose(wide.xs('omitted',axis=1,level=1).sum(1),1-fixed.source_recall,atol=1e-15,rtol=0)
        for (rs,cs),v in wide.items():
            e,l,h=ci(v,wide.index);summary.append(dict(dataset=ds,model=m,family=family,rule=rule,K=k,reference_state=rs,control_state=cs,estimate=e,ci_low=l,ci_high=h,queries=len(wide),pairs=int(ref.n_highest.sum()),weighting='query_macro'))
        checks.append(dict(dataset=ds,model=m,family=family,rule=rule,K=k,six_cells=True,row_sums=True,column_sums=True))
    csv(pd.DataFrame(summary),HERE/'tables/transitions_complete.csv');f.to_parquet(HERE/'data/transitions_per_query.parquet',index=False);dump(HERE/'qa/transition_checks.json',checks)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--resume-controls',action='store_true');args=ap.parse_args()
    if args.resume_controls:pq=pd.read_parquet(HERE/'data/effectiveness_per_query.parquet')
    else:inclusion();pq=graded();membership(pq)
    controls(pq);transitions()
    dump(HERE/'qa/saved_analysis_complete.json',dict(time_utc=now(),all_checks_passed=True,no_new_inference=True))

if __name__=='__main__':main()
