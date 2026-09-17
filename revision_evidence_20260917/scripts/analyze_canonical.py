"""All frozen canonical rules, with pure sorting separated from historical M2."""
from common import *
from analyze_membership import per_query, summarize

def main():
    summary=[];pqout=[];states=[];availability=[];sources=[];meanref=[];transition_checks=[]
    serial=pd.read_csv(DATA/'canonical_serialization_audit.csv') if (DATA/'canonical_serialization_audit.csv').exists() else None
    for ds in ['wands','esci']:
      catalog_size=len(products(ds))
      for model in ['minilm','bge_base','gte_modernbert']:
        raw=rawwide(ds,model)
        entries=[]
        for rule in RULES:
            path=HERE/'experiments'/f'{ds}_{model}_{rule}_source.json'
            if not path.exists():
                availability.append(dict(dataset=ds,model=model,rule=rule,status='missing computation',source=''))
                continue
            record=json.loads(path.read_text(encoding='utf8'));rankpath=ROOT/record['rank_source']['relative_path']
            assert sha(rankpath)==record['rank_source']['sha256']
            if serial is not None:
                row=serial[(serial.dataset==ds)&(serial.rule==rule)].iloc[0]
                assert row.text_source_sha256==record['text_source_sha256'] and row.all_text_equal
            entries.append((rule,rankpath,record['rank_origin'],'pure complete-entry sorting'))
            availability.append(dict(dataset=ds,model=model,rule=rule,status='available',source=rankpath.relative_to(ROOT).as_posix()))
            sources.append(source(path))
        # M2 is a separate historical framing/section-placement control, never pure order evidence.
        m2=ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_M2C0.parquet'
        if m2.exists():entries.append(('historical_M2_template',m2,'inherited ranks','changes framing and section placement'))
        for rule,path,origin,family in entries:
          alt=pairread(path);alt=alt[alt.label.eq(highest_label(ds))].set_index(['query_id','product_id']).sort_index()
          base=raw.set_index(['query_id','product_id']).sort_index()
          assert alt.index.equals(base.index),(ds,model,rule,'exact Hq alignment')
          f=base.copy();f[rule]=alt['rank'];f=f.reset_index()
          for pop,ids in populations(ds):
            z=f[f.query_id.isin(ids)];eligible=sorted(map(int,z.query_id.unique()));w=weights(eligible)
            for k in [20,100]:
              q=per_query(z,'C0',rule,k)
              meta=dict(dataset=ds,model=model,encoder_profile='native',population=pop,K=k,reference='C0',alternative=rule,
                control_family=family,rank_origin=origin,intervention='catalog_wide',catalog_size=catalog_size)
              summary.extend(summarize(q,w,meta));pqout.append(q.reset_index().assign(**meta))
              inc=z[STEMS].le(k).mean(1)
              temp=pd.DataFrame({'query_id':z.query_id,'raw_mean_schedule_recall':inc,'canonical_recall':z[rule].le(k).astype(float)})
              qr=temp.groupby('query_id',sort=True).mean();qr['delta_vs_mean_schedules']=qr.canonical_recall-qr.raw_mean_schedule_recall
              n=z.groupby('query_id',sort=True).size().to_numpy()
              for agg in ['query_macro','pair_micro']:
                row={**meta,'reference':'raw_mean_over_seven_schedules','aggregation':agg,'eligible_queries':len(qr),'highest_relevance_pairs':len(z)}
                for col in qr.columns:
                    v=qr[col].to_numpy();value=v.mean() if agg=='query_macro' else (v*n).sum()/n.sum()
                    boot=w@v/len(v) if agg=='query_macro' else (w@(v*n))/(w@n)
                    row[col]=float(value);row[col+'_ci_low'],row[col+'_ci_high']=ci(boot)
                meanref.append(row)
              if rule!='historical_M2_template':
                # All seven incoming schedules are exactly the same canonical text.
                # A single materialized canonical embedding/index is reused, so state membership is identical.
                for agg in ['query_macro','pair_micro']:
                    v=q.alternative_recall.to_numpy();n=q.relevant_count.to_numpy()
                    always=float(v.mean() if agg=='query_macro' else (v*n).sum()/n.sum())
                    b=w@v/len(v) if agg=='query_macro' else (w@(v*n))/(w@n)
                    lo,hi=ci(b)
                    states.append({**meta,'aggregation':agg,'eligible_queries':len(q),'highest_relevance_pairs':len(z),
                       'C0_Recall':always,'mean_over_seven_schedules_Recall':always,'VI':0.,'Always':always,'Never':1-always,
                       'Always_ci_low':lo,'Always_ci_high':hi,'Never_ci_low':1-hi,'Never_ci_high':1-lo,
                       'VI_ci_low':0.,'VI_ci_high':0.,'invariance_basis':'all incoming schedules map to identical text; one fixed materialized index per rule'})
              if rule=='lexical_ascending' and pop=='primary_existing_evaluation' and model in ['minilm','bge_base']:
                  oldpath=ROOT/f'phase4/results/{ds}_{model}_transitions.csv'
                  old=pd.read_csv(oldpath);old=old[(old.method=='canonical_raw')&(old.K==k)].set_index('query_id').sort_index()
                  assert list(old.index)==list(q.index)
                  assert np.array_equal(old.rescued.to_numpy(),q.newly_included.to_numpy())
                  assert np.array_equal(old.newly_missed.to_numpy(),q.newly_omitted.to_numpy())
                  transition_checks.append(dict(dataset=ds,model=model,K=k,queries=len(q),all_saved_gain_loss_exact=True,source=oldpath.relative_to(ROOT).as_posix()))
        print('CANONICAL ANALYSIS',ds,model,len(entries),flush=True)
    pd.DataFrame(availability).to_csv(DATA/'canonical_control_availability.csv',index=False)
    if pqout:pd.concat(pqout,ignore_index=True).to_csv(DATA/'canonical_membership_per_query.csv',index=False)
    pd.DataFrame(summary).to_csv(DATA/'canonical_membership_summary.csv',index=False)
    pd.DataFrame(meanref).to_csv(DATA/'canonical_mean_schedule_reference.csv',index=False)
    pd.DataFrame(states).to_csv(DATA/'canonical_inclusion_states.csv',index=False)
    pd.DataFrame(transition_checks).to_csv(DATA/'canonical_raw_transition_reconciliation.csv',index=False)
    dump(DATA/'canonical_sources.json',dict(protocol=source(HERE/'PROSPECTIVE_PROTOCOL.json'),sources=sources,
       comparison_scope='Same Hq support as the raw catalog analysis; primary WANDS held-out and existing ESCI evaluation sample.',
       no_selection='All three frozen rules are reported; no held-out winner is selected.',
       old_template='M2 changes framing/section placement and is separate from pure sorting controls.',
       invariance='Canonical VI=0 follows from verified identical serialized text and reuse of one fixed index. It is not an empirical guarantee about repeated floating-point forward passes.',
       intervals='10,000 paired query-cluster draws; seed 2026091701; descriptive uncorrected 95% percentile. Including zero does not demonstrate equivalence.'))

if __name__=='__main__':main()
