from common import *

def states(f,k):
    g=f.assign(included=f['rank']<=k).groupby(['query_id','product_id']).included.agg(['all','any'])
    g['persistent_inclusion']=g['all'].astype(float);g['VI']=(g['any']&~g['all']).astype(float);g['persistent_omission']=(~g['any']).astype(float)
    return g[['persistent_inclusion','VI','persistent_omission']]
def qstates(g):
    q=g.groupby(level='query_id').mean();q['n_highest']=g.groupby(level='query_id').size();return q.reset_index()
def main():
    allq=[];summary=[];increments=[];comparisons=[];cmpq=[];identities=[];valid=[];execution_differences=[]
    for m in ['minilm','bge_base']:
        folder=HERE/'harmonized'/m;done=read(folder/'complete.json')
        differences=pd.read_csv(folder/'original7_rank_differences.csv',dtype={'product_id':str})
        for ds in ['wands','esci']:
            f=pairread(folder/f'{ds}_ranks.parquet');old=pairread(ROOT/f'phase3/results/target_only_permutations/{ds}_{m}_pairs.parquet')
            old=old[old.query_id.isin(f.query_id.unique())]
            changed=differences[differences.dataset.eq(ds)]
            denominator=len(f[f.schedule.isin(STEMS)])
            for k in [20,100]:
                crossed=changed[changed[f'decision{k}_changed']]
                execution_differences.append(dict(dataset=ds,model=m,K=k,original7_pair_schedule_records=denominator,changed_exact_ranks=len(changed),changed_cutoff_decisions=len(crossed),changed_decision_record_share=len(crossed)/denominator,queries_with_changed_cutoff_decision=int(crossed.query_id.nunique()),target_pairs_with_changed_cutoff_decision=len(crossed[['query_id','product_id']].drop_duplicates()),note='Historical7 versus harmonized7, same original serialized orders. These differences are execution-related, not additional-order effects; counts are record-level, not query-macro VI.'))
            olds=old.melt(id_vars=['query_id','product_id'],value_vars=STEMS,var_name='schedule',value_name='rank')
            families=[('7',f.schedule_index<7),('16',f.schedule_index<16),('32',f.schedule_index<32)] if ds=='wands' else [('7',f.schedule.isin(STEMS)),('exhaustive_target',f.schedule.str.startswith('E'))]
            for scope in ['full','common_all_fitting']:
                ix=f[['query_id','product_id','fully_fitting']].drop_duplicates();ix=ix if scope=='full' else ix[ix.fully_fitting]
                ix=pd.MultiIndex.from_frame(ix[['query_id','product_id']]);prev={};origin={}
                for fam,mask in families:
                    part=f[mask];part=part[part.set_index(['query_id','product_id']).index.isin(ix)]
                    for k in [20,100]:
                        g=states(part,k);q=qstates(g);cols=['persistent_inclusion','VI','persistent_omission'];meta=dict(dataset=ds,model=m,support=scope,family=fam,K=k,profile=done['profile'])
                        assert (g.sum(1)==1).all();summary.extend(summarize(q,cols,meta));allq.append(q.assign(**meta))
                        if fam=='7':
                            o=olds[olds.set_index(['query_id','product_id']).index.isin(ix)];oq=qstates(states(o,k));assert oq.query_id.tolist()==q.query_id.tolist()
                            dq=q.copy();dq[cols]=q[cols].to_numpy()-oq[cols].to_numpy();comparisons.extend(summarize(dq,cols,dict(dataset=ds,model=m,support=scope,K=k,contrast='harmonized7 minus historical7')))
                            cmpq.append(dq.assign(dataset=ds,model=m,support=scope,K=k));origin[k]=g.copy()
                        if k in prev:
                            pf,pg=prev[k];assert g.index.equals(pg.index);assert (g.VI>=pg.VI).all() and (g.persistent_inclusion<=pg.persistent_inclusion).all() and (g.persistent_omission<=pg.persistent_omission).all()
                            dq=qstates(g-pg);increments.extend(summarize(dq,cols,dict(dataset=ds,model=m,support=scope,K=k,contrast=fam+' minus '+pf)))
                            ids=g[g.VI>pg.VI].reset_index()[['query_id','product_id']];ids=ids.assign(dataset=ds,model=m,support=scope,K=k,contrast=fam+' minus '+pf);identities.append(ids)
                        if fam=='32':
                            assert g.index.equals(origin[k].index)
                            dq=qstates(g-origin[k]);increments.extend(summarize(dq,cols,dict(dataset=ds,model=m,support=scope,K=k,contrast='32 minus 7')))
                            ids=g[g.VI>origin[k].VI].reset_index()[['query_id','product_id']]
                            identities.append(ids.assign(dataset=ds,model=m,support=scope,K=k,contrast='32 minus 7'))
                        prev[k]=(fam,g)
            valid.append(dict(dataset=ds,model=m,queries=int(f.query_id.nunique()),pairs=int(len(f[['query_id','product_id']].drop_duplicates())),records=len(f),all_families_nested=True,profile=done['profile'],repeated_inputs_exactly_equal=done['all_repeated_inputs_identical']))
    csv(pd.DataFrame(summary),HERE/'tables/harmonized_summary.csv');csv(pd.DataFrame(increments),HERE/'tables/harmonized_increments.csv');csv(pd.DataFrame(comparisons),HERE/'tables/historical_to_harmonized.csv')
    csv(pd.concat(allq,ignore_index=True),HERE/'data/harmonized_per_query.csv');csv(pd.concat(cmpq,ignore_index=True),HERE/'data/historical_to_harmonized_per_query.csv');pd.concat(identities,ignore_index=True).to_parquet(HERE/'data/harmonized_added_crossing_identities.parquet',index=False)
    dump(HERE/'qa/harmonized_validation.json',valid)
    csv(pd.DataFrame(execution_differences),HERE/'tables/harmonized_execution_differences.csv')
    print(pd.DataFrame(summary).query("metric=='VI' and K==20").to_string(index=False))
if __name__=='__main__':main()
