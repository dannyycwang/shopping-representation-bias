from common import *
import time,argparse

def combine_parts():
 if not all((HERE/f'qa/coverage_validation_{m}.json').exists() for m in ['minilm','bge_base']):return False
 for name in ['coverage_summary','coverage_increments']:
  pd.concat([pd.read_csv(HERE/f'tables/{name}_{m}.csv') for m in ['minilm','bge_base']],ignore_index=True).to_csv(HERE/f'tables/{name}.csv',index=False)
 for name in ['coverage_pair_states','coverage_increment_pair_identities']:
  pd.concat([pd.read_parquet(HERE/f'data/{name}_{m}.parquet') for m in ['minilm','bge_base']],ignore_index=True).to_parquet(HERE/f'data/{name}.parquet',index=False)
 dump(HERE/'qa/coverage_validation.json',sum([read(HERE/f'qa/coverage_validation_{m}.json') for m in ['minilm','bge_base']],[]))
 print('COVERAGE PARTS COMBINED',flush=True);return True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['minilm','bge_base']);args=ap.parse_args()
 verify()
 if args.model is None and combine_parts():return
 verify();plan=pd.read_parquet(HERE/'data/coverage_plan.parquet');aliases=pd.read_parquet(HERE/'data/coverage_input_aliases.parquet');supports=pd.read_parquet(HERE/'data/coverage_common_support.parquet');prov=provenance();summaries=[];perpairs=[];increments=[];incrementq=[];audit=[]
 for ds in ['wands','esci']:
  p,q=load(ds);pi={str(x['product_id']):i for i,x in enumerate(p)};qi={int(v):i for i,v in enumerate(q.query_id)}
  for model in ([args.model] if args.model else ['minilm','bge_base']):
   done=read(HERE/f'qa/{model}_coverage_encoding_complete.json');assert done['all_blocks_complete']
   h=supports[(supports.dataset==ds)&(supports.model==model)].copy();qidlist=sorted(map(int,h.query_id.unique()));qsel={v:i for i,v in enumerate(qidlist)};rp=prov[(prov.dataset==ds)&(prov.model==model)].set_index('schedule');qfull=np.load(ROOT/rp.loc['C0','query_embedding']);qv=qfull[[qi[v] for v in qidlist]]
   av=aliases[(aliases.dataset==ds)&(aliases.model==model)].reset_index(drop=True);keypos={v:i for i,v in enumerate(av.input_sha256)};pv={s:np.load(ROOT/rp.loc[s,'product_embedding'],mmap_mode='r') for s in STEMS}
   dest=HERE/f'cache/{ds}_{model}_coverage_input_scores.npy';scores=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(qidlist),len(av)))
   # Score historical input aliases in the same FP32 normalized-query dot-product protocol.
   old=av[av.kind.eq('historical_exact_model_input')]
   for start in range(0,len(old),1024):
    g=old.iloc[start:start+1024];vectors=np.array([pv[r.representative_schedule][pi[r.representative_product_id]] for r in g.itertuples()],dtype='float32');scores[:,g.index.to_numpy()]=qv@vectors.T
   found=set()
   for path in sorted((HERE/f'cache/coverage/{ds}_{model}').glob('block_*.npz')):
    meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256'];z=np.load(path);keys=z['input_sha256'].tolist();idx=[keypos[x] for x in keys];scores[:,idx]=qv@z['vectors'].T;found.update(keys)
   assert found==set(av[av.kind.eq('new_forward')].input_sha256)
   scores.flush();dump(dest.with_suffix('.json'),dict(sha256=sha(dest),shape=list(scores.shape),query_ids=qidlist,input_alias_indices='row order of matching dataset/model in coverage_input_aliases.parquet',dtype='float32',protocol='cached normalized queries @ native normalized product vectors; CPU BLAS4',encoding_manifest=done))
   pp=plan[plan.dataset==ds].copy();oldref=pp[pp.schedule.isin(STEMS)].drop_duplicates(['product_id','input_sha_'+model]).set_index(['product_id','input_sha_'+model]).schedule.to_dict();oldtexts=pp[pp.schedule.isin(STEMS)].groupby('product_id').text_sha256.agg(set).to_dict()
   records=h[['query_id','product_id','fully_fitting']].merge(pp,on='product_id',validate='many_to_many');records['score']=scores[[qsel[x] for x in records.query_id],[keypos[x] for x in records['input_sha_'+model]]]
   historical=pd.read_parquet(HERE/f'data/{ds}_{model}_boundary_records.parquet');historical=historical[(historical.K==20)&historical.query_id.isin(qidlist)].set_index(['query_id','product_id','schedule'])
   vals=records.score.to_numpy(dtype='float32');source=[];saved=np.full(len(records),-1,dtype=int)
   for i,row in enumerate(records.itertuples()):
    ref=oldref.get((row.product_id,getattr(row,'input_sha_'+model)))
    if row.schedule in STEMS:ref=row.schedule
    if ref is not None:
     prior=historical.loc[(row.query_id,row.product_id,ref)];vals[i]=prior.score;source.append('saved_same_model_input:'+ref)
    else:source.append('native_forward_input_alias')
    if row.schedule in STEMS:saved[i]=int(historical.loc[(row.query_id,row.product_id,row.schedule),'saved_rank'])
   records['score']=vals;records['score_source']=source;records['saved_original_rank']=saved;records['catalog_index']=records.product_id.map(pi);records['reconstructed_rank']=0
   base=qfull@pv['C0'].T
   for k in [20,100]:records[f'threshold_K{k}']=0.;records[f'threshold_index_K{k}']=0
   for qid,g in records.groupby('query_id',sort=False):
    ix=g.index.to_numpy();bb=base[qi[int(qid)]];order=np.argsort(-bb,kind='stable');ind=g.catalog_index.to_numpy();position=np.empty(len(p),dtype=int);position[order]=np.arange(1,len(p)+1);records.loc[ix,'reconstructed_rank']=replacement(bb,ind,g.score.to_numpy(dtype='float32'),order)
    for k in [20,100]:
     ti=order[np.where(position[ind]<=k,k,k-1)];records.loc[ix,f'threshold_K{k}']=bb[ti];records.loc[ix,f'threshold_index_K{k}']=ti
   records['rank']=np.where(records.saved_original_rank>0,records.saved_original_rank,records.reconstructed_rank);records['original_rank_discrepancy']=(records.saved_original_rank>0)&(records.saved_original_rank!=records.reconstructed_rank)
   records['added_distinct_text']=[s not in STEMS and tx not in oldtexts[pid] for s,tx,pid in zip(records.schedule,records.text_sha256,records.product_id)]
   for k in [20,100]:
    threshold=records[f'threshold_K{k}'];inc=(records.score>threshold)|((records.score==threshold)&(records.catalog_index<records[f'threshold_index_K{k}']));assert np.array_equal(inc.to_numpy(),records.reconstructed_rank.to_numpy()<=k)
    records[f'margin_K{k}']=records.score.astype(float)-threshold
    assert ((records['rank']<=k)==inc).all(),'New boundary contradiction; do not change tolerance.'
   records['model']=model;records.to_parquet(HERE/f'data/{ds}_{model}_coverage_ranks.parquet',index=False)
   families=[('7',records.schedule_index<7),('16',records.schedule_index<16),('32',records.schedule_index<32)] if ds=='wands' else [('7',records.schedule.isin(STEMS)),('exhaustive_target',records.schedule.isin(STEMS)|records.added_distinct_text)]
   previous={}
   for scope in ['full','common_all_fitting']:
    supported=h if scope=='full' else h[h.fully_fitting];index=pd.MultiIndex.from_frame(supported[['query_id','product_id']]);ids=sorted(supported.query_id.unique());N=supported.groupby('query_id').size().loc[ids].to_numpy();w=weights(ids)
    original_state={};last=None
    for family,mask in families:
     f=records[mask];f=f[f.set_index(['query_id','product_id']).index.isin(index)]
     for k in [20,100]:
      grp=f.assign(included=f['rank']<=k).groupby(['query_id','product_id'],sort=True).agg(n_schedules=('included','size'),n_included=('included','sum'),distinct_texts=('text_sha256','nunique'))
      grp['persistent_inclusion']=(grp.n_included==grp.n_schedules).astype(float);grp['persistent_omission']=(grp.n_included==0).astype(float);grp['VI']=1-grp.persistent_inclusion-grp.persistent_omission;grp['tested_order_inclusion_frequency']=grp.n_included/grp.n_schedules
      cols=['persistent_inclusion','VI','persistent_omission','tested_order_inclusion_frequency'];qq=grp.groupby(level='query_id')[cols].mean().loc[ids]
      for agg in ['query_macro','pair_micro']:
       v=qq.to_numpy();point=v.mean(0) if agg=='query_macro' else N@v/N.sum();boot=w@v/len(ids) if agg=='query_macro' else (w@(v*N[:,None]))/(w@N)[:,None];lo,hi=np.quantile(boot,[.025,.975],axis=0)
       for col,m,l,hh in zip(cols,point,lo,hi):summaries.append(dict(dataset=ds,model=model,support=scope,family=family,K=k,aggregation=agg,metric=col,mean=m,ci_low=l,ci_high=hh,queries=len(ids),pairs=len(grp),distinct_texts_mean=float(grp.distinct_texts.mean()),bootstrap_seed=SEED,bootstrap_draws=DRAWS,intervention='separate target-only counterfactuals',not_coherent_catalog_recall=True))
      out=grp.reset_index();out['dataset']=ds;out['model']=model;out['support']=scope;out['family']=family;out['K']=k;perpairs.append(out)
      if family=='7':original_state[k]=grp[cols].copy()
      if (scope,k) in previous:
       prevname,prev=previous[(scope,k)];assert grp.index.equals(prev.index)
       assert (grp.VI>=prev.VI).all() and (grp.persistent_inclusion<=prev.persistent_inclusion).all() and (grp.persistent_omission<=prev.persistent_omission).all()
       added=(grp.VI>prev.VI);delta=pd.DataFrame({'incremental_crossing':added.astype(float),'from_original_always_in':(added&original_state[k].persistent_inclusion.eq(1)).astype(float),'from_original_always_out':(added&original_state[k].persistent_omission.eq(1)).astype(float)},index=grp.index);dq=delta.groupby(level='query_id').mean().loc[ids]
       for agg in ['query_macro','pair_micro']:
        v=dq.to_numpy();point=v.mean(0) if agg=='query_macro' else N@v/N.sum();boot=w@v/len(ids) if agg=='query_macro' else (w@(v*N[:,None]))/(w@N)[:,None];lo,hi=np.quantile(boot,[.025,.975],axis=0)
        for col,m,l,hh in zip(dq.columns,point,lo,hi):increments.append(dict(dataset=ds,model=model,support=scope,K=k,contrast=family+' minus '+prevname,aggregation=agg,quantity=col,mean=m,ci_low=l,ci_high=hh,queries=len(ids),pairs=len(grp)))
       tmp=delta.reset_index();tmp['dataset']=ds;tmp['model']=model;tmp['support']=scope;tmp['K']=k;tmp['contrast']=family+' minus '+prevname;incrementq.append(tmp)
      previous[(scope,k)]=(family,grp)
   audit.append(dict(dataset=ds,model=model,records=len(records),original_rank_discrepancies=int(records.original_rank_discrepancy.sum()),boundary_discrepancies=0,monotonicity_passed=True,original_seven_ranks_preserved=True,new_input_vectors=len(found)))
   print('COVERAGE SCORED',audit[-1],flush=True)
 suffix='_'+args.model if args.model else ''
 pd.DataFrame(summaries).to_csv(HERE/f'tables/coverage_summary{suffix}.csv',index=False);pd.DataFrame(increments).to_csv(HERE/f'tables/coverage_increments{suffix}.csv',index=False);pd.concat(perpairs).to_parquet(HERE/f'data/coverage_pair_states{suffix}.parquet',index=False);pd.concat(incrementq).to_parquet(HERE/f'data/coverage_increment_pair_identities{suffix}.parquet',index=False);dump(HERE/f'qa/coverage_validation{suffix}.json',audit)
 if args.model:combine_parts()

if __name__=='__main__':main()
