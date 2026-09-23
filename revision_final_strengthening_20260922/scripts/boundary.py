from common import *
import time

def main():
 verify();fn=old_rank_function();rng=np.random.default_rng(2026092201);tests=0
 for n in [1,2,3,10,100]:
  for _ in range(30):
   base=rng.integers(-3,4,n).astype('f4');v=rng.integers(-4,5,n).astype('f4');ind=np.arange(n);a=replacement(base,ind,v);assert np.array_equal(a,fn(base,ind,v))
   for i,x,r in zip(ind,v,a):
    other=base.copy();other[i]=x;assert 1+np.flatnonzero(np.argsort(-other,kind='stable')==i)[0]==r;tests+=1
 dump(HERE/'qa/boundary_algorithm_tests.json',dict(tied_replacement_cases=tests,passed=True,old_target_removed=True))
 prov=provenance();tok=pd.read_csv(OLD/'data/all_seven_token_lengths.csv',dtype={'product_id':str});audit=[];allpairs=[];diffs=[]
 for ds in ['wands','esci']:
  p,q=load(ds);pids=np.array([str(x['product_id']) for x in p]);pi={v:i for i,v in enumerate(pids)};qi={int(v):i for i,v in enumerate(q.query_id)}
  for model in MODELS:
   start=time.perf_counter();rprov=prov[(prov.dataset==ds)&(prov.model==model)].set_index('schedule');qv=np.load(ROOT/rprov.loc['C0','query_embedding']);pv=np.load(ROOT/rprov.loc['C0','product_embedding'],mmap_mode='r');base=qv@pv.T
   t=target_frame(ds,model);index=pd.MultiIndex.from_frame(t[['query_id','product_id']]);assert len(t)==(21299 if ds=='wands' else 4434)
   a=np.array([qi[x] for x in t.query_id]);b=np.array([pi[x] for x in t.product_id]);scores=np.empty((len(t),7),dtype='f4');lengths=np.empty((len(t),7),dtype=int)
   tt=tok[(tok.dataset==ds)&(tok.model==model)].set_index('product_id').loc[t.product_id]
   for si,s in enumerate(STEMS):
    raw=pairread(ROOT/rprov.loc[s,'rank_source']).set_index(['query_id','product_id']).loc[index]
    scores[:,si]=base[a,b] if s=='C0' else raw.score.to_numpy(dtype='f4');lengths[:,si]=tt['tokens_'+s]
   cos=np.zeros(len(t));v0=np.asarray(pv[b],dtype=float);norm0=np.linalg.norm(v0,axis=1)
   for s in STEMS[1:]:
    v=np.asarray(np.load(ROOT/rprov.loc[s,'product_embedding'],mmap_mode='r')[b],dtype=float);dist=1-np.einsum('ij,ij->i',v0,v)/(norm0*np.linalg.norm(v,axis=1));cos=np.maximum(cos,dist)
   reconstructed=np.empty_like(scores,dtype='int32');thresholds={k:np.empty(len(t),dtype='f4') for k in [20,100]};thresholdidx={k:np.empty(len(t),dtype='int32') for k in [20,100]}
   for qid,g in t.groupby('query_id',sort=False):
    ix=g.index.to_numpy();bb=base[qi[int(qid)]];order=np.argsort(-bb,kind='stable');position=np.empty(len(p),dtype='int32');position[order]=np.arange(1,len(p)+1)
    reconstructed[ix]=replacement(bb,b[ix,None],scores[ix],order)
    for k in [20,100]:
     rankpos=np.where(position[b[ix]]<=k,k,k-1);thresholdidx[k][ix]=order[rankpos];thresholds[k][ix]=bb[order[rankpos]]
   saved=t[STEMS].to_numpy(dtype='int32');mismatch=saved!=reconstructed
   for ii,si in zip(*np.where(mismatch)):
    diffs.append(dict(dataset=ds,model=model,query_id=int(t.query_id.iloc[ii]),product_id=t.product_id.iloc[ii],schedule=STEMS[si],saved_rank=int(saved[ii,si]),reconstructed_rank=int(reconstructed[ii,si]),score=float(scores[ii,si]),C0_score=float(scores[ii,0]),exact_tie_competitors=int(np.sum(base[a[ii]]==scores[ii,si]))-(scores[ii,0]==scores[ii,si])))
   records=[]
   for k in [20,100]:
    margin=scores.astype(float)-thresholds[k][:,None].astype(float);incl=(scores>thresholds[k][:,None])|((scores==thresholds[k][:,None])&(b[:,None]<thresholdidx[k][:,None]));assert np.array_equal(incl,reconstructed<=k)
    unresolved=incl!=(saved<=k)
    for si,s in enumerate(STEMS):
     records.append(pd.DataFrame(dict(dataset=ds,model=model,query_id=t.query_id,product_id=t.product_id,schedule=s,K=k,catalog_index=b,score=scores[:,si],competitor_threshold_score=thresholds[k],competitor_threshold_id=pids[thresholdidx[k]],competitor_threshold_index=thresholdidx[k],signed_margin=margin[:,si],reconstructed_rank=reconstructed[:,si],saved_rank=saved[:,si],inclusion=saved[:,si]<=k,reconstructed_inclusion=incl[:,si],token_length=lengths[:,si],tokenizer_cap=model_spec(model)['max_tokens'],fully_fitting=tt.fully_fits.to_numpy(),rank_mismatch=mismatch[:,si],unresolved_boundary=unresolved[:,si],exact_threshold_tie=scores[:,si]==thresholds[k],precision='saved fp32; no epsilon in ranking',catalog_size=len(p),competitors=len(p)-1)))
    state=np.where((saved<=k).all(1),'persistent_inclusion',np.where((saved>k).all(1),'persistent_omission','crossing'))
    summary=pd.DataFrame(dict(dataset=ds,model=model,query_id=t.query_id,product_id=t.product_id,K=k,state=state,source_margin=margin[:,0],source_inclusion=saved[:,0]<=k,min_score=scores.min(1),max_score=scores.max(1),score_range=scores.max(1).astype(float)-scores.min(1).astype(float),max_abs_score_deviation=np.max(abs(scores.astype(float)-scores[:,0,None]),axis=1),best_rank=saved.min(1),worst_rank=saved.max(1),included_schedules=(saved<=k).sum(1),max_embedding_cosine_distance=cos,fully_fitting=tt.fully_fits.to_numpy(),max_tokens_seven=lengths.max(1),any_rank_mismatch=mismatch.any(1),unresolved_boundary=unresolved.any(1)))
    allpairs.append(summary)
   rec=pd.concat(records,ignore_index=True);rec.to_parquet(HERE/f'data/{ds}_{model}_boundary_records.parquet',index=False)
   audit.append(dict(dataset=ds,model=model,pairs=len(t),records=len(rec),rank_mismatches=int(mismatch.sum()),boundary_mismatches=int(rec.unresolved_boundary.sum()),exact_threshold_ties=int(rec.exact_threshold_tie.sum()),elapsed_seconds=time.perf_counter()-start))
   print('BOUNDARY',audit[-1],flush=True)
 pd.DataFrame(audit).to_csv(HERE/'qa/boundary_reconstruction.csv',index=False)
 pd.DataFrame(diffs,columns=['dataset','model','query_id','product_id','schedule','saved_rank','reconstructed_rank','score','C0_score','exact_tie_competitors']).to_csv(HERE/'qa/boundary_rank_discrepancies.csv',index=False)
 pairs=pd.concat(allpairs,ignore_index=True);pairs.to_parquet(HERE/'data/boundary_pair_summary.parquet',index=False)
 # Explicitly weighted state-conditional pair distributions; query clusters stay whole.
 rows=[];qrows=[];cols=['score_range','max_abs_score_deviation','source_margin','max_embedding_cosine_distance','best_rank','worst_rank','included_schedules']
 for (ds,model,k),group in pairs.groupby(['dataset','model','K']):
  for scope in ['full','all_seven_fitting']:
   f=group if scope=='full' else group[group.fully_fitting];counts=f.groupby('query_id').size();ids=sorted(counts.index);w=weights(ids)
   for state in ['persistent_inclusion','crossing','persistent_omission']:
    g=f[f.state==state];n=g.groupby('query_id').size().reindex(ids,fill_value=0).to_numpy();N=counts.loc[ids].to_numpy()
    for agg in ['query_macro','pair_micro']:
     denom=N if agg=='query_macro' else np.ones(len(ids));num=n/denom;mass=num.sum()/np.sum(N/denom);bootmass=(w@num)/(w@(N/denom));lo,hi=np.quantile(bootmass,[.025,.975])
     for qid,nn,NN in zip(ids,n,N):
      if agg=='query_macro':qrows.append(dict(dataset=ds,model=model,K=k,support=scope,state=state,query_id=int(qid),state_pairs=int(nn),support_pairs=int(NN),fraction=nn/NN))
     for col in cols:
      vals=g[col].to_numpy();pairw=(1/g.query_id.map(counts).to_numpy()) if agg=='query_macro' else np.ones(len(g));idx=np.argsort(vals,kind='stable');cw=np.cumsum(pairw[idx]);quant=[float(vals[idx][min(np.searchsorted(cw,x*cw[-1],side='left'),len(vals)-1)]) for x in [.05,.25,.5,.75,.95]] if len(vals) else [None]*5
      sums=g.groupby('query_id')[col].sum().reindex(ids,fill_value=0).to_numpy()/denom;bd=w@num;boot=np.divide(w@sums,bd,out=np.full(DRAWS,np.nan),where=bd>0);ciq=np.nanquantile(boot,[.025,.975]) if len(vals) else [None,None]
      rows.append(dict(dataset=ds,model=model,K=k,support=scope,state=state,aggregation=agg,queries=len(ids),support_pairs=len(f),state_pairs=len(g),state_mass=mass,state_mass_ci_low=lo,state_mass_ci_high=hi,quantity=col,conditional_mean=float(np.sum(sums)/np.sum(num)) if len(g) else None,conditional_ci_low=ciq[0],conditional_ci_high=ciq[1],p05=quant[0],p25=quant[1],median=quant[2],p75=quant[3],p95=quant[4],weighting='equal query then equal pair, conditional on state' if agg=='query_macro' else 'equal pair, conditional on state'))
 pd.DataFrame(rows).to_csv(HERE/'tables/boundary_distributions.csv',index=False);pd.DataFrame(qrows).to_parquet(HERE/'data/boundary_states_per_query.parquet',index=False)
 finish()
 print('BOUNDARY COMPLETE',len(pairs),flush=True)

def finish():
 cases=[]
 for model,qid,pid,limits in [('minilm',359,'12575',(15,174,139)),('gte_modernbert',409,'24318',(11,939,1059))]:
  f=pd.read_parquet(HERE/f'data/wands_{model}_boundary_records.parquet');g=f[(f.query_id==qid)&(f.product_id==pid)&(f.K==20)].set_index('schedule').loc[STEMS]
  assert (int(g.saved_rank.min()),int(g.saved_rank.max()),int(g.token_length.max()))==limits
  assert g.fully_fitting.all();cases.extend(g.reset_index().to_dict('records'))
 pd.DataFrame(cases).to_csv(HERE/'data/historical_case_boundary.csv',index=False)
 panel=pd.read_csv(HERE/'XAI_CASE_MANIFEST.csv',dtype={'product_id':str});variants=[]
 for r in panel.itertuples():
  f=pd.read_parquet(HERE/f'data/wands_{r.model}_boundary_records.parquet');g=f[(f.query_id==r.query_id)&(f.product_id==r.product_id)&(f.K==20)].set_index('schedule').loc[STEMS]
  for which,pos in [('minimum',np.argmin(g.score.to_numpy())),('maximum',np.argmax(g.score.to_numpy()))]:variants.append({**g.reset_index().iloc[pos].to_dict(),'selection':r.selection,'display':r.display,'variant':which})
 pd.DataFrame(variants).to_csv(HERE/'data/xai_variant_manifest.csv',index=False)
 print('BOUNDARY CASES COMPLETE',len(variants),flush=True)

if __name__=='__main__':
 if '--finish-only' in sys.argv:finish()
 else:main()
