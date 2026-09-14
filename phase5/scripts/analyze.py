from common5 import *
from scipy.stats import norm as normal

frozen=json.loads((CONFIG/'sample.json').read_text());short={'heuristic_authoritative':'G1','optimized_technical':'G2'};methods=['Raw','Canonical','G1','CG1','G2','CG2','GuardG1','GuardCG1','GuardG2','GuardCG2'];summary=[];orders=[];states=[];transitions=[];fully=[];common_fully=[];common_valid_metrics=[];query_files={}
for ds in frozen['datasets']:
 for model in ['bge_base','minilm']:
  f=pd.read_parquet(OUT/f'{ds}_{model}_target_ranks.parquet');length=pd.read_parquet(OUT/f'{ds}_{model}_primary_text_lengths.parquet')
  if 'text_sha256' in f:f=f.merge(length[['text_sha256','fully_fits']],on='text_sha256',how='left');assert f.fully_fits.notna().all()
  for method,g in f.groupby('method',sort=False):
   wide=g.pivot(index=['query_id','product_id'],columns='stem',values='rank')[STEMS];assert wide.notna().all().all();qidx=wide.index.get_level_values('query_id')
   row={'dataset':ds,'model':model,'method':method,'pairs':len(wide),'queries':qidx.nunique(),'rank_spread_median':float((wide.max(axis=1)-wide.min(axis=1)).median())}
   for k in [20,100]:
    inc=wide.le(k);qp=inc.groupby(level=0).mean();perq=qp.mean(axis=1);vi=inc.any(axis=1)&~inc.all(axis=1);qvi=vi.groupby(level=0).mean();rob=inc.all(axis=1).groupby(level=0).mean();mu,lo,hi=boot(perq)
    row.update({f'inclusion@{k}_micro':float(inc.to_numpy().mean()),f'inclusion@{k}_macro':mu,f'inclusion@{k}_ci_low':lo,f'inclusion@{k}_ci_high':hi,f'VI@{k}_micro':float(vi.mean()),f'VI@{k}_macro':float(qvi.mean()),f'robust_coverage@{k}_macro':float(rob.mean()),f'schedule_min_inclusion@{k}_macro':float(qp.mean(0).min()),f'always@{k}':int(inc.all(axis=1).sum()),f'sometimes@{k}':int(vi.sum()),f'never@{k}':int((~inc.any(axis=1)).sum())})
    for stem in STEMS:orders.append({'dataset':ds,'model':model,'method':method,'stem':stem,'K':k,'pair_micro':float(inc[stem].mean()),'query_macro':float(qp[stem].mean())})
    for state,count in [('always',inc.all(axis=1).sum()),('sometimes',vi.sum()),('never',(~inc.any(axis=1)).sum())]:states.append({'dataset':ds,'model':model,'method':method,'K':k,'state':state,'pairs':int(count)})
   summary.append(row)
  for method in [m for m in methods if m not in ['Raw']]:
   a=f[f.method.eq('Raw')][['query_id','product_id','stem','rank']].rename(columns={'rank':'raw_rank'});b=f[f.method.eq(method)].merge(a,on=['query_id','product_id','stem'],validate='one_to_one')
   for k in [20,100]:
    b['rescue']=(b.raw_rank>k)&(b['rank']<=k);b['miss']=(b.raw_rank<=k)&(b['rank']>k);pq=b.groupby('query_id').agg(rescued=('rescue','sum'),missed=('miss','sum'),n=('product_id','size'));net=(pq.rescued-pq.missed)/pq.n
    transitions.append({'dataset':ds,'model':model,'method':method,'K':k,'order_target_trials':len(b),'pairs':b[['query_id','product_id']].drop_duplicates().shape[0],'queries':b.query_id.nunique(),'rescued':int(b.rescue.sum()),'newly_missed':int(b['miss'].sum()),'macro_net':float(net.mean())})
  for method,g in f.groupby('method'):
   if method=='Raw':continue
   pfit=g.groupby(['query_id','product_id']).fully_fits.all();support=pfit[pfit].index
   for k in [20,100]:
    z=g.set_index(['query_id','product_id']).loc[g.set_index(['query_id','product_id']).index.isin(support)].reset_index();macro=z.assign(hit=z['rank']<=k).groupby(['query_id','stem']).hit.mean().groupby('query_id').mean()
    fully.append({'dataset':ds,'model':model,'method':method,'K':k,'pairs':len(support),'queries':len(macro),'macro_inclusion':float(macro.mean()) if len(macro) else np.nan})
  primary=['Raw','Canonical','G1','CG1','G2','CG2'];fit=f[f.method.isin(primary)].groupby(['method','query_id','product_id']).fully_fits.all().unstack(0);support=fit.index[fit[primary].all(axis=1)]
  for method in primary:
   z=f[f.method.eq(method)].set_index(['query_id','product_id']);z=z.loc[z.index.isin(support)].reset_index()
   for k in [20,100]:
    macro=z.assign(hit=z['rank']<=k).groupby(['query_id','stem']).hit.mean().groupby('query_id').mean();common_fully.append({'dataset':ds,'model':model,'method':method,'K':k,'common_pairs':len(support),'queries':len(macro),'macro_inclusion':float(macro.mean()) if len(macro) else np.nan})
  for n in ['1','2']:
   gv=f[f.method.eq('G'+n)].groupby(['query_id','product_id']).fact_pass.all();cv=f[f.method.eq('CG'+n)].groupby(['query_id','product_id']).fact_pass.all();support=gv.index[gv&cv.reindex(gv.index).fillna(False)]
   for method in ['Raw','Canonical','G'+n,'CG'+n]:
    z=f[f.method.eq(method)].set_index(['query_id','product_id']);z=z.loc[z.index.isin(support)].reset_index()
    for k in [20,100]:
     macro=z.assign(hit=z['rank']<=k).groupby(['query_id','stem']).hit.mean().groupby('query_id').mean();common_valid_metrics.append({'dataset':ds,'model':model,'optimizer':'G'+n,'method':method,'K':k,'common_pairs':len(support),'queries':len(macro),'macro_inclusion':float(macro.mean()) if len(macro) else np.nan})
  qp=f.assign(hit=f['rank']<=100).groupby(['method','query_id','stem']).hit.mean().groupby(['method','query_id']).mean().unstack(0);query_files[(ds,model)]=qp
pd.DataFrame(summary).to_csv(OUT/'optimization_robustness_summary.csv',index=False);pd.DataFrame(orders).to_csv(OUT/'per_order_inclusion.csv',index=False);pd.DataFrame(states).to_csv(OUT/'visibility_states.csv',index=False);pd.DataFrame(transitions).to_csv(OUT/'rescued_missed.csv',index=False);pd.DataFrame(fully).to_csv(OUT/'fully_fitting.csv',index=False);pd.DataFrame(common_fully).to_csv(OUT/'common_fully_fitting.csv',index=False);pd.DataFrame(common_valid_metrics).to_csv(OUT/'common_valid_metrics.csv',index=False)

def sign_p(d,seed=20260907):
 d=np.asarray(d,float);rng=np.random.default_rng(seed);obs=abs(d.mean());vals=[]
 for _ in range(20):vals.extend(abs((d*rng.choice([-1,1],(500,len(d)))).mean(axis=1)))
 return (1+sum(x>=obs for x in vals))/(1+len(vals))
def holm(ps):
 out=np.empty(len(ps));order=np.argsort(ps);running=0
 for rank,i in enumerate(order):running=max(running,(len(ps)-rank)*ps[i]);out[i]=min(1,running)
 return out
contrasts=[]
for (ds,model),q in query_files.items():
 cells=[('G1','Raw'),('CG1','G1'),('CG1','Canonical'),('G2','Raw'),('CG2','G2'),('CG2','Canonical')];tmp=[]
 for a,b in cells:
  d=(q[a]-q[b]).dropna();mu,lo,hi=boot(d);tmp.append({'dataset':ds,'model':model,'method_a':a,'method_b':b,'K':100,'delta':mu,'ci_low':lo,'ci_high':hi,'p_raw':sign_p(d),'queries':len(d)})
 adj=holm([x['p_raw'] for x in tmp])
 for x,p in zip(tmp,adj):x['p_holm']=p;contrasts.append(x)
pd.DataFrame(contrasts).to_csv(OUT/'primary_contrasts.csv',index=False)

# Existing Phase-4 full-catalog pure raw-atom sorting provides the deployable
# Recall comparator for C.  This is a paired re-analysis of frozen ranks, not a
# target-intervention rate and not the M2 field-framing template.
catalog=[]
split=json.loads((P4/'config/splits.json').read_text())
for ds in ['wands','esci']:
 for model in ['bge_base','minilm']:
  folder=P4/'results'/ds
  a=pd.read_csv(folder/f'{model}_canonical_raw_per_query.csv').set_index('query_id')
  b=pd.read_csv(folder/f'{model}_Original_per_query.csv').set_index('query_id')
  ids=a.index.difference(split['wands_dev']) if ds=='wands' else a.index
  for metric in ['Recall@20','Recall@100','cNDCG@10']:
   d=(a.loc[ids,metric]-b.loc[ids,metric]).dropna();mu,lo,hi=boot(d)
   catalog.append({'dataset':ds,'model':model,'method_a':'canonical_raw','method_b':'Original','metric':metric,'queries':len(d),'delta':mu,'ci_low':lo,'ci_high':hi,'source':'phase4 frozen full-catalog ranks'})
pd.DataFrame(catalog).to_csv(OUT/'canonical_catalog_contrasts.csv',index=False)

gens=pd.DataFrame([json.loads(x) for x in (OUT/'generations.jsonl').open(encoding='utf8')]);rows=[];common=[];components=[]
for ds in frozen['datasets']:
 for source,label in [('heuristic_authoritative','G1'),('optimized_technical','G2')]:
  g=gens[(gens.dataset==ds)&(gens.method==source)&(gens.replicate=='primary')];raw=g[g.input_kind=='raw'];can=g[g.input_kind=='canonical'];valid=raw.groupby('product_id').apply(lambda x:len(x)==7 and all(v['pass'] for v in x.fact_check),include_groups=False)
  for kind,x in [('G',raw),('CG',can)]:
   execution=np.array([(r.finish_reason!='ok') or bool(r.fact_check['empty']) for r in x.itertuples()]);guard=np.array([e or not r.fact_check['pass'] for e,r in zip(execution,x.itertuples())])
   rows.append({'dataset':ds,'method':kind+label[-1],'outputs':len(x),'fact_pass_rate':float(np.mean([v['pass'] for v in x.fact_check])),'execution_fallback_rate':float(execution.mean()),'guarded_fallback_rate':float(guard.mean()),'generation_truncation_rate':float((x.finish_reason=='length').mean()),'mean_input_tokens':float(x.input_tokens.mean()),'mean_output_tokens':float(x.output_tokens.mean()),'p95_output_tokens':float(x.output_tokens.quantile(.95)),'mean_output_characters':float(x.output.map(len).mean()),'generation_seconds':float(x.generation_seconds_batch_share.sum()),'estimated_api_cost_usd':0.0})
   for name in ['new_numbers','missing_numbers','new_colors','missing_colors','new_units','missing_units','new_models','missing_models','missing_attributes','missing_negations']:
    components.append({'dataset':ds,'method':kind+label[-1],'component':name,'failure_rate':float(np.mean([bool(v.get(name,[])) for v in x.fact_check]))})
   components.append({'dataset':ds,'method':kind+label[-1],'component':'title_below_80pct','failure_rate':float(np.mean([v.get('title_pass_fraction',0)<.8 for v in x.fact_check]))})
   components.append({'dataset':ds,'method':kind+label[-1],'component':'source_conflict_present','failure_rate':float(np.mean([v.get('source_has_conflict',False) for v in x.fact_check]))})
  common.append({'dataset':ds,'method':label,'all_seven_fact_valid_products':int(valid.sum()),'all_seven_fact_valid_rate':float(valid.mean()),'total_products':len(valid)})
pd.DataFrame(rows).to_csv(OUT/'factuality_cost.csv',index=False);pd.DataFrame(common).to_csv(OUT/'common_valid_support.csv',index=False);pd.DataFrame(components).to_csv(OUT/'fact_failure_components.csv',index=False)

noise=[];same=[];noise_fact=[]
for ds in frozen['datasets']:
 p=OUT/f'{ds}_bge_base_noise_ranks.parquet'
 if not p.exists():continue
 f=pd.read_parquet(p)
 for method,g in f.groupby('method'):
  raw=g[g.stem.isin(STEMS)]
  for rep,h in raw.groupby('replicate'):
   w=h.pivot(index=['query_id','product_id'],columns='stem',values='rank')[STEMS];vi=w.le(100).any(axis=1)&~w.le(100).all(axis=1);noise.append({'dataset':ds,'method':method,'replicate':rep,'pairs':len(w),'VI@100_micro':float(vi.mean()),'rank_spread_mean':float((w.max(axis=1)-w.min(axis=1)).mean())})
  for input_name,h in [('C0',g[g.stem.eq('C0')]),('canonical',g[g.stem.eq('canonical')])]:
   w=h.pivot(index=['query_id','product_id'],columns='replicate',values='rank');cross=w.le(100).any(axis=1)&~w.le(100).all(axis=1);same.append({'dataset':ds,'method':method,'input':input_name,'pairs':len(w),'replicate_crossing@100':float(cross.mean()),'rank_spread_mean':float((w.max(axis=1)-w.min(axis=1)).mean())})
pd.DataFrame(noise).to_csv(OUT/'generation_noise_order_effect.csv',index=False);pd.DataFrame(same).to_csv(OUT/'generation_noise_same_input.csv',index=False)
ng=gens[gens.replicate.str.startswith('noise')]
for (ds,m,rep,kind),g in ng.groupby(['dataset','method','replicate','input_kind']):noise_fact.append({'dataset':ds,'method':short.get(m,m),'replicate':rep,'input_kind':kind,'outputs':len(g),'fact_pass_rate':float(np.mean([x['pass'] for x in g.fact_check])),'truncation_rate':float((g.finish_reason=='length').mean())})
pd.DataFrame(noise_fact).to_csv(OUT/'generation_noise_factuality.csv',index=False)
print(pd.DataFrame(summary)[['dataset','model','method','inclusion@100_macro','VI@100_macro']].to_string(index=False))
