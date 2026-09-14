from common5 import *
import argparse,gc

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['primary','noise'],default='primary');args=ap.parse_args()
frozen=json.loads((CONFIG/'sample.json').read_text());gens=[json.loads(x) for x in (OUT/'generations.jsonl').open(encoding='utf8')]
short={'heuristic_authoritative':'G1','optimized_technical':'G2'}
cfg=json.loads((P2/'config/phase2.json').read_text());cfg['encoding']['batch_size_bge_base']=48;cfg['encoding']['batch_size_minilm']=128
profile={'key':'native','chunk_tokens':None,'overlap':0,'pooling':'model_native'}

def generated_text(g,canonical,guard=False):
    execution=g['finish_reason']!='ok' or g['fact_check']['empty'];fallback=execution or (guard and not g['fact_check']['pass'])
    return (canonical if fallback else g['output']),execution,fallback

for ds,spec in frozen['datasets'].items():
 p,q,j=load(ds);qids=spec['query_ids'];highest='Exact' if ds=='wands' else 'E';pairs=j[j.query_id.isin(qids)&j.label.eq(highest)][['query_id','product_id','label']].copy();pairs['product_id']=pairs.product_id.astype(str)
 rm={str(x['product_id']):x for x in p};pi={str(x['product_id']):i for i,x in enumerate(p)};qi={int(x):i for i,x in enumerate(q.query_id)}
 if args.mode=='primary':
  entries=[]
  primary={(x['product_id'],x['method'],x['stem']):x for x in gens if x['dataset']==ds and x['replicate']=='primary'}
  assert len(primary)==spec['unique_products']*2*8,(ds,len(primary))
  for pid in spec['product_ids']:
   can=canonical_text(rm[pid])
   for stem in STEMS:entries.append({'product_id':pid,'method':'Canonical','stem':stem,'text':can,'fact_pass':True,'execution_fallback':False,'guard_fallback':False})
   for source_method,label in short.items():
    cg=primary[(pid,source_method,'canonical')]
    for stem in STEMS:
     g=primary[(pid,source_method,stem)]
     for guard in [False,True]:
      txt,ex,fb=generated_text(g,can,guard);entries.append({'product_id':pid,'method':('Guard' if guard else '')+label,'stem':stem,'text':txt,'fact_pass':g['fact_check']['pass'],'execution_fallback':ex,'guard_fallback':fb})
      txt,ex,fb=generated_text(cg,can,guard);entries.append({'product_id':pid,'method':('GuardC' if guard else 'C')+label,'stem':stem,'text':txt,'fact_pass':cg['fact_check']['pass'],'execution_fallback':ex,'guard_fallback':fb})
  entries=pd.DataFrame(entries);models=['bge_base','minilm'];outstem='target_ranks'
 else:
  chosen={(x['dataset'],x['product_id']) for x in frozen['noise_products']};pairs=pairs[pairs.product_id.map(lambda x:(ds,x) in chosen)].copy()
  if pairs.empty:continue
  entries=[]
  noise=[x for x in gens if x['dataset']==ds and x['replicate'].startswith('noise')]
  for g in noise:
   can=canonical_text(rm[g['product_id']]);txt,ex,fb=generated_text(g,can,False);entries.append({'product_id':g['product_id'],'method':short[g['method']],'stem':g['stem'],'replicate':g['replicate'],'text':txt,'fact_pass':g['fact_check']['pass'],'execution_fallback':ex,'guard_fallback':fb})
  entries=pd.DataFrame(entries);models=['bge_base'];outstem='noise_ranks'
 entries['text_sha256']=entries.text.map(sha);unique=entries[['text_sha256','text']].drop_duplicates('text_sha256').sort_values('text_sha256').reset_index(drop=True)
 entries.to_parquet(OUT/f'{ds}_{args.mode}_text_manifest.parquet',index=False);unique.to_json(OUT/f'{ds}_{args.mode}_unique_texts.jsonl',orient='records',lines=True,force_ascii=False)
 for model_key in models:
  specm=next(x for x in cfg['models'] if x['key']==model_key);enc=DenseEncoder(specm,cfg,P5/'embeddings');t0=time.time();vec=np.asarray(enc.encode(unique.text.tolist(),f'{ds}_{args.mode}_{model_key}',profile,sha('\0'.join(unique.text))),dtype='f4');elapsed=time.time()-t0
  token_lengths=[len(x) for x in enc.tokenizer(unique.text.tolist(),add_special_tokens=True,truncation=False,verbose=False)['input_ids']];max_tokens=specm['max_tokens'];vmap={h:vec[i] for i,h in enumerate(unique.text_sha256)}
  length_rows=pd.DataFrame({'text_sha256':unique.text_sha256,'retriever_tokens':token_lengths,'retriever_max_tokens':max_tokens,'fully_fits':np.asarray(token_lengths)<=max_tokens})
  raw_lookup=None
  if args.mode=='primary':
   raw_lookup=pd.DataFrame([{'product_id':pid,'stem':stem,'text_sha256':sha(raw_text(rm[pid],stem)),'text':raw_text(rm[pid],stem)} for pid in spec['product_ids'] for stem in STEMS])
   ru=raw_lookup[['text_sha256','text']].drop_duplicates('text_sha256');rl=[len(x) for x in enc.tokenizer(ru.text.tolist(),add_special_tokens=True,truncation=False,verbose=False)['input_ids']]
   length_rows=pd.concat([length_rows,pd.DataFrame({'text_sha256':ru.text_sha256,'retriever_tokens':rl,'retriever_max_tokens':max_tokens,'fully_fits':np.asarray(rl)<=max_tokens})]).drop_duplicates('text_sha256')
  enc.close();del vec;gc.collect();length_rows.to_parquet(OUT/f'{ds}_{model_key}_{args.mode}_text_lengths.parquet',index=False)
  qv,pv=original_cache(ds,model_key);qs=np.array([qi[x] for x in qids]);base=np.asarray(qv[qs],dtype='f4')@np.asarray(pv,dtype='f4').T;qrow={qid:i for i,qid in enumerate(qids)}
  long=[]
  if args.mode=='primary':
   raw=pd.read_parquet(P3/f'results/target_only_permutations/{ds}_{model_key}_pairs.parquet');raw=raw[raw.query_id.isin(qids)]
   for stem in STEMS:
    z=raw[['query_id','product_id',stem]].rename(columns={stem:'rank'});z['product_id']=z.product_id.astype(str);z['method']='Raw';z['stem']=stem;z['score']=np.nan;z['fact_pass']=True;z['execution_fallback']=False;z['guard_fallback']=False;z=z.merge(raw_lookup[['product_id','stem','text_sha256']],on=['product_id','stem'],validate='many_to_one');long.append(z)
  for (method,stem,*rep),eg in entries.groupby(['method','stem']+(['replicate'] if args.mode=='noise' else []),sort=False):
   m=pairs.merge(eg.drop_duplicates('product_id'),on='product_id',validate='many_to_one');indices=np.array([pi[x] for x in m.product_id]);values=np.array([float(qv[qi[int(r.query_id)]]@vmap[r.text_sha256]) for r in m.itertuples()],dtype='f4');ranks=np.empty(len(m),int)
   for qid,g in m.assign(_value=values,_index=indices).groupby('query_id',sort=False):ranks[g.index]=rank_target(base[qrow[int(qid)]],g._index.to_numpy(),g._value.to_numpy(dtype='f4'))
   z=m[['query_id','product_id','fact_pass','execution_fallback','guard_fallback','text_sha256']].copy();z['method']=method;z['stem']=stem;z['rank']=ranks;z['score']=values
   if args.mode=='noise':z['replicate']=rep[0]
   long.append(z)
  result=pd.concat(long,ignore_index=True);result.to_parquet(OUT/f'{ds}_{model_key}_{outstem}.parquet',index=False)
  pd.DataFrame([{'dataset':ds,'model':model_key,'mode':args.mode,'encoded_unique_texts':len(unique),'embedding_seconds':elapsed,'retriever_max_tokens':max_tokens,'fraction_unique_texts_truncated':float(np.mean(np.array(token_lengths)>max_tokens)),'mean_tokens':float(np.mean(token_lengths)),'p95_tokens':float(np.quantile(token_lengths,.95))}]).to_csv(OUT/f'{ds}_{model_key}_{args.mode}_embedding_cost.csv',index=False)
  print(ds,model_key,args.mode,len(result),'rows',flush=True)
