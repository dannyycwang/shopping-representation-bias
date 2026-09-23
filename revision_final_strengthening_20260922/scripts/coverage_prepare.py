from common import *
import itertools,time
from transformers import AutoTokenizer

def text_for(p,s):
 if s.startswith('E'):
  alternatives=sorted(set(itertools.permutations(p['attributes'])))
  return rep._plain(p,list(alternatives[int(s[1:])]))
 return serialize(p,s)

def main():
 verify();assert not (HERE/'data/coverage_plan.parquet').exists(),'Reuse existing frozen plan.'
 # Exact model-input aliases are a computation optimization, with no outcome selection.
 dump(HERE/'COVERAGE_EXECUTION_ADDENDUM.json',dict(before_new_outcomes=True,protocol_sha256=sha(HERE/'PROTOCOL.json'),rule='Every full text SHA256 is retained. Texts whose complete truncated input_ids, token_type_ids and attention masks are identical under the pinned profile alias one vector; historical original-seven ranks remain authoritative. New input groups use one representative forward pass. This saves redundant computation without changing model inputs, string counts, supports, seeds or schedules.',selection='first record in dataset, catalog product order, old schedules first, then fixed new schedule order; no scores examined',cache='full text hash plus encoder profile resolves via explicit model-input-hash alias manifest'))
 toks={m:AutoTokenizer.from_pretrained(model_spec(m)['name'],revision=model_spec(m)['revision'],local_files_only=True) for m in ['minilm','bge_base']}
 rows=[];est=[];supportrows=[];start=time.perf_counter()
 for ds in ['wands','esci']:
  p,q=load(ds);h=pd.read_csv(HERE/('data/'+ds+'_'+('coverage_support.csv' if ds=='wands' else 'primary_support.csv')),dtype={'product_id':str})
  wanted=set(h.product_id);products=[x for x in p if str(x['product_id']) in wanted];pending=[]
  def flush():
   if not pending:return
   texts=[x.pop('_text') for x in pending]
   for model,tok in toks.items():
    enc=tok(texts,add_special_tokens=True,truncation=False,verbose=False)
    cap=model_spec(model)['max_tokens']
    # Native truncation must preserve the terminal special token: use the tokenizer itself.
    trunc=tok(texts,add_special_tokens=True,truncation=True,max_length=cap,verbose=False)
    for i,row in enumerate(pending):
     row['tokens_'+model]=len(enc['input_ids'][i]);inputrecord={key:trunc[key][i] for key in ['input_ids','token_type_ids','attention_mask'] if key in trunc}
     row['input_sha_'+model]=textsha(json.dumps(inputrecord,sort_keys=True,separators=(',',':')))
   rows.extend(pending);pending.clear()
  for i,product in enumerate(products):
   schedules=STEMS+(['P'+str(s) for s in range(2026092201,2026092226)] if ds=='wands' else ['E'+str(j) for j in range(len(set(itertools.permutations(product['attributes']))))])
   assert ds!='esci' or len(product['attributes'])<=3
   for si,s in enumerate(schedules):
    text=text_for(product,s);pending.append(dict(dataset=ds,product_id=str(product['product_id']),schedule=s,schedule_index=si,text_sha256=textsha(text),_text=text,attribute_count=len(product['attributes'])))
   if len(pending)>=512:flush()
   if i%1000==0:print('TOKEN PLAN',ds,i,len(products),round(time.perf_counter()-start),flush=True)
  flush()
 plan=pd.DataFrame(rows);aliases=[]
 for ds in ['wands','esci']:
  group=plan[plan.dataset==ds]
  for model in ['minilm','bge_base']:
   inputcol='input_sha_'+model;tokencol='tokens_'+model;old=group[group.schedule.isin(STEMS)].drop_duplicates(inputcol);oldmap=old.set_index(inputcol)
   unique=group.drop_duplicates(inputcol);new=unique[~unique[inputcol].isin(oldmap.index)]
   for row in unique.itertuples(index=False):
    ih=getattr(row,inputcol)
    if ih in oldmap.index:
     ref=oldmap.loc[ih];kind='historical_exact_model_input';pid=ref.product_id;s=ref.schedule
    else:kind='new_forward';pid=row.product_id;s=row.schedule
    aliases.append(dict(dataset=ds,model=model,input_sha256=ih,kind=kind,representative_product_id=pid,representative_schedule=s,token_length=int(getattr(row,tokencol))))
   cap=model_spec(model)['max_tokens'];maxlens=group.groupby('product_id')[tokencol].max()
   h=pd.read_csv(HERE/('data/'+ds+'_'+('coverage_support.csv' if ds=='wands' else 'primary_support.csv')),dtype={'product_id':str});h['fully_fitting']=h.product_id.map(maxlens)<=cap;h['model']=model;h['max_tokens_all']=h.product_id.map(maxlens);h['cap']=cap;supportrows.append(h.assign(dataset=ds))
   counts=group.groupby('product_id').text_sha256.nunique();oldcounts=group[group.schedule.isin(STEMS)].groupby('product_id').text_sha256.nunique()
   counts=pd.DataFrame({'distinct_full_family_texts':counts,'distinct_old_seven_texts':oldcounts}).reset_index();counts['dataset']=ds;counts.to_csv(HERE/f'data/{ds}_distinct_order_counts.csv',index=False)
   est.append(dict(dataset=ds,model=model,products=group.product_id.nunique(),schedule_records=len(group),distinct_full_texts=group.text_sha256.nunique(),distinct_model_inputs=len(unique),new_model_inputs=len(new),historical_reused_model_inputs=len(oldmap),new_capped_tokens=int(np.minimum(new[tokencol],cap).sum()),new_untruncated_tokens=int(new[tokencol].sum()),fully_fitting_pairs=int(h.fully_fitting.sum()),fully_fitting_queries=h[h.fully_fitting].query_id.nunique(),full_pairs=len(h),full_queries=h.query_id.nunique(),runtime_estimate_seconds=float(np.minimum(new[tokencol],cap).sum()/(50000 if model=='minilm' else 9000)),runtime_estimate_basis='pre-scoring planning rate only; actual measured runtime reported later'))
 plan.to_parquet(HERE/'data/coverage_plan.parquet',index=False);pd.DataFrame(aliases).to_parquet(HERE/'data/coverage_input_aliases.parquet',index=False);pd.concat(supportrows).to_parquet(HERE/'data/coverage_common_support.parquet',index=False);pd.DataFrame(est).to_csv(HERE/'tables/coverage_compute_estimate.csv',index=False)
 dump(HERE/'data/COVERAGE_PLAN_FROZEN.json',dict(frozen_before_new_scoring=True,plan_sha256=sha(HERE/'data/coverage_plan.parquet'),alias_sha256=sha(HERE/'data/coverage_input_aliases.parquet'),support_sha256=sha(HERE/'data/coverage_common_support.parquet'),tokenizer_revisions={m:model_spec(m)['revision'] for m in toks},estimates=est,elapsed_seconds=time.perf_counter()-start))
 print('COVERAGE PLAN COMPLETE',est,flush=True)

if __name__=='__main__':main()
