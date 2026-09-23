"""Query-conditioned word-embedding IG with frozen native retrieval outcomes."""
from common import *
import os,time,traceback
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from transformers import AutoTokenizer,AutoModel
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def spans(product,schedule):
 ids=list(range(len(product['attributes'])))
 if schedule=='C1':ids.reverse()
 elif schedule.startswith('C2s'):ids=rep._random_order(ids,str(product['product_id']),20260910+int(schedule[-1]))
 attrs=[product['attributes'][i] for i in ids];blocks=rep._blocks(product,attrs);text='';parts=[]
 for field,value in blocks:
  if text:parts.append((len(text),len(text)+1,'separator','section separator'));text+='\n'
  if field=='attributes':
   for i,occ in enumerate(ids):
    if i:sep=product['attribute_separator'];parts.append((len(text),len(text)+len(sep),'separator','attribute separator'));text+=sep
    value=product['attributes'][occ];parts.append((len(text),len(text)+len(value),f'attr_{occ:03d}',value));text+=value
  else:parts.append((len(text),len(text)+len(value),'fixed_'+field,value));text+=value
 assert text==serialize(product,schedule)
 return text,parts

def scalar(model,inputs,qv,which,embedding=None):
 out=model(**inputs,**({'inputs_embeds':embedding} if embedding is not None else {})).last_hidden_state
 pooled=(out.float()*inputs['attention_mask'].unsqueeze(-1)).sum(1)/inputs['attention_mask'].sum(1,keepdim=True) if which=='minilm' else out[:,0].float()
 return torch.nn.functional.normalize(pooled,dim=1)@qv

def main():
 verify();assert torch.cuda.is_available(),'CUDA unavailable for bounded GPU workload'
 manifest=pd.read_csv(HERE/'data/xai_variant_manifest.csv',dtype={'product_id':str});p,q=load('wands');lookup={str(x['product_id']):x for x in p};qi={int(v):i for i,v in enumerate(q.query_id)};queries=q.set_index('query_id')['query'].to_dict();prov=provenance();folder=HERE/'data/xai';folder.mkdir(exist_ok=True);allqa=[];failures=[]
 dump(HERE/'qa/xai_runtime.json',dict(torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name(0),diagnostic_weights='Pinned FP32 checkpoint converted to native FP16, then exact FP16-rounded parameters represented in FP32 for gradients',query_vector='unchanged cached normalized FP32 vector',tf32=False,forward_calls_only=True,training=False))
 for which in ['minilm','bge_base']:
  spec=model_spec(which);tokenizer=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True);model=AutoModel.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True).cuda().half().eval();model.requires_grad_(False)
  qpath=prov[(prov.dataset=='wands')&(prov.model==which)&(prov.schedule=='C0')].iloc[0].query_embedding;qcache=np.load(ROOT/qpath);prepared=[]
  for row in manifest[manifest.model==which].itertuples(index=False):
   product=lookup[row.product_id];text,parts=spans(product,row.schedule);enc=tokenizer(text,return_tensors='pt',return_offsets_mapping=True,return_special_tokens_mask=True,truncation=True,max_length=spec['max_tokens']);offsets=enc.pop('offset_mapping')[0].tolist();special=enc.pop('special_tokens_mask')[0].bool();inp={k:v.cuda() for k,v in enc.items()};inp['position_ids']=torch.arange(inp['input_ids'].shape[1],device='cuda')[None,:];qv=torch.from_numpy(qcache[qi[int(row.query_id)]]).cuda();assert inp['input_ids'].shape[1]==row.token_length
   with torch.inference_mode():native=float(scalar(model,inp,qv,which).item())
   prepared.append((row,text,parts,offsets,special,inp,qv,native))
  model.float()
  for row,text,parts,offsets,special,inp,qv,native in prepared:
   key=f'{which}_{row.query_id}_{row.product_id}_{row.variant}';emb=model.get_input_embeddings()(inp['input_ids']).detach();keep={k:v for k,v in inp.items() if k!='input_ids'}
   with torch.no_grad():fid=float(scalar(model,inp,qv,which).item());femb=float(scalar(model,keep,qv,which,emb).item())
   error=abs(femb-row.score);inc=(femb>row.competitor_threshold_score) or (femb==row.competitor_threshold_score and row.catalog_index<row.competitor_threshold_index)
   guard=bool(inc==row.inclusion and abs(row.signed_margin)>5*error)
   forward=dict(model=which,query_id=int(row.query_id),product_id=row.product_id,query=queries[int(row.query_id)],variant=row.variant,schedule=row.schedule,display=row.display,native_saved_score=float(row.score),fresh_native_fp16_score=native,diagnostic_fp32_score=femb,input_ids_score=fid,input_ids_vs_embeds_error=abs(fid-femb),diagnostic_native_error=error,fresh_fp16_native_error=abs(native-row.score),native_rank=int(row.saved_rank),native_inclusion=bool(row.inclusion),diagnostic_inclusion=bool(inc),native_margin=float(row.signed_margin),margin_gt_five_errors=abs(row.signed_margin)>5*error,variant_boundary_guard=guard,token_count=int(row.token_length),cap=spec['max_tokens'],model_attention_implementation=str(model.config._attn_implementation),query_embedding_sha256=sha(ROOT/qpath))
   assert abs(fid-femb)<=1e-6,('input_ids/inputs_embeds forward disagreement',forward)
   tokenids=inp['input_ids'][0].tolist();tokenstrings=tokenizer.convert_ids_to_tokens(tokenids);entries=[]
   for (start,end),isspecial in zip(offsets,special.tolist()):
    if isspecial:entries.append(('special','special token'))
    else:
     inside=[(name,label) for a,b,name,label in parts if start>=a and end<=b and end>start]
     entries.append(inside[0] if len(inside)==1 else ('boundary','token spans content boundary'))
   for baseline in ['zero','pad']:
    path=folder/(key+'_'+baseline+'_qa.json')
    if path.exists():allqa.append(read(path));continue
    t=time.perf_counter();base=emb.clone();change=(~special).cuda()&inp['attention_mask'][0].bool();base[:,change,:]=0 if baseline=='zero' else model.get_input_embeddings().weight[tokenizer.pad_token_id].detach()
    attempts=[]
    try:
     with torch.no_grad():fb=float(scalar(model,keep,qv,which,base).item())
     for n in [64,128,256]:
      nodes,ww=np.polynomial.legendre.leggauss(n);nodes=(nodes+1)/2;ww=ww/2;grads=torch.zeros_like(emb);batch=8 if which=='minilm' else 2
      for first in range(0,n,batch):
       aa=torch.tensor(nodes[first:first+batch],dtype=torch.float32,device='cuda')[:,None,None];xx=(base+aa*(emb-base)).detach().requires_grad_(True);binputs={k:v.expand(len(aa),-1) for k,v in keep.items()};scores=scalar(model,binputs,qv,which,xx);grad=torch.autograd.grad(scores.sum(),xx)[0];weight=torch.tensor(ww[first:first+batch],dtype=torch.float32,device='cuda')[:,None,None];grads+=(grad*weight).sum(0,keepdim=True)
      ig=((emb-base)*grads).detach()[0].cpu().numpy();token=ig.sum(1).astype(float);residual=float(token.sum()-(femb-fb));tol=max(1e-4,.01*abs(femb-fb));passed=abs(residual)<=tol;attempts.append(dict(steps=n,residual=residual,tolerance=tol,passed=passed))
      if passed:break
     qa=dict(**forward,baseline=baseline,baseline_score=fb,score_difference=femb-fb,attribution_sum=float(token.sum()),completeness_residual=residual,completeness_tolerance=tol,completeness_passed=passed,steps=n,attempts=attempts,elapsed_seconds=time.perf_counter()-t,status='passed' if passed else 'completeness_failed')
     tr=[]
     for i,(tid,tok,offs,entry,signed) in enumerate(zip(tokenids,tokenstrings,offsets,entries,token)):
      tr.append(dict(model=which,query_id=int(row.query_id),product_id=row.product_id,variant=row.variant,schedule=row.schedule,baseline=baseline,token_index=i,token_id=tid,token=tok,char_start=offs[0],char_end=offs[1],entry_id=entry[0],entry_text=entry[1],attribution=float(signed),absolute_attribution=abs(float(signed)),feature_absolute_sum=float(np.abs(ig[i]).sum())))
     df=pd.DataFrame(tr);df.to_parquet(folder/(key+'_'+baseline+'_tokens.parquet'),index=False)
     ef=df.groupby(['model','query_id','product_id','variant','schedule','baseline','entry_id','entry_text'],sort=False,as_index=False)[['attribution','absolute_attribution','feature_absolute_sum']].sum();ef.to_parquet(folder/(key+'_'+baseline+'_entries.parquet'),index=False)
     assert abs(ef.attribution.sum()-token.sum())<1e-10
    except Exception as e:
     qa=dict(**forward,baseline=baseline,status='failed',error=repr(e),traceback=traceback.format_exc(),attempts=attempts,elapsed_seconds=time.perf_counter()-t);failures.append(qa);torch.cuda.empty_cache()
    dump(path,qa);allqa.append(qa);print('XAI',key,baseline,qa['status'],qa.get('steps'),round(qa['elapsed_seconds'],1),flush=True)
  del model,prepared;torch.cuda.empty_cache()
 df=pd.DataFrame(allqa);df.drop(columns=['attempts'],errors='ignore').to_csv(HERE/'qa/xai_forward_completeness.csv',index=False);dump(HERE/'qa/xai_failures.json',failures)
 alltoken=pd.concat([pd.read_parquet(p) for p in folder.glob('*_tokens.parquet')],ignore_index=True);alltoken.to_parquet(HERE/'data/xai_token_attributions.parquet',index=False)
 allentries=pd.concat([pd.read_parquet(p) for p in folder.glob('*_entries.parquet')],ignore_index=True);allentries.to_parquet(HERE/'data/xai_entry_attributions.parquet',index=False)
 guards=[]
 for key,g in df.groupby(['model','query_id','product_id']):guards.append(dict(model=key[0],query_id=int(key[1]),product_id=key[2],both_variant_boundary_guard=bool(g.variant_boundary_guard.all()),all_completeness_passed=bool(g.get('completeness_passed',pd.Series(False,index=g.index)).fillna(False).all()),runs=len(g),display=g.display.iloc[0]))
 pd.DataFrame(guards).to_csv(HERE/'qa/xai_case_guards.csv',index=False)
 print('XAI COMPLETE',len(df),'input/baseline runs',flush=True)

if __name__=='__main__':main()
