from common5 import *
import argparse,torch,datetime
from transformers import AutoTokenizer,AutoModelForCausalLM

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['primary','noise','all'],default='all');ap.add_argument('--batch-size',type=int,default=8);args=ap.parse_args()
frozen=json.loads((CONFIG/'sample.json').read_text());assert sha((ROOT/'OPTIMIZATION_ROBUSTNESS_PROTOCOL.md').read_text(encoding='utf8'))==(CONFIG/'protocol.sha256').read_text().strip()
prompts=prompt_sources();tasks=[];maps={d:record_map(d) for d in frozen['datasets']}
if args.mode in ['primary','all']:
 for ds,spec in frozen['datasets'].items():
  for pid in spec['product_ids']:
   r=maps[ds][pid]
   for method,template in prompts.items():
    for stem in STEMS:tasks.append({'dataset':ds,'product_id':pid,'method':method,'stem':stem,'input_kind':'raw','replicate':'primary','seed':20260907,'decoding':frozen['primary_decoding'],'source':raw_text(r,stem),'template':template})
    tasks.append({'dataset':ds,'product_id':pid,'method':method,'stem':'canonical','input_kind':'canonical','replicate':'primary','seed':20260907,'decoding':frozen['primary_decoding'],'source':canonical_text(r),'template':template})
if args.mode in ['noise','all']:
 for item in frozen['noise_products']:
  ds=item['dataset'];pid=item['product_id'];r=maps[ds][pid]
  for method,template in prompts.items():
   for rep,seed in enumerate(frozen['noise_decoding']['seeds'],1):
    dec={k:v for k,v in frozen['noise_decoding'].items() if k!='seeds'}
    for stem in STEMS:tasks.append({'dataset':ds,'product_id':pid,'method':method,'stem':stem,'input_kind':'raw','replicate':f'noise{rep}','seed':seed,'decoding':dec,'source':raw_text(r,stem),'template':template})
    tasks.append({'dataset':ds,'product_id':pid,'method':method,'stem':'canonical','input_kind':'canonical','replicate':f'noise{rep}','seed':seed,'decoding':dec,'source':canonical_text(r),'template':template})

for t in tasks:
 user=t['template'].format(description=t['source']);t['user_prompt']=user;t['cache_key']=cache_key(t['source'],t['template'],t['decoding'],t['replicate']);t['input_sha256']=sha(t['source'])

def task_id(x):
 return (x['dataset'],x['product_id'],x['method'],x['stem'],x['replicate'])
def materialize(task,source,cache_reused,seconds=None,created=None):
 r=maps[task['dataset']][task['product_id']];fc=fact_check(r,task['source'],source['output'],source['finish_reason'])
 rec={k:v for k,v in task.items() if k not in ['template','source','user_prompt','chat','estimated_tokens']}
 rec.update({k:source[k] for k in ['model_id','model_revision','prompt_sha256','system_sha256','output','output_sha256','input_tokens','output_tokens','finish_reason']})
 rec.update({'generation_seconds_batch_share':source['generation_seconds_batch_share'] if seconds is None else seconds,'fact_check':fc,'cache_reused':bool(cache_reused),'cache_source_task':source.get('cache_source_task',list(task_id(source)) if all(k in source for k in ['dataset','product_id','method','stem','replicate']) else None),'created_utc':created or source.get('created_utc')})
 return rec

path=OUT/'generations.jsonl';existing=[]
if path.exists():existing=[json.loads(line) for line in path.open(encoding='utf8')]
# One output is authoritative for each content-addressed key.  A task row is
# still materialized for every order, so metric populations never depend on
# whether two orderings happen to yield identical input text.
cache={}
for x in existing:cache.setdefault(x['cache_key'],x)
expected_ids={task_id(t) for t in tasks}
unrelated=[x for x in existing if task_id(x) not in expected_ids]
cached_tasks=[];charged=set()
for t in tasks:
 if t['cache_key'] in cache:
  src=cache[t['cache_key']];reuse=t['cache_key'] in charged
  cached_tasks.append(materialize(t,src,reuse,0.0 if reuse else src.get('generation_seconds_batch_share',0.0)))
  charged.add(t['cache_key'])
if existing:
 tmp=path.with_suffix('.normalized.tmp')
 with tmp.open('w',encoding='utf8') as f:
  for x in unrelated:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  for x in cached_tasks:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 tmp.replace(path)
groups=defaultdict(list)
for t in tasks:
 if t['cache_key'] not in cache:groups[t['cache_key']].append(t)
pending=[v[0] for v in groups.values()]
print(f'tasks={len(tasks)} materialized_cached={len(cached_tasks)} unique_cached={len(cache)} unique_pending={len(pending)}',flush=True)
if not pending:raise SystemExit(0)
tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV);tok.padding_side='left';tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,torch_dtype=torch.float16,device_map='cuda').eval()
for t in pending:
 t['chat']=tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':t['user_prompt']}],tokenize=False,add_generation_prompt=True);t['estimated_tokens']=len(tok.encode(t['chat'],add_special_tokens=False))
assert max(t['estimated_tokens'] for t in pending)+512<=32768
pending.sort(key=lambda x:x['estimated_tokens'])
start_all=time.time()
with path.open('a',encoding='utf8') as f:
 for bi in range(0,len(pending),args.batch_size):
  batch=pending[bi:bi+args.batch_size];enc=tok([x['chat'] for x in batch],return_tensors='pt',padding=True).to('cuda');max_new=int(batch[0]['decoding']['max_new_tokens']);do_sample=bool(batch[0]['decoding']['do_sample'])
  # Sorting keeps a batch on one decoding regime. Each row is generated with a
  # recorded seed; for stochastic controls batch size is forced to one below.
  if do_sample and len(batch)>1:raise RuntimeError('Run --mode noise with --batch-size 1 for independent seeds')
  torch.manual_seed(int(batch[0]['seed']));t0=time.time();kwargs={'max_new_tokens':max_new,'do_sample':do_sample,'pad_token_id':tok.eos_token_id}
  if do_sample:kwargs.update(temperature=float(batch[0]['decoding']['temperature']),top_p=float(batch[0]['decoding']['top_p']))
  with torch.inference_mode():gen=model.generate(**enc,**kwargs)
  elapsed=time.time()-t0;input_width=enc.input_ids.shape[1]
  for row,(task,seq) in enumerate(zip(batch,gen)):
   new=seq[input_width:];answer=tok.decode(new,skip_special_tokens=True).strip();ended=bool((new==tok.eos_token_id).any());finish='ok' if ended or len(new)<max_new else 'length';r=maps[task['dataset']][task['product_id']];fc=fact_check(r,task['source'],answer,finish)
   now=datetime.datetime.now(datetime.timezone.utc).isoformat();source={**{k:v for k,v in task.items() if k not in ['template','source','user_prompt','chat','estimated_tokens']},'model_id':MODEL_ID,'model_revision':MODEL_REV,'prompt_sha256':sha(task['template']),'system_sha256':sha(SYSTEM_PROMPT),'output':answer,'output_sha256':sha(answer),'input_tokens':int((enc.attention_mask[row]).sum()),'output_tokens':int(len(new)),'generation_seconds_batch_share':elapsed/len(batch),'finish_reason':finish,'fact_check':fc,'created_utc':now,'cache_source_task':list(task_id(task))}
   for ti,member in enumerate(groups[task['cache_key']]):
    rec=materialize(member,source,ti>0,source['generation_seconds_batch_share'] if ti==0 else 0.0,now)
    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   f.flush();cache[task['cache_key']]=source
  if bi%(args.batch_size*10)==0:print(f'{bi+len(batch)}/{len(pending)} elapsed={time.time()-start_all:.1f}s',flush=True)
print(f'complete {len(pending)} new outputs in {time.time()-start_all:.1f}s',flush=True)
