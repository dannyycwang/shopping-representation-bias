from common import *
from coverage_prepare import text_for
import os,time,argparse
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['minilm','bge_base'],required=True);args=ap.parse_args();which=args.model
 verify();frozen=read(HERE/'data/COVERAGE_PLAN_FROZEN.json');assert sha(HERE/'data/coverage_plan.parquet')==frozen['plan_sha256'];assert sha(HERE/'data/coverage_input_aliases.parquet')==frozen['alias_sha256']
 assert read(HERE/'qa/coverage_plan_validation.json')['all_original_token_lengths_match']
 assert torch.cuda.is_available(),'CUDA unavailable'
 spec=importlib.util.spec_from_file_location('native_frozen_encoding',ROOT/'phase2/src/encoding.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 enc=mod.DenseEncoder(model_spec(which),CFG,HERE/'cache/native_metadata');plan=pd.read_parquet(HERE/'data/coverage_plan.parquet');aliases=pd.read_parquet(HERE/'data/coverage_input_aliases.parquet');prov=provenance();runtime=[];checks=[]
 profile=dict(model=model_spec(which),max_tokens=model_spec(which)['max_tokens'],pooling=model_spec(which)['native_pooling'],query_prefix='',product_prefix='',precision='fp16_model_fp32_pooling',torch=torch.__version__,transformers=__import__('transformers').__version__,encoder_source_sha256=sha(ROOT/'phase2/src/encoding.py'),protocol_sha256=sha(HERE/'PROTOCOL.json'))
 fingerprint=textsha(json.dumps(profile,sort_keys=True));dump(HERE/f'cache/{which}_encoder_profile.json',dict(**profile,fingerprint=fingerprint))
 for ds in ['wands','esci']:
  p,q=load(ds);lookup={str(x['product_id']):x for x in p};pi={str(x['product_id']):i for i,x in enumerate(p)};a=aliases[(aliases.dataset==ds)&(aliases.model==which)&(aliases.kind=='new_forward')].copy();a['capped']=np.minimum(a.token_length,model_spec(which)['max_tokens']);a=a.sort_values(['capped','input_sha256']).reset_index(drop=True);folder=HERE/f'cache/coverage/{ds}_{which}';folder.mkdir(parents=True,exist_ok=True)
  # Forward audit uses a hash-fixed subset of historical C0 target texts.
  old=plan[(plan.dataset==ds)&(plan.schedule=='C0')].copy();old['audit_hash']=old.product_id.map(lambda v:textsha(f'coverage-forward-audit|{ds}|{which}|{v}'));old=old.sort_values('audit_hash').head(16)
  ep=ROOT/prov[(prov.dataset==ds)&(prov.model==which)&(prov.schedule=='C0')].iloc[0].product_embedding;pv=np.load(ep,mmap_mode='r');texts=[serialize(lookup[x],'C0') for x in old.product_id];fresh=enc._forward_native(texts,'model_native',len(texts));cached=np.array([pv[pi[x]] for x in old.product_id]);qv=np.load(ROOT/prov[(prov.dataset==ds)&(prov.model==which)&(prov.schedule=='C0')].iloc[0].query_embedding)
  for i,pid in enumerate(old.product_id):checks.append(dict(dataset=ds,model=which,product_id=pid,max_abs_vector_error=float(np.max(abs(fresh[i]-cached[i]))),cosine_distance=float(1-np.dot(fresh[i],cached[i])/(np.linalg.norm(fresh[i])*np.linalg.norm(cached[i]))),max_abs_query_score_error=float(np.max(abs(qv@fresh[i]-qv@cached[i]))),historical_replaced=False))
  for block,start in enumerate(range(0,len(a),1024)):
   part=a.iloc[start:start+1024];dest=folder/f'block_{block:05d}.npz';meta=dest.with_suffix('.json');keys=part.input_sha256.tolist()
   if dest.exists() and meta.exists():
    record=read(meta);assert record['profile_fingerprint']==fingerprint and record['input_hashes']==keys and sha(dest)==record['sha256'];print('CACHE',ds,which,block,flush=True);runtime.append(record);continue
   t=time.perf_counter();texts=[text_for(lookup[r.representative_product_id],r.representative_schedule) for r in part.itertuples()];vectors=[]
   # Same native tokenizer/model/pooling implementation; OOM splits are deterministic.
   effective_batch=min(enc.base_batch_size,24) if which=='bge_base' else enc.base_batch_size
   for off in range(0,len(texts),effective_batch):
    batch=texts[off:off+effective_batch];vectors.append(enc._forward_native(batch,'model_native',len(batch)))
   v=np.concatenate(vectors);assert v.dtype==np.float32 and np.isfinite(v).all();assert np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-3)
   np.savez_compressed(dest,input_sha256=np.array(keys),vectors=v);record=dict(dataset=ds,model=which,block=block,input_hashes=keys,profile_fingerprint=fingerprint,rows=len(v),seconds=time.perf_counter()-t,sha256=sha(dest),path=dest.relative_to(ROOT).as_posix(),shape=list(v.shape),effective_batch_size=effective_batch);dump(meta,record);runtime.append(record)
   torch.cuda.empty_cache()
   print('ENCODED',ds,which,start+len(v),'/',len(a),'seconds',round(record['seconds'],1),flush=True)
 pd.DataFrame(checks).to_csv(HERE/f'qa/{which}_coverage_forward_audit.csv',index=False);dump(HERE/f'qa/{which}_coverage_encoding_complete.json',dict(model=which,profile_fingerprint=fingerprint,new_inputs=sum(x['rows'] for x in runtime),seconds=sum(x['seconds'] for x in runtime),blocks=len(runtime),all_blocks_complete=True))
 enc.close();print('ENCODING COMPLETE',which,flush=True)

if __name__=='__main__':main()
