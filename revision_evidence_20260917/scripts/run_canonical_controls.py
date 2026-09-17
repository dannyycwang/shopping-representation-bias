"""Frozen pure-order controls. No learned architecture or historical output writes."""
from common import *
from datetime import datetime, timezone
import argparse, gc, os, time, platform
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['gte_modernbert','minilm','bge_base']);ap.add_argument('--reuse-only',action='store_true');ap.add_argument('--batch-size-cap',type=int);args=ap.parse_args()
    if args.batch_size_cap is not None:assert args.batch_size_cap>0
    protocol=HERE/'PROSPECTIVE_PROTOCOL.json'
    assert sha(protocol)==(HERE/'PROSPECTIVE_PROTOCOL.sha256').read_text().strip()
    frozen=json.loads(protocol.read_text(encoding='utf8'))
    for path,digest in frozen['source_hashes'].items(): assert sha(ROOT/path)==digest,(path,'frozen design/source changed')
    import torch, transformers
    from src.encoding import DenseEncoder
    torch.set_num_threads(4)
    folder=HERE/'experiments';emb=folder/'embeddings';emb.mkdir(exist_ok=True)
    profile=next(x for x in CFG['chunk_controls'] if x['key']=='native')
    runtime=dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
                 platform=platform.platform(),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    if args.batch_size_cap is not None:
        addendum=HERE/'BGE_EXECUTION_ADDENDUM.json'
        policy=json.loads(addendum.read_text(encoding='utf8'))
        assert args.models==['bge_base'] and args.batch_size_cap==policy['effective_max_batch_size']
        assert policy['protocol_sha256']==sha(protocol)
        runtime['batching_addendum']=source(addendum)
        runtime['model_scope']=args.models
        runtime['configured_max_batch_size']=CFG['encoding']['batch_size_bge_base']
        runtime['effective_max_batch_size']=args.batch_size_cap
    dump(folder/'runtime.json',runtime)
    for model in args.models:
      spec=next(x for x in CFG['models'] if x['key']==model);enc=None
      for ds in ['wands','esci']:
        p=products(ds);qs=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_queries.csv')
        judgments=pd.read_csv(ROOT/f'phase2/data/processed/{ds}_judgments.csv',dtype={'product_id':str})
        prov=next(x for x in PROVENANCE['sources'] if x['dataset']==ds and x['model']==model)
        qpath=ROOT/prov['query_embedding'];qv=np.load(qpath)
        assert len(qv)==len(qs)
        for rule in RULES:
          result=folder/f'{ds}_{model}_{rule}_source.json'
          if result.exists():
              recorded=json.loads(result.read_text(encoding='utf8'));assert recorded['protocol_sha256']==sha(protocol)
              rankpath=ROOT/recorded['rank_source']['relative_path'];assert sha(rankpath)==recorded['rank_source']['sha256']
              print('CONTROL COMPLETE',ds,model,rule,flush=True);continue
          text=[canonical_text(x,rule) for x in p]
          digest=hashlib.sha256('\0'.join(text).encode()).hexdigest()
          inherited_name={'lexical_ascending':'canonical_raw','field_priority_type':'rule_type'}.get(rule)
          old=ROOT/f'phase4/results/{ds}/{model}_{inherited_name}_pairs.parquet'
          if args.reuse_only and not (inherited_name and old.exists()):continue
          start=time.monotonic()
          baseline_audit={}
          if inherited_name and old.exists():
              metas=list((ROOT/'phase4/results/embeddings').glob(f'{ds}_{model}_{inherited_name}_*.json'))
              matches=[x for x in metas if json.loads(x.read_text())['source_sha256']==digest]
              assert len(matches)==1,(ds,model,rule,'text cache match')
              meta=json.loads(matches[0].read_text());assert meta['model']==spec and meta['profile']==profile
              origin='reused inherited exact ranks';rankpath=old;embedding_meta=source(matches[0])
              oldbase=pairread(ROOT/f'phase4/results/{ds}/{model}_Original_pairs.parquet').set_index(['query_id','product_id']).sort_index()
              phase2=pairread(ROOT/f'phase2/results/phase2_pair_ranks/{ds}_{model}_native_C0.parquet').set_index(['query_id','product_id']).sort_index()
              assert oldbase.index.equals(phase2.index)
              h=phase2.label.eq(highest_label(ds))
              baseline_audit=dict(all_judged_rank_mismatches=int((oldbase['rank']!=phase2['rank']).sum()),
                 highest_label_rank_mismatches=int((oldbase.loc[h,'rank']!=phase2.loc[h,'rank']).sum()),
                 inclusion20_mismatches=int(((oldbase['rank']<=20)!=(phase2['rank']<=20)).sum()),
                 inclusion100_mismatches=int(((oldbase['rank']<=100)!=(phase2['rank']<=100)).sum()))
              assert baseline_audit['highest_label_rank_mismatches']==0
              assert baseline_audit['inclusion20_mismatches']==baseline_audit['inclusion100_mismatches']==0
          else:
              if enc is None:
                  enc=DenseEncoder(spec,CFG,emb)
                  if args.batch_size_cap is not None:enc.base_batch_size=min(enc.base_batch_size,args.batch_size_cap)
              print('NEW CONTROL',ds,model,rule,'products',len(p),flush=True)
              pv=np.asarray(enc.encode(text,f'{ds}_{model}_{rule}',profile,digest),dtype='f4')
              # Rank every original catalog item for every original query. Ties use original catalog index.
              scores=np.asarray(qv,dtype='f4')@pv.T
              order=np.argsort(-scores,axis=1,kind='stable')
              rank=np.empty_like(order,dtype=np.int32)
              np.put_along_axis(rank,order,np.broadcast_to(np.arange(1,len(p)+1),order.shape),axis=1)
              pi={str(x['product_id']):i for i,x in enumerate(p)};qi={int(v):i for i,v in enumerate(qs.query_id)}
              a=np.array([qi[int(v)] for v in judgments.query_id]);b=np.array([pi[str(v)] for v in judgments.product_id])
              f=judgments[['query_id','product_id','label']].copy();f['rank']=rank[a,b];f['score']=scores[a,b]
              assert not f.duplicated(['query_id','product_id']).any()
              rankpath=folder/f'{ds}_{model}_{rule}_pairs.parquet';assert not rankpath.exists()
              f.to_parquet(rankpath,index=False)
              metas=list(emb.glob(f'{ds}_{model}_{rule}_*.json'));assert len(metas)==1
              embedding_meta=source(metas[0]);origin='new controlled retrieval, frozen pretrained encoder; not regenerated raw ranks'
              del pv,scores,order,rank;gc.collect()
          dump(result,dict(dataset=ds,model=model,rule=rule,rank_origin=origin,rank_source=source(rankpath),
               query_embedding=source(qpath),query_embedding_metadata=source(qpath.with_suffix('.json')),
               product_embedding_metadata=embedding_meta,protocol_sha256=sha(protocol),text_source_sha256=digest,
               catalog_size=len(p),query_ids=list(map(int,qs.query_id)),seconds=time.monotonic()-start,baseline_numerical_audit=baseline_audit,
               completed_utc=datetime.now(timezone.utc).isoformat(),runtime=runtime,
               tie_rule='Descending fp32 score then ascending original catalog index.',incoming_schedules=STEMS))
          print('CONTROL SAVED',ds,model,rule,origin,flush=True)
      if enc is not None: del enc
      gc.collect()
      if torch.cuda.is_available():torch.cuda.empty_cache()

if __name__=='__main__':main()
