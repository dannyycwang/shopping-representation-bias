"""Bounded forward-only provenance audit; does not change IG profile or saved ranks."""
from pathlib import Path
import argparse, gzip, json, os, sys, time
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel,AutoTokenizer
from run_teal_captum import read,dump,writecsv,loadmod,sha,STEMS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    repo=args.repo.resolve();out=args.output.resolve();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    common=loadmod(repo/'revision_final_strengthening_20260922/scripts/common.py','native_precision_common')
    ps,qs=common.load('wands');pids=[str(p['product_id']) for p in ps];pi=pids.index('24318')
    lp=repo/'revision_evidence_20260917/data/all_seven_token_lengths.csv'
    lengths=pd.read_csv(lp,dtype={'product_id':str});lengths=lengths[(lengths.dataset=='wands')&(lengths.model=='gte_modernbert')].set_index('product_id').loc[pids]
    assert len(lengths)==42994 and lengths.index.is_unique
    spec=read(out/'PROTOCOL.json')['model'];tok=AutoTokenizer.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True)
    m=AutoModel.from_pretrained(spec['name'],revision=spec['revision'],local_files_only=True,attn_implementation='sdpa').cuda().half().eval()
    m.requires_grad_(False);q=np.load(out/'fixed_query_embedding.npy');qv=torch.from_numpy(q).cuda()
    native=pd.read_csv(out/'native_records.csv',float_precision='round_trip').set_index('schedule')
    prov=common.provenance();prov=prov[(prov.dataset=='wands')&(prov.model=='gte_modernbert')].set_index('schedule')
    companions={};batches={};rows=[]
    for s in STEMS:
        lens=lengths['tokens_'+s].to_numpy();order=np.argsort(np.minimum(lens,8192),kind='stable');offset=0
        while offset<len(order):
            length=int(lens[order[offset]]);bs=1 if length>1500 else 2 if length>1024 else 4 if length>768 else 8 if length>500 else 24
            batch=order[offset:offset+bs]
            if pi in batch:break
            offset+=bs
        companions[s]={'catalog_indices':batch.tolist(),'product_ids':[pids[i] for i in batch],
                       'saved_token_lengths':[int(lens[i]) for i in batch],'stable_sorted_offset':offset,'target_slot':int(np.flatnonzero(batch==pi)[0])}
        texts=[common.serialize(ps[int(i)],s) for i in batch];enc=tok(texts,padding=True,truncation=False,return_tensors='pt')
        assert enc['attention_mask'].sum(-1).tolist()==companions[s]['saved_token_lengths']
        batches[s]=enc
    for compile_mode in [False,True]:
        m.config.reference_compile=compile_mode
        for s in STEMS:
            cached=np.load(repo/prov.loc[s,'product_embedding'],mmap_mode='r')[pi]
            for mode in ['singleton','original_length_bucket']:
                enc=tok(common.serialize(ps[pi],s),truncation=False,return_tensors='pt') if mode=='singleton' else batches[s]
                index=0 if mode=='singleton' else companions[s]['target_slot']
                with torch.inference_mode():
                    hidden=m(**{k:v.cuda() for k,v in enc.items()}).last_hidden_state
                    vectors=torch.nn.functional.normalize(hidden[:,0].float(),dim=1)
                    score=float((vectors[index]@qv).item());vector=vectors[index].cpu().numpy()
                rows.append({'schedule':s,'compile':compile_mode,'batch_mode':mode,'batch_size':len(enc['input_ids']),
                    'padded_length':enc['input_ids'].shape[1],'fresh_score':score,'saved_score':float(native.loc[s,'score']),
                    'fresh_minus_saved':score-float(native.loc[s,'score']),'max_abs_vector_error':float(np.abs(vector-cached).max()),
                    'vectors_bitwise_equal':bool(np.array_equal(vector,cached))})
                writecsv(out/'native_precision_audit.csv',rows)
                print(rows[-1],flush=True)
    dump(out/'environment/native_precision_audit.json',{'scope':'28 forward-only checks of original seven orders; no extra IG or local moves',
        'token_lengths_source':str(lp),'token_lengths_source_sha256':sha(lp),'companions':companions,
        'all_historical_tokens_verified':True,'attention':'sdpa','pooling':'FP32 CLS then L2','tf32':False,
        'IG_profile_unchanged':True,'saved_native_ranks_unchanged':True})
if __name__=='__main__':main()
