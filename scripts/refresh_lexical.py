"""Recompute lexical/fused outputs with deterministic query-term accumulation order.

This also verifies bag-of-words controls over the entire catalog, not just toy tests.
Dense artifacts are reused unchanged. Run after run_order_control.py.
"""
import json
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.retrieval import bm25,rank,hybrid
from src.evaluation import evaluate
from src.cache import signature,ready,tag

def main():
    cfg=json.loads((ROOT/'configs/phase1.json').read_text())
    parser=argparse.ArgumentParser(); parser.add_argument('--conditions',nargs='+',default=cfg['representations']+['C_order'])
    args=parser.parse_args()
    fingerprint=signature(ROOT,cfg)
    p=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').sort_values('product_id').reset_index(drop=True)
    q=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').sort_values('query_id').reset_index(drop=True)
    lab=pd.read_csv(ROOT/'data/raw/label.csv',sep='\t'); c=lab.groupby(['query_id','product_id']).label.nunique()
    lab=lab[~lab.set_index(['query_id','product_id']).index.isin(c[c>1].index)].drop_duplicates(['query_id','product_id']).copy()
    lab['grade']=lab.label.map({'Exact':2,'Partial':1,'Irrelevant':0})
    verification=[]; baseline=None
    for cond in args.conditions:
        file=ROOT/f'data/representations/{cond}.jsonl'
        if not file.exists(): continue
        texts=[json.loads(line)['text'] for line in file.open(encoding='utf-8')]
        scores=bm25(texts,q['query'].tolist(),cfg['bm25_k1'],cfg['bm25_b']); order=rank(scores)
        if cond=='R0': baseline=(scores.copy(),order.copy())
        if cond in ['R4','C_order']:
            assert baseline is not None, 'Include R0 first when verifying a bag-of-words control.'
            np.testing.assert_array_equal(scores,baseline[0]); np.testing.assert_array_equal(order,baseline[1])
            verification.append({'control':cond,'identical_scores':True,'identical_ranks':True,'query_product_comparisons':int(scores.size)})
        for retriever in ['BM25','Hybrid']:
            if retriever=='Hybrid':
                dense=ROOT/f'results/raw/{cond}_Dense.npz'
                if not dense.exists(): continue
                if dense.with_suffix('.meta.json').exists():
                    assert ready(ROOT,dense,fingerprint), 'Dense cache is stale; rerun retrieval before fusion.'
                scores=hybrid(order,np.load(dense)['order'],cfg['rrf_constant']); order=rank(scores)
            frame,pairs,top=evaluate(order,scores,p,q,lab)
            stem=f'{cond}_{retriever}'
            frame.to_csv(ROOT/f'results/per_query/{stem}.csv',index=False)
            pairs.to_csv(ROOT/f'results/raw/{stem}_exact.csv',index=False); top.to_csv(ROOT/f'results/raw/{stem}_top100.csv',index=False)
            np.savez_compressed(ROOT/f'results/raw/{stem}.npz',order=order)
            tag(ROOT/f'results/raw/{stem}.npz',fingerprint)
            print(stem,frame['cNDCG@10'].mean(),flush=True)
    verification_path=ROOT/'results/control_verification.json'
    old=json.loads(verification_path.read_text()) if verification_path.exists() else []
    combined={x['control']:x for x in old+verification}
    verification_path.write_text(json.dumps(list(combined.values()),indent=2))

if __name__=='__main__': main()
