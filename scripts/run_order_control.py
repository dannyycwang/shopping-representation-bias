"""Supplementary strict length/fact control. Run after the main experiment."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from collections import Counter
from src.representations import build,FIELDS,audit,tokens
from src.retrieval import bm25,rank,hybrid,Encoder
from src.evaluation import evaluate
from src.cache import signature, ready, tag

def main():
    cfg=json.loads((ROOT/'configs/phase1.json').read_text())
    p=pd.read_csv(ROOT/'data/raw/product.csv',sep='\t').sort_values('product_id').reset_index(drop=True)
    p[FIELDS]=p[FIELDS].fillna(''); q=pd.read_csv(ROOT/'data/raw/query.csv',sep='\t').sort_values('query_id').reset_index(drop=True)
    lab=pd.read_csv(ROOT/'data/raw/label.csv',sep='\t'); conflict=lab.groupby(['query_id','product_id']).label.nunique()
    lab=lab[~lab.set_index(['query_id','product_id']).index.isin(conflict[conflict>1].index)].drop_duplicates(['query_id','product_id']).copy()
    lab['grade']=lab.label.map({'Exact':2,'Partial':1,'Irrelevant':0})
    texts=[]
    with (ROOT/'data/representations/C_order.jsonl').open('w',encoding='utf-8') as f:
        for row in p.to_dict('records'):
            t=build(row,'C_order'); original=build(row,'R0')
            assert audit(row,t)['pass'] and Counter(tokens(t))==Counter(tokens(original)) and len(t)==len(original)
            texts.append(t); f.write(json.dumps({'product_id':row['product_id'],'text':t},ensure_ascii=False)+'\n')
    orders={}; fingerprint=signature(ROOT,cfg)
    for r in ['BM25','Dense','Hybrid']:
        path=ROOT/f'results/raw/C_order_{r}.npz'
        if ready(ROOT,path,fingerprint): orders[r]=np.load(path)['order']; continue
        if r=='BM25': scores=bm25(texts,q['query'].tolist(),cfg['bm25_k1'],cfg['bm25_b'])
        elif r=='Dense':
            enc=Encoder(cfg,ROOT/'data/embeddings')
            originals=[json.loads(line)['text'] for line in (ROOT/'data/representations/R0.jsonl').open(encoding='utf-8')]
            for offset in range(0,len(texts),512):
                aa=enc.tokenizer(originals[offset:offset+512],add_special_tokens=False,truncation=False,verbose=False)['input_ids']
                bb=enc.tokenizer(texts[offset:offset+512],add_special_tokens=False,truncation=False,verbose=False)['input_ids']
                assert all(Counter(a)==Counter(b) for a,b in zip(aa,bb))
            (ROOT/'results/order_control_verification.json').write_text(json.dumps({'products':len(texts), 'same_character_count':True,'same_lexical_token_multiset':True,'same_wordpiece_multiset':True},indent=2))
            qe=enc.encode(q['query'].tolist(),'queries'); pe=enc.encode(texts,'C_order'); scores=qe@pe.T
        else: scores=hybrid(orders['BM25'],orders['Dense'],cfg['rrf_constant'])
        order=rank(scores); orders[r]=order
        # Full score/rank invariance is checked by refresh_lexical.py in one process.
        frame,pairs,top=evaluate(order,scores,p,q,lab)
        stem=f'C_order_{r}'; frame.to_csv(ROOT/f'results/per_query/{stem}.csv',index=False)
        pairs.to_csv(ROOT/f'results/raw/{stem}_exact.csv',index=False); top.to_csv(ROOT/f'results/raw/{stem}_top100.csv',index=False)
        np.savez_compressed(path,order=order)
        tag(path,fingerprint)
        print(stem,frame['cNDCG@10'].mean(),flush=True)

if __name__=='__main__': main()
