import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.representations import build, audit, atoms, FIELDS, features, tokens
from src.retrieval import bm25, rank, hybrid, Encoder
from src.evaluation import evaluate
from src.cache import signature, ready, tag

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default=str(ROOT/'configs/phase1.json'))
    parser.add_argument('--stage',choices=['prepare','bm25','dense','all'],default='all')
    args=parser.parse_args(); cfg=json.loads(Path(args.config).read_text())
    for folder in ['data/raw','data/representations','data/embeddings','results/raw','results/per_query','results/tables','results/figures','results/qualitative']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data/raw'
    for file in ['product.csv','query.csv','label.csv','README.md','LICENSE']:
        if not (raw/file).exists():
            url=f'https://raw.githubusercontent.com/wayfair/WANDS/{cfg["dataset_commit"]}/'+('dataset/' if file.endswith('.csv') else '')+file
            urllib.request.urlretrieve(url,raw/file)
    p=pd.read_csv(raw/'product.csv',sep='\t').sort_values('product_id').reset_index(drop=True)
    q=pd.read_csv(raw/'query.csv',sep='\t').sort_values('query_id').reset_index(drop=True)
    raw_labels=pd.read_csv(raw/'label.csv',sep='\t')
    conflicts=raw_labels.groupby(['query_id','product_id']).label.nunique()
    ambiguous=set(conflicts[conflicts>1].index)
    raw_labels[raw_labels.set_index(['query_id','product_id']).index.isin(ambiguous)].to_csv(ROOT/'results/tables/ambiguous_judgments.csv',index=False)
    lab=raw_labels[~raw_labels.set_index(['query_id','product_id']).index.isin(ambiguous)].drop_duplicates(['query_id','product_id']).copy()
    lab['grade']=lab.label.map({'Exact':2,'Partial':1,'Irrelevant':0})
    assert p.product_id.is_unique and q.query_id.is_unique
    assert not lab.duplicated(['query_id','product_id']).any() and lab.grade.notna().all()
    assert set(lab.product_id)<=set(p.product_id) and set(lab.query_id)==set(q.query_id)
    stats=[]
    for col in p:
        s=p[col]; item={'field':col,'missing':int(s.isna().sum()),'missing_pct':float(s.isna().mean()*100),
                       'unique_nonmissing':int(s.nunique()),'empty_strings':int(s.astype(str).str.strip().eq('').sum())}
        if s.dtype=='object': item.update(mean_chars=float(s.fillna('').str.len().mean()),median_chars=float(s.fillna('').str.len().median()))
        else: item.update(minimum=float(s.min()),maximum=float(s.max()),mean=float(s.mean()))
        stats.append(item)
    pd.DataFrame(stats).to_csv(ROOT/'results/tables/field_statistics.csv',index=False)
    meta={'products':len(p),'queries':len(q),'raw_judgments':len(raw_labels),'judgments':len(lab),'labels':lab.label.value_counts().to_dict(),
          'duplicate_pair_rows':int(raw_labels.duplicated(['query_id','product_id']).sum()),'excluded_conflicting_pairs':len(ambiguous),
          'dataset_commit':cfg['dataset_commit'],'python':sys.version,'platform':platform.platform(),
          'config':cfg,'raw_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in raw.iterdir() if f.is_file()},
          'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','torch','transformers','matplotlib']}}
    (ROOT/'results/manifest.json').write_text(json.dumps(meta,indent=2))
    p[FIELDS]=p[FIELDS].fillna('')
    records=p.to_dict('records'); query_texts=q['query'].tolist()
    representations={}; fidelity=[]; lengths=[]
    source_path=ROOT/'data/representations/source_map.jsonl'
    with source_path.open('w',encoding='utf-8') as f:
        for row in records: f.write(json.dumps({'product_id':row['product_id'],'atoms':[{'source':s,'text':v} for s,v in atoms(row)]},ensure_ascii=False)+'\n')
    for cond in cfg['representations']:
        texts=[build(row,cond) for row in records]
        representations[cond]=texts
        with (ROOT/f'data/representations/{cond}.jsonl').open('w',encoding='utf-8') as f:
            for row,text in zip(records,texts):
                check=audit(row,text); assert check['pass'],(cond,row['product_id'])
                fidelity.append({'representation':cond,'product_id':row['product_id'],**check})
                f.write(json.dumps({'product_id':row['product_id'],'text':text},ensure_ascii=False)+'\n')
        lens=np.array([len(tokens(t)) for t in texts]); lengths.append({'representation':cond,'mean_tokens':lens.mean(),'median_tokens':np.median(lens),'p95_tokens':np.quantile(lens,.95)})
        print(f'{cond} built and audited: {len(texts)} products; mean lexical tokens={lens.mean():.1f}',flush=True)
    pd.DataFrame(fidelity).to_csv(ROOT/'results/tables/fidelity.csv',index=False)
    pd.DataFrame(lengths).to_csv(ROOT/'results/tables/lengths.csv',index=False)
    if args.stage=='prepare': return
    fingerprint=signature(ROOT,cfg)
    encoder=None; qemb=None
    for cond,texts in representations.items():
        orders={}
        for retriever in ['BM25','Dense']:
            if args.stage=='bm25' and retriever!='BM25': continue
            target=ROOT/f'results/raw/{cond}_{retriever}.npz'
            if ready(ROOT,target,fingerprint):
                orders[retriever]=np.load(target)['order']; print(f'Loaded {cond}/{retriever}',flush=True); continue
            if retriever=='BM25': scores=bm25(texts,query_texts,cfg['bm25_k1'],cfg['bm25_b'])
            else:
                if encoder is None:
                    encoder=Encoder(cfg,ROOT/'data/embeddings'); qemb=encoder.encode(query_texts,'queries')
                pemb=encoder.encode(texts,cond); scores=qemb@pemb.T
            order=rank(scores); orders[retriever]=order
            per_query,pairs,top=evaluate(order,scores,p,q,lab)
            stem=f'{cond}_{retriever}'
            per_query.to_csv(ROOT/f'results/per_query/{stem}.csv',index=False)
            pairs.to_csv(ROOT/f'results/raw/{stem}_exact.csv',index=False)
            top.to_csv(ROOT/f'results/raw/{stem}_top100.csv',index=False)
            np.savez_compressed(target,order=order)
            tag(target,fingerprint)
            print(f'{stem}: cNDCG10={per_query["cNDCG@10"].mean():.4f}, Recall20={per_query["Recall@20"].mean():.4f}',flush=True)
        if len(orders)==2:
            target=ROOT/f'results/raw/{cond}_Hybrid.npz'
            if not ready(ROOT,target,fingerprint):
                scores=hybrid(orders['BM25'],orders['Dense'],cfg['rrf_constant']); order=rank(scores)
                per_query,pairs,top=evaluate(order,scores,p,q,lab)
                stem=f'{cond}_Hybrid'
                per_query.to_csv(ROOT/f'results/per_query/{stem}.csv',index=False)
                pairs.to_csv(ROOT/f'results/raw/{stem}_exact.csv',index=False)
                top.to_csv(ROOT/f'results/raw/{stem}_top100.csv',index=False)
                np.savez_compressed(target,order=order)
                tag(target,fingerprint)
                print(f'{stem}: cNDCG10={per_query["cNDCG@10"].mean():.4f}, Recall20={per_query["Recall@20"].mean():.4f}',flush=True)
    print('Phase 1 retrieval finished.',flush=True)

if __name__=='__main__': main()
