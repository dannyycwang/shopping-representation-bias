from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]; P2=ROOT/"phase2"; OUT=ROOT/"phase3"/"results"
STEMS=["C0","C1","C2s1","C2s2","C2s3","C2s4","C2s5"]
MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"; REVISION="233902d25c440f23af6f7d6e94d2946bac0bee0a"


def bootstrap(values,seed=20260963,samples=10000):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)];rng=np.random.default_rng(seed);means=[]
    for start in range(0,samples,500):
        n=min(500,samples-start);means.extend(x[rng.integers(0,len(x),size=(n,len(x)))].mean(1))
    return [float(x.mean()),*map(float,np.quantile(means,[.025,.975]))]


def load_rep(dataset,stem):
    ids=[]; texts=[]
    with gzip.open(P2/"data/representations"/dataset/f"{stem}.jsonl.gz","rt",encoding="utf8") as f:
        for line in f:
            x=json.loads(line); ids.append(x["product_id"]); texts.append(x["text"])
    return ids,texts


def embedding_file(dataset,stem):
    files=list((P2/"results/embeddings").glob(f"{dataset}_bge_base_native_{stem}_*.npy"))
    if stem!="C0": assert len(files)==1,(stem,files)
    if len(files)==1:return files[0]
    # C0 has a historical duplicate. Select the file matching a frozen stored score.
    pair=pd.read_parquet(P2/"results/phase2_pair_ranks"/f"{dataset}_bge_base_native_C0.parquet").iloc[0]
    ids,_=load_rep(dataset,"C0"); pi={str(x):i for i,x in enumerate(ids)}[str(pair.product_id)]
    queries=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv"); qi={int(x):i for i,x in enumerate(queries.query_id)}[int(pair.query_id)]
    q=select_query_embedding(dataset)
    return min(files,key=lambda f:abs(float(q[qi]@np.load(f,mmap_mode="r")[pi])-float(pair.score)))


def select_query_embedding(dataset):
    qfiles=list((P2/"results/embeddings").glob(f"{dataset}_bge_base_native_queries_*.npy")); assert qfiles
    if len(qfiles)==1:return np.load(qfiles[0],mmap_mode="r")
    pair=pd.read_parquet(P2/"results/phase2_pair_ranks"/f"{dataset}_bge_base_native_C1.parquet").iloc[0]
    ids,_=load_rep(dataset,"C1"); pi={str(x):i for i,x in enumerate(ids)}[str(pair.product_id)]
    p=np.load(next((P2/"results/embeddings").glob(f"{dataset}_bge_base_native_C1_*.npy")),mmap_mode="r")
    queries=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv"); qi={int(x):i for i,x in enumerate(queries.query_id)}[int(pair.query_id)]
    best=min(qfiles,key=lambda f:abs(float(np.load(f,mmap_mode="r")[qi]@p[pi])-float(pair.score)))
    return np.load(best,mmap_mode="r")


def retrieve_top100(dataset,stem):
    path=OUT/"phase3_reranking"/f"{dataset}_{stem}_dense_top100.parquet"
    if path.exists():return pd.read_parquet(path)
    queries=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv"); q=np.asarray(select_query_embedding(dataset),dtype=np.float32)
    ids,_=load_rep(dataset,stem); p=np.load(embedding_file(dataset,stem),mmap_mode="r")
    rows=[]
    for start in range(0,len(q),32):
        scores=q[start:start+32]@np.asarray(p,dtype=np.float32).T
        idx=np.argpartition(-scores,99,axis=1)[:,:100]
        for local in range(len(scores)):
            order=idx[local][np.argsort(-scores[local,idx[local]],kind="stable")]
            for rank,j in enumerate(order,1): rows.append((int(queries.iloc[start+local].query_id),ids[int(j)],rank,float(scores[local,j])))
    d=pd.DataFrame(rows,columns=["query_id","product_id","dense_rank","dense_score"]); d.to_parquet(path,index=False); return d


def rerank(dataset,stem,model,tokenizer,device,batch_size=256):
    path=OUT/"phase3_reranking"/f"{dataset}_{stem}_reranked.parquet"
    if path.exists():return pd.read_parquet(path)
    dense=retrieve_top100(dataset,stem); queries=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv").set_index("query_id")["query"].to_dict()
    ids,texts=load_rep(dataset,stem); text_lookup={str(i):t for i,t in zip(ids,texts)}
    pairs=[(str(queries[int(q)]),text_lookup[str(p)]) for q,p in zip(dense.query_id,dense.product_id)]
    import torch
    scores=[]; started=time.time()
    for start in range(0,len(pairs),batch_size):
        block=pairs[start:start+batch_size]
        inp=tokenizer([x[0] for x in block],[x[1] for x in block],padding=True,truncation=True,max_length=256,return_tensors="pt")
        inp={k:v.to(device) for k,v in inp.items()}
        with torch.inference_mode(): logits=model(**inp).logits.float().squeeze(-1)
        scores.extend(logits.cpu().numpy().tolist())
        if start==0 or start%16384<batch_size:print(dataset,stem,start+len(block),"/",len(pairs),f"{time.time()-started:.0f}s",flush=True)
    dense["rerank_score"]=scores
    dense=dense.sort_values(["query_id","rerank_score","product_id"],ascending=[True,False,True],kind="stable")
    dense["rerank_rank"]=dense.groupby("query_id").cumcount()+1
    dense.to_parquet(path,index=False);return dense


def analyze(dataset):
    highest="Exact" if dataset=="wands" else "E"; parts=[]
    for stem in STEMS:
        base=pd.read_parquet(P2/"results/phase2_pair_ranks"/f"{dataset}_bge_base_native_{stem}.parquet")
        r=pd.read_parquet(OUT/"phase3_reranking"/f"{dataset}_{stem}_reranked.parquet")
        d=base.merge(r[["query_id","product_id","rerank_rank"]],on=["query_id","product_id"],how="left")
        d["rerank_rank_full"]=d.rerank_rank.fillna(10**9)
        parts.append(d[["query_id","product_id","label","grade","rank","rerank_rank_full"]].rename(columns={"rank":f"dense_{stem}","rerank_rank_full":f"rerank_{stem}"}))
    w=parts[0]
    for d in parts[1:]:w=w.merge(d,on=["query_id","product_id","label","grade"],validate="one_to_one")
    h=w[w.label.eq(highest)].copy(); dense=h[[f"dense_{s}" for s in STEMS]].to_numpy(); rer=h[[f"rerank_{s}" for s in STEMS]].to_numpy()
    common=(dense<=100).all(1); recoverable=(dense<=100).any(1)
    h["Irrecoverable@100"]=recoverable&~common
    before=((dense<=20).any(1)&(dense>20).any(1)); after=((rer<=20).any(1)&(rer>20).any(1))
    hc=h[common]; db=dense[common]; ra=rer[common]
    irre_q=h.groupby("query_id")["Irrecoverable@100"].mean();irre_macro,irre_lo,irre_hi=bootstrap(irre_q)
    common_frame=h.loc[common,["query_id"]].copy();common_frame["dense_vi20"]=before[common];common_frame["rerank_vi20"]=after[common]
    dense_q=common_frame.groupby("query_id").dense_vi20.mean();rerank_q=common_frame.groupby("query_id").rerank_vi20.mean()
    dense_macro,dense_lo,dense_hi=bootstrap(dense_q,20260964);rer_macro,rer_lo,rer_hi=bootstrap(rerank_q,20260965)
    result={"dataset":dataset,"first_stage":"BAAI/bge-base-en-v1.5","reranker":MODEL,"reranker_revision":REVISION,
            "highest_pairs":len(h),"ever_in_top100":int(recoverable.sum()),"common_top100":int(common.sum()),
            "Irrecoverable@100_micro_all_highest":float(h["Irrecoverable@100"].mean()),
            "Irrecoverable@100_given_ever_retrieved":float(h.loc[recoverable,"Irrecoverable@100"].mean()),
            "Irrecoverable@100_query_macro":irre_macro,"Irrecoverable@100_macro_ci_low":irre_lo,"Irrecoverable@100_macro_ci_high":irre_hi,
            "common_pairs_dense_rank_range_median":float(np.median(db.max(1)-db.min(1))) if len(db) else np.nan,
            "common_pairs_rerank_range_median":float(np.median(ra.max(1)-ra.min(1))) if len(ra) else np.nan,
            "common_pairs_dense_VI@20":float(before[common].mean()) if common.any() else np.nan,
            "common_pairs_rerank_VI@20":float(after[common].mean()) if common.any() else np.nan,
            "common_query_macro_dense_VI@20":dense_macro,"common_query_macro_dense_VI@20_ci_low":dense_lo,"common_query_macro_dense_VI@20_ci_high":dense_hi,
            "common_query_macro_rerank_VI@20":rer_macro,"common_query_macro_rerank_VI@20_ci_low":rer_lo,"common_query_macro_rerank_VI@20_ci_high":rer_hi,
            "all_highest_C0_dense_recall@20":float((dense[:,0]<=20).mean()),
            "all_highest_C0_rerank_recall@20":float((rer[:,0]<=20).mean())}
    pd.DataFrame([result]).to_csv(OUT/"phase3_tables"/f"{dataset}_reranker_pipeline.csv",index=False)
    h.to_parquet(OUT/"phase3_reranking"/f"{dataset}_pipeline_highest_pairs.parquet",index=False)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--datasets",nargs="+",default=["wands","esci"]);args=ap.parse_args()
    (OUT/"phase3_reranking").mkdir(parents=True,exist_ok=True);(OUT/"phase3_tables").mkdir(exist_ok=True)
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    device="cuda" if torch.cuda.is_available() else "cpu";tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION).to(device).eval()
    if device=="cuda":model.half()
    for dataset in args.datasets:
        for stem in STEMS:rerank(dataset,stem,model,tok,device)
        analyze(dataset)


if __name__=="__main__":main()
