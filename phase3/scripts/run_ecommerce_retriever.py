from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]; P2=ROOT/"phase2"; OUT=ROOT/"phase3"/"results"
sys.path.insert(0,str(P2))
from src.evaluation import evaluate, mean_bootstrap

MODEL="Marqo/marqo-ecommerce-embeddings-B"
REVISION="856b7151a2d1698bf940bc1b3813021c47991481"
STEMS=["C0","C1","C2s1","C2s2","C2s3","C2s4","C2s5"]


def load_texts(dataset,stem):
    path=P2/"data/representations"/dataset/f"{stem}.jsonl.gz"; ids=[]; texts=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            x=json.loads(line); ids.append(x["product_id"]); texts.append(x["text"])
    return ids,texts


class Encoder:
    def __init__(self,batch=384):
        import open_clip,torch
        self.torch=torch; self.open_clip=open_clip; self.device="cuda" if torch.cuda.is_available() else "cpu"; self.batch=batch
        self.model,_,_=open_clip.create_model_and_transforms(f"hf-hub:{MODEL}",device=self.device)
        self.model.eval(); self.tokenizer=open_clip.get_tokenizer(f"hf-hub:{MODEL}")
        if self.device=="cuda": self.model.half()
    def encode(self,texts,path):
        if path.exists(): return np.load(path,mmap_mode="r")
        out=np.empty((len(texts),768),dtype=np.float32); started=time.time()
        pos=0
        while pos<len(texts):
            block=texts[pos:pos+self.batch]
            try:
                tokens=self.tokenizer(block).to(self.device)
                with self.torch.inference_mode(): values=self.model.encode_text(tokens,normalize=True)
                out[pos:pos+len(block)]=values.float().cpu().numpy(); pos+=len(block)
                if pos==len(block) or pos%4096<len(block): print(f"encoded {pos}/{len(texts)} {time.time()-started:.0f}s",flush=True)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache(); self.batch=max(16,self.batch//2); print("reduced batch",self.batch,flush=True)
        np.save(path,out); return np.load(path,mmap_mode="r")


def analyze(dataset,model_key="marqo_ecommerce_b"):
    table=[]; highest_label="Exact" if dataset=="wands" else "E"; frames=[]
    for stem in STEMS:
        pair=pd.read_parquet(OUT/"phase3_pair_ranks"/f"{dataset}_{model_key}_{stem}.parquet")
        frames.append(pair[["query_id","product_id","label","grade","query_type","rank"]].rename(columns={"rank":stem}))
    wide=frames[0]
    for d in frames[1:]: wide=wide.merge(d,on=["query_id","product_id","label","grade","query_type"],validate="one_to_one")
    ranks=wide[STEMS].to_numpy(); highest=wide[wide.label.eq(highest_label)].copy()
    hr=highest[STEMS].to_numpy(); highest["rank_range"]=hr.max(1)-hr.min(1)
    row={"dataset":dataset,"retriever":model_key,"model":MODEL,"revision":REVISION,"highest_pairs":len(highest),"rank_range_median":highest.rank_range.median()}
    for k in [10,20,50]:
        inside=(hr<=k).sum(1); highest[f"VI@{k}"]=((inside>0)&(inside<len(STEMS))).astype(float)
        q=highest.groupby("query_id")[f"VI@{k}"].mean(); lo,hi=mean_bootstrap(q,20260906+k,10000)
        row.update({f"VI@{k}_micro":highest[f"VI@{k}"].mean(),f"VI@{k}_query_macro":q.mean(),f"VI@{k}_macro_ci_low":lo,f"VI@{k}_macro_ci_high":hi})
    per=pd.read_csv(OUT/"phase3_per_query"/f"{dataset}_{model_key}_C0.csv")
    row.update({"cNDCG@10":per["cNDCG@10"].mean(),"cNDCG@20":per["cNDCG@20"].mean(),"HighestRecall@20":per["HighestRecall@20"].mean()})
    pd.DataFrame([row]).to_csv(OUT/"phase3_tables"/f"{dataset}_{model_key}_sensitivity.csv",index=False)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--datasets",nargs="+",default=["wands","esci"]); args=ap.parse_args()
    for folder in ["phase3_embeddings","phase3_pair_ranks","phase3_per_query","phase3_top100","phase3_tables"]: (OUT/folder).mkdir(parents=True,exist_ok=True)
    enc=Encoder()
    metadata={"model":MODEL,"revision":REVISION,"library":"open_clip_torch 2.32.0","context_tokens":64,"precision":"fp16","pooling":"model encode_text normalized"}
    (OUT/"phase3_embeddings"/"marqo_ecommerce_b_metadata.json").write_text(json.dumps(metadata,indent=2))
    for dataset in args.datasets:
        queries=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv"); judgments=pd.read_csv(P2/"data/processed"/f"{dataset}_judgments.csv")
        qemb=enc.encode(queries["query"].fillna("").tolist(),OUT/"phase3_embeddings"/f"{dataset}_marqo_ecommerce_b_queries.npy")
        expected=None
        for stem in STEMS:
            pair_path=OUT/"phase3_pair_ranks"/f"{dataset}_marqo_ecommerce_b_{stem}.parquet"
            if pair_path.exists(): print("result cache hit",pair_path.name,flush=True); continue
            ids,texts=load_texts(dataset,stem); assert expected is None or ids==expected; expected=ids
            pemb=enc.encode(texts,OUT/"phase3_embeddings"/f"{dataset}_marqo_ecommerce_b_{stem}.npy")
            scores=np.asarray(qemb,dtype=np.float32)@np.asarray(pemb,dtype=np.float32).T
            pq,pairs,top=evaluate(scores,ids,queries,judgments,dataset)
            pq.to_csv(OUT/"phase3_per_query"/f"{dataset}_marqo_ecommerce_b_{stem}.csv",index=False)
            pairs.to_parquet(pair_path,index=False); top.to_parquet(OUT/"phase3_top100"/f"{dataset}_marqo_ecommerce_b_{stem}.parquet",index=False)
            print(dataset,stem,"cNDCG10",pq["cNDCG@10"].mean(),flush=True); del scores
        print(json.dumps(analyze(dataset),indent=2),flush=True)


if __name__=="__main__": main()
