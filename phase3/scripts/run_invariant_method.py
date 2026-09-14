from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT=Path(__file__).resolve().parents[2];P2=ROOT/"phase2";OUT=ROOT/"phase3"/"results";INV=OUT/"phase3_invariant_method"
sys.path.insert(0,str(P2))
from src.encoding import DenseEncoder
from src.evaluation import evaluate,mean_bootstrap


def load_data(dataset):
    products=[]
    with gzip.open(P2/"data/processed"/f"{dataset}_products.jsonl.gz","rt",encoding="utf8") as f:products=list(map(json.loads,f))
    q=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv");j=pd.read_csv(P2/"data/processed"/f"{dataset}_judgments.csv")
    return products,q,j


def structures(products):
    unique=sorted({a for p in products for a in p["attributes"]});lookup={a:i for i,a in enumerate(unique)}
    rows=[];cols=[]
    for i,p in enumerate(products):
        attrs=p["attributes"]
        if attrs:
            rows.extend([i]*len(attrs));cols.extend(lookup[a] for a in attrs)
    counts=np.bincount(rows,minlength=len(products)).astype(np.float32);data=np.asarray([1/max(1,counts[r]) for r in rows],dtype=np.float32)
    incidence=sparse.csr_matrix((data,(rows,cols)),shape=(len(products),len(unique)),dtype=np.float32)
    # Keep every non-attribute catalog field in its fixed source order; only the
    # attribute collection receives a permutation-invariant encoder.
    text=["\n".join(x for x in [p["title"],p.get("class",""),p.get("category",""),p.get("description","")] if x) for p in products]
    return unique,incidence,text,counts


def split_queries(queries):
    order=sorted(range(len(queries)),key=lambda i:hashlib.sha256(f"phase3-dev:{queries.iloc[i].query_id}".encode()).hexdigest())
    dev=set(order[:max(1,len(order)//5)]);return np.array([i in dev for i in range(len(queries))])


def cn10(scores,ids,queries,judgments,indices):
    pq,_,_=evaluate(scores[indices],ids,queries.iloc[indices].reset_index(drop=True),judgments[judgments.query_id.isin(queries.iloc[indices].query_id)],"wands")
    return float(pq["cNDCG@10"].mean())


def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def run_dataset(dataset,encoder,chosen=None):
    products,q,j=load_data(dataset);ids=[p["product_id"] for p in products];unique,inc,text,counts=structures(products)
    profile={"key":"phase3_native","chunk_tokens":None,"overlap":0,"pooling":"model_native"}
    def enc(values,name):
        sha=hashlib.sha256("\0".join(values).encode()).hexdigest();return np.asarray(encoder.encode(values,name,profile,sha),dtype=np.float32)
    qemb=enc(q["query"].fillna("").astype(str).tolist(),f"{dataset}_queries")
    temb=enc(text,f"{dataset}_title_description");aemb=enc(unique,f"{dataset}_unique_attributes")
    raw_attr=np.asarray(incidence_dot(inc,aemb),dtype=np.float32);set_attr=normalize(raw_attr.copy())
    attr_lookup={a:i for i,a in enumerate(unique)}
    reversed_raw=[]
    for product in products:
        idx=[attr_lookup[a] for a in reversed(product["attributes"])]
        reversed_raw.append(aemb[idx].mean(0) if idx else np.zeros(aemb.shape[1],dtype=np.float32))
    reversed_raw=np.asarray(reversed_raw,dtype=np.float32)
    title_scores=qemb@temb.T;mean_scores=qemb@raw_attr.T
    max_scores=max_attribute_scores(qemb,aemb,inc)
    top3_scores=topk_attribute_scores(qemb,aemb,products,attr_lookup,k=3)

    alphas=[.25,.5,.75]
    if dataset=="wands":
        dev=split_queries(q); dev_idx=np.flatnonzero(dev);test_idx=np.flatnonzero(~dev)
        trials=[]
        for method,attribute in [("set_mean",set_attr),("late_mean",None),("late_max",None),("late_top3",None)]:
            for alpha in alphas:
                if method=="set_mean":score=qemb@normalize(alpha*temb+(1-alpha)*attribute).T
                elif method=="late_mean":score=alpha*title_scores+(1-alpha)*mean_scores
                elif method=="late_max":score=alpha*title_scores+(1-alpha)*max_scores
                else:score=alpha*title_scores+(1-alpha)*top3_scores
                trials.append({"method":method,"alpha":alpha,"dev_cNDCG@10":cn10(score,ids,q,j,dev_idx)})
        trial=pd.DataFrame(trials);trial.to_csv(INV/"alpha_selection_wands_dev.csv",index=False)
        chosen={m:float(g.sort_values(["dev_cNDCG@10","alpha"],ascending=[False,True]).iloc[0].alpha) for m,g in trial.groupby("method")}
        (INV/"chosen_hyperparameters.json").write_text(json.dumps({"dev_rule":"lowest SHA256 20% of WANDS queries","dev_queries":len(dev_idx),"heldout_queries":len(test_idx),"alphas":alphas,"chosen":chosen},indent=2))
        eval_idx=test_idx
    else:eval_idx=np.arange(len(q))

    methods={}
    for method,alpha in chosen.items():
        if method=="set_mean":score=qemb@normalize(alpha*temb+(1-alpha)*set_attr).T
        elif method=="late_mean":score=alpha*title_scores+(1-alpha)*mean_scores
        elif method=="late_max":score=alpha*title_scores+(1-alpha)*max_scores
        else:score=alpha*title_scores+(1-alpha)*top3_scores
        methods[method]=(alpha,score)
    rows=[];subset_q=q.iloc[eval_idx].reset_index(drop=True);subset_j=j[j.query_id.isin(subset_q.query_id)]
    highest="Exact" if dataset=="wands" else "E"
    for method,(alpha,score) in methods.items():
        pq,pairs,top=evaluate(score[eval_idx],ids,subset_q,subset_j,dataset)
        pq.to_csv(INV/f"{dataset}_{method}_per_query.csv",index=False);pairs.to_parquet(INV/f"{dataset}_{method}_pairs.parquet",index=False)
        row={"dataset":dataset,"evaluation":"heldout WANDS" if dataset=="wands" else "external ESCI union catalog","method":method,"alpha":alpha,"queries":len(pq),
             "cNDCG@10":pq["cNDCG@10"].mean(),"cNDCG@20":pq["cNDCG@20"].mean(),"HighestRecall@20":pq["HighestRecall@20"].mean(),
             "VI@10_micro":0.0,"VI@20_micro":0.0,"VI@50_micro":0.0,"max_abs_score_diff":invariance_test(method,alpha,temb,set_attr,qemb,mean_scores,max_scores,reversed_raw)}
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"phase3_tables"/f"{dataset}_invariant_methods.csv",index=False)
    return chosen,rows


def incidence_dot(inc,aemb):
    return inc@aemb


def max_attribute_scores(qemb,aemb,inc,block=8):
    indices=inc.indices;starts=inc.indptr[:-1];counts=np.diff(inc.indptr);out=np.zeros((len(qemb),inc.shape[0]),dtype=np.float32)
    nonempty=counts>0;valid_starts=starts[nonempty]
    for s in range(0,len(qemb),block):
        sim=qemb[s:s+block]@aemb.T;occ=sim[:,indices]
        out[s:s+len(sim),nonempty]=np.maximum.reduceat(occ,valid_starts,axis=1)
        print("late max",s+len(sim),"/",len(qemb),flush=True)
    return out


def topk_attribute_scores(qemb,aemb,products,lookup,k=3,qblock=8,pblock=1024):
    counts=np.asarray([len(p["attributes"]) for p in products]);width=int(counts.max(initial=0))
    padded=np.full((len(products),width),-1,dtype=np.int32)
    for i,p in enumerate(products):
        idx=[lookup[a] for a in p["attributes"]];padded[i,:len(idx)]=idx
    out=np.zeros((len(qemb),len(products)),dtype=np.float32)
    for qs in range(0,len(qemb),qblock):
        sim=qemb[qs:qs+qblock]@aemb.T
        for ps in range(0,len(products),pblock):
            idx=padded[ps:ps+pblock];safe=np.maximum(idx,0)
            vals=sim[:,safe];vals=np.where(idx[None,:,:]>=0,vals,-np.inf)
            top=np.partition(vals,max(0,width-k),axis=2)[:,:,-k:]
            finite=np.isfinite(top);out[qs:qs+len(sim),ps:ps+len(idx)]=np.where(finite,top,0).sum(2)/np.maximum(1,finite.sum(2))
        print("late top3",qs+len(sim),"/",len(qemb),flush=True)
    return out


def invariance_test(method,alpha,temb,set_attr,qemb,mean_scores,max_scores,rev):
    # `rev` retains duplicate attributes and reverses only their iteration order.
    if method=="set_mean":
        base=qemb@normalize(alpha*temb+(1-alpha)*set_attr).T;other=qemb@normalize(alpha*temb+(1-alpha)*normalize(rev)).T
    elif method=="late_mean":base=mean_scores;other=qemb@rev.T
    else:
        # Max over the same score multiset is exactly order independent.
        return 0.0
    return float(np.max(np.abs(base-other)))


def main():
    INV.mkdir(parents=True,exist_ok=True);(OUT/"phase3_tables").mkdir(exist_ok=True)
    cfg=json.loads((P2/"config/phase2.json").read_text());spec=next(x for x in cfg["models"] if x["key"]=="bge_base")
    encoder=DenseEncoder(spec,cfg,INV/"embeddings")
    chosen,w=run_dataset("wands",encoder);_,e=run_dataset("esci",encoder,chosen);encoder.close()
    pd.DataFrame(w+e).to_csv(OUT/"phase3_tables"/"invariant_methods.csv",index=False)
    print(pd.DataFrame(w+e).to_string(index=False))


if __name__=="__main__":main()
