from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2];P2=ROOT/"phase2";OUT=ROOT/"phase3"/"results";INV=OUT/"phase3_invariant_method"
sys.path.insert(0,str(P2))
from src.evaluation import paired_stats

BASE=["C0","C1","C2s1","C2s2","C2s3","C2s4","C2s5"]
METHOD_STEMS={"Original":BASE,"M1 factual":["M1"+x for x in BASE],"M2 canonical":["M2"+x for x in BASE]}


def qvi(dataset,stems):
    highest="Exact" if dataset=="wands" else "E";frames=[]
    for s in stems:
        d=pd.read_parquet(P2/"results/phase2_pair_ranks"/f"{dataset}_bge_base_native_{s}.parquet")
        d=d[d.label.eq(highest)][["query_id","product_id","rank"]].rename(columns={"rank":s});frames.append(d)
    w=frames[0]
    for d in frames[1:]:w=w.merge(d,on=["query_id","product_id"],validate="one_to_one")
    ranks=w[stems].to_numpy();w["VI"]=(ranks<=20).any(1)&(ranks>20).any(1)
    return w.groupby("query_id").VI.mean()


def eval_queries(dataset):
    q=pd.read_csv(P2/"data/processed"/f"{dataset}_queries.csv")
    if dataset=="wands":
        order=sorted(range(len(q)),key=lambda i:hashlib.sha256(f"phase3-dev:{q.iloc[i].query_id}".encode()).hexdigest())
        q=q.iloc[order[len(q)//5:]]
    return set(q.query_id)


def main():
    rows=[]
    for dataset in ["wands","esci"]:
        ids=eval_queries(dataset);base_pq=pd.read_csv(P2/"results/phase2_per_query"/f"{dataset}_bge_base_native_C0.csv").set_index("query_id")
        base_pq=base_pq[base_pq.index.isin(ids)];base_vi=qvi(dataset,BASE);base_vi=base_vi[base_vi.index.isin(ids)]
        sources={}
        for name,stems in METHOD_STEMS.items():
            stem=stems[0];pq=pd.read_csv(P2/"results/phase2_per_query"/f"{dataset}_bge_base_native_{stem}.csv").set_index("query_id");pq=pq[pq.index.isin(ids)]
            sources[name]=(pq,qvi(dataset,stems))
        for method in ["set_mean","late_mean","late_max","late_top3"]:
            pq=pd.read_csv(INV/f"{dataset}_{method}_per_query.csv").set_index("query_id");sources[method]=(pq,pd.Series(0.0,index=base_vi.index))
        for method,(pq,vi) in sources.items():
            common=base_pq.index.intersection(pq.index);delta,lo,hi,p=paired_stats(base_pq.loc[common,"cNDCG@10"],pq.loc[common,"cNDCG@10"],20260960,10000)
            vc=base_vi.index.intersection(vi.index);dv,vlo,vhi,vp=paired_stats(base_vi.loc[vc],vi.loc[vc],20260961,10000)
            rows.append({"dataset":dataset,"method":method,"queries":len(common),"cNDCG@10":pq.loc[common,"cNDCG@10"].mean(),"delta_cNDCG@10":delta,"cNDCG_ci_low":lo,"cNDCG_ci_high":hi,"p_cNDCG":p,
                         "query_macro_VI@20":vi.loc[vc].mean(),"delta_query_macro_VI@20":dv,"VI_ci_low":vlo,"VI_ci_high":vhi,"p_VI":vp,
                         "relevance_loss_confirmed":bool(hi<0),"VI_reduced":bool(dv<0),"aligned_success":bool(hi>=0 and dv<0)})
    d=pd.DataFrame(rows);d.to_csv(OUT/"phase3_tables/robustness_relevance_pareto.csv",index=False);print(d.to_string(index=False))


if __name__=="__main__":main()
