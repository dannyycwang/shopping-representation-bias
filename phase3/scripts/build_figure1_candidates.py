from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]; P2=ROOT/"phase2"; OUT=ROOT/"phase3"/"results"
MODELS=["minilm","bge_base","gte_modernbert"]
STEMS=["C0","C1","C2s1","C2s2","C2s3","C2s4","C2s5"]


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"phase3_figures").mkdir(exist_ok=True)
    queries=pd.read_csv(P2/"data/processed/wands_queries.csv").set_index("query_id")
    with gzip.open(P2/"data/processed/wands_products.jsonl.gz","rt",encoding="utf-8") as f:
        products={int(x["product_id"]):x for x in map(json.loads,f)}
    cells=[]
    for model in MODELS:
        frames=[]
        for stem in STEMS:
            d=pd.read_parquet(P2/"results/phase2_pair_ranks"/f"wands_{model}_native_{stem}.parquet")
            d=d[d.label.eq("Exact")][["query_id","product_id","rank"]].rename(columns={"rank":stem})
            frames.append(d)
        w=frames[0]
        for d in frames[1:]: w=w.merge(d,on=["query_id","product_id"],validate="one_to_one")
        w["retriever"]=model; w["rank_min"]=w[STEMS].min(1); w["rank_max"]=w[STEMS].max(1)
        w["rank_range"]=w.rank_max-w.rank_min; w["crossing20"]=(w.rank_min<=20)&(w.rank_max>20)
        cells.append(w[w.crossing20])
    allc=pd.concat(cells,ignore_index=True)
    agg=allc.groupby(["query_id","product_id"]).agg(models_crossing=("retriever","nunique"),
        largest_range=("rank_range","max"),best_rank=("rank_min","min"),worst_rank=("rank_max","max")).reset_index()
    agg["query"]=agg.query_id.map(queries["query"]); agg["query_words"]=agg["query"].str.split().str.len()
    agg["product_title"]=agg.product_id.map(lambda x:products[int(x)]["title"])
    details=allc.pivot(index=["query_id","product_id"],columns="retriever",values=STEMS)
    details.columns=[f"{m}_{s}" for s,m in details.columns]; details=details.reset_index()
    result=agg.merge(details,on=["query_id","product_id"])
    result=result.sort_values(["models_crossing","query_words","largest_range"],ascending=[False,False,False])
    preferred=pd.DataFrame({"query_id":[162,181],"product_id":[34536,7756]})
    result["specified_case"]=result.set_index(["query_id","product_id"]).index.isin(preferred.set_index(["query_id","product_id"]).index)
    top=pd.concat([result[result.specified_case],result[(result.query_words>=2)&~result.specified_case]]).drop_duplicates(["query_id","product_id"]).head(20)
    top.to_csv(ROOT/"figure1_candidates.csv",index=False)

    chosen=result[(result.query_id==162)&(result.product_id==34536)].iloc[0]
    fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.axis("off")
    ax.text(.5,.93,'Query: “turquoise chair”',ha='center',fontsize=15,fontweight='bold')
    ax.text(.5,.80,'Same product · Human relevance: Exact',ha='center',fontsize=11)
    ax.text(.25,.63,'Representation A',ha='center',fontsize=12,fontweight='bold',color='#0072B2')
    ax.text(.75,.63,'Representation B',ha='center',fontsize=12,fontweight='bold',color='#D55E00')
    ax.text(.25,.51,'Random attribute order s4\nSame facts and attributes',ha='center',fontsize=10)
    ax.text(.75,.51,'Random attribute order s1\nSame facts and attributes',ha='center',fontsize=10)
    ax.annotate('',xy=(.43,.50),xytext=(.57,.50),arrowprops=dict(arrowstyle='<->',color='#777'))
    ax.text(.5,.43,'Only attribute serialization changed',ha='center',fontsize=9,color='#555')
    ax.text(.25,.27,'Rank #3',ha='center',fontsize=22,fontweight='bold',color='#0072B2')
    ax.text(.75,.27,'Rank #1303',ha='center',fontsize=22,fontweight='bold',color='#D55E00')
    ax.text(.25,.15,'VISIBLE at Top-20',ha='center',fontsize=11,fontweight='bold',color='#0072B2')
    ax.text(.75,.15,'HIDDEN at Top-20',ha='center',fontsize=11,fontweight='bold',color='#D55E00')
    ax.text(.5,.03,'Same product · Same facts · Same query · Same MiniLM model',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"phase3_figures/figure1_same_product.png",dpi=300,bbox_inches='tight')
    fig.savefig(OUT/"phase3_figures/figure1_same_product.pdf",bbox_inches='tight'); plt.close(fig)

    lines=["# Figure 1 Candidate Audit","", "All candidates are WANDS Exact pairs, cross Top-20 under the frozen seven-member primary family, and retain the same fact/token multiset.","",
           "## Verified specified cases","", "- `turquoise chair`, product 34536: MiniLM ranks C0=486, C1=295, C2s1=1303, C2s2=21, C2s3=484, C2s4=3, C2s5=4. BGE is rank 1 for every variant; GTE ranges 1–6. The 3↔1303 illustration is verified.",
           "- `e12/candelabra`, product 7756: MiniLM C0=14 and C1=1881; the simple original-versus-reverse contrast is verified. Other models do not cross Top-20.","",
           "## Selection rule","", "Candidates are ordered by number of models showing a Top-20 crossing, then query length and largest rank range. The two prespecified examples are retained at the top for explicit audit. Full ranks and metadata are in `figure1_candidates.csv`.","",
           "## Recommendation","", "Use `turquoise chair` as Figure 1 because the fact-equivalent MiniLM swing is extreme and the same product remains highly ranked and stable under BGE/GTE, making model dependence visible. Use `e12/candelabra` as an appendix case because it uses the especially simple original-versus-reverse intervention."]
    (ROOT/"FIGURE1_CANDIDATES.md").write_text("\n".join(lines),encoding="utf-8")
    print(top[["query","product_id","models_crossing","largest_range","best_rank","worst_rank"]].to_string(index=False))


if __name__=="__main__": main()
