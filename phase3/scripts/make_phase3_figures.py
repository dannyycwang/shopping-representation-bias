from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[2];P2=ROOT/"phase2";R=ROOT/"phase3/results";OUT=R/"phase3_figures";OUT.mkdir(parents=True,exist_ok=True)
COLORS={"minilm":"#0072B2","bge_base":"#D55E00","gte_modernbert":"#009E73","marqo_ecommerce_b":"#CC79A7"}
LABELS={"minilm":"MiniLM","bge_base":"BGE-base","gte_modernbert":"GTE-ModernBERT","marqo_ecommerce_b":"Marqo-Ecommerce-B"}


def style():
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":8,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.dpi":300})


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight",facecolor="white");fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",facecolor="white");plt.close(fig)


def sensitivity():
    d=pd.read_csv(P2/"results/phase2_tables/table1_representation_sensitivity_native.csv")
    m=pd.read_csv(R/"phase3_tables/ecommerce_retriever_sensitivity.csv");d=pd.concat([d,m],ignore_index=True,sort=False)
    d.to_csv(OUT/"figure2_source.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
    models=list(LABELS);y=np.arange(4)
    for ax,dataset,title in zip(axes,["wands","esci"],["WANDS","ESCI union catalog"]):
        s=d[d.dataset.eq(dataset)].set_index("retriever").loc[models]
        x=100*s["VI@20_query_macro"];lo=100*s["VI@20_macro_ci_low"];hi=100*s["VI@20_macro_ci_high"]
        for i,model in enumerate(models):ax.errorbar(x.iloc[i],i,xerr=[[x.iloc[i]-lo.iloc[i]],[hi.iloc[i]-x.iloc[i]]],fmt="o",ms=6,capsize=2.5,color=COLORS[model])
        ax.axvline(1,color="#999",ls="--",lw=.8);ax.grid(axis="x",color="#ddd",lw=.6);ax.set_title(title,loc="left",fontweight="bold");ax.set_xlabel("Query-macro VI@20 (%)")
    axes[0].set_yticks(y,[LABELS[x] for x in models]);axes[0].invert_yaxis();fig.suptitle("Fact-equivalent serialization changes visibility across retriever families",fontweight="bold",y=1.02);fig.tight_layout();save(fig,"figure2_cross_model_vi")


def pipeline():
    rows=[pd.read_csv(R/"phase3_tables"/f"{ds}_reranker_pipeline.csv").iloc[0] for ds in ["wands","esci"]]
    pd.DataFrame(rows).to_csv(OUT/"figure3_source.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0));x=np.arange(2);width=.31
    dense=[100*r["common_query_macro_dense_VI@20"] for r in rows];rer=[100*r["common_query_macro_rerank_VI@20"] for r in rows]
    axes[0].bar(x-width/2,dense,width,label="Dense Top-100 ranks",color="#0072B2");axes[0].bar(x+width/2,rer,width,label="After reranking",color="#E69F00")
    axes[0].set_xticks(x,["WANDS","ESCI"]);axes[0].set_ylabel("Query-macro VI@20 (%)");axes[0].set_title("(a) Common Top-100 products",loc="left",fontweight="bold");axes[0].legend(frameon=False);axes[0].grid(axis="y",color="#ddd",lw=.6)
    val=[100*r["Irrecoverable@100_query_macro"] for r in rows];lo=[100*r["Irrecoverable@100_macro_ci_low"] for r in rows];hi=[100*r["Irrecoverable@100_macro_ci_high"] for r in rows]
    axes[1].bar(x,val,color=["#CC3311","#EE7733"],width=.5);axes[1].errorbar(x,val,yerr=[np.array(val)-lo,np.array(hi)-val],fmt="none",ecolor="#222",capsize=3)
    axes[1].set_xticks(x,["WANDS","ESCI"]);axes[1].set_ylabel("Query-macro Irrecoverable@100 (%)");axes[1].set_title("(b) Candidate loss",loc="left",fontweight="bold");axes[1].grid(axis="y",color="#ddd",lw=.6)
    fig.suptitle("Reranking reduces instability but cannot recover missing candidates",fontweight="bold",y=1.02);fig.tight_layout();save(fig,"figure3_reranker_pipeline")


def pareto():
    d=pd.read_csv(R/"phase3_tables/robustness_relevance_pareto.csv");d.to_csv(OUT/"figure4_source.csv",index=False)
    # Keep the main-paper panel readable. The source CSV retains every ablation;
    # late-interaction variants are reported in the appendix table.
    main_methods=["Original","M1 factual","M2 canonical","set_mean"]
    d=d[d.method.isin(main_methods)]
    names={"Original":"Original","M1 factual":"M1 factual","M2 canonical":"M2 canonical","set_mean":"Set mean (ours)"}
    markers={"Original":"o","M1 factual":"s","M2 canonical":"^","set_mean":"*"}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2))
    for ax,dataset,title in zip(axes,["wands","esci"],["WANDS held-out","ESCI union catalog"]):
        s=d[d.dataset.eq(dataset)]
        for _,r in s.iterrows():
            ax.scatter(r["cNDCG@10"],100*r["query_macro_VI@20"],marker=markers[r.method],s=95 if r.method=="set_mean" else 45,
                       color="#009E73" if r.method=="set_mean" else "#666",edgecolor="black" if r.method=="set_mean" else "none",label=names[r.method])
            offset=(4,-13) if r.method=="set_mean" else (4,4)
            ax.annotate(names[r.method],(r["cNDCG@10"],100*r["query_macro_VI@20"]),xytext=offset,textcoords="offset points",fontsize=7)
        ax.set_xlabel("cNDCG@10 (higher is better)");ax.set_ylabel("Query-macro VI@20 (%) (lower is better)");ax.set_title(title,loc="left",fontweight="bold");ax.grid(color="#ddd",lw=.5)
    fig.suptitle("Permutation invariance exposes the robustness–relevance frontier",fontweight="bold",y=1.02);fig.tight_layout();save(fig,"figure4_pareto")


if __name__=="__main__":style();sensitivity();pipeline();pareto();print("Phase III figures generated")
