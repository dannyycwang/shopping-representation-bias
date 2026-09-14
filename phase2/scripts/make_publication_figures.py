from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


P2 = Path(__file__).resolve().parents[1]
TABLES = P2 / "results" / "phase2_tables"
OUT = P2 / "results" / "publication_figures"
DATA = OUT / "plotting_data"

MODELS = {"minilm": "MiniLM", "bge_base": "BGE-base", "gte_modernbert": "GTE-ModernBERT"}
DATASETS = {"wands": "WANDS", "esci": "ESCI"}
METHODS = {
    "M1_factual_field_sentence": "M1: factual sentence",
    "M2_normalized_attributes": "M2: canonical attributes",
}
COLORS = {"minilm": "#0072B2", "bge_base": "#D55E00", "gte_modernbert": "#009E73"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.dpi": 160, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_sensitivity() -> None:
    d = pd.read_csv(TABLES / "table1_representation_sensitivity_native.csv")
    d.to_csv(DATA / "figure_1_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(MODELS))
    offsets = {"wands": -0.12, "esci": 0.12}
    markers = {"wands": "o", "esci": "s"}
    for dataset in DATASETS:
        s = d[d.dataset.eq(dataset)].set_index("retriever").loc[list(MODELS)]
        yy = y + offsets[dataset]
        axes[0].errorbar(
            100 * s["VI@20_query_macro"], yy,
            xerr=np.vstack([100 * (s["VI@20_query_macro"] - s["VI@20_macro_ci_low"]),
                            100 * (s["VI@20_macro_ci_high"] - s["VI@20_query_macro"])]),
            fmt=markers[dataset], ms=5, capsize=2.5, lw=1.2, color="#222222",
            mfc="#FFFFFF" if dataset == "esci" else "#222222", label=DATASETS[dataset], zorder=3)
        axes[1].scatter(s["rank_range_median"], yy, marker=markers[dataset], s=28,
                        c="#222222" if dataset == "wands" else "white",
                        edgecolors="#222222", linewidths=1, label=DATASETS[dataset], zorder=3)
    for ax in axes:
        ax.set_yticks(y, [MODELS[m] for m in MODELS])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#DDDDDD", linewidth=.6, zorder=0)
    axes[0].set_xlabel("Query-macro VI@20 (%)")
    axes[0].set_title("(a) Visibility instability", loc="left", fontweight="bold")
    axes[0].axvline(1, color="#999999", linestyle="--", linewidth=.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Median rank range (log scale)")
    axes[1].set_title("(b) Rank displacement", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Equivalent product records produce model- and dataset-dependent visibility", y=1.02, fontweight="bold")
    fig.tight_layout()
    save(fig, "figure_1_sensitivity")


def fig2_controls() -> None:
    d = pd.read_csv(TABLES / "chunk_pooling_controls.csv")
    profiles = ["native", "mean_126_o32", "mean_254_o0", "mean_254_o64", "cls_254_o0"]
    labels = ["Native", "Mean 126/32", "Mean 254/0", "Mean 254/64", "CLS 254/0"]
    d.to_csv(DATA / "figure_2_controls.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(profiles))
    for model in ["minilm", "bge_base"]:
        s = d[d.retriever.eq(model)].set_index("profile").loc[profiles]
        axes[0].plot(x, 100 * s["VI@20_micro"], marker="o", lw=1.6, color=COLORS[model], label=MODELS[model])
        axes[1].plot(x, s["rank_range_median"], marker="o", lw=1.6, color=COLORS[model], label=MODELS[model])
    for ax in axes:
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
        ax.axvspan(-.35, .35, color="#EEEEEE", zorder=-1)
    axes[0].set_ylabel("Micro VI@20 (%)")
    axes[0].set_title("(a) Threshold instability", loc="left", fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median rank range (log scale)")
    axes[1].set_title("(b) Rank displacement", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("Chunking and pooling do not remove the WANDS effect", y=1.02, fontweight="bold")
    fig.tight_layout()
    save(fig, "figure_2_controls")


def fig3_mitigation() -> None:
    d = pd.read_csv(TABLES / "table4_mitigation_native.csv")
    d.to_csv(DATA / "figure_3_mitigation.csv", index=False)
    rows = [(ds, m, rep) for ds in DATASETS for m in MODELS for rep in METHODS]
    d = d.set_index(["dataset", "retriever", "representation"]).loc[rows].reset_index()
    y = np.arange(len(d))
    labels = [f"{DATASETS[r.dataset]} · {MODELS[r.retriever]} · {METHODS[r.representation].split(':')[0]}" for r in d.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.55), sharey=True, gridspec_kw={"wspace": .08})
    for i, r in enumerate(d.itertuples()):
        color = "#009E73" if r.mitigation_success else "#CC3311"
        axes[0].errorbar(r.delta_cNDCG_at_10, i,
                         xerr=[[r.delta_cNDCG_at_10-r.ci_low_cNDCG_at_10], [r.ci_high_cNDCG_at_10-r.delta_cNDCG_at_10]],
                         fmt="o", color=color, ms=4, capsize=2, lw=1)
        axes[1].errorbar(100*r.delta_VI_at_20, i,
                         xerr=[[100*(r.delta_VI_at_20-r.ci_low_VI_at_20)], [100*(r.ci_high_VI_at_20-r.delta_VI_at_20)]],
                         fmt="o", color=color, ms=4, capsize=2, lw=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color="#555555", lw=.8)
        ax.grid(axis="x", color="#E0E0E0", lw=.6)
    axes[0].set_xlabel("Δ cNDCG@10")
    axes[0].set_title("(a) Relevance", loc="left", fontweight="bold")
    axes[1].set_xlabel("Δ query-macro VI@20 (pp)")
    axes[1].set_title("(b) Instability", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#009E73',lw=0,label='Protocol success'),
                        Line2D([0],[0],marker='o',color='#CC3311',lw=0,label='Failed alignment')],
               frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.56,.925))
    fig.suptitle("Stability gains must be read together with relevance", y=.995, fontweight="bold")
    fig.subplots_adjust(top=.84, left=.32, right=.98, bottom=.13, wspace=.08)
    save(fig, "figure_3_mitigation_tradeoff")


def fig4_direct_full() -> None:
    d = pd.read_csv(TABLES / "single_product_summary_native.csv")
    d = d[d.subset.eq("highest")].copy()
    d.to_csv(DATA / "figure_4_direct_full.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(6)
    for ax, dataset in zip(axes, DATASETS):
        s = d[d.dataset.eq(dataset)]
        labels=[]
        for i, (model, rep) in enumerate([(m, r) for m in MODELS for r in METHODS]):
            row=s[(s.retriever.eq(model)) & (s.representation.eq(rep))].iloc[0]
            a,b=100*row["single_crossing@20"],100*row["full_crossing@20"]
            ax.plot([a,b],[i,i],color="#999999",lw=1.2,zorder=1)
            ax.scatter(a,i,c="#0072B2",marker="o",s=28,zorder=2)
            ax.scatter(b,i,c="#D55E00",marker="s",s=28,zorder=2)
            labels.append(f"{MODELS[model]} · {METHODS[rep].split(':')[0]}")
        ax.set_yticks(y,labels)
        ax.invert_yaxis(); ax.grid(axis="x",color="#E0E0E0",lw=.6)
        ax.set_xlabel("Crossing@20 (%)")
        ax.set_title(DATASETS[dataset],loc="left",fontweight="bold")
    axes[1].scatter([],[],c="#0072B2",marker="o",label="Direct target intervention")
    axes[1].scatter([],[],c="#D55E00",marker="s",label="Full-catalog transformation")
    axes[1].legend(frameon=False,loc="lower right",fontsize=7.5)
    fig.suptitle("Direct and catalog-wide interventions answer different causal questions", y=1.02, fontweight="bold")
    fig.tight_layout()
    save(fig, "figure_4_direct_vs_full")


def fig5_query_types() -> None:
    d = pd.read_csv(TABLES / "table2_query_type_native.csv")
    d.to_csv(DATA / "figure_5_query_types.csv", index=False)
    qorder = ["product_type", "attribute_constraint", "brand_entity_proxy", "multi_constraint", "long_descriptive", "style"]
    qlabels = ["Product type", "Attribute constraint", "Brand/entity", "Multi-constraint", "Long/descriptive", "Style"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for ax,dataset in zip(axes,DATASETS):
        s=d[d.dataset.eq(dataset)].pivot(index="query_type",columns="retriever",values="VI@20_query_macro").reindex(qorder)
        arr=100*s[list(MODELS)].to_numpy()
        im=ax.imshow(arr,aspect="auto",cmap="YlOrRd",vmin=0,vmax=max(30,np.nanmax(arr)))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:.1f}",ha="center",va="center",fontsize=7,color="white" if arr[i,j]>20 else "#222")
        ax.set_xticks(range(3),[MODELS[m] for m in MODELS],rotation=25,ha="right")
        ax.set_yticks(range(6),qlabels)
        ax.set_title(DATASETS[dataset],loc="left",fontweight="bold")
    cbar=fig.colorbar(im, ax=axes.tolist(), fraction=.028, pad=.03)
    cbar.set_label("Query-macro VI@20 (%)")
    fig.suptitle("Instability extends beyond brand and style queries",y=1.01,fontweight="bold")
    fig.tight_layout(rect=[0,0,.94,1])
    save(fig,"figure_5_query_types")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True); style()
    fig1_sensitivity(); fig2_controls(); fig3_mitigation(); fig4_direct_full(); fig5_query_types()
    print(f"Wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
