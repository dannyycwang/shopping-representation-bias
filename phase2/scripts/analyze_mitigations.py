from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2))
from src.evaluation import mean_bootstrap, paired_stats
from src.mitigations import BASE_VARIANTS, METHODS, mitigation_slug


ORIGINAL_STEMS = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - position) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def load_wide(dataset: str, model: str, profile: str, stems: list[str]) -> pd.DataFrame:
    frames = []
    keys = ["query_id", "product_id", "label", "grade", "query_type"]
    for stem in stems:
        path = P2 / f"results/phase2_pair_ranks/{dataset}_{model}_{profile}_{stem}.parquet"
        frame = pd.read_parquet(path)[keys + ["rank"]].rename(columns={"rank": stem})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner", validate="one_to_one")
    ranks = merged[stems].to_numpy(dtype=np.float64)
    merged["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)
    merged["rank_std"] = ranks.std(axis=1)
    for k in [10, 20, 50]:
        inside = (ranks <= k).sum(axis=1)
        merged[f"VI@{k}"] = ((inside > 0) & (inside < len(stems))).astype(float)
    return merged


def canonical_query(dataset: str, model: str, profile: str, representation: str) -> pd.DataFrame:
    stem = "C0" if representation == "C0_original" else mitigation_slug(representation, "C0")
    return pd.read_csv(P2 / f"results/phase2_per_query/{dataset}_{model}_{profile}_{stem}.csv")


def product_lookup(dataset: str) -> dict[str, dict]:
    path = P2 / f"data/processed/{dataset}_products.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(item["product_id"]): item for item in map(json.loads, handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="native")
    parser.add_argument("--datasets", nargs="+", default=["wands", "esci"])
    parser.add_argument("--models", nargs="+", default=["minilm", "bge_base", "gte_modernbert"])
    args = parser.parse_args()
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    tables = P2 / "results/phase2_tables"
    figures = P2 / "results/phase2_figures"
    qualitative = P2 / "results/phase2_qualitative"
    for folder in [tables, figures, qualitative]:
        folder.mkdir(parents=True, exist_ok=True)

    alignment_rows, mitigation_rows, sar_rows, direct_rows = [], [], [], []
    sensitivity_by_cell: dict[tuple[str, str, str], pd.DataFrame] = {}
    query_vi_by_cell: dict[tuple[str, str, str], pd.Series] = {}
    representations = ["C0_original", *METHODS]
    for dataset in args.datasets:
        highest_label = "Exact" if dataset == "wands" else "E"
        for model in args.models:
            for representation in representations:
                stems = ORIGINAL_STEMS if representation == "C0_original" else [
                    mitigation_slug(representation, variant) for variant in BASE_VARIANTS
                ]
                sensitivity = load_wide(dataset, model, args.profile, stems)
                highest = sensitivity[sensitivity.label.eq(highest_label)].copy()
                query_vi = highest.groupby("query_id")["VI@20"].mean()
                sensitivity_by_cell[(dataset, model, representation)] = highest
                query_vi_by_cell[(dataset, model, representation)] = query_vi
                per_query = canonical_query(dataset, model, args.profile, representation)
                # SAR uses the same relevance-eligible query population as the
                # reported cNDCG. Queries without a highest-label product have no
                # highest-label VI event and receive a zero instability penalty.
                # The previous inner join silently restricted lambda=0 to the
                # highest-label-eligible subset (379 rather than 480 WANDS queries).
                aligned = per_query.set_index("query_id").join(query_vi.rename("VI@20"), how="left")
                aligned["VI@20"] = aligned["VI@20"].fillna(0.0)
                row = {
                    "dataset": dataset,
                    "representation": representation,
                    "retriever": model,
                    "cNDCG@10": float(per_query["cNDCG@10"].mean()),
                    "RelevantRecall@20": float(per_query["RelevantRecall@20"].mean()),
                    "HighestRecall@20": float(per_query["HighestRecall@20"].mean()),
                    "RelevantHidden@20": float(per_query["RelevantHidden@20"].mean()),
                    "RelevanceVisibilitySpearman": float(per_query["RelevanceVisibilitySpearman"].mean()),
                    "VI@20_micro": float(highest["VI@20"].mean()),
                    "VI@20_query_macro": float(query_vi.mean()),
                    "rank_range_median": float(highest.rank_range.median()),
                }
                low, high = mean_bootstrap(query_vi, cfg["seed"] + 220, cfg["bootstrap_samples"])
                row["VI@20_macro_ci_low"] = low
                row["VI@20_macro_ci_high"] = high
                alignment_rows.append(row)
                for value in cfg["sar_lambdas"]:
                    scores = aligned["cNDCG@10"] - float(value) * aligned["VI@20"]
                    low, high = mean_bootstrap(scores, cfg["seed"] + int(value * 1000), cfg["bootstrap_samples"])
                    sar_rows.append({
                        "dataset": dataset, "retriever": model,
                        "representation": representation, "lambda": value,
                        "queries": int(scores.notna().sum()),
                        "SAR": float(scores.mean()), "ci_low": low, "ci_high": high,
                    })

            base_query = canonical_query(dataset, model, args.profile, "C0_original").set_index("query_id")
            base_vi = query_vi_by_cell[(dataset, model, "C0_original")]
            for method in METHODS:
                method_query = canonical_query(dataset, model, args.profile, method).set_index("query_id")
                method_vi = query_vi_by_cell[(dataset, model, method)]
                common = base_query.index.intersection(method_query.index)
                metrics = [
                    "cNDCG@10", "RelevantRecall@20", "HighestRecall@20",
                    "RelevantHidden@20", "RelevanceVisibilitySpearman",
                ]
                values = {
                    "dataset": dataset, "retriever": model, "representation": method,
                    "queries": len(common),
                }
                for metric in metrics:
                    delta, low, high, p = paired_stats(
                        base_query.loc[common, metric], method_query.loc[common, metric],
                        cfg["seed"] + 310, cfg["permutation_samples"],
                    )
                    key = metric.replace("@", "_at_")
                    values[f"delta_{key}"] = delta
                    values[f"ci_low_{key}"] = low
                    values[f"ci_high_{key}"] = high
                    values[f"p_{key}"] = p
                vi_common = base_vi.index.intersection(method_vi.index)
                delta, low, high, p = paired_stats(
                    base_vi.loc[vi_common], method_vi.loc[vi_common],
                    cfg["seed"] + 320, cfg["permutation_samples"],
                )
                values.update({
                    "delta_VI_at_20": delta, "ci_low_VI_at_20": low,
                    "ci_high_VI_at_20": high, "p_VI_at_20": p,
                    "relevance_not_confirmed_worse": values["ci_high_cNDCG_at_10"] >= 0,
                    "VI_reduced": delta < 0,
                    "VI_reduction_confirmed": high < 0,
                })
                values["mitigation_success"] = bool(
                    values["relevance_not_confirmed_worse"] and values["VI_reduced"]
                )
                mitigation_rows.append(values)

            for method in METHODS:
                path = P2 / (
                    f"results/phase2_single_product/{dataset}_{model}_{args.profile}_"
                    f"{mitigation_slug(method, 'C0')}.parquet"
                )
                direct = pd.read_parquet(path)
                for subset_name, subset in [("all_relevant", direct), ("highest", direct[direct.is_highest])]:
                    direct_rows.append({
                        "dataset": dataset, "retriever": model, "representation": method,
                        "subset": subset_name, "pairs": len(subset),
                        "single_rank_delta_median": float(subset.base_to_single_delta.median()),
                        "full_rank_delta_median": float(subset.base_to_full_delta.median()),
                        "single_improved_fraction": float((subset.base_to_single_delta > 0).mean()),
                        "single_worsened_fraction": float((subset.base_to_single_delta < 0).mean()),
                        "single_crossing@20": float(subset["single_crossing@20"].mean()),
                        "full_crossing@20": float(subset["full_crossing@20"].mean()),
                    })

    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(tables / f"table3_relevance_visibility_alignment_{args.profile}.csv", index=False)
    mitigation = pd.DataFrame(mitigation_rows)
    for metric in ["cNDCG_at_10", "VI_at_20"]:
        mitigation[f"holm_p_{metric}"] = mitigation.groupby(["dataset", "retriever"])[f"p_{metric}"].transform(
            lambda values: holm(values.tolist())
        )
    mitigation.to_csv(tables / f"table4_mitigation_{args.profile}.csv", index=False)
    pd.DataFrame(sar_rows).to_csv(tables / f"stability_adjusted_relevance_{args.profile}.csv", index=False)
    pd.DataFrame(direct_rows).to_csv(tables / f"single_product_summary_{args.profile}.csv", index=False)

    markers = {"C0_original": "o", "M1_factual_field_sentence": "s", "M2_normalized_attributes": "^"}
    colors = {model: plt.cm.tab10(i) for i, model in enumerate(args.models)}
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(6 * len(args.datasets), 4.8), squeeze=False)
    for axis, dataset in zip(axes[0], args.datasets):
        subset = alignment[alignment.dataset.eq(dataset)]
        for _, row in subset.iterrows():
            axis.scatter(
                row["cNDCG@10"], row["HighestRecall@20"],
                s=45 + 1300 * row["VI@20_query_macro"],
                marker=markers[row["representation"]], color=colors[row["retriever"]], alpha=0.8,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Pooled cNDCG@10")
        axis.set_ylabel("Highest-relevance recall@20")
        axis.grid(alpha=0.2)
    handles = [
        plt.Line2D([], [], marker=marker, linestyle="", color="black", label=label)
        for label, marker in markers.items()
    ]
    handles.extend([
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=model)
        for model, color in colors.items()
    ])
    axes[0, -1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Relevance–visibility alignment; marker size is query-macro VI@20")
    fig.tight_layout()
    fig.savefig(figures / f"figure_D_alignment_mitigation_{args.profile}.png", dpi=180)
    plt.close(fig)

    # Deterministic extreme-case selection for illustration only.
    example_rows = []
    for dataset in args.datasets:
        queries = pd.read_csv(P2 / f"data/processed/{dataset}_queries.csv").set_index("query_id")
        products = product_lookup(dataset)
        candidates = []
        for model in args.models:
            for method in METHODS:
                direct = pd.read_parquet(
                    P2 / (
                        f"results/phase2_single_product/{dataset}_{model}_{args.profile}_"
                        f"{mitigation_slug(method, 'C0')}.parquet"
                    )
                )
                direct = direct[direct.is_highest & (direct.base_rank > 20) & (direct.single_product_rank <= 20)]
                if len(direct):
                    direct = direct.assign(retriever=model, representation=method)
                    candidates.append(direct)
        if not candidates:
            continue
        candidates = pd.concat(candidates, ignore_index=True).sort_values(
            ["base_to_single_delta", "query_id", "product_id"], ascending=[False, True, True]
        )
        selected = candidates.iloc[0]
        record = products[str(selected.product_id)]
        example_rows.append({
            "dataset": dataset, "retriever": selected.retriever,
            "representation": selected.representation,
            "query_id": int(selected.query_id), "query": queries.loc[int(selected.query_id), "query"],
            "product_id": selected.product_id, "product_title": record["title"],
            "base_rank": int(selected.base_rank),
            "single_product_rank": int(selected.single_product_rank),
            "full_catalog_rank": int(selected.full_catalog_rank),
            "source_description": record["description"],
            "source_attributes": " | ".join(record["attributes"]),
        })
    examples = pd.DataFrame(example_rows)
    examples.to_csv(qualitative / "strong_examples.csv", index=False)
    if len(examples):
        figure_rows = []
        for row in examples.itertuples(index=False):
            for intervention, rank in [
                ("C0", row.base_rank), ("single product", row.single_product_rank),
                ("full catalog", row.full_catalog_rank),
            ]:
                figure_rows.append({"case": f"{row.dataset.upper()}\n{row.query}", "intervention": intervention, "rank": rank})
        figure_data = pd.DataFrame(figure_rows)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(examples))
        width = 0.24
        for offset, intervention in enumerate(["C0", "single product", "full catalog"]):
            values = figure_data[figure_data.intervention.eq(intervention)].set_index("case")["rank"]
            ax.bar(x + (offset - 1) * width, values.to_numpy(), width, label=intervention)
        ax.set_yscale("log")
        ax.set_xticks(x, [f"{row.dataset.upper()}\n{row.query}" for row in examples.itertuples()])
        ax.set_ylabel("Rank (log scale; lower is better)")
        ax.legend()
        ax.set_title("Same judged product and facts, different representation")
        fig.tight_layout()
        fig.savefig(figures / f"figure_A_same_product_{args.profile}.png", dpi=180)
        plt.close(fig)
        lines = ["# Deterministically selected qualitative examples", "", "Selected only for illustration; query-paired results provide the evidence.", ""]
        for row in examples.itertuples(index=False):
            lines.extend([
                f"## {row.dataset.upper()}: {row.query}", "",
                f"- Product: {row.product_title} (`{row.product_id}`)",
                f"- Retriever / intervention: {row.retriever} / {row.representation}",
                f"- Ranks: C0 {row.base_rank}; single-product {row.single_product_rank}; full-catalog {row.full_catalog_rank}",
                f"- Source attributes: {row.source_attributes}", "",
            ])
        (qualitative / "strong_examples.md").write_text("\n".join(lines), encoding="utf-8")

    print(alignment.to_string(index=False), flush=True)
    print(mitigation[["dataset", "retriever", "representation", "delta_cNDCG_at_10",
                      "delta_VI_at_20", "mitigation_success"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
