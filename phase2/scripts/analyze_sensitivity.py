from __future__ import annotations

import argparse
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
from src.representations import slug


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(p_values) - position) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def load_rank_wide(dataset: str, model: str, profile: str, conditions: list[str]) -> pd.DataFrame:
    frames = []
    for condition in conditions:
        path = P2 / f"results/phase2_pair_ranks/{dataset}_{model}_{profile}_{slug(condition)}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        keep = ["query_id", "product_id", "label", "grade", "query_type", "rank"]
        frame = frame[keep].rename(columns={"rank": slug(condition)})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["query_id", "product_id", "label", "grade", "query_type"],
            how="inner",
            validate="one_to_one",
        )
    if len(merged) != len(frames[0]):
        raise AssertionError("rank files do not cover the same judged pairs")
    return merged


def add_sensitivity(frame: pd.DataFrame, rank_columns: list[str]) -> pd.DataFrame:
    ranks = frame[rank_columns].to_numpy(dtype=np.float64)
    frame = frame.copy()
    frame["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)
    frame["rank_std"] = ranks.std(axis=1)
    frame["reciprocal_rank_variance"] = (1 / ranks).var(axis=1)
    variants = ranks.shape[1]
    for k in [10, 20, 50]:
        inside = (ranks <= k).sum(axis=1)
        frame[f"VI@{k}"] = ((inside > 0) & (inside < variants)).astype(float)
        frame[f"pairwise_crossing@{k}"] = 2 * inside * (variants - inside) / (variants * (variants - 1))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="native")
    parser.add_argument("--datasets", nargs="+", default=["wands", "esci"])
    parser.add_argument("--models", nargs="+", default=["minilm", "bge_base", "gte_modernbert"])
    args = parser.parse_args()
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    conditions = cfg["primary_variant_family"]
    rank_columns = [slug(condition) for condition in conditions]
    tables = P2 / "results/phase2_tables"
    figures = P2 / "results/phase2_figures"
    per_pair_dir = P2 / "results/phase2_per_pair_sensitivity"
    for folder in [tables, figures, per_pair_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    table1, strata, query_types, gates, aggregate, contrasts = [], [], [], [], [], []
    all_highest = []
    for dataset in args.datasets:
        highest_label = "Exact" if dataset == "wands" else "E"
        low_label = "Irrelevant" if dataset == "wands" else "I"
        for model in args.models:
            frame = add_sensitivity(load_rank_wide(dataset, model, args.profile, conditions), rank_columns)
            frame.to_parquet(per_pair_dir / f"{dataset}_{model}_{args.profile}.parquet", index=False)
            highest = frame[frame.label.eq(highest_label)].copy()
            highest["dataset"] = dataset
            highest["model"] = model
            all_highest.append(highest)
            macro = highest.groupby("query_id")[["VI@10", "VI@20", "VI@50"]].mean()
            row = {
                "dataset": dataset,
                "retriever": model,
                "profile": args.profile,
                "highest_pairs": len(highest),
                "rank_range_median": float(highest.rank_range.median()),
                "rank_std_median": float(highest.rank_std.median()),
                "reciprocal_rank_variance_median": float(highest.reciprocal_rank_variance.median()),
            }
            for k in [10, 20, 50]:
                low, high = mean_bootstrap(macro[f"VI@{k}"], cfg["seed"] + k, cfg["bootstrap_samples"])
                row[f"VI@{k}_micro"] = float(highest[f"VI@{k}"].mean())
                row[f"VI@{k}_query_macro"] = float(macro[f"VI@{k}"].mean())
                row[f"VI@{k}_macro_ci_low"] = low
                row[f"VI@{k}_macro_ci_high"] = high
                row[f"pairwise_crossing@{k}_micro"] = float(highest[f"pairwise_crossing@{k}"].mean())
            row["go_cell"] = bool(
                row["VI@20_micro"] >= cfg["go_thresholds"]["minimum_micro_vi20"]
                and row["VI@20_macro_ci_low"] > cfg["go_thresholds"]["minimum_query_macro_vi20_ci_low"]
            )
            table1.append(row)
            for label, group in frame.groupby("label"):
                grouped = group.groupby("query_id")["VI@20"].mean()
                low, high = mean_bootstrap(grouped, cfg["seed"] + 20, cfg["bootstrap_samples"])
                strata.append({
                    "dataset": dataset, "retriever": model, "label": label,
                    "pairs": len(group), "VI@20_micro": float(group["VI@20"].mean()),
                    "VI@20_query_macro": float(grouped.mean()), "ci_low": low, "ci_high": high,
                    "rank_range_median": float(group.rank_range.median()),
                })
            for query_type, group in highest.groupby("query_type"):
                grouped = group.groupby("query_id")["VI@20"].mean()
                query_types.append({
                    "dataset": dataset, "retriever": model, "query_type": query_type,
                    "queries": group.query_id.nunique(), "pairs": len(group),
                    "VI@20_micro": float(group["VI@20"].mean()),
                    "VI@20_query_macro": float(grouped.mean()),
                    "rank_range_median": float(group.rank_range.median()),
                })
            pivot = frame.groupby(["query_id", "label"])["VI@20"].mean().unstack()
            if highest_label in pivot and low_label in pivot:
                valid = pivot[[highest_label, low_label]].dropna()
                if len(valid):
                    delta, low, high, p = paired_stats(
                        valid[low_label], valid[highest_label], cfg["seed"], cfg["permutation_samples"]
                    )
                    contrasts.append({
                        "dataset": dataset, "retriever": model,
                        "contrast": f"{highest_label}-minus-{low_label} query-macro VI@20",
                        "queries": len(valid), "delta": delta, "ci_low": low, "ci_high": high, "p": p,
                    })
            for condition in conditions:
                path = P2 / f"results/phase2_per_query/{dataset}_{model}_{args.profile}_{slug(condition)}.csv"
                per_query = pd.read_csv(path)
                aggregate.append({
                    "dataset": dataset, "retriever": model, "representation": condition,
                    "cNDCG@10": float(per_query["cNDCG@10"].mean()),
                    "RelevantRecall@20": float(per_query["RelevantRecall@20"].mean()),
                    "HighestRecall@20": float(per_query["HighestRecall@20"].mean()),
                    "RelevantHidden@20": float(per_query["RelevantHidden@20"].mean()),
                    "RelevanceVisibilitySpearman": float(per_query["RelevanceVisibilitySpearman"].mean()),
                })
    table1_frame = pd.DataFrame(table1)
    table1_frame.to_csv(tables / f"table1_representation_sensitivity_{args.profile}.csv", index=False)
    pd.DataFrame(strata).to_csv(tables / f"relevance_stratified_{args.profile}.csv", index=False)
    pd.DataFrame(query_types).to_csv(tables / f"table2_query_type_{args.profile}.csv", index=False)
    pd.DataFrame(aggregate).to_csv(tables / f"alignment_by_representation_{args.profile}.csv", index=False)
    contrast_frame = pd.DataFrame(contrasts)
    if len(contrast_frame):
        contrast_frame["holm_within_dataset"] = contrast_frame.groupby("dataset")["p"].transform(
            lambda values: holm(values.tolist())
        )
    contrast_frame.to_csv(tables / f"relevance_stratum_contrasts_{args.profile}.csv", index=False)

    for dataset, group in table1_frame.groupby("dataset"):
        passed = int(group.go_cell.sum())
        gates.append({"dataset": dataset, "models_passing": passed,
                      "required": cfg["go_thresholds"]["required_models"],
                      "pass": passed >= cfg["go_thresholds"]["required_models"]})
    overall = len(gates) == 2 and all(item["pass"] for item in gates)
    gate_payload = {"profile": args.profile, "dataset_gates": gates, "phase2a_pass": overall}
    (tables / f"phase2a_gate_{args.profile}.json").write_text(json.dumps(gate_payload, indent=2))

    high = pd.concat(all_highest, ignore_index=True)
    fig, axes = plt.subplots(
        1, len(args.datasets), figsize=(5 * len(args.datasets), 4), sharey=True, squeeze=False
    )
    axes = axes[0]
    for axis, (dataset, group) in zip(axes, high.groupby("dataset")):
        for model, values in group.groupby("model"):
            ordered = np.sort(values.rank_range.to_numpy())
            axis.plot(ordered, np.linspace(0, 1, len(ordered)), label=model)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Rank range across attribute-order variants")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(figures / f"figure_B_rank_range_{args.profile}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{row.dataset}\n{row.retriever}" for row in table1_frame.itertuples()]
    ax.bar(np.arange(len(table1_frame)), table1_frame["VI@20_micro"])
    ax.axhline(cfg["go_thresholds"]["minimum_micro_vi20"], color="black", linestyle="--", label="3% gate")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Highest-relevance micro VI@20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / f"figure_C_vi20_{args.profile}.png", dpi=180)
    plt.close(fig)

    query_frame = pd.DataFrame(query_types)
    for dataset in args.datasets:
        subset = query_frame[query_frame.dataset.eq(dataset)]
        pivot = subset.pivot(index="query_type", columns="retriever", values="VI@20_query_macro")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=max(0.01, float(pivot.max().max())), cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.iloc[i,j]:.3f}", ha="center", va="center")
        ax.set_title(f"{dataset.upper()} query-macro VI@20")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / f"figure_E_query_type_{dataset}_{args.profile}.png", dpi=180)
        plt.close(fig)
    print(json.dumps(gate_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
