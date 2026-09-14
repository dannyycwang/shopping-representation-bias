"""Analyze locked VI-B artifacts without retraining or modifying prior phases."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import ROOT, HERE, OUT, CONFIG, STEMS, prior, sha, dump, now, validate, FAMILIES

BASELINES = ["Original", "Canonical", "Set-Mean", "Set-Attention", "Phase-VI DeepSets", "Phase-VI Set Transformer"]
MEASURES = ["Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100", "worst_schedule@100"]


def prior_frame(key, split, mode, old_selection):
    family = mode.split("_")[0]
    if key.startswith("Phase-VI"):
        method = "S4_s42" if key.endswith("DeepSets") else "S1_s42"
        if split == "dev":
            return pd.read_parquet(ROOT / "phase6/checkpoints" / old_selection["selected"][method]["name"] / "dev_pairs.parquet")
        # Invariant prior methods have one C0 index: own-C0 target inclusion equals catalog inclusion.
        return pd.read_parquet(ROOT / "phase6/results" / method / "catalog_pairs.parquet")
    name = "Set-Attention_s42" if key == "Set-Attention" else key
    return pd.read_parquet(ROOT / "phase5_mitigation/results" / name / f"{family}_pairs.parquet")


def main():
    cfg = validate()
    lock = json.loads((HERE / "selection.json").read_text())
    finished = json.loads((HERE / "evaluation_complete.json").read_text())
    assert finished["selection_sha256"] == sha(HERE / "selection.json")
    old = json.loads((ROOT / "phase6/selection.json").read_text())
    audit = json.loads((HERE / "PHASE6B_TRAINING_AUDIT.json").read_text())
    features = pd.read_csv(ROOT / "phase4/results/wands_product_features.csv", dtype={"product_id": str})
    assert features.fully_fits.dtype == bool
    fitids = set(features.loc[features.fully_fits, "product_id"])
    keys = BASELINES + list(lock["selected"])
    rows, frames, pairframes = [], {}, {}
    for split in ["dev", "test"]:
        for mode in ["catalog", "catalog_fully_fitting", "target", "target_fully_fitting"]:
            if split == "dev" and mode != "catalog":
                continue
            if mode.startswith("target") and not lock["target_keys"]:
                continue
            for key in keys:
                baseline = key in BASELINES
                if baseline:
                    f = prior_frame(key, split, mode, old)
                    n, neg, epoch, params, residual_base = 0, 0., 0, 0, "none"
                    if key == "Set-Attention":
                        n, neg, epoch, params = 546, 1., 1, 98561
                    elif key.startswith("Phase-VI"):
                        x = old["selected"]["S4_s42" if key.endswith("DeepSets") else "S1_s42"]
                        n, neg, epoch, params = 546, 1., x["best_epoch"], x["parameters"]
                    metadata = dict(family=None, regime="historical", seed=42 if n else None, lr=None)
                else:
                    selected = lock["selected"][key]
                    if mode.startswith("target") and key not in lock["target_keys"]:
                        continue
                    path = (HERE / "checkpoints" / selected["name"] / "dev_pairs.parquet") if split == "dev" else OUT / key / f"{mode.split('_')[0]}_pairs.parquet"
                    f = pd.read_parquet(path)
                    n, neg, epoch, params, residual_base = selected["examples"], selected["mean_negatives"], selected["best_epoch"], selected["parameters"], "frozen Set-Attention"
                    metadata = {k: selected[k] for k in ["family", "regime", "seed", "lr"]}
                f["product_id"] = f.product_id.astype(str)
                split_ids = cfg["validation"] if split == "dev" else cfg["test"]
                f = f[f.query_id.isin(split_ids)].reset_index(drop=True)
                if "fully_fitting" in mode:
                    f = f[f.product_id.isin(fitids)].reset_index(drop=True)
                _, pq = prior.metric_frame(f)
                if split == "test":
                    assert (len(f), len(pq)) == ((6797, 239) if "fully_fitting" in mode else (21299, 308))
                else:
                    assert len(pq) == 17
                frames[split, mode, key], pairframes[split, mode, key] = pq, f
                measures = {m: 100 * float(pq[m].mean()) for m in MEASURES}
                if mode.startswith("target"):
                    measures = {m.replace("Recall", "Inclusion"): v for m, v in measures.items()}
                rows.append(dict(Method=key, Split=split, Mode=mode, Residual_Base=residual_base, Training_Examples=n,
                                 Negatives_per_example=neg, Epoch=epoch, Params=params, **metadata,
                                 Pairs=len(f), Queries=len(pq), **measures,
                                 minimum_macro_schedule100=100 * float(pq[[f"{s}_Recall@100" for s in STEMS]].mean().min()),
                                 units="percent; bootstrap deltas in percentage points"))
    results = pd.DataFrame(rows)
    results.to_csv(HERE / "PHASE6B_RESULTS.csv", index=False)
    scale_rows = []
    for family in FAMILIES:
        for fraction in [.25, .5, 1.]:
            key = f"{family}_T2_s42" if fraction == 1 else f"{family}_T2_f{fraction:g}_s42"
            candidate = lock["selected"][key]
            curve = pd.read_csv(HERE / "checkpoints" / candidate["name"] / "epochs.csv")
            trained = curve[curve.epoch > 0].sort_values(["Recall@100", "epoch"], ascending=[False, True]).iloc[0]
            for split in ["dev", "test"]:
                row = results[(results.Method == key) & (results.Split == split) & (results.Mode == "catalog")].iloc[0]
                scale_rows.append(dict(Method=FAMILIES[family], Key=key, Split=split, Training_Fraction=fraction,
                                       Examples=int(row.Training_Examples), Epoch=int(row.Epoch), Learning_Rate=row.lr,
                                       Best_Trained_Dev_R100=100 * float(trained["Recall@100"]), Best_Trained_Dev_Epoch=int(trained.epoch),
                                       **{m: row[m] for m in ["Recall@100", "Robust@100", "Never@100"]}))
    scale = pd.DataFrame(scale_rows)
    scale.to_csv(HERE / "PHASE6B_SCALE.csv", index=False)
    comparisons = []
    # Primary selected winners and their confirmation seeds, decided before trained test evaluation.
    primary_keys = list(lock["family_winners"].values()) + [k for k, v in lock["selected"].items() if v["seed"] != 42]
    for split in ["dev", "test"]:
        for mode in ["catalog", "catalog_fully_fitting", "target", "target_fully_fitting"]:
            if (split, mode, "Original") not in frames:
                continue
            for key in primary_keys:
                if (split, mode, key) not in frames:
                    continue
                family = lock["selected"][key]["family"]
                reference_keys = BASELINES[:4] + ["Phase-VI DeepSets" if family == "D" else "Phase-VI Set Transformer"]
                for reference in reference_keys:
                    for metric in ["Recall@100", "Robust@100", "Never@100"]:
                        append_ci(comparisons, frames, split, mode, key, reference, metric)
            if mode.startswith("target"):
                continue
            for family in FAMILIES:
                for key, ref in [(f"{family}_T1_s42", f"{family}_T0_s42"), (f"{family}_T2_s42", f"{family}_T1_s42"),
                                 (f"{family}_T2_f0.5_s42", f"{family}_T2_f0.25_s42"), (f"{family}_T2_s42", f"{family}_T2_f0.5_s42")]:
                    append_ci(comparisons, frames, split, mode, key, ref, "Recall@100")
            # Seed-mean is mean performance across trained models, NOT an ensemble product vector.
            for family, winner in lock["family_winners"].items():
                seed_keys = [winner] + [k for k in primary_keys if k.startswith(family + "_") and lock["selected"][k]["seed"] != 42]
                if len(seed_keys) < 2:
                    continue
                mean_key = family + "_primary_seed_mean"
                frames[split, mode, mean_key] = sum(frames[split, mode, k] for k in seed_keys) / len(seed_keys)
                reference_keys = BASELINES[:4] + ["Phase-VI DeepSets" if family == "D" else "Phase-VI Set Transformer"]
                for reference in reference_keys:
                    for metric in ["Recall@100", "Robust@100", "Never@100"]:
                        append_ci(comparisons, frames, split, mode, mean_key, reference, metric)
    ci = pd.DataFrame(comparisons)
    ci.to_csv(HERE / "PHASE6B_BOOTSTRAP.csv", index=False)
    seed_rows = []
    for split in ["dev", "test"]:
        for mode in ["catalog", "catalog_fully_fitting"]:
            for family, winner in lock["family_winners"].items():
                members = [winner] + [k for k in primary_keys if k.startswith(family + "_") and lock["selected"][k]["seed"] != 42]
                z = results[(results.Method.isin(members)) & (results.Split == split) & (results.Mode == mode)]
                if z.empty:
                    continue
                seed_rows.append(dict(Family=family, Split=split, Mode=mode, N=len(z),
                                      **{m + suffix: float(getattr(z[m], op)()) for m in ["Recall@100", "Robust@100", "Never@100"]
                                         for suffix, op in [("_mean", "mean"), ("_std", "std"), ("_min", "min"), ("_max", "max")]}))
    pd.DataFrame(seed_rows).to_csv(HERE / "seed_summary.csv", index=False)
    primary_table_keys = ["Set-Attention", "Phase-VI DeepSets", "D_T0_s42", "D_T1_s42", "D_T2_s42",
                          "Phase-VI Set Transformer", "S_T0_s42", "S_T1_s42", "S_T2_s42"]
    table = results[(results.Split == "test") & (results.Mode == "catalog")].set_index("Method").loc[primary_table_keys].reset_index()
    table[["Method", "Residual_Base", "Training_Examples", "Negatives_per_example", "Epoch", "Recall@20", "Recall@100", "VI@20", "Robust@100", "Never@100"]].to_csv(HERE / "TABLE_VI_B1.csv", index=False)
    costs = pd.DataFrame([json.loads(line) for line in (HERE / "cost.jsonl").read_text().splitlines()])
    costs.groupby(["name", "stage"]).seconds.sum().reset_index().to_csv(HERE / "cost_summary.csv", index=False)
    plot_scale(scale, cfg)
    # Complete curves from every candidate, including unselected LRs and epoch0.
    all_curves = []
    for candidate in lock["candidates"]:
        x = pd.read_csv(HERE / "checkpoints" / candidate["name"] / "epochs.csv")
        # Raw epoch CSVs used FP32 column reductions. Report the actual global alpha
        # from the step log (not the slightly rounded mean of 42,994 equal values).
        steps = pd.read_csv(HERE / "checkpoints" / candidate["name"] / "steps.csv")
        actual_alpha = steps.groupby("epoch").alpha.last().to_dict()
        x["alpha_global"] = x.epoch.map(actual_alpha).fillna(cfg["alpha_initial"])
        x.insert(0, "candidate", candidate["name"])
        all_curves.append(x)
    pd.concat(all_curves, ignore_index=True).to_csv(HERE / "all_development_curves.csv", index=False)
    print(table[["Method", "Epoch", "Recall@100", "Robust@100", "Never@100"]].to_string(index=False))
    print("SCALE", scale.to_string(index=False))
    print("SEEDS", pd.DataFrame(seed_rows).to_string(index=False))


def append_ci(rows, frames, split, mode, key, reference, metric):
    a, b = frames[split, mode, key], frames[split, mode, reference]
    assert set(a.index) == set(b.index)
    mean, lo, hi = prior.bootstrap(a[metric] - b.loc[a.index, metric], seed=20260963, samples=10000)
    rows.append(dict(Method=key, Reference=reference, Split=split, Mode=mode,
                     Metric=metric.replace("Recall", "Inclusion") if mode.startswith("target") else metric,
                     Delta_pp=100 * mean, CI_low_pp=100 * lo, CI_high_pp=100 * hi,
                     Queries=len(a), Draws=10000, Seed=20260963, multiplicity_correction="none"))


def plot_scale(scale, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for family, color in [("DeepSets", "#1865ab"), ("SetTransformer", "#bf5735")]:
        x = scale[(scale.Method == family) & (scale.Split == "dev")].sort_values("Examples")
        marker = "o" if family == "DeepSets" else "x"
        axes[0].plot(x.Examples, x["Recall@100"], marker + "-", color=color, label="Residual " + family.replace("SetTransformer", "Set Transformer"), linewidth=1.7)
        axes[1].plot(x.Examples, x.Best_Trained_Dev_R100, marker + "-", color=color, linewidth=1.7)
    for ax in axes:
        ax.axhline(100 * cfg["dev_baselines"]["Set-Attention_s42"], color="#397e74", ls="--", label="Set-Attention")
        ax.axhline(100 * cfg["dev_baselines"]["Canonical"], color="#7b6d55", ls=":", label="Canonical")
        ax.grid(alpha=.2)
        ax.set_ylabel("Dev Recall@100 (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Phase VI-B scaling: selected checkpoint (epoch 0 allowed)")
    axes[0].legend(fontsize=8, loc="best")
    if scale.Epoch.eq(0).all():
        axes[0].text(.02, .12, "All selected checkpoints are epoch 0; residual lines overlap Set-Attention.",
                     transform=axes[0].transAxes, fontsize=8)
    axes[1].set_title("Diagnostic: best trained epoch, excluding epoch 0")
    axes[1].set_xlabel("Training query-positive examples (T2; seed 42)")
    fig.tight_layout()
    fig.savefig(HERE / "PHASE6B_SCALING.png", dpi=180)
    fig.savefig(HERE / "PHASE6B_SCALING.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
