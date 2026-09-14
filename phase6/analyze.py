"""Analysis of locked Phase VI runs and existing raw-track baseline ranks only."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run import ROOT, HERE, OUT, CONFIG, BASELINES, STEMS, prior, sha, dump, now, validate_config

LABELS = {"Original": "Original", "Canonical": "Canonical", "Set-Mean": "Set-Mean",
          "Set-Attention_s42": "Set-Attention", "S1": "Set Transformer",
          "S2": "Set Transformer + Field-ID", "S3": "Circular Projection", "S4": "DeepSets"}
METRICS = ["Recall@20", "Recall@100", "VI@20", "VI@100", "Robust@100", "Never@100", "worst_schedule@100"]


def method_label(key):
    return LABELS.get(key, LABELS.get(key.split("_")[0], key))


def main():
    cfg = validate_config()
    selection = json.loads((HERE / "selection.json").read_text())
    complete = json.loads((HERE / "evaluation_complete.json").read_text())
    assert complete["selection_sha256"] == sha(HERE / "selection.json")
    keys = BASELINES + list(selection["selected"])
    fitting = pd.read_csv(ROOT / "phase4/results/wands_product_features.csv", dtype={"product_id": str})
    assert fitting.fully_fits.dtype == bool
    fitids = set(fitting.loc[fitting.fully_fits, "product_id"])
    rows, uncertainty, frames, pairframes = [], [], {}, {}
    for split in ["validation", "test"]:
        for mode in ["catalog", "catalog_fully_fitting", "target", "target_fully_fitting"]:
            if split == "validation" and mode != "catalog":
                continue
            family = mode.split("_")[0]
            for key in keys:
                baseline = key in BASELINES
                if baseline:
                    path = ROOT / f"phase5_mitigation/results/{key}/{family}_pairs.parquet"
                elif split == "validation":
                    path = HERE / "checkpoints" / selection["selected"][key]["name"] / "dev_pairs.parquet"
                else:
                    path = OUT / key / f"{family}_pairs.parquet"
                if not path.exists():
                    assert family == "target" and not baseline
                    continue
                frame = pd.read_parquet(path)
                frame["product_id"] = frame.product_id.astype(str)
                frame = frame[frame.query_id.isin(cfg[split])].reset_index(drop=True)
                if "fully_fitting" in mode:
                    frame = frame[frame.product_id.isin(fitids)].reset_index(drop=True)
                assert not frame.duplicated(["query_id", "product_id"]).any()
                _, pq = prior.metric_frame(frame)
                expected = (6797, 239) if "fully_fitting" in mode else (21299, 308)
                if split == "test":
                    assert (len(frame), len(pq)) == expected, (key, split, mode, len(frame), len(pq))
                else:
                    assert len(pq) == 17
                frames[split, mode, key] = pq
                pairframes[split, mode, key] = frame
                params = (98561 if key == "Set-Attention_s42" else 0) if baseline else selection["selected"][key]["parameters"]
                values = {m: 100 * float(pq[m].mean()) for m in METRICS}
                if family == "target":
                    values["Inclusion@20"] = values.pop("Recall@20")
                    values["Inclusion@100"] = values.pop("Recall@100")
                row = dict(method=key, label=method_label(key), split="dev" if split == "validation" else split,
                           mode=mode, seed=42 if key == "Set-Attention_s42" else (None if baseline else selection["selected"][key]["seed"]),
                           Params=params, pairs=len(frame), queries=len(pq), **values,
                           minimum_macro_schedule100=100 * float(pq[[f"{s}_Recall@100" for s in STEMS]].mean().min()),
                           units="percent; deltas in percentage points", vectors_per_product=1)
                rows.append(row)
            available = [k for k in keys if (split, mode, k) in frames]
            for key in available:
                if key in BASELINES:
                    continue
                for ref in BASELINES:
                    a, b = frames[split, mode, key], frames[split, mode, ref]
                    assert set(a.index) == set(b.index)
                    for metric in METRICS[:-1]:
                        mean, lo, hi = prior.bootstrap(a[metric] - b.loc[a.index, metric], seed=cfg["bootstrap_seed"], samples=cfg["bootstrap_draws"])
                        uncertainty.append(dict(method=key, reference=ref, split="dev" if split == "validation" else split, mode=mode,
                                                metric=metric.replace("Recall", "Inclusion") if family == "target" else metric,
                                                delta_pp=100 * mean, ci_low_pp=100 * lo, ci_high_pp=100 * hi,
                                                queries=len(a), draws=10000, bootstrap_seed=cfg["bootstrap_seed"], correction="none"))
            for key, ref in [("S2_s42", "S1_s42"), ("S3_s42", "S4_s42"), ("S3_s42", "S2_s42"), ("S4_s42", "S2_s42")]:
                if key not in available or ref not in available:
                    continue
                a, b = frames[split, mode, key], frames[split, mode, ref]
                mean, lo, hi = prior.bootstrap(a["Recall@100"] - b.loc[a.index, "Recall@100"], seed=cfg["bootstrap_seed"], samples=10000)
                uncertainty.append(dict(method=key, reference=ref, split="dev" if split == "validation" else split, mode=mode,
                                        metric="Recall@100", delta_pp=100 * mean, ci_low_pp=100 * lo, ci_high_pp=100 * hi,
                                        queries=len(a), draws=10000, bootstrap_seed=cfg["bootstrap_seed"], correction="none"))
    results = pd.DataFrame(rows)
    # Compute all table deltas from the matching population, including the first baseline row.
    for (split, mode), group in results.groupby(["split", "mode"]):
        metric = "Inclusion@100" if mode.startswith("target") else "Recall@100"
        for ref in BASELINES[:2]:
            ref_value = float(group.loc[group.method.eq(ref), metric].iloc[0])
            delta_column = ("Delta Inclusion@100 vs " if mode.startswith("target") else "Delta R@100 vs ") + ref
            results.loc[group.index, delta_column] = group[metric] - ref_value
    results.to_csv(ROOT / "PHASE6_RESULTS.csv", index=False)
    ci = pd.DataFrame(uncertainty)
    ci.to_csv(ROOT / "PHASE6_BOOTSTRAP.csv", index=False)
    primary = results[(results.split == "test") & (results["mode"] == "catalog") & (~results.method.str.endswith(("_s43", "_s44")))].copy()
    expected = {"Original": 63.620, "Canonical": 63.237, "Set-Mean": 61.820, "Set-Attention_s42": 62.173}
    for key, number in expected.items():
        assert abs(primary.set_index("method").loc[key, "Recall@100"] - number) < .00051
    columns = ["label", "Params", "Recall@20", "Recall@100", "Delta R@100 vs Original", "Delta R@100 vs Canonical", "VI@20", "VI@100", "Robust@100", "Never@100"]
    primary[columns].rename(columns={"label": "Method"}).to_csv(HERE / "TABLE_VI_1.csv", index=False)
    transitions = []
    query_changes = []
    original = pairframes["test", "catalog", "Original"].set_index(["query_id", "product_id"])
    for key in selection["selected"]:
        f = pairframes["test", "catalog", key]
        base = original.loc[pd.MultiIndex.from_frame(f[["query_id", "product_id"]])]
        a, b = f[STEMS].to_numpy() <= 100, base[STEMS].to_numpy() <= 100
        changes = pd.DataFrame(dict(query_id=f.query_id, rescued=(a & ~b).mean(1), newly_missed=(~a & b).mean(1)))
        changes["net"] = changes.rescued - changes.newly_missed
        changes["formerly_robust_now_never"] = b.all(1) & ~a.any(1)
        changes["formerly_never_now_robust"] = ~b.any(1) & a.all(1)
        macro = changes.groupby("query_id").mean()
        transitions.append(dict(method=key, **{col: 100 * float(val) for col, val in macro.mean().items()}))
        for qid, row in macro.iterrows():
            query_changes.append(dict(method=key, query_id=qid, **{col: 100 * float(val) for col, val in row.items()}))
    pd.DataFrame(transitions).to_csv(HERE / "failure_transitions.csv", index=False)
    pd.DataFrame(query_changes).to_csv(HERE / "query_changes.csv", index=False)
    costs = pd.DataFrame([json.loads(line) for line in (HERE / "cost.jsonl").read_text().splitlines()])
    costs.groupby(["run", "stage"]).seconds.sum().reset_index().to_csv(HERE / "cost_summary.csv", index=False)
    seed_table = results[(results.split == "test") & (results["mode"] == "catalog") & results.method.str.startswith(("S1_", "S2_", "S3_", "S4_"))].copy()
    seed_table["family"] = seed_table.method.str.split("_").str[0]
    seed_table.groupby("family")[["Recall@100", "Robust@100"]].agg(["mean", "std", "min", "max", "count"]).to_csv(HERE / "seed_summary.csv")
    figure(primary)
    print(primary[columns].to_string(index=False))
    print("DEVELOPMENT")
    print(results[(results.split == "dev")][["method", "Recall@100"]].to_string(index=False))
    print("PRIMARY BOOTSTRAP")
    print(ci[(ci.split == "test") & (ci["mode"] == "catalog") & (ci.metric == "Recall@100")].to_string(index=False))


def figure(primary):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {"Original": "#242424", "Canonical": "#7b6d55", "Set-Mean": "#768390", "Set-Attention_s42": "#397e74",
              "S1_s42": "#1865ab", "S2_s42": "#be4b38", "S3_s42": "#8c57aa", "S4_s42": "#b07712"}
    for i, row in primary.reset_index(drop=True).iterrows():
        x, y = row["Recall@100"], row["Robust@100"]
        ax.scatter(x, y, color=colors[row.method], s=48, zorder=3)
        offset = (6, 6)
        if row.method == "Canonical":
            offset = (7, 15)
        elif row.method == "Set-Mean":
            offset = (-75, -22)
        elif row.method == "Set-Attention_s42":
            offset = (7, -9)
        ax.annotate(row.label, (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=.45, color=colors[row.method]))
    low = min(primary["Recall@100"].min(), primary["Robust@100"].min()) - 3
    high = max(primary["Recall@100"].max(), primary["Robust@100"].max()) + 3
    ax.plot([low, high], [low, high], color="#c7c7c7", lw=.8, ls="--", zorder=0)
    ax.set(xlabel="Recall@100 (%)", ylabel="Robust@100 (%)", xlim=(low, high + 5), ylim=(low, high + 1),
           title="Phase VI screening · WANDS / frozen BGE · seed42")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "PHASE6_SCREENING.png", dpi=180)
    fig.savefig(HERE / "PHASE6_SCREENING.pdf")
    plt.close(fig)


def manifest():
    names = ["PHASE6_REPORT.md", "PHASE6_RESULTS.csv", "PHASE6_BOOTSTRAP.csv", "PHASE6_INVARIANCE.json", "PHASE6_CONFIG.json"]
    outputs = [ROOT / n for n in names]
    assert all(p.exists() for p in outputs)
    outputs += [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and not p.name.endswith(".tmp")]
    cfg = validate_config()
    selection = json.loads((HERE / "selection.json").read_text())
    dump(ROOT / "FINAL_DELIVERY_MANIFEST.json", dict(phase=6, completed_utc=now(), status="complete",
          repository=str(ROOT), track=cfg["track"], prior_sources_unchanged=True, test_used_for_selection=False,
          config_sha256=sha(CONFIG), selection_sha256=sha(HERE / "selection.json"), seeds=selection["gates"],
          outputs=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(outputs)],
          reused_sources=cfg["sources"], reproducibility=[
              "phase2/.venv/Scripts/python.exe -X utf8 -m pytest phase6/test_models.py phase6/test_metrics.py -q",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py prepare",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py train",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/run.py evaluate",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/analyze.py",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/verify.py",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/write_report.py",
              "phase2/.venv/Scripts/python.exe -X utf8 phase6/analyze.py manifest"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        manifest()
    else:
        main()
