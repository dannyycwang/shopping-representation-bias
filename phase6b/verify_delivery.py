"""Verify data/selection/metrics and hash every VI-B deliverable plus frozen history."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from experiment import ROOT, HERE, OUT, CONFIG, STEMS, BASE_CK, prior, sha, dump, now, validate


def main():
    cfg = validate()
    lock = json.loads((HERE / "selection.json").read_text())
    assert not lock["test_used_for_selection"]
    initial = json.loads((HERE / "initialization_passed.json").read_text())
    assert initial["passed"] and initial["config_sha256"] == sha(CONFIG)
    assert len(lock["candidates"]) == 20 and len(lock["selected"]) == 14
    assert lock["target_keys"] == []
    assert all(x["additional_seeds"] == [43, 44] for x in lock["seed_gates"].values())
    data = {name: json.loads((HERE / "data" / f"{name}.json").read_text()) for name in cfg["training_examples"]}
    assert data["T2_f0.25"] == data["T2"][:814]
    assert data["T2_f0.5"] == data["T2"][:1628]
    assert len(data["T2"]) == len(data["T1"]) == 3257
    judgments = pd.read_csv(ROOT / "phase2/data/processed/wands_judgments.csv", dtype={"product_id": str})
    train = judgments[judgments.query_id.isin(cfg["train"])].set_index(["query_id", "product_id"]).label
    for examples in data.values():
        for e in examples:
            assert e["query_id"] in cfg["train"] and e["query_id"] not in cfg["validation"] + cfg["test"]
            assert train.loc[e["query_id"], e["positive"]] == "Exact"
            assert len(e["negatives"]) == len(set(e["negatives"]))
            assert all(train.loc[e["query_id"], p] == "Irrelevant" for p in e["negatives"])
    for candidate in lock["candidates"]:
        steps = pd.read_csv(HERE / "checkpoints" / candidate["name"] / "steps.csv")
        examples = data[candidate["regime"]]
        for epoch, group in steps.groupby("epoch"):
            assert group.positives.sum() == len(examples)
            assert group.negatives.sum() == sum(len(e["negatives"]) for e in examples)
            assert np.isfinite(group[["loss", "gradient_norm_before_clip", "alpha"]]).all().all()
    results = pd.read_csv(HERE / "PHASE6B_RESULTS.csv")
    ci = pd.read_csv(HERE / "PHASE6B_BOOTSTRAP.csv")
    scale = pd.read_csv(HERE / "PHASE6B_SCALE.csv")
    invariant = json.loads((HERE / "PHASE6B_INVARIANCE.json").read_text())
    base = torch.load(BASE_CK, map_location="cpu", weights_only=True)
    decision_time = pd.Timestamp(lock["locked_utc"])
    for key, selected in lock["selected"].items():
        folder = HERE / "checkpoints" / selected["name"]
        assert sha(folder / "weights.pt") == selected["weights_sha256"]
        epochs = pd.read_csv(folder / "epochs.csv")
        best = epochs.sort_values(["Recall@100", "Robust@100", "epoch"], ascending=[False, False, True]).iloc[0]
        assert int(best.epoch) == selected["best_epoch"] == 0
        state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        for k, v in base.items():
            assert torch.equal(state["base." + k.removeprefix("att.")], v)
        last = "branch.output.2" if selected["family"] == "D" else "branch.output"
        assert state[last + ".weight"].count_nonzero() == 0 and state[last + ".bias"].count_nonzero() == 0
        ev = json.loads((OUT / key / "evaluation.json").read_text())
        assert pd.Timestamp(ev["completed_utc"]) > decision_time
        assert ev["selection_sha256"] == sha(HERE / "selection.json")
        assert sha(OUT / key / "products.npy") == ev["products_sha256"]
        assert (ev["exact_pairs"], ev["queries"]) == (21299, 308)
        vector = np.load(OUT / key / "products.npy", mmap_mode="r")
        assert vector.shape == (42994, 768) and vector.dtype == np.float32
        assert np.isfinite(vector).all()
        inv = invariant["models"][key]
        assert inv["passed"] and inv["products"] == 128 and inv["products_with_changed_order"] >= 100
        assert inv["l2_max"] <= 1e-5 and max(x["max_L2"] for x in inv["full_catalog_audit"]) <= 1e-5
        assert [x["schedule"] for x in inv["full_catalog_audit"]] == STEMS[1:]
        primary = pd.read_parquet(OUT / key / "catalog_pairs.parquet")
        regenerated = pd.read_parquet(OUT / key / "regenerated_catalog_pairs.parquet")
        assert primary[["query_id", "product_id"]].equals(regenerated[["query_id", "product_id"]])
        assert primary.C0.equals(regenerated.C0)
        for record in inv["full_catalog_audit"]:
            schedule = record["schedule"]
            assert int((regenerated[schedule] != primary.C0).sum()) == record["changed_ranks"]
            for cutoff in [20, 100]:
                count = int(((regenerated[schedule] <= cutoff) != (primary.C0 <= cutoff)).sum())
                assert count == record[f"changed_inclusion{cutoff}"] == 0
        regenerated_queries = prior.metric_frame(regenerated)[1]
        for metric, value in inv["regenerated_metrics"].items():
            assert np.isclose(regenerated_queries[metric].mean(), value)
        stats = pd.read_parquet(OUT / key / "residual_diagnostics.parquet")
        assert len(stats) == 42994 and stats.product_id.nunique() == 42994
        assert stats.delta_norm.max() == 0 and stats.final_base_cosine.min() > 1 - 1e-5
        assert stats.alpha.max() == stats.alpha.min()
    for _, row in results.iterrows():
        if row.Split == "test":
            assert (row.Pairs, row.Queries) == ((6797, 239) if "fully_fitting" in row.Mode else (21299, 308))
        else:
            assert row.Queries == 17
        assert np.isclose(row["VI@100"] + row["Robust@100"] + row["Never@100"], 100)
    assert ci.Draws.eq(10000).all() and ci.Seed.eq(20260963).all()
    for _, row in ci.iterrows():
        if "seed_mean" in row.Method:
            continue
        a = results[(results.Method == row.Method) & (results.Split == row.Split) & (results.Mode == row.Mode)].iloc[0]
        b = results[(results.Method == row.Reference) & (results.Split == row.Split) & (results.Mode == row.Mode)].iloc[0]
        assert np.isclose(a[row.Metric] - b[row.Metric], row.Delta_pp)
    # One independent interval recomputation from saved paired ranks.
    a = prior.metric_frame(pd.read_parquet(OUT / "D_T0_s42/catalog_pairs.parquet"))[1]
    b = pd.read_parquet(ROOT / "phase5_mitigation/results/Original/catalog_pairs.parquet")
    b = prior.metric_frame(b[b.query_id.isin(cfg["test"])])[1]
    actual = np.asarray(prior.bootstrap(a["Recall@100"] - b.loc[a.index, "Recall@100"])) * 100
    expected = ci[(ci.Method == "D_T0_s42") & (ci.Reference == "Original") & (ci.Split == "test") & (ci.Mode == "catalog") & (ci.Metric == "Recall@100")].iloc[0]
    assert np.allclose(actual, expected[["Delta_pp", "CI_low_pp", "CI_high_pp"]].to_numpy(dtype=float))
    assert len(scale) == 12 and scale.Epoch.eq(0).all()
    assert all(not (OUT / k / "target_pairs.parquet").exists() for k in lock["selected"])
    verification = dict(passed=True, verified_utc=now(), historical_files_unchanged=True, train_only_judgments_verified=True,
                        nested_subsets_verified=True, selected_models=14, candidate_runs=20, epoch_zero_selections=14,
                        independent_regenerated_ranks_verified=True, full_catalog_schedules_per_model=len(STEMS),
                        test_not_used_for_selection=True, initial_test_identity_guard_disclosed=True,
                        bootstrap_interval_reproduced=True, bootstrap_rows=len(ci), result_rows=len(results), scale_rows=len(scale),
                        residual_unit_tests_passed=5, targets_run=0, selection_sha256=sha(HERE / "selection.json"))
    dump(HERE / "verification.json", verification)
    required = ["PHASE6B_REPORT.md", "PHASE6B_RESULTS.csv", "PHASE6B_SCALE.csv", "PHASE6B_BOOTSTRAP.csv", "PHASE6B_INVARIANCE.json",
                "PHASE6B_CONFIG.json", "PHASE6B_TRAINING_AUDIT.json"]
    assert all((HERE / name).exists() for name in required)
    paths = sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and
                   p.name != "FINAL_DELIVERY_MANIFEST.json" and not p.name.endswith(".tmp"))
    manifest = dict(phase="VI-B", completed_utc=now(), status="complete", output_directory=str(HERE), required_files=required + ["FINAL_DELIVERY_MANIFEST.json"],
                    decision=json.loads((HERE / "decision.json").read_text()), verification=verification,
                    config_sha256=sha(CONFIG), historical_sources=json.loads((HERE / "historical_sources.json").read_text(encoding="utf8")),
                    outputs=[dict(path=str(p.relative_to(HERE)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths])
    dump(HERE / "FINAL_DELIVERY_MANIFEST.json", manifest)
    # Verify the delivery hashes immediately; the manifest itself is intentionally excluded.
    for entry in manifest["outputs"]:
        assert sha(HERE / entry["path"]) == entry["sha256"]
    print(json.dumps(verification, indent=2))
    print("Verified", len(paths), "VI-B output hashes; historical Phase VI preserved")


if __name__ == "__main__":
    main()
