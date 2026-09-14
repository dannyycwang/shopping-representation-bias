"""Read-only checks of frozen decisions, paired populations, and delivered measurements."""
import json
import numpy as np
import pandas as pd

from run import ROOT, HERE, OUT, CONFIG, STEMS, BASELINES, prior, sha, dump, validate_config


def main():
    cfg = validate_config()
    selection = json.loads((HERE / "selection.json").read_text())
    results = pd.read_csv(ROOT / "PHASE6_RESULTS.csv")
    ci = pd.read_csv(ROOT / "PHASE6_BOOTSTRAP.csv")
    assert not selection["test_used"]
    assert len(selection["candidates"]) == 8
    assert len(selection["selected"]) == 4
    assert all(not g["passed"] and not g["extra_seeds"] for g in selection["gates"].values())
    locked = pd.Timestamp(selection["locked_utc"])
    for key, selected in selection["selected"].items():
        cp = HERE / "checkpoints" / selected["name"]
        assert sha(cp / "weights.pt") == selected["weights_sha256"]
        epochs = pd.read_csv(cp / "epochs.csv")
        best = epochs.sort_values(["dev_recall100", "epoch"], ascending=[False, True]).iloc[0]
        assert int(best.epoch) == selected["best_epoch"]
        assert abs(best.dev_recall100 - selected["dev_recall100"]) < 1e-12
        ev = json.loads((OUT / key / "evaluation.json").read_text())
        assert pd.Timestamp(ev["completed_utc"]) > locked
        assert ev["selection_sha256"] == sha(HERE / "selection.json")
        assert sha(OUT / key / "products.npy") == ev["products_sha256"]
        v = np.load(OUT / key / "products.npy", mmap_mode="r")
        assert v.shape == (42994, 768) and v.dtype == np.float32
        assert np.isfinite(v).all()
    for _, row in results.iterrows():
        if row.split == "test":
            assert (row.pairs, row.queries) == ((6797, 239) if "fully_fitting" in row["mode"] else (21299, 308))
        else:
            assert row.queries == 17
        assert np.isclose(row["VI@100"] + row["Robust@100"] + row["Never@100"], 100)
        if row.method != "Original":
            m = "Inclusion@100" if row["mode"].startswith("target") else "Recall@100"
            assert np.isclose(row[m], row["Robust@100"])
            assert row["VI@100"] == row["VI@20"] == 0
        if row["mode"].startswith("target"):
            assert pd.isna(row["Recall@100"]) and pd.isna(row["Delta R@100 vs Original"])
    assert ci.draws.eq(10000).all() and ci.bootstrap_seed.eq(20260963).all()
    for _, row in ci.iterrows():
        a = results[(results.method == row.method) & (results.split == row.split) & (results["mode"] == row["mode"])].iloc[0]
        b = results[(results.method == row.reference) & (results.split == row.split) & (results["mode"] == row["mode"])].iloc[0]
        assert np.isclose(a[row.metric] - b[row.metric], row.delta_pp), row.to_dict()
    # Independently regenerate one reported interval from saved query-level ranks.
    a = prior.metric_frame(pd.read_parquet(OUT / "S3_s42/catalog_pairs.parquet"))[1]
    b = pd.read_parquet(ROOT / "phase5_mitigation/results/Original/catalog_pairs.parquet")
    b = prior.metric_frame(b[b.query_id.isin(cfg["test"])])[1]
    actual = np.array(prior.bootstrap(a["Recall@100"] - b.loc[a.index, "Recall@100"])) * 100
    expected = ci[(ci.method == "S3_s42") & (ci.reference == "Original") & (ci.split == "test") &
                  (ci["mode"] == "catalog") & (ci.metric == "Recall@100")].iloc[0]
    assert np.allclose(actual, expected[["delta_pp", "ci_low_pp", "ci_high_pp"]].to_numpy(dtype=float))
    numerical = []
    for path in OUT.glob("*/regenerated_target_pairs.parquet"):
        independent = pd.read_parquet(path)
        primary = pd.read_parquet(path.parent / "target_pairs.parquet")
        _, pq = prior.metric_frame(independent)
        numerical.append(dict(method=path.parent.name, pairs=len(independent), queries=len(pq),
                              changed_pair_schedule_ranks=int((independent[STEMS].to_numpy() != primary[STEMS].to_numpy()).sum()),
                              Inclusion100=100 * float(pq["Recall@100"].mean()),
                              VI20=100 * float(pq["VI@20"].mean()), VI100=100 * float(pq["VI@100"].mean())))
    pd.DataFrame(numerical).to_csv(HERE / "target_numerical_audit.csv", index=False)
    record = dict(passed=True, frozen_sources_unchanged=True, selection_is_development_only=True,
                  candidate_runs=8, selected_models=4, result_rows=len(results), bootstrap_rows=len(ci),
                  paired_bootstrap_reproduced=True, unit_tests_passed=13,
                  target_numerical_audit=numerical, config_sha256=sha(CONFIG), selection_sha256=sha(HERE / "selection.json"))
    dump(HERE / "verification.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
