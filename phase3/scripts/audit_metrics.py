from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "phase2"
OUT = ROOT / "phase3" / "results" / "phase3_audit"
MODELS = ["minilm", "bge_base", "gte_modernbert"]
DATASETS = ["wands", "esci"]
STEMS = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]


def dcg(gains: np.ndarray, k: int) -> float:
    values = np.asarray(gains[:k], dtype=float)
    if not len(values):
        return 0.0
    return float(np.sum(values / np.log2(np.arange(2, len(values) + 2))))


def independently_recompute(dataset: str, model: str, profile: str = "native") -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [P2 / "results" / "phase2_pair_ranks" / f"{dataset}_{model}_{profile}_{s}.parquet" for s in STEMS]
    base = pd.read_parquet(paths[0])
    expected = ["query_id", "product_id", "label", "grade", "rank"]
    assert set(expected).issubset(base.columns)
    assert not base.duplicated(["query_id", "product_id"]).any()

    wide = base[["query_id", "product_id", "label", "grade", "rank"]].rename(columns={"rank": "C0"})
    for stem, path in zip(STEMS[1:], paths[1:]):
        part = pd.read_parquet(path)[["query_id", "product_id", "label", "grade", "rank"]]
        wide = wide.merge(part.rename(columns={"rank": stem}),
                          on=["query_id", "product_id", "label", "grade"], validate="one_to_one")
    rank_values = wide[STEMS].to_numpy()
    wide["VI@20"] = ((rank_values <= 20).any(axis=1) & (rank_values > 20).any(axis=1)).astype(float)

    highest = "Exact" if dataset == "wands" else "E"
    query_rows = []
    for query_id, group in wide.groupby("query_id", sort=True):
        ordered = group.sort_values(["C0", "product_id"], kind="stable")
        ideal = np.sort(group.grade.to_numpy(dtype=float))[::-1]
        relevant = group[group.grade.gt(0)]
        highest_rows = group[group.label.eq(highest)]
        row = {
            "query_id": int(query_id),
            "cNDCG@10": dcg(ordered.grade.to_numpy(), 10) / dcg(ideal, 10) if dcg(ideal, 10) else np.nan,
            "cNDCG@20": dcg(ordered.grade.to_numpy(), 20) / dcg(ideal, 20) if dcg(ideal, 20) else np.nan,
            "Recall@20": float((relevant.C0 <= 20).mean()) if len(relevant) else np.nan,
            "HighestRecall@20": float((highest_rows.C0 <= 20).mean()) if len(highest_rows) else np.nan,
            "RelevantHidden@20": float(1 - (highest_rows.C0 <= 20).mean()) if len(highest_rows) else np.nan,
            "VI@20_query": float(highest_rows["VI@20"].mean()) if len(highest_rows) else np.nan,
            "has_highest": bool(len(highest_rows)),
        }
        query_rows.append(row)
    per_query = pd.DataFrame(query_rows)
    highest_pairs = wide[wide.label.eq(highest)]
    aggregate = pd.DataFrame([{
        "dataset": dataset, "retriever": model, "profile": profile,
        "evaluated_queries": int(per_query["cNDCG@10"].notna().sum()),
        "highest_eligible_queries": int(per_query.has_highest.sum()),
        "cNDCG@10": float(per_query["cNDCG@10"].mean()),
        "cNDCG@20": float(per_query["cNDCG@20"].mean()),
        "Recall@20": float(per_query["Recall@20"].mean()),
        "HighestRecall@20": float(per_query["HighestRecall@20"].mean()),
        "RelevantHidden@20": float(per_query["RelevantHidden@20"].mean()),
        "VI@20_micro": float(highest_pairs["VI@20"].mean()),
        "VI@20_query_macro": float(per_query["VI@20_query"].mean()),
        # Define SAR over every relevance-eligible query. A query with no highest-label
        # pair has no highest-label instability event, hence a zero VI penalty.
        "SAR(lambda=0)": float((per_query["cNDCG@10"] - 0.0 * per_query["VI@20_query"].fillna(0)).mean()),
    }])
    return per_query, aggregate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            per_query, aggregate = independently_recompute(dataset, model)
            per_query.to_csv(OUT / f"{dataset}_{model}_native_recomputed_per_query.csv", index=False)
            rows.append(aggregate)
    audit = pd.concat(rows, ignore_index=True)
    audit.to_csv(OUT / "recomputed_metrics.csv", index=False)

    table3 = pd.read_csv(P2 / "results" / "phase2_tables" / "table3_relevance_visibility_alignment_native.csv")
    table3 = table3[table3.representation.eq("C0_original")]
    sar_old = pd.read_csv(P2 / "results" / "phase2_tables" / "stability_adjusted_relevance_native.csv")
    sar_old = sar_old[(sar_old.representation.eq("C0_original")) & sar_old["lambda"].eq(0)]
    checks = audit.merge(table3, on=["dataset", "retriever"], suffixes=("_audit", "_table3"))
    checks = checks.merge(sar_old[["dataset", "retriever", "SAR"]], on=["dataset", "retriever"])
    for metric in ["cNDCG@10", "HighestRecall@20", "RelevantHidden@20", "VI@20_micro", "VI@20_query_macro"]:
        checks[f"diff_{metric}"] = checks[f"{metric}_audit"] - checks[f"{metric}_table3"]
        assert np.allclose(checks[f"{metric}_audit"], checks[f"{metric}_table3"], atol=1e-12, equal_nan=True), metric
    assert np.allclose(audit["SAR(lambda=0)"], audit["cNDCG@10"], atol=1e-15)
    checks["old_SAR_minus_all_query_cNDCG"] = checks["SAR"] - checks["cNDCG@10_audit"]
    checks.to_csv(OUT / "summary_reconciliation.csv", index=False)

    sar_now_matches = bool(np.allclose(checks["SAR"], checks["cNDCG@10_audit"], atol=1e-12))
    assert sar_now_matches
    result = {
        "status": "PASS",
        "independent_recomputation_matches_phase2": True,
        "new_SAR_lambda0_equals_relevance_component": True,
        "SAR_lambda0_matches_after_fix": sar_now_matches,
        "resolved_SAR_issue": "the former inner join restricted cNDCG to highest-label-eligible queries; SAR now left-joins VI and assigns zero penalty when no highest-label pair exists",
        "cells": len(checks),
        "max_absolute_primary_difference": float(max(abs(checks.filter(regex="^diff_").to_numpy()).ravel())),
    }
    (OUT / "p0_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
