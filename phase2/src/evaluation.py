from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable").astype(np.int32)


def inverse(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order)
    values = np.broadcast_to(np.arange(1, order.shape[1] + 1, dtype=np.int32), order.shape)
    np.put_along_axis(ranks, order, values, axis=1)
    return ranks


def evaluate(scores: np.ndarray, product_ids: list, queries: pd.DataFrame,
             judgments: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = stable_order(scores)
    ranks = inverse(order)
    lookup = {str(product_id): index for index, product_id in enumerate(product_ids)}
    groups = {int(query_id): group for query_id, group in judgments.groupby("query_id")}
    highest_label = "Exact" if dataset == "wands" else "E"
    query_rows, pair_rows, top_rows = [], [], []
    for query_index, query_row in enumerate(queries.itertuples(index=False)):
        group = groups[int(query_row.query_id)]
        indices = np.asarray([lookup[str(value)] for value in group.product_id], dtype=np.int32)
        gains = group.grade.to_numpy(dtype=np.float64)
        labels = group.label.astype(str).to_numpy()
        judged = dict(zip(indices.tolist(), gains.tolist()))
        condensed = np.asarray([judged[int(index)] for index in order[query_index] if int(index) in judged])
        ideal = np.sort(gains)[::-1]
        row = {
            "query_id": int(query_row.query_id),
            "query": query_row.query,
            "query_type": query_row.query_type,
            "n_judged": len(group),
            "n_highest": int(np.sum(labels == highest_label)),
        }
        relevant = indices[gains > 0]
        highest = indices[labels == highest_label]
        for k in [10, 20, 50]:
            width = min(k, len(ideal))
            discount = 1 / np.log2(np.arange(2, width + 2))
            idcg = float(np.sum(ideal[:width] * discount))
            row[f"cNDCG@{k}"] = float(np.sum(condensed[:width] * discount) / idcg) if idcg else np.nan
            row[f"RelevantRecall@{k}"] = float(np.mean(ranks[query_index, relevant] <= k)) if len(relevant) else np.nan
            row[f"HighestRecall@{k}"] = float(np.mean(ranks[query_index, highest] <= k)) if len(highest) else np.nan
            row[f"RelevantHidden@{k}"] = 1 - row[f"HighestRecall@{k}"] if len(highest) else np.nan
            row[f"JudgedCoverage@{k}"] = float(sum(int(index) in judged for index in order[query_index, :k]) / k)
        row["RelevanceVisibilitySpearman"] = (
            float(spearmanr(gains, -ranks[query_index, indices]).statistic)
            if len(np.unique(gains)) > 1 else np.nan
        )
        query_rows.append(row)
        for index, label, gain in zip(indices, labels, gains):
            pair_rows.append({
                "query_id": int(query_row.query_id),
                "product_id": product_ids[int(index)],
                "label": label,
                "grade": gain,
                "rank": int(ranks[query_index, index]),
                "score": float(scores[query_index, index]),
                "query_type": query_row.query_type,
            })
        for position, index in enumerate(order[query_index, :50], 1):
            top_rows.append({
                "query_id": int(query_row.query_id),
                "product_id": product_ids[int(index)],
                "rank": position,
                "score": float(scores[query_index, index]),
            })
    return pd.DataFrame(query_rows), pd.DataFrame(pair_rows), pd.DataFrame(top_rows)


def mean_bootstrap(values, seed: int, samples: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        means.extend(array[rng.integers(0, len(array), size=(count, len(array)))].mean(axis=1))
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def paired_stats(first, second, seed: int, samples: int = 10000) -> tuple[float, float, float, float]:
    difference = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    difference = difference[np.isfinite(difference)]
    rng = np.random.default_rng(seed)
    boots, null = [], []
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        boots.extend(difference[rng.integers(0, len(difference), size=(count, len(difference)))].mean(axis=1))
        null.extend((difference * rng.choice([-1, 1], size=(count, len(difference)))).mean(axis=1))
    mean = float(difference.mean())
    low, high = (float(value) for value in np.quantile(boots, [0.025, 0.975]))
    p = float((1 + np.sum(np.abs(null) >= abs(mean) - 1e-15)) / (samples + 1))
    return mean, low, high, p
