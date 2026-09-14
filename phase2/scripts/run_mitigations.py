from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2))
from src.encoding import DenseEncoder
from src.evaluation import evaluate
from src.mitigations import BASE_VARIANTS, METHODS, mitigation_slug


def load_texts(path: Path) -> tuple[list, list[str], str]:
    product_ids, texts = [], []
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            digest.update(line.encode())
            item = json.loads(line)
            product_ids.append(item["product_id"])
            texts.append(item["text"])
    return product_ids, texts, digest.hexdigest()


def save_evaluation(dataset: str, model: str, profile_key: str, stem: str,
                    scores: np.ndarray, product_ids: list, queries: pd.DataFrame,
                    judgments: pd.DataFrame) -> None:
    output = P2 / "results"
    pair_path = output / f"phase2_pair_ranks/{dataset}_{model}_{profile_key}_{stem}.parquet"
    query_path = output / f"phase2_per_query/{dataset}_{model}_{profile_key}_{stem}.csv"
    if pair_path.exists() and query_path.exists():
        print(f"result cache hit {dataset}_{model}_{profile_key}_{stem}", flush=True)
        return
    per_query, pairs, top = evaluate(scores, product_ids, queries, judgments, dataset)
    per_query.to_csv(query_path, index=False)
    pairs.to_parquet(pair_path, index=False)
    top.to_parquet(
        output / f"phase2_top50/{dataset}_{model}_{profile_key}_{stem}.parquet", index=False
    )
    print(
        f"{dataset}_{model}_{profile_key}_{stem}: "
        f"cNDCG10={per_query['cNDCG@10'].mean():.4f} "
        f"HighestRecall20={per_query['HighestRecall@20'].mean():.4f}",
        flush=True,
    )


def direct_ranks(base_scores: np.ndarray, alternative_embeddings: np.ndarray,
                 query_embeddings: np.ndarray, product_ids: list,
                 judgments: pd.DataFrame, dataset: str, full_pair_path: Path) -> pd.DataFrame:
    product_lookup = {str(value): index for index, value in enumerate(product_ids)}
    query_ids = sorted(judgments.query_id.unique())
    query_lookup = {int(value): index for index, value in enumerate(query_ids)}
    highest = "Exact" if dataset == "wands" else "E"
    relevant = judgments[judgments.grade.gt(0)].copy()
    full = pd.read_parquet(full_pair_path)[["query_id", "product_id", "rank"]]
    full = full.rename(columns={"rank": "full_catalog_rank"})
    rows = []
    brute_checks = 0
    for query_id, group in relevant.groupby("query_id", sort=True):
        query_index = query_lookup[int(query_id)]
        target_indices = np.asarray(
            [product_lookup[str(value)] for value in group.product_id], dtype=np.int32
        )
        alternative_scores = (
            np.asarray(alternative_embeddings[target_indices], dtype=np.float32)
            @ np.asarray(query_embeddings[query_index], dtype=np.float32)
        )
        catalog_scores = np.asarray(base_scores[query_index], dtype=np.float32)
        order = np.argsort(-catalog_scores, kind="stable")
        base_ranks = np.empty(len(order), dtype=np.int32)
        base_ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
        sorted_scores = catalog_scores[order]
        greater = np.searchsorted(-sorted_scores, -alternative_scores, side="left")
        # Remove the target's original score when it lies above its alternative
        # score; every other catalog item stays in C0.
        greater -= (catalog_scores[target_indices] > alternative_scores).astype(np.int64)
        ranks = greater + 1
        # Exact fp32 ties are uncommon but stable catalog-id ordering remains
        # part of the protocol. Repair those ranks without scanning the catalog.
        for local, (target_index, score) in enumerate(zip(target_indices, alternative_scores)):
            left = int(np.searchsorted(-sorted_scores, -score, side="left"))
            right = int(np.searchsorted(-sorted_scores, -score, side="right"))
            if right > left:
                equal_indices = order[left:right]
                equal_before = int(np.sum(equal_indices < target_index))
                ranks[local] = left - int(catalog_scores[target_index] > score) + equal_before + 1
            if brute_checks < 5:
                modified = catalog_scores.copy()
                modified[target_index] = score
                brute_order = np.argsort(-modified, kind="stable")
                brute_rank = int(1 + np.flatnonzero(brute_order == target_index)[0])
                if int(ranks[local]) != brute_rank:
                    raise AssertionError((int(ranks[local]), brute_rank, target_index, float(score)))
                brute_checks += 1
        for source, rank, alt_score in zip(group.itertuples(index=False), ranks, alternative_scores):
            target_index = product_lookup[str(source.product_id)]
            rows.append({
                "query_id": int(query_id),
                "product_id": source.product_id,
                "label": source.label,
                "grade": float(source.grade),
                "is_highest": source.label == highest,
                "base_rank": int(base_ranks[target_index]),
                "single_product_rank": int(rank),
                "single_product_score": float(alt_score),
            })
    result = pd.DataFrame(rows)
    return result.merge(full, on=["query_id", "product_id"], how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wands", "esci"], required=True)
    parser.add_argument("--model", choices=["minilm", "bge_base", "gte_modernbert"], required=True)
    parser.add_argument("--profile", default="native")
    args = parser.parse_args()
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    model_spec = next(item for item in cfg["models"] if item["key"] == args.model)
    profile = next(item for item in cfg["chunk_controls"] if item["key"] == args.profile)
    queries = pd.read_csv(P2 / f"data/processed/{args.dataset}_queries.csv").sort_values("query_id")
    judgments = pd.read_csv(P2 / f"data/processed/{args.dataset}_judgments.csv")
    for folder in ["phase2_per_query", "phase2_pair_ranks", "phase2_top50", "embeddings",
                   "phase2_single_product"]:
        (P2 / "results" / folder).mkdir(parents=True, exist_ok=True)
    encoder = DenseEncoder(model_spec, cfg, P2 / "results/embeddings")
    query_texts = queries["query"].fillna("").astype(str).tolist()
    query_sha = hashlib.sha256("\0".join(query_texts).encode()).hexdigest()
    query_profile = dict(profile)
    query_profile["chunk_tokens"] = None
    qemb = encoder.encode(
        query_texts, f"{args.dataset}_{args.model}_{args.profile}_queries", query_profile, query_sha
    )

    embedded_by_sha: dict[str, np.ndarray] = {}
    baseline_ids, baseline_texts, baseline_sha = load_texts(
        P2 / f"data/representations/{args.dataset}/C0.jsonl.gz"
    )
    baseline_embeddings = encoder.encode(
        baseline_texts, f"{args.dataset}_{args.model}_{args.profile}_C0", profile, baseline_sha
    )
    base_scores = np.asarray(qemb, dtype=np.float32) @ np.asarray(baseline_embeddings, dtype=np.float32).T
    canonical_embeddings = {}
    for method in METHODS:
        for base_variant in BASE_VARIANTS:
            stem = mitigation_slug(method, base_variant)
            product_ids, texts, source_sha = load_texts(
                P2 / f"data/mitigations/{args.dataset}/{stem}.jsonl.gz"
            )
            if product_ids != baseline_ids:
                raise AssertionError("mitigation product order differs from C0")
            if source_sha in embedded_by_sha:
                embeddings = embedded_by_sha[source_sha]
                print(f"identical-source reuse {stem}", flush=True)
            else:
                embeddings = encoder.encode(
                    texts, f"{args.dataset}_{args.model}_{args.profile}_{stem}", profile, source_sha
                )
                embedded_by_sha[source_sha] = embeddings
            scores = np.asarray(qemb, dtype=np.float32) @ np.asarray(embeddings, dtype=np.float32).T
            save_evaluation(
                args.dataset, args.model, args.profile, stem, scores,
                product_ids, queries, judgments,
            )
            if base_variant == "C0":
                canonical_embeddings[method] = embeddings

    for method, embeddings in canonical_embeddings.items():
        stem = mitigation_slug(method, "C0")
        pair_path = P2 / (
            f"results/phase2_pair_ranks/{args.dataset}_{args.model}_{args.profile}_{stem}.parquet"
        )
        direct = direct_ranks(
            base_scores, embeddings, qemb, baseline_ids, judgments, args.dataset, pair_path
        )
        direct["dataset"] = args.dataset
        direct["retriever"] = args.model
        direct["representation"] = method
        direct["base_to_single_delta"] = direct.base_rank - direct.single_product_rank
        direct["base_to_full_delta"] = direct.base_rank - direct.full_catalog_rank
        for k in [10, 20, 50]:
            direct[f"single_crossing@{k}"] = (
                (direct.base_rank <= k) != (direct.single_product_rank <= k)
            )
            direct[f"full_crossing@{k}"] = (
                (direct.base_rank <= k) != (direct.full_catalog_rank <= k)
            )
        direct.to_parquet(
            P2 / f"results/phase2_single_product/{args.dataset}_{args.model}_{args.profile}_{stem}.parquet",
            index=False,
        )
    encoder.close()


if __name__ == "__main__":
    main()
