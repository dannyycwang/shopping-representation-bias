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
from src.representations import slug


def load_representation(dataset: str, condition: str) -> tuple[list, list[str], str]:
    path = P2 / f"data/representations/{dataset}/{slug(condition)}.jsonl.gz"
    product_ids, texts = [], []
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            digest.update(line.encode())
            item = json.loads(line)
            product_ids.append(item["product_id"])
            texts.append(item["text"])
    return product_ids, texts, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wands", "esci"], required=True)
    parser.add_argument("--model", choices=["minilm", "bge_base", "gte_modernbert"], required=True)
    parser.add_argument("--profile", default="native")
    parser.add_argument("--family", choices=["primary", "all"], default="all")
    args = parser.parse_args()
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    model_spec = next(item for item in cfg["models"] if item["key"] == args.model)
    profile = next(item for item in cfg["chunk_controls"] if item["key"] == args.profile)
    conditions = cfg["representations"] if args.family == "all" else cfg["primary_variant_family"]
    queries = pd.read_csv(P2 / f"data/processed/{args.dataset}_queries.csv")
    judgments = pd.read_csv(P2 / f"data/processed/{args.dataset}_judgments.csv")
    output = P2 / "results"
    for folder in ["phase2_per_query", "phase2_pair_ranks", "phase2_top50", "embeddings"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    encoder = DenseEncoder(model_spec, cfg, output / "embeddings")
    query_texts = queries["query"].fillna("").astype(str).tolist()
    query_sha = hashlib.sha256("\0".join(query_texts).encode()).hexdigest()
    query_profile = dict(profile)
    query_profile["chunk_tokens"] = None
    qemb = encoder.encode(query_texts, f"{args.dataset}_{args.model}_{args.profile}_queries", query_profile, query_sha)
    expected_ids = None
    for condition in conditions:
        stem = f"{args.dataset}_{args.model}_{args.profile}_{slug(condition)}"
        pair_path = output / f"phase2_pair_ranks/{stem}.parquet"
        per_query_path = output / f"phase2_per_query/{stem}.csv"
        if pair_path.exists() and per_query_path.exists():
            print(f"result cache hit {stem}", flush=True)
            continue
        product_ids, texts, source_sha = load_representation(args.dataset, condition)
        if expected_ids is None:
            expected_ids = product_ids
        elif product_ids != expected_ids:
            raise AssertionError("product order changed between representations")
        pemb = encoder.encode(texts, stem, profile, source_sha)
        scores = np.asarray(qemb, dtype=np.float32) @ np.asarray(pemb, dtype=np.float32).T
        per_query, pairs, top = evaluate(scores, product_ids, queries, judgments, args.dataset)
        per_query.to_csv(per_query_path, index=False)
        pairs.to_parquet(pair_path, index=False)
        top.to_parquet(output / f"phase2_top50/{stem}.parquet", index=False)
        print(
            f"{stem}: cNDCG10={per_query['cNDCG@10'].mean():.4f} "
            f"HighestRecall20={per_query['HighestRecall@20'].mean():.4f}",
            flush=True,
        )
    encoder.close()


if __name__ == "__main__":
    main()
