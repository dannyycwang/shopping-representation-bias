from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


P2 = Path(__file__).resolve().parents[1]
ROOT = P2.parent
sys.path.insert(0, str(P2))
from src.representations import audit, build, classify_query, slug


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def wands(cfg: dict) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(ROOT / "data/raw/product.csv", sep="\t").sort_values("product_id")
    queries = pd.read_csv(ROOT / "data/raw/query.csv", sep="\t").sort_values("query_id")
    labels = pd.read_csv(ROOT / "data/raw/label.csv", sep="\t")
    conflicts = labels.groupby(["query_id", "product_id"]).label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    mask = labels.set_index(["query_id", "product_id"]).index.isin(bad)
    labels = labels[~mask].drop_duplicates(["query_id", "product_id"]).copy()
    labels["grade"] = labels.label.map(cfg["wands_gains"])
    records = []
    for row in products.itertuples(index=False):
        item = row._asdict()
        raw_features = clean(item["product_features"])
        records.append({
            "product_id": int(item["product_id"]),
            "title": clean(item["product_name"]),
            "class": clean(item["product_class"]),
            "category": clean(item["_3"] if "_3" in item else item.get("category hierarchy", "")),
            "description": clean(item["product_description"]),
            "attributes": raw_features.split("|") if raw_features else [],
            "attribute_separator": "|",
            "section_order": ["title", "class", "category", "description", "attributes"],
        })
    queries = queries[["query_id", "query"]].copy()
    return records, queries, labels[["query_id", "product_id", "label", "grade"]]


def esci(cfg: dict) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    base = P2 / "data/esci_repo/shopping_queries_dataset"
    spec = cfg["datasets"]["esci"]
    examples = pd.read_parquet(
        base / "shopping_queries_dataset_examples.parquet",
        filters=[
            ("product_locale", "==", spec["locale"]),
            (spec["version_flag"], "==", 1),
            ("split", "==", spec["split"]),
        ],
    )
    query_ids = sorted(
        examples.query_id.unique(),
        key=lambda q: hashlib.sha256(f"{cfg['seed']}:{int(q)}".encode()).hexdigest(),
    )[: spec["query_sample_size"]]
    examples = examples[examples.query_id.isin(query_ids)].copy()
    assert not examples.duplicated(["query_id", "product_id"]).any()
    product_ids = sorted(examples.product_id.unique())
    dataset = ds.dataset(base / "shopping_queries_dataset_products.parquet", format="parquet")
    table = dataset.to_table(
        filter=(ds.field("product_locale") == spec["locale"]) & ds.field("product_id").isin(product_ids)
    )
    products = table.to_pandas().drop_duplicates("product_id").set_index("product_id")
    missing = set(product_ids) - set(products.index)
    assert not missing, f"Missing {len(missing)} selected products"
    records = []
    for product_id in product_ids:
        row = products.loc[product_id]
        attributes = []
        if clean(row.product_brand):
            attributes.append(clean(row.product_brand))
        if clean(row.product_color):
            attributes.append(clean(row.product_color))
        if clean(row.product_bullet_point):
            attributes.append(clean(row.product_bullet_point))
        records.append({
            "product_id": product_id,
            "title": clean(row.product_title),
            "class": "",
            "category": "",
            "description": clean(row.product_description),
            "attributes": attributes,
            "attribute_separator": "\n",
            "section_order": ["title", "description", "attributes"],
        })
    queries = examples[["query_id", "query"]].drop_duplicates().sort_values("query_id")
    labels = examples[["query_id", "product_id", "esci_label"]].copy()
    labels = labels.rename(columns={"esci_label": "label"})
    labels["grade"] = labels.label.map(cfg["esci_official_task1_gains"])
    return records, queries, labels


def write_dataset(name: str, records: list[dict], queries: pd.DataFrame,
                  labels: pd.DataFrame, cfg: dict) -> dict:
    processed = P2 / "data/processed"
    reps = P2 / "data/representations" / name
    audits = P2 / "results/equivalence"
    for folder in [processed, reps, audits]:
        folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(processed / f"{name}_products.jsonl.gz", "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    queries = queries.copy()
    queries["query_type"] = queries["query"].map(classify_query)
    queries.to_csv(processed / f"{name}_queries.csv", index=False)
    labels.to_csv(processed / f"{name}_judgments.csv", index=False)

    summaries = []
    for condition in cfg["representations"]:
        path = reps / f"{slug(condition)}.jsonl.gz"
        digest = hashlib.sha256()
        counts = {"products": 0, "passed": 0, "characters_equal": 0, "tokens_equal": 0}
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                original = build(record, "C0_original", cfg["attribute_permutation_seeds"])
                text = build(record, condition, cfg["attribute_permutation_seeds"])
                check = audit(record, condition, text, original)
                if not check["pass"]:
                    raise AssertionError((name, condition, record["product_id"], check))
                payload = json.dumps({"product_id": record["product_id"], "text": text}, ensure_ascii=False)
                handle.write(payload + "\n")
                digest.update((payload + "\n").encode())
                counts["products"] += 1
                counts["passed"] += int(check["pass"])
                counts["characters_equal"] += int(check["character_multiset_equal"] is True)
                counts["tokens_equal"] += int(check["token_multiset_equal"] is True)
        summaries.append({"dataset": name, "representation": condition, **counts,
                          "representation_sha256": digest.hexdigest()})
        print(name, condition, counts, flush=True)
    pd.DataFrame(summaries).to_csv(audits / f"{name}_equivalence_summary.csv", index=False)
    return {
        "dataset": name,
        "products": len(records),
        "queries": len(queries),
        "judgments": len(labels),
        "labels": labels.label.value_counts().to_dict(),
        "query_types": queries.query_type.value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wands", "esci", "all"], default="all")
    args = parser.parse_args()
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    outputs = []
    if args.dataset in {"wands", "all"}:
        outputs.append(write_dataset("wands", *wands(cfg), cfg))
    if args.dataset in {"esci", "all"}:
        outputs.append(write_dataset("esci", *esci(cfg), cfg))
    path = P2 / "results/dataset_statistics.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = {item["dataset"]: item for item in existing + outputs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(merged.values()), indent=2))


if __name__ == "__main__":
    main()
