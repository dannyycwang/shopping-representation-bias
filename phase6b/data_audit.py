"""Freeze judged-only supervision without changing any historical experiment."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase6"))
import run as p6


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def audit_examples(examples, labels):
    positives = [e["positive"] for e in examples]
    negatives = [p for e in examples for p in e["negatives"]]
    qcounts = Counter(e["query_id"] for e in examples)
    occurrences = {}
    for e in examples:
        qid = e["query_id"]
        assert labels.loc[qid, e["positive"]] == "Exact"
        assert len(e["negatives"]) == len(set(e["negatives"]))
        assert e["positive"] not in e["negatives"]
        for p in e["negatives"]:
            assert labels.loc[qid, p] == "Irrelevant"
        for p in [e["positive"]] + e["negatives"]:
            occurrences.setdefault(p, set()).add(qid)
    return dict(examples=len(examples), unique_queries=len(qcounts), unique_positive_products=len(set(positives)),
                unique_negative_products=len(set(negatives)), negative_presentations=len(negatives),
                negatives_per_example=dict(min=min(len(e["negatives"]) for e in examples),
                                           mean=float(np.mean([len(e["negatives"]) for e in examples])),
                                           max=max(len(e["negatives"]) for e in examples)),
                examples_per_query={str(k): v for k, v in sorted(qcounts.items())},
                repeated_products_across_queries=sum(len(q) > 1 for q in occurrences.values()),
                label_presentations=dict(Exact=len(positives), Irrelevant=len(negatives), Partial=0, unjudged=0))


def main():
    destination = HERE / "PHASE6B_TRAINING_AUDIT.json"
    if destination.exists():
        print(destination.read_text())
        return
    old_manifest = json.loads((ROOT / "FINAL_DELIVERY_MANIFEST.json").read_text(encoding="utf8"))
    for entry in old_manifest["outputs"] + old_manifest["reused_sources"]:
        assert p6.sha(ROOT / entry["path"]) == entry["sha256"]
    protected = {e["path"]: e for e in old_manifest["outputs"] + old_manifest["reused_sources"]}
    protected["FINAL_DELIVERY_MANIFEST.json"] = dict(path="FINAL_DELIVERY_MANIFEST.json", sha256=p6.sha(ROOT / "FINAL_DELIVERY_MANIFEST.json"))
    p6.dump(HERE / "historical_sources.json", list(protected.values()))
    cfg = json.loads((ROOT / "phase5_mitigation/config.json").read_text())
    judgments = pd.read_csv(ROOT / "phase2/data/processed/wands_judgments.csv", dtype={"product_id": str})
    train = judgments[judgments.query_id.isin(cfg["train"])].copy()
    assert not set(train.query_id) & (set(cfg["validation"]) | set(cfg["test"]))
    labels = train.set_index(["query_id", "product_id"]).label
    original = json.loads((ROOT / "phase5_mitigation/triples.json").read_text())
    reconstructed = []
    for qid in cfg["train"]:
        group = train[train.query_id.eq(qid)]
        pos = sorted(group[group.label.eq("Exact")].product_id, key=lambda p: key(f"positive42:{qid}:{p}"))[:16]
        neg = sorted(group[group.label.eq("Irrelevant")].product_id, key=lambda p: key(f"negative42:{qid}:{p}"))
        if neg:
            reconstructed += [dict(query_id=int(qid), positive=p, negative=neg[i % len(neg)]) for i, p in enumerate(pos)]
    assert original == reconstructed and len(original) == 546
    t0 = [dict(query_id=e["query_id"], positive=e["positive"], negatives=[e["negative"]]) for e in original]
    old_neg = {(e["query_id"], e["positive"]): e["negative"] for e in original}
    expanded, excluded = [], []
    for qid, group in train.groupby("query_id"):
        neg = sorted(group[group.label.eq("Irrelevant")].product_id)
        pos = sorted(group[group.label.eq("Exact")].product_id)
        if not neg:
            excluded += [dict(query_id=int(qid), product_id=p, reason="no explicitly Irrelevant training judgments for this query") for p in pos]
            continue
        for pid in pos:
            pool = sorted(neg, key=lambda n: key(f"phase6b-negative42:{qid}:{pid}:{n}"))
            old = old_neg.get((qid, pid))
            if old:
                pool.remove(old)
                pool.insert(0, old)
            expanded.append(dict(query_id=int(qid), positive=pid, negatives=pool[:8]))
    expanded.sort(key=lambda e: key(f'phase6b-scale42:{e["query_id"]}:{e["positive"]}'))
    datasets = {"T0": t0, "T1": [dict(e, negatives=e["negatives"][:1]) for e in expanded], "T2": expanded}
    for fraction in [.25, .5]:
        datasets[f"T2_f{fraction:g}"] = expanded[:max(1, int(len(expanded) * fraction))]
    for name, examples in datasets.items():
        p6.dump(HERE / "data" / f"{name}.json", examples)
    pd.DataFrame(excluded).to_csv(HERE / "data/excluded_positives.csv", index=False)
    audit = dict(frozen_utc=p6.now(), train_query_ids=cfg["train"], train_queries=72, judged_pairs=len(train),
                 label_distribution={label: dict(judged_pairs=len(g), unique_products=g.product_id.nunique(), queries=g.query_id.nunique())
                                     for label, g in train.groupby("label")},
                 original_546_source="phase5_mitigation/screen.py freeze: at most 16 Exact products/query in SHA256 positive42 order; one explicit Irrelevant in cyclic negative42 hash order; skip queries with no negatives",
                 original_triples_exactly_reconstructed=True, excluded_positive_pairs=len(excluded),
                 excluded_queries=sorted({e["query_id"] for e in excluded}),
                 no_dev_or_test_training=True, partial_products_used=False, unjudged_negatives_used=False,
                 T1_contains_all_T0_pairs_and_their_original_negative=True,
                 scale_order="global SHA256(phase6b-scale42:query_id:positive_id); nested prefixes; floor(fraction*N)",
                 cap="No positive cap after excluding queries with no explicit Irrelevant judgments; at most 8 distinct explicit negatives per pair",
                 datasets={name: audit_examples(examples, labels) for name, examples in datasets.items()})
    t1 = {(e["query_id"], e["positive"], e["negatives"][0]) for e in datasets["T1"]}
    assert all((e["query_id"], e["positive"], e["negatives"][0]) in t1 for e in t0)
    p6.dump(destination, audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"train_query_ids", "datasets"}}, indent=2))
    print(json.dumps({name: {k: v for k, v in value.items() if k != "examples_per_query"} for name, value in audit["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
