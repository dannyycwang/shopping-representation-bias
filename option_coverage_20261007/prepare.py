"""Catalog-only inspection and extraction. Intentionally never reads rankings."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
from core import extract, normalize, selection_hash

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    spec = json.loads((HERE / "extraction_spec_v1.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / "data/raw/product.csv", sep="\t", keep_default_na=False).set_index("product_id")
    with gzip.open(ROOT / "phase2/data/processed/wands_products.jsonl.gz", "rt", encoding="utf-8") as f:
        products = list(map(json.loads, f))
    keys = collections.defaultdict(list)
    records, audit, coverage = [], [], []
    for product in products:
        pid = product["product_id"]
        original = raw.loc[pid, "product_features"]
        assert product["attributes"] == (original.split("|") if original else []), pid
        for entry in product["attributes"]:
            key, sep, val = entry.partition(":")
            if sep:
                keys[normalize(key)].append((pid, key, val))
        for field in spec["fields"]:
            records.append(dict(product_id=pid, field=field, spec_version=spec["version"],
                                **extract(product["attributes"], field, spec)))
    census = []
    for key, rows in sorted(keys.items()):
        census.append(dict(key=key, entries=len(rows), products=len({p for p,k,v in rows}),
                           raw_key_variants=json.dumps(sorted({k for p,k,v in rows})),
                           examples=json.dumps(collections.Counter(v for p,k,v in rows).most_common(5)),
                           mapped=key in spec["fields"]))
    pd.DataFrame(census).to_csv(out / "raw_key_census.csv", index=False)
    with gzip.open(out / "product_fields.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for field in spec["fields"]:
        rows = [x for x in records if x["field"] == field]
        counter = collections.Counter(x["status"] for x in rows)
        reasons = collections.Counter(v for x in rows for v in x["reasons"])
        coverage.append(dict(field=field, total_products=len(rows), raw_key_present=sum(bool(x["sources"]) for x in rows),
                             valid=counter["valid"], unknown=counter["unknown"], **reasons))
        available = sorted((x for x in rows if x["sources"]),
                           key=lambda x: selection_hash(f"20261007:attribute-audit:{field}:{x['product_id']}"))[:30]
        for row in available:
            pid = row["product_id"]
            original = raw.loc[pid, "product_features"]
            assert all(e["raw_entry"] == original.split("|")[e["entry_index"]] for e in row["sources"])
            audit.append(dict(**row, title=raw.loc[pid, "product_name"], product_class=raw.loc[pid, "product_class"],
                              original_product_features=original, source_match=True))
    pd.DataFrame(coverage).fillna(0).to_csv(out / "catalog_coverage.csv", index=False)
    with (out / "attribute_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump(out / "extraction_freeze.json", dict(spec_sha256=hashlib.sha256((HERE / "extraction_spec_v1.json").read_bytes()).hexdigest(),
         catalog_products=len(products), raw_processed_attribute_mismatches=0, audit_records=len(audit),
         ordering="Prepared before option-result computation; catalog-only audit"))
    print(pd.DataFrame(coverage).fillna(0).to_string(index=False))
    for row in audit:
        print(row["field"], row["product_id"], row["product_class"], row["status"], row["values"],
              [e["raw_entry"] for e in row["sources"]])


if __name__ == "__main__":
    main()
