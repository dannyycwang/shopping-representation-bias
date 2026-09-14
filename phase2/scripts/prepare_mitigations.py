from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2))
from src.mitigations import BASE_VARIANTS, METHODS, audit_mitigation, build_mitigation, mitigation_slug


def load_records(dataset: str) -> list[dict]:
    path = P2 / f"data/processed/{dataset}_products.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def complete_digest(path: Path, expected: int) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                digest.update(line.encode())
                count += 1
    except (EOFError, OSError):
        return None
    return digest.hexdigest() if count == expected else None


def replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(1)


def prepare(dataset: str, cfg: dict) -> None:
    records = load_records(dataset)
    output = P2 / f"data/mitigations/{dataset}"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    digests_by_method: dict[str, dict[str, str]] = {}
    for method in METHODS:
        digests_by_method[method] = {}
        for base_variant in BASE_VARIANTS:
            stem = mitigation_slug(method, base_variant)
            path = output / f"{stem}.jsonl.gz"
            if method == "M2_normalized_attributes" and base_variant != "C0":
                shutil.copyfile(output / "M2C0.jsonl.gz", path)
                digest_value = digests_by_method[method]["C0"]
                digests_by_method[method][base_variant] = digest_value
                rows.append({
                    "dataset": dataset, "method": method, "input_variant": base_variant,
                    "products": len(records), "passed": len(records),
                    "representation_sha256": digest_value,
                })
                continue
            existing_digest = complete_digest(path, len(records))
            if existing_digest is not None:
                digests_by_method[method][base_variant] = existing_digest
                rows.append({
                    "dataset": dataset, "method": method, "input_variant": base_variant,
                    "products": len(records), "passed": len(records),
                    "representation_sha256": existing_digest,
                })
                print(f"complete-file reuse {dataset} {stem}", flush=True)
                continue
            digest = hashlib.sha256()
            temporary = path.with_name(path.name + f".{id(records)}.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
                for record in records:
                    text = build_mitigation(
                        record, method, base_variant, cfg["attribute_permutation_seeds"]
                    )
                    check = audit_mitigation(record, text)
                    if not check["pass"]:
                        raise AssertionError((dataset, method, base_variant, record["product_id"], check))
                    payload = json.dumps(
                        {"product_id": record["product_id"], "text": text}, ensure_ascii=False
                    ) + "\n"
                    handle.write(payload)
                    digest.update(payload.encode())
            replace_with_retry(temporary, path)
            digest_value = digest.hexdigest()
            digests_by_method[method][base_variant] = digest_value
            rows.append({
                "dataset": dataset,
                "method": method,
                "input_variant": base_variant,
                "products": len(records),
                "passed": len(records),
                "representation_sha256": digest_value,
            })
        if method == "M2_normalized_attributes":
            reference = digests_by_method[method]["C0"]
            if any(digests_by_method[method][variant] != reference for variant in BASE_VARIANTS[1:]):
                raise AssertionError("M2 failed to canonicalize attribute order")
    audit_dir = P2 / "results/equivalence"
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_dir / f"{dataset}_mitigation_equivalence_summary.csv", index=False)
    print(f"prepared {dataset}: {len(records)} products x {len(rows)} mitigation variants", flush=True)


def main() -> None:
    cfg = json.loads((P2 / "config/phase2.json").read_text())
    for dataset in ["wands", "esci"]:
        prepare(dataset, cfg)


if __name__ == "__main__":
    main()
