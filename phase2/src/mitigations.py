from __future__ import annotations

import re
from collections import Counter

from src.representations import _random_order


BASE_VARIANTS = ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]
METHODS = ["M1_factual_field_sentence", "M2_normalized_attributes"]


def mitigation_slug(method: str, base_variant: str) -> str:
    prefix = "M1" if method.startswith("M1_") else "M2"
    if method not in METHODS or base_variant not in BASE_VARIANTS:
        raise KeyError((method, base_variant))
    return f"{prefix}{base_variant}"


def ordered_attributes(record: dict, base_variant: str, seeds: list[int]) -> list[str]:
    attributes = list(record["attributes"])
    if base_variant == "C0":
        return attributes
    if base_variant == "C1":
        return list(reversed(attributes))
    if base_variant.startswith("C2s"):
        index = int(base_variant[3:]) - 1
        return _random_order(attributes, str(record["product_id"]), seeds[index])
    raise KeyError(base_variant)


def build_mitigation(record: dict, method: str, base_variant: str,
                     seeds: list[int]) -> str:
    attributes = ordered_attributes(record, base_variant, seeds)
    if method == "M1_factual_field_sentence":
        lines = []
        if record["title"]:
            lines.append(f'The product name is {record["title"]}.')
        if record.get("class", ""):
            lines.append(f'The product class is {record["class"]}.')
        if record.get("category", ""):
            lines.append(f'The category hierarchy is {record["category"]}.')
        if record["description"]:
            lines.append(f'The source description is: {record["description"]}')
        if attributes:
            lines.append("The listed product attributes follow. " + ". ".join(attributes))
        return "\n".join(lines)
    if method == "M2_normalized_attributes":
        # Canonicalization deliberately removes the incoming attribute-order
        # degree of freedom while retaining duplicates and raw values.
        attributes = sorted(attributes, key=lambda value: (value.casefold(), value))
        lines = []
        if record["title"]:
            lines.append(f'product name: {record["title"]}')
        if record.get("class", ""):
            lines.append(f'product class: {record["class"]}')
        if record.get("category", ""):
            lines.append(f'category hierarchy: {record["category"]}')
        if attributes:
            lines.extend(["attributes:", *attributes])
        if record["description"]:
            lines.append(f'source description: {record["description"]}')
        return "\n".join(lines)
    raise KeyError(method)


def audit_mitigation(record: dict, text: str) -> dict:
    atoms = [
        record["title"], record.get("class", ""), record.get("category", ""),
        record["description"], *record["attributes"],
    ]
    atoms = [value for value in atoms if value]
    numbers = Counter(re.findall(r"\d+(?:\.\d+)?", " ".join(atoms)))
    output_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", text))
    missing = sum(value not in text for value in atoms)
    return {
        "source_atoms": len(atoms),
        "missing_atoms": int(missing),
        "numeric_multiset_equal": numbers == output_numbers,
        "pass": bool(missing == 0 and numbers == output_numbers),
    }
