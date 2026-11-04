from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, OrderedDict


TOKEN_RE = re.compile(r"[a-z0-9]+")


def slug(condition: str) -> str:
    fixed = {
        "C0_original": "C0",
        "C1_reverse_attribute_order": "C1",
        "C3_reverse_sentence_order": "C3",
        "C4_section_order": "C4",
        "C5_json_key_order_ascending": "C5asc",
        "C5_json_key_order_descending": "C5desc",
    }
    if condition.startswith("C2_random_attribute_order_s"):
        return "C2s" + condition.rsplit("s", 1)[1]
    return fixed[condition]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _random_order(values: list[str], product_id: str, seed: int) -> list[str]:
    result = list(values)
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def _reverse_sentences_exact(text: str) -> str:
    if not text:
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    core = text.strip()
    parts = re.split(r"(?<=[.!?])(\s+)", core)
    sentences = parts[0::2]
    separators = parts[1::2]
    if len(sentences) < 2:
        return text
    reversed_sentences = list(reversed(sentences))
    rebuilt = reversed_sentences[0]
    for separator, sentence in zip(separators, reversed_sentences[1:]):
        rebuilt += separator + sentence
    result = leading + rebuilt + trailing
    assert Counter(result) == Counter(text)
    return result


def _blocks(record: dict, attributes: list[str] | None = None,
            description: str | None = None) -> list[tuple[str, str]]:
    attrs = record["attributes"] if attributes is None else attributes
    desc = record["description"] if description is None else description
    values = {
        "title": record["title"],
        "class": record.get("class", ""),
        "category": record.get("category", ""),
        "description": desc,
        "attributes": record["attribute_separator"].join(attrs),
    }
    return [(key, values[key]) for key in record["section_order"] if values.get(key, "")]


def _plain(record: dict, attributes: list[str] | None = None,
           description: str | None = None, reverse_sections: bool = False) -> str:
    blocks = _blocks(record, attributes, description)
    if reverse_sections:
        blocks.reverse()
    return "\n".join(value for _, value in blocks)


def build(record: dict, condition: str, permutation_seeds: list[int]) -> str:
    if condition == "C0_original":
        return _plain(record)
    if condition == "C1_reverse_attribute_order":
        return _plain(record, list(reversed(record["attributes"])))
    if condition.startswith("C2_random_attribute_order_s"):
        index = int(condition.rsplit("s", 1)[1]) - 1
        return _plain(
            record,
            _random_order(record["attributes"], str(record["product_id"]), permutation_seeds[index]),
        )
    if condition == "C3_reverse_sentence_order":
        return _plain(record, description=_reverse_sentences_exact(record["description"]))
    if condition == "C4_section_order":
        return _plain(record, reverse_sections=True)
    if condition.startswith("C5_json_key_order_"):
        values = {
            "title": record["title"],
            "class": record.get("class", ""),
            "category": record.get("category", ""),
            "description": record["description"],
            "attributes": record["attributes"],
        }
        keys = sorted(values, reverse=condition.endswith("descending"))
        payload = OrderedDict((key, values[key]) for key in keys)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raise KeyError(condition)


def source_atoms(record: dict) -> list[str]:
    scalar = [
        record["title"],
        record.get("class", ""),
        record.get("category", ""),
        record["description"],
    ]
    return [value for value in scalar + list(record["attributes"]) if value]


def audit(record: dict, condition: str, text: str, original: str) -> dict:
    atoms = source_atoms(record)
    missing = sum(value not in text for value in atoms)
    original_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", " ".join(atoms)))
    text_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", text))
    if condition.startswith("C5_"):
        payload = json.loads(text)
        values_ok = (
            payload["title"] == record["title"]
            and payload["class"] == record.get("class", "")
            and payload["category"] == record.get("category", "")
            and payload["description"] == record["description"]
            and payload["attributes"] == record["attributes"]
        )
        character_multiset_equal = None
        token_multiset_equal = None
    else:
        character_multiset_equal = Counter(text) == Counter(original)
        token_multiset_equal = Counter(tokens(text)) == Counter(tokens(original))
        # C3 intentionally changes the contiguous description string by moving
        # complete sentences; exact character and token multisets are its
        # stronger losslessness check. Other plain variants retain every atom
        # as an exact substring in addition to the multiset checks.
        values_ok = character_multiset_equal and (
            condition == "C3_reverse_sentence_order" or missing == 0
        )
    return {
        "product_id": record["product_id"],
        "representation": condition,
        "source_atoms": len(atoms),
        "missing_atoms": int(missing),
        "values_exact": bool(values_ok),
        "numeric_multiset_equal": original_numbers == text_numbers,
        "character_multiset_equal": character_multiset_equal,
        "token_multiset_equal": token_multiset_equal,
        "pass": bool(values_ok and original_numbers == text_numbers and
                     (character_multiset_equal is not False) and
                     (token_multiset_equal is not False)),
    }


STYLE_TERMS = {
    "bohemian", "coastal", "contemporary", "farmhouse", "glam", "industrial",
    "modern", "rustic", "scandinavian", "traditional", "transitional", "vintage",
}
ATTRIBUTE_TERMS = {
    "black", "blue", "brown", "clear", "gold", "gray", "green", "grey", "red",
    "silver", "white", "wood", "wooden", "metal", "leather", "plastic", "acrylic",
    "small", "large", "tall", "short", "round", "square", "queen", "king", "twin",
}
PRODUCT_TERMS = {
    "bed", "bench", "bookcase", "bottle", "cabinet", "chair", "charger", "curtain",
    "desk", "dresser", "fan", "lamp", "light", "mattress", "mirror", "phone", "rug",
    "shelf", "shoes", "sofa", "stand", "storage", "table", "toy", "watch",
}


def classify_query(query: str) -> str:
    words = tokens(query)
    wordset = set(words)
    if wordset & STYLE_TERMS:
        return "style"
    if len(words) >= 8:
        return "long_descriptive"
    if len(words) >= 5:
        return "multi_constraint"
    if wordset & ATTRIBUTE_TERMS or any(any(ch.isdigit() for ch in word) for word in words):
        return "attribute_constraint"
    if len(words) <= 2 and not (wordset & PRODUCT_TERMS):
        return "brand_entity_proxy"
    return "product_type"
