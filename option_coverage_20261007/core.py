"""Pure set logic and conservative extraction; no retrieval or model imports."""
import hashlib
import unicodedata


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def extract(attributes, field, spec):
    entries, reasons = [], set()
    for i, entry in enumerate(attributes):
        key, colon, value = entry.partition(":")
        if normalize(key) not in spec["fields"][field]:
            continue
        entries.append(dict(entry_index=i, raw_entry=entry, raw_key=key,
                            raw_value=value, normalized_value=normalize(value)))
        if not colon:
            reasons.add("unparseable")
    values = {e["normalized_value"] for e in entries}
    unknown = set(spec["missing_sentinels"]) | set(spec["field_specific_unknown"][field])
    if not entries:
        reasons.add("missing_key")
    if values & unknown:
        reasons.add("unknown_value")
    if len(values) > 1:
        reasons.add("conflicting_values")
    return dict(status="valid" if not reasons else "unknown", reasons=sorted(reasons),
                values=sorted(values) if not reasons else [], sources=entries)


def validate_ranking(ranking, catalog, k):
    if len(ranking) != k:
        raise ValueError(f"Incomplete ranking: {len(ranking)} != {k}; no padding")
    if len(set(ranking)) != k:
        raise ValueError("Duplicate product IDs in ranking")
    unknown = set(ranking) - set(catalog)
    if unknown:
        raise ValueError(f"Unknown catalog IDs: {sorted(unknown)}")


def membership(before, after, highest):
    a, b = set(before) & set(highest), set(after) & set(highest)
    lost, gained = a - b, b - a
    assert len(b) - len(a) == len(gained) - len(lost)
    return dict(before=sorted(a), after=sorted(b), lost_products=sorted(lost),
                gained_products=sorted(gained), n_before=len(a), n_after=len(b),
                n_lost_products=len(lost), n_gained_products=len(gained),
                membership_changed=bool(lost or gained),
                cancellation=len(lost) == len(gained) > 0,
                recall_equal=len(a) == len(b))


def options(before, after, records):
    a, b = set(before), set(after)
    unknown_ids = (a | b) - records.keys()
    reasons = set()
    if not a:
        reasons.add("empty_before")
    if not b:
        reasons.add("empty_after")
    bad = {}
    for pid in sorted(a | b):
        if pid in unknown_ids:
            bad[pid] = ["unknown_id"]
        elif records[pid]["status"] != "valid":
            bad[pid] = records[pid]["reasons"]
        if pid in bad:
            reasons.update(bad[pid])
    known_a = a - bad.keys()
    known_b = b - bad.keys()
    oa = set().union(*(set(records[p]["values"]) for p in known_a))
    ob = set().union(*(set(records[p]["values"]) for p in known_b))
    eligible = not reasons
    row = dict(eligible=eligible, exclusion_reasons=sorted(reasons),
               invalid_products=bad, known_before=len(known_a), known_after=len(known_b),
               missing_before=len(a)-len(known_a), missing_after=len(b)-len(known_b),
               known_fraction_before=len(known_a)/len(a) if a else None,
               known_fraction_after=len(known_b)/len(b) if b else None,
               observed_before=sorted(oa), observed_after=sorted(ob))
    # Observed values are coverage diagnostics, never verified loss on incomplete sets.
    row.update({k: None for k in ["options_before", "options_after", "lost_options", "gained_options",
               "option_changed", "any_lost", "any_gained", "n_lost", "n_gained", "n_options_before",
               "n_options_after", "net_options", "unchanged_options"]})
    if eligible:
        lost, gained = oa-ob, ob-oa
        assert len(ob)-len(oa) == len(gained)-len(lost)
        row.update(options_before=sorted(oa), options_after=sorted(ob),
                   lost_options=sorted(lost), gained_options=sorted(gained),
                   option_changed=bool(lost or gained), unchanged_options=oa==ob,
                   any_lost=bool(lost), any_gained=bool(gained), n_lost=len(lost), n_gained=len(gained),
                   n_options_before=len(oa), n_options_after=len(ob), net_options=len(ob)-len(oa))
    return row


def selection_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
