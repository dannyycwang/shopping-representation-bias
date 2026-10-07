import json
import unittest
from pathlib import Path
from core import extract, membership, normalize, options, validate_ranking

SPEC = json.loads((Path(__file__).parent / "extraction_spec_v1.json").read_text())


def records(mapping):
    return {k: dict(status="valid", values=v, reasons=[]) for k,v in mapping.items()}


class SetLogic(unittest.TestCase):
    def test_retained_supplier(self):
        r = records({"a": ["wood"], "b": ["wood"], "c": ["metal"]})
        self.assertEqual(options(["a","b"], ["b","c"], r)["lost_options"], [])

    def test_equal_counts_different_identities(self):
        r = records({"a": ["red"], "b": ["blue"]})
        x = options(["a"], ["b"], r)
        self.assertEqual(x["net_options"], 0)
        self.assertTrue(x["any_lost"] and x["any_gained"])

    def test_value_deduplication(self):
        r = records({"a": ["wood"], "b": ["wood"]})
        self.assertEqual(options(["a","b"], ["b"], r)["n_options_before"], 1)

    def test_missing_and_conflicts(self):
        for attrs in [[], ["color: red", "color: blue"], ["color: red", "color: unknown"]]:
            r = {"a": extract(attrs, "color", SPEC), "b": extract(["color: red"], "color", SPEC)}
            x = options(["a"], ["b"], r)
            self.assertFalse(x["eligible"])
            self.assertIsNone(x["lost_options"])

    def test_identical(self):
        m = membership(["a","b"], ["a","b"], ["a","b"])
        self.assertFalse(m["membership_changed"])
        x = options(m["before"], m["after"], records({"a":["red"],"b":["blue"]}))
        self.assertFalse(x["option_changed"])

    def test_swap_symmetry(self):
        r = records({"a":["wood"],"b":["metal"]})
        a,b = options(["a"],["b"],r), options(["b"],["a"],r)
        self.assertEqual(a["eligible"],b["eligible"])
        self.assertEqual(a["lost_options"],b["gained_options"])
        self.assertEqual(a["net_options"],-b["net_options"])
        r["a"] = extract([], "material", SPEC)
        self.assertEqual(options(["a"],["b"],r)["eligible"], options(["b"],["a"],r)["eligible"])

    def test_cancellation_independent_of_rank_order(self):
        m = membership(["a","x"], ["x","b"], ["a","b"])
        self.assertTrue(m["cancellation"])
        self.assertTrue(m["recall_equal"])
        self.assertFalse(membership(["a"],["a"],["a"])["cancellation"])

    def test_empty(self):
        x = options([], [], {})
        self.assertFalse(x["eligible"])
        self.assertEqual(x["exclusion_reasons"], ["empty_after", "empty_before"])

    def test_unknown_ids_and_rankings(self):
        self.assertIn("unknown_id", options(["z"],["z"],{})["exclusion_reasons"])
        for ranking in [["a"], ["a","a"], ["a","z"]]:
            with self.assertRaises(ValueError): validate_ranking(ranking, {"a","b"}, 2)

    def test_normalization_and_preserved_distinctions(self):
        self.assertEqual(normalize("  DARK\tGRAY "), "dark gray")
        self.assertNotEqual(normalize("wood"), normalize("engineered wood"))
        self.assertNotEqual(normalize("rectangle"), normalize("rectangular"))
        self.assertEqual(normalize("e\u0301"), "\u00e9")

    def test_scalar_delimiters_and_aliases(self):
        x = extract(["color: blue/white, red", "finish: black"], "color", SPEC)
        self.assertEqual(x["values"], ["blue/white, red"])
        self.assertEqual(extract(["framematerial: wood"], "material", SPEC)["status"], "unknown")

    def test_field_specific_none(self):
        self.assertEqual(extract(["color: none"], "color", SPEC)["values"], ["none"])
        self.assertEqual(extract(["material: none"], "material", SPEC)["status"], "unknown")

    def test_duplicate_identical_and_malformed(self):
        self.assertEqual(extract(["shape: round","shape: ROUND"],"shape",SPEC)["values"],["round"])
        self.assertIn("unparseable", extract(["shape"], "shape", SPEC)["reasons"])

    def test_cancellation_not_recomputed_after_metadata_filtering(self):
        m = membership(["a","c"], ["b"], ["a","b","c"])
        self.assertFalse(m["cancellation"])
        r=records({"a":["red"],"b":["blue"]})
        self.assertFalse(options(m["before"],m["after"],r)["eligible"])


if __name__ == "__main__": unittest.main()
