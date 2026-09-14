import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2))
from src.evaluation import inverse, stable_order
from src.mitigations import audit_mitigation, build_mitigation
from src.representations import audit, build, tokens


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "product_id": "P1",
            "title": "Clear chair",
            "class": "Seating",
            "category": "Furniture",
            "description": "First fact. Second fact? Third fact!",
            "attributes": ["material:acrylic", "color:clear", "size:20 in"],
            "attribute_separator": "|",
            "section_order": ["title", "class", "category", "description", "attributes"],
        }
        self.seeds = [11, 12, 13, 14, 15]

    def test_plain_controls_are_exact_multisets(self):
        original = build(self.record, "C0_original", self.seeds)
        for condition in [
            "C1_reverse_attribute_order",
            "C2_random_attribute_order_s1",
            "C3_reverse_sentence_order",
            "C4_section_order",
        ]:
            text = build(self.record, condition, self.seeds)
            self.assertEqual(Counter(text), Counter(original))
            self.assertEqual(Counter(tokens(text)), Counter(tokens(original)))
            self.assertTrue(audit(self.record, condition, text, original)["pass"])

    def test_random_permutations_are_deterministic(self):
        first = build(self.record, "C2_random_attribute_order_s2", self.seeds)
        second = build(self.record, "C2_random_attribute_order_s2", self.seeds)
        self.assertEqual(first, second)

    def test_json_orders_decode_to_identical_values(self):
        ascending = build(self.record, "C5_json_key_order_ascending", self.seeds)
        descending = build(self.record, "C5_json_key_order_descending", self.seeds)
        self.assertNotEqual(ascending, descending)
        self.assertEqual(json.loads(ascending), json.loads(descending))
        self.assertTrue(audit(self.record, "C5_json_key_order_ascending", ascending,
                              build(self.record, "C0_original", self.seeds))["pass"])

    def test_stable_tie_break_and_inverse(self):
        scores = np.array([[0.5, 0.5, 0.2], [0.0, 1.0, 1.0]], dtype=np.float32)
        order = stable_order(scores)
        np.testing.assert_array_equal(order, [[0, 1, 2], [1, 2, 0]])
        np.testing.assert_array_equal(inverse(order), [[1, 2, 3], [3, 1, 2]])

    def test_mitigation_fidelity_and_canonicalization(self):
        m1 = build_mitigation(self.record, "M1_factual_field_sentence", "C1", self.seeds)
        self.assertTrue(audit_mitigation(self.record, m1)["pass"])
        m2_outputs = {
            build_mitigation(self.record, "M2_normalized_attributes", variant, self.seeds)
            for variant in ["C0", "C1", "C2s1", "C2s2", "C2s3", "C2s4", "C2s5"]
        }
        self.assertEqual(len(m2_outputs), 1)
        self.assertTrue(audit_mitigation(self.record, m2_outputs.pop())["pass"])


if __name__ == "__main__":
    unittest.main()
