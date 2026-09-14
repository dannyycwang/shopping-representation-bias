"""Compare efficient target insertion with a complete stable reranking."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_target_only_permutations import rank_target


def test_target_only_matches_brute_force():
    rng = np.random.default_rng(20260963)
    for size in [2, 3, 10, 100]:
        for _ in range(100):
            base = rng.integers(-3, 4, size=size).astype('f4')
            indices = np.arange(size)
            values = rng.integers(-4, 5, size=size).astype('f4')
            actual = rank_target(base, indices, values)
            expected = []
            for index, value in zip(indices, values):
                changed = base.copy()
                changed[index] = value
                expected.append(np.flatnonzero(np.argsort(-changed, kind='stable') == index)[0] + 1)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                rank_target(base, indices, base),
                np.argsort(np.argsort(-base, kind='stable')) + 1,
            )
