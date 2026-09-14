"""Regression checks of inherited scientific definitions used in Phase VI."""
import numpy as np
import pandas as pd

from run import STEMS, prior


def test_pair_first_query_macro_and_worst_schedule_order():
    # Unequal relevant-product counts distinguish macro from micro averaging.
    frame = pd.DataFrame({"query_id": [1, 2, 2], "product_id": ["a", "b", "c"]})
    for i, schedule in enumerate(STEMS):
        frame[schedule] = [1 if i < 3 else 101, 1, 101]
    pair, query = prior.metric_frame(frame)
    assert pair["VI@100"].tolist() == [True, False, False]
    assert pair["Robust@100"].tolist() == [False, True, False]
    assert pair["Never@100"].tolist() == [False, False, True]
    assert np.isclose(query["Recall@100"].mean(), (3 / 7 + .5) / 2)
    assert np.isclose(query["VI@100"].mean(), .5)
    assert np.isclose(query["worst_schedule@100"].mean(), .25)
    assert np.allclose(query["VI@100"] + query["Robust@100"] + query["Never@100"], 1)


def test_stable_ties_and_target_reinsertion_match_brute_force():
    base = np.array([.8, .8, .5, .5, -.1], dtype="f4")
    assert prior.ranks(base[None])[1].tolist() == [[1, 2, 3, 4, 5]]
    indices = np.array([0, 1, 2, 3, 4])
    values = np.array([.5, .9, .8, .5, .8], dtype="f4")
    target = prior.rank_target(base, indices, values)
    brute = []
    for i, value in zip(indices, values):
        copy = base.copy()
        copy[i] = value
        brute.append(prior.ranks(copy[None])[1][0, i])
    assert target.tolist() == brute


def test_query_paired_bootstrap_is_deterministic_and_keeps_pairing():
    a = pd.Series([.8, .2, .6], index=[3, 1, 2])
    b = pd.Series([.1, .5, .7], index=[1, 2, 3])
    difference = a - b.loc[a.index]
    mean, lo, hi = prior.bootstrap(difference, samples=10000)
    assert np.allclose([mean, lo, hi], [.1, .1, .1])
