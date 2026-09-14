import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_metrics.py"
SPEC = importlib.util.spec_from_file_location("audit_metrics", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_independent_wands_minilm_recomputation_and_lambda_zero():
    per_query, aggregate = MOD.independently_recompute("wands", "minilm")
    assert len(per_query) == 480
    assert int(per_query.has_highest.sum()) == 379
    assert np.isclose(aggregate.iloc[0]["cNDCG@10"], 0.7792599092618921, atol=1e-12)
    assert np.isclose(aggregate.iloc[0]["SAR(lambda=0)"], aggregate.iloc[0]["cNDCG@10"], atol=1e-15)


def test_micro_and_macro_are_distinct_estimands():
    _, aggregate = MOD.independently_recompute("wands", "minilm")
    row = aggregate.iloc[0]
    assert np.isclose(row["VI@20_micro"], 0.1326266195524146, atol=1e-12)
    assert np.isclose(row["VI@20_query_macro"], 0.1916757414132084, atol=1e-12)
    assert not np.isclose(row["VI@20_micro"], row["VI@20_query_macro"])
