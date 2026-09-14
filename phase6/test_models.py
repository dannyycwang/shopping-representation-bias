import itertools

import pytest
import torch
from torch.nn import functional as F

from models import StructuralAggregator, field_key


@pytest.mark.parametrize("method", ["S1", "S2", "S3", "S4"])
def test_arbitrary_permutations_padding_empty_and_duplicate_multiplicity(method):
    torch.manual_seed(42)
    model = StructuralAggregator(method, 5, hidden=16, dim=24).eval()
    a = torch.randn(2, 4, 24)
    a[0, 2] = a[0, 0]  # Duplicate atom; the same field ID follows it.
    ids = torch.tensor([[1, 2, 1, 0], [0, 0, 0, 0]])
    mask = torch.tensor([[True, True, True, False], [False] * 4])
    non = F.normalize(torch.randn(2, 24), dim=1)
    with torch.no_grad():
        base = model(a, mask, non, ids)
        assert torch.isfinite(base).all()
        torch.testing.assert_close(base[1], non[1])
        for order in itertools.permutations(range(4)):
            ix = list(order)
            out = model(a[:, ix], mask[:, ix], non, ids[:, ix])
            torch.testing.assert_close(base, out, atol=1e-6, rtol=1e-6)
        a[:, 3] = 10000
        torch.testing.assert_close(base, model(a, mask, non, ids))
        shorter = mask.clone()
        shorter[0, 2] = False
        assert not torch.allclose(base[0], model(a, shorter, non, ids)[0], atol=1e-6)


def test_field_keys_are_schema_not_value_or_position():
    assert field_key(" color : red:blue") == "color"
    assert field_key("color:green") == "color"
    assert field_key(" : unknown") is None
    assert field_key("no schema") is None


@pytest.mark.parametrize("method", ["S1", "S2", "S3", "S4"])
def test_trainable_paths_and_semantic_content_sensitivity(method):
    torch.manual_seed(43)
    model = StructuralAggregator(method, 5, hidden=16, dim=24)
    a = torch.randn(2, 3, 24, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    ids = torch.tensor([[0, 2, 2], [1, 3, 4]])
    non = F.normalize(torch.randn(2, 24), dim=1)
    query = torch.randn(2, 24)
    (model(a, mask, non, ids) * query).sum().backward()
    assert a.grad.abs().sum() > 0
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
    if method in {"S1", "S2"}:
        assert model.blocks[0].attention.in_proj_weight.grad.abs().sum() > 0


def test_s1_s2_common_initialization_matches():
    torch.manual_seed(42)
    a = StructuralAggregator("S1", 5)
    torch.manual_seed(42)
    b = StructuralAggregator("S2", 5)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
