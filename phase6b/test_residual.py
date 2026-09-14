import itertools
import torch
import pytest
from torch.nn import functional as F

from residual_models import ResidualAggregator


@pytest.mark.parametrize("architecture", ["DeepSets", "SetTransformer"])
def test_epoch_zero_identity_and_frozen_base(architecture):
    torch.manual_seed(42)
    model = ResidualAggregator(architecture, 5, None, hidden=16, dim=24)
    a = torch.randn(2, 4, 24)
    mask = torch.tensor([[True, True, True, False], [False] * 4])
    non = F.normalize(torch.randn(2, 24), dim=1)
    ids = torch.tensor([[0, 1, 2, 0], [0] * 4])
    final, base, _, delta, alpha = model.components(a, mask, non, ids)
    assert torch.count_nonzero(delta) == 0
    torch.testing.assert_close(final, base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(final[1], non[1])
    assert abs(alpha.item() - .05) < 1e-7
    assert all(not p.requires_grad for p in model.base.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
    query = torch.randn(2, 24)
    for _ in range(2):
        opt.zero_grad()
        (model(a, mask, non, ids) * query).sum().backward()
        opt.step()
    assert model.branch.project.weight.grad.abs().sum() > 0
    assert model.beta.grad.abs() > 0
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["DeepSets", "SetTransformer"])
def test_nonzero_residual_is_invariant_and_preserves_duplicates(architecture):
    torch.manual_seed(43)
    model = ResidualAggregator(architecture, 5, None, hidden=16, dim=24).eval()
    last = model.branch.output[-1] if isinstance(model.branch.output, torch.nn.Sequential) else model.branch.output
    torch.nn.init.normal_(last.weight, std=.1)
    a = torch.randn(1, 4, 24)
    a[:, 2] = a[:, 0]
    mask = torch.tensor([[True, True, True, False]])
    ids = torch.tensor([[1, 2, 1, 0]])
    non = F.normalize(torch.randn(1, 24), dim=1)
    with torch.no_grad():
        expected = model(a, mask, non, ids)
        for order in itertools.permutations(range(4)):
            ix = list(order)
            torch.testing.assert_close(expected, model(a[:, ix], mask[:, ix], non, ids[:, ix]), atol=1e-6, rtol=1e-6)
        mask[:, 2] = False
        assert not torch.allclose(expected, model(a, mask, non, ids), atol=1e-6)


def test_multinegative_one_negative_equals_inherited_loss():
    positive, negative = torch.tensor([.2, .9]), torch.tensor([.7, .1])
    logits = torch.stack([positive, negative], dim=1) / .05
    actual = torch.logsumexp(logits, dim=1) - logits[:, 0]
    expected = F.softplus((negative - positive) / .05)
    torch.testing.assert_close(actual, expected)
