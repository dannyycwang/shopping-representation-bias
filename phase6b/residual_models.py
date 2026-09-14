"""Reuse exactly the Phase VI S1/S4 branches; preserve frozen Set-Attention values."""
from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase6"))
from models import StructuralAggregator


class ResidualAggregator(nn.Module):
    def __init__(self, architecture, fields, base_state, hidden=128, dim=768):
        super().__init__()
        assert architecture in {"DeepSets", "SetTransformer"}
        self.branch = StructuralAggregator("S4" if architecture == "DeepSets" else "S1", fields, hidden=hidden, dim=dim)
        # The existing module performs its original attribute operations. Calling its
        # output module via a hook exposes the unnormalized 768-D result without
        # duplicating/changing attention, pooling, field inventory or hidden layers.
        self.base = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        if base_state is not None:
            self.base.load_state_dict({k.removeprefix("att."): v for k, v in base_state.items()})
        self.base.requires_grad_(False)
        self.beta = nn.Parameter(torch.tensor(math.log(.05 / .95)))
        last = self.branch.output[-1] if isinstance(self.branch.output, nn.Sequential) else self.branch.output
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def components(self, atoms, mask, non_attribute, field_ids):
        with torch.no_grad():
            weights = self.base(atoms).squeeze(-1).masked_fill(~mask, -1e9).softmax(1) * mask
            base_attr = F.normalize((atoms * weights.unsqueeze(-1)).sum(1), dim=1)
            base_final = F.normalize(.5 * non_attribute + .5 * base_attr, dim=1)
        captured = []
        handle = self.branch.output.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            # Only the branch's raw output is used; its old normalization/mix is discarded.
            self.branch(atoms, mask, non_attribute, field_ids)
        finally:
            handle.remove()
        delta = captured[0] * mask.any(1).unsqueeze(-1)
        alpha = self.beta.sigmoid()
        attribute = F.normalize(base_attr + alpha * delta, dim=1)
        final = F.normalize(.5 * non_attribute + .5 * attribute, dim=1)
        return final, base_final, base_attr, delta, alpha

    def forward(self, atoms, mask, non_attribute, field_ids):
        return self.components(atoms, mask, non_attribute, field_ids)[0]
