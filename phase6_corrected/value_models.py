"""Context determines scalar weights; original frozen BGE atoms are the only values."""
from pathlib import Path
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase6'))
from models import AttentionBlock, field_key


class ValueAttention(nn.Module):
    def __init__(self, method, fields, hidden=128, dim=768):
        super().__init__()
        assert method in {'C1', 'C2', 'C3', 'C4'}
        self.method = method
        self.project = nn.Linear(dim, hidden)
        if method == 'C1':
            self.phi = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.scorer = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        elif method in {'C2', 'C3'}:
            self.context = AttentionBlock(hidden, heads=4)
            self.scorer = nn.Linear(hidden, 1)
        else:
            self.phi = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.scorer = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        last = self.scorer[-1] if isinstance(self.scorer, nn.Sequential) else self.scorer
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Create identity parameters last to match common C2/C3 initial weights.
        if method == 'C3':
            self.field_embedding = nn.Embedding(fields, hidden)
            nn.init.normal_(self.field_embedding.weight, std=.02)
        if method == 'C4':
            self.theta = nn.Parameter(torch.arange(fields, dtype=torch.float32) * (2 * math.pi / fields))

    def components(self, atoms, mask, non_attribute, field_ids):
        x = self.project(atoms)
        if self.method in {'C2', 'C3'}:
            if self.method == 'C3':
                x = x + self.field_embedding(field_ids)
            safe = mask.clone()
            safe[~mask.any(1), 0] = True
            x = x * mask.unsqueeze(-1)
            context = self.context(x, x, safe)
            scores = self.scorer(context).squeeze(-1)
        else:
            if self.method == 'C4':
                angle = self.theta[field_ids]
                x = torch.cat([x, angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)], dim=-1)
            u = self.phi(x)
            c = (u * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            local = atoms if self.method == 'C1' else u
            scores = self.scorer(torch.cat([local, c[:, None, :].expand(-1, atoms.shape[1], -1)], dim=-1)).squeeze(-1)
        weights = scores.masked_fill(~mask, -1e9).softmax(1) * mask
        # No contextual state, field embedding, or learned residual reaches this value path.
        attribute = F.normalize((weights.unsqueeze(-1) * atoms).sum(1), dim=1)
        final = F.normalize(.5 * non_attribute + .5 * attribute, dim=1)
        return final, weights, attribute

    def forward(self, atoms, mask, non_attribute, field_ids):
        return self.components(atoms, mask, non_attribute, field_ids)[0]
