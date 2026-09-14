"""Frozen-field structural aggregators; no serialization positions or encoder weights."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def field_key(atom):
    """A native key, not the value or incoming occurrence index. Empty/malformed -> UNK."""
    key, separator, _ = atom.partition(":")
    return key.strip() if separator and key.strip() else None


class AttentionBlock(nn.Module):
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, query, values, safe_mask):
        attended, _ = self.attention(query, values, values, key_padding_mask=~safe_mask,
                                     need_weights=False)
        x = self.norm1(query + attended)
        return self.norm2(x + self.ff(x))


class StructuralAggregator(nn.Module):
    def __init__(self, method, fields, hidden=128, blocks=1, dim=768):
        super().__init__()
        assert method in {"S1", "S2", "S3", "S4"}
        self.method = method
        self.project = nn.Linear(dim, hidden)
        if method in {"S1", "S2"}:
            self.blocks = nn.ModuleList([AttentionBlock(hidden) for _ in range(blocks)])
            self.seed = nn.Parameter(torch.empty(1, 1, hidden))
            nn.init.normal_(self.seed, std=.02)
            self.pool = AttentionBlock(hidden)
            self.output = nn.Linear(hidden, dim)
        else:
            identity_dim = 2 if method == "S3" else hidden
            self.phi = nn.Sequential(nn.Linear(hidden + identity_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
            self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # Create identity parameters last so S1/S2 common modules initialize identically.
        if method in {"S2", "S4"}:
            self.field_embedding = nn.Embedding(fields, hidden)
            nn.init.normal_(self.field_embedding.weight, std=.02)
        if method == "S3":
            self.theta = nn.Parameter(torch.arange(fields, dtype=torch.float32) * (2 * math.pi / fields))

    def forward(self, atoms, mask, non_attribute, field_ids):
        x = self.project(atoms)
        nonempty = mask.any(1)
        if self.method in {"S1", "S2"}:
            if self.method == "S2":
                x = x + self.field_embedding(field_ids)
            # A masked sentinel prevents NaNs on empty sets; its result is zeroed below.
            safe_mask = mask.clone()
            safe_mask[~nonempty, 0] = True
            x = x * mask.unsqueeze(-1)
            for block in self.blocks:
                x = block(x, x, safe_mask)
            pooled = self.pool(self.seed.expand(len(x), -1, -1), x, safe_mask).squeeze(1)
        else:
            if self.method == "S3":
                angles = self.theta[field_ids]
                identity = torch.stack((angles.cos(), angles.sin()), dim=-1)
            else:
                identity = self.field_embedding(field_ids)
            z = self.phi(torch.cat((x, identity), dim=-1))
            pooled = (z * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        attr = self.output(pooled) * nonempty.unsqueeze(-1)
        # Identical two-stage normalization and fixed mix to prior Set-Attention.
        return F.normalize(.5 * non_attribute + .5 * F.normalize(attr, dim=1), dim=1)
