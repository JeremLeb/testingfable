"""Small MLP building blocks: per-modality encoders and decoders.

All modalities in this sandbox are low-dimensional vectors, so plain MLPs
suffice (no conv retina). Each modality has its own encoder producing a
shared-width embedding; embeddings are summed and projected to form the
fused observation embedding fed to the RSSM posterior. Each modality also
has its own decoder from the latent state, which is what forces cross-modal
fusion: one latent must reconstruct every sense.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class MultiEncoder(nn.Module):
    def __init__(self, spaces: dict[str, int], embed_dim: int, hidden: int):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: mlp(dim, hidden, embed_dim, layers=2)
            for name, dim in spaces.items()
        })
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        emb = None
        for name, enc in self.encoders.items():
            e = enc(obs[name])
            emb = e if emb is None else emb + e
        return self.norm(emb)


class MultiDecoder(nn.Module):
    """Reconstructs every modality from the latent state (h, z)."""

    def __init__(self, spaces: dict[str, int], feat_dim: int, hidden: int):
        super().__init__()
        self.decoders = nn.ModuleDict({
            name: mlp(feat_dim, hidden, dim, layers=2)
            for name, dim in spaces.items()
        })

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: dec(feat) for name, dec in self.decoders.items()}
