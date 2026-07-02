"""Per-modality encoders and decoders, dispatched by modality type.

Vector modalities (touch, proprio, intero, smell, ray-vision) use MLPs.
Image modalities (the pixel retina) use a small CNN encoder and a mirrored
transposed-conv decoder. Every modality's encoder maps into one shared
embedding width; the embeddings are summed into the fused observation
embedding fed to the RSSM posterior. Every modality is also decoded from the
latent state, which is what forces cross-modal fusion: one latent must
reconstruct every sense, image included.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .spaces import ModalitySpec, normalize_spaces


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def _flatten_lead(x: torch.Tensor, keep: int) -> tuple[torch.Tensor, tuple]:
    """Flatten all but the last `keep` dims into one batch dim."""
    lead = x.shape[:-keep]
    return x.reshape(-1, *x.shape[-keep:]), lead


class ImageEncoder(nn.Module):
    def __init__(self, shape: tuple, embed_dim: int, depth: int = 16):
        super().__init__()
        c, h, w = shape
        layers: list[nn.Module] = []
        ch, chans, res = c, depth, h
        while res > 4:
            layers += [nn.Conv2d(ch, chans, 4, stride=2, padding=1),
                       nn.SiLU()]
            ch, chans, res = chans, chans * 2, res // 2
        self.conv = nn.Sequential(*layers)
        self.res, self.ch = res, ch
        self.out = nn.Linear(ch * res * res, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, lead = _flatten_lead(x, keep=3)
        h = self.conv(flat).flatten(1)
        return self.out(h).reshape(*lead, -1)


class ImageDecoder(nn.Module):
    def __init__(self, feat_dim: int, shape: tuple, depth: int = 16):
        super().__init__()
        c, h, w = shape
        self.shape = shape
        n_up = int(round(math.log2(h / 4)))
        base_ch = depth * (2 ** (n_up - 1)) if n_up > 0 else depth
        self.base_ch, self.res = base_ch, 4
        self.fc = nn.Linear(feat_dim, base_ch * 4 * 4)
        layers: list[nn.Module] = []
        ch = base_ch
        for i in range(n_up):
            out_ch = c if i == n_up - 1 else ch // 2
            layers.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2,
                                             padding=1))
            if i < n_up - 1:
                layers.append(nn.SiLU())
            ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        flat, lead = _flatten_lead(feat, keep=1)
        h = self.fc(flat).reshape(-1, self.base_ch, self.res, self.res)
        img = torch.sigmoid(self.deconv(h))
        return img.reshape(*lead, *self.shape)


class MultiEncoder(nn.Module):
    def __init__(self, spaces: dict, embed_dim: int, hidden: int,
                 cnn_depth: int = 16):
        super().__init__()
        self.specs = normalize_spaces(spaces)
        self.encoders = nn.ModuleDict()
        for name, spec in self.specs.items():
            if spec.kind == "image":
                self.encoders[name] = ImageEncoder(spec.shape, embed_dim,
                                                   cnn_depth)
            else:
                self.encoders[name] = mlp(spec.dim, hidden, embed_dim,
                                          layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        emb = None
        for name, enc in self.encoders.items():
            e = enc(obs[name])
            emb = e if emb is None else emb + e
        return self.norm(emb)


class MultiDecoder(nn.Module):
    """Reconstructs every modality from the latent state (h, z)."""

    def __init__(self, spaces: dict, feat_dim: int, hidden: int,
                 cnn_depth: int = 16):
        super().__init__()
        self.specs = normalize_spaces(spaces)
        self.decoders = nn.ModuleDict()
        for name, spec in self.specs.items():
            if spec.kind == "image":
                self.decoders[name] = ImageDecoder(feat_dim, spec.shape,
                                                   cnn_depth)
            else:
                self.decoders[name] = mlp(feat_dim, hidden, spec.dim,
                                          layers=2)

    def forward(self, feat: torch.Tensor) -> dict:
        return {name: dec(feat) for name, dec in self.decoders.items()}
