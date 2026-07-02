"""Random Network Distillation novelty bonus (Burda et al. 2018).

A fixed, randomly-initialized target network embeds the latent state; a
predictor is trained to match it on visited states. Prediction error is
high on states the predictor has seen rarely, so the error -- whitened by a
running std -- is a novelty bonus added to reward inside imagination and the
real env. It is a *thin exploration layer*: ~99% of the learning signal is
still the world model's self-supervised prediction of the sensory stream.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..model.networks import mlp
from ..utils.running import RunningNorm


class RND:
    def __init__(self, cfg: Config, feat_dim: int, device: str = "cpu"):
        ic = cfg.intrinsic
        self.scale = ic.scale
        self.device = device
        self.target = mlp(feat_dim, ic.hidden, ic.out_dim, layers=2).to(device)
        self.predictor = mlp(feat_dim, ic.hidden, ic.out_dim,
                             layers=2).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=ic.lr)
        self.norm = RunningNorm()

    def _error(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tgt = self.target(feat)
        pred = self.predictor(feat)
        return ((pred - tgt) ** 2).mean(-1)

    @torch.no_grad()
    def reward(self, feat: torch.Tensor) -> torch.Tensor:
        """Whitened novelty bonus with the same leading shape as feat[..., :-1]."""
        shape = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        err = self._error(flat)
        bonus = self.scale * err / self.norm.std
        return bonus.reshape(shape)

    def train_step(self, feat: torch.Tensor) -> dict:
        err = self._error(feat)
        loss = err.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.norm.update(err.detach().cpu().numpy())
        return {"intrinsic_reward": float((self.scale * err / self.norm.std)
                                          .mean().detach()),
                "rnd_loss": float(loss.detach())}
