"""Ensemble-disagreement novelty bonus.

An ensemble of predictors is trained to regress a shared fixed random
embedding of the latent state. Where states have been visited often the
predictors converge and agree; on novel states they disagree. The variance
across the ensemble is the novelty bonus (whitened by a running std).

This is the model-uncertainty flavour of curiosity toggled via
`intrinsic.method: disagreement`. The richer form measures disagreement of
an ensemble of forward dynamics models predicting the next latent from
(state, action); that would slot in here by conditioning each member on the
action and predicting the next feat -- noted as future work in the README.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..model.networks import mlp
from ..utils.running import RunningNorm


class Disagreement:
    def __init__(self, cfg: Config, feat_dim: int, device: str = "cpu"):
        ic = cfg.intrinsic
        self.scale = ic.scale
        self.device = device
        self.target = mlp(feat_dim, ic.hidden, ic.out_dim, layers=2).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.members = nn.ModuleList([
            mlp(feat_dim, ic.hidden, ic.out_dim, layers=2)
            for _ in range(ic.ensemble)]).to(device)
        self.opt = torch.optim.Adam(self.members.parameters(), lr=ic.lr)
        self.norm = RunningNorm()

    def _disagreement(self, feat: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(feat) for m in self.members], 0)  # (E, N, D)
        return preds.var(0).mean(-1)                             # (N,)

    @torch.no_grad()
    def reward(self, feat: torch.Tensor) -> torch.Tensor:
        shape = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        dis = self._disagreement(flat)
        return (self.scale * dis / self.norm.std).reshape(shape)

    @torch.no_grad()
    def epistemic(self, feat: torch.Tensor) -> torch.Tensor:
        """Whitened ensemble disagreement WITHOUT the hand-set curiosity scale
        -- the information-gain (epistemic) term for the EFE objective (B6)."""
        shape = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        dis = self._disagreement(flat)
        return (dis / self.norm.std).reshape(shape)

    def train_step(self, feat: torch.Tensor) -> dict:
        with torch.no_grad():
            tgt = self.target(feat)
        loss = sum(((m(feat) - tgt) ** 2).mean() for m in self.members)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            dis = self._disagreement(feat)
        self.norm.update(dis.cpu().numpy())
        return {"intrinsic_reward": float((self.scale * dis / self.norm.std)
                                          .mean()),
                "disagreement_loss": float(loss.detach())}
