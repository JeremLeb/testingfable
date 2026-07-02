"""Distribution utilities for the DreamerV3-style upgrade.

Contains:
  * symlog / symexp        -- squashing transform for wide-magnitude targets
  * TwoHotHead             -- discrete-regression head over symlog-spaced bins
  * categorical latent ops -- straight-through one-hot sampling + balanced KL
  * gaussian_kl            -- the original diagonal-Gaussian balanced KL

The reward in this sandbox spans a very wide range (small per-step drive
drift vs large food/collision spikes after the reward_scale multiply), which
is exactly the regime where plain MSE regression struggles and symlog +
two-hot classification shines: it is robust to outliers and needs no reward
normalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import mlp


# --------------------------------------------------------------------- symlog

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ------------------------------------------------------------------- two-hot

class TwoHotHead(nn.Module):
    """Predicts a scalar as a distribution over fixed bins in symlog space.

    Regressing symlog(target) with a two-hot cross-entropy (DreamerV3) is far
    more robust to the reward outliers this environment produces than MSE.
    """

    def __init__(self, in_dim: int, hidden: int, bins: int = 51,
                 low: float = -8.0, high: float = 8.0, layers: int = 2):
        super().__init__()
        self.net = mlp(in_dim, hidden, bins, layers=layers)
        self.register_buffer("centers", torch.linspace(low, high, bins))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def mean(self, feat: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.forward(feat), dim=-1)
        return symexp((probs * self.centers).sum(-1))

    def _twohot(self, target_symlog: torch.Tensor) -> torch.Tensor:
        centers = self.centers
        t = target_symlog.clamp(centers[0], centers[-1])
        idx = torch.searchsorted(centers, t.contiguous(), right=True)
        upper = idx.clamp(max=centers.numel() - 1)
        lower = (idx - 1).clamp(min=0)
        c_low, c_high = centers[lower], centers[upper]
        span = (c_high - c_low).clamp(min=1e-8)
        w_high = (t - c_low) / span
        w_low = 1.0 - w_high
        # bins coincide at the clamped ends -> put all mass on that bin
        same = lower == upper
        w_low = torch.where(same, torch.ones_like(w_low), w_low)
        w_high = torch.where(same, torch.zeros_like(w_high), w_high)
        out = torch.zeros(*t.shape, centers.numel(), device=t.device)
        out.scatter_(-1, lower.unsqueeze(-1), w_low.unsqueeze(-1))
        out.scatter_add_(-1, upper.unsqueeze(-1), w_high.unsqueeze(-1))
        return out

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.forward(feat)
        target_twohot = self._twohot(symlog(target)).detach()
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_twohot * log_probs).sum(-1)


# ----------------------------------------------------------- categorical latent

def categorical_sample(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Straight-through one-hot sample. logits: (..., G, C) -> (..., G*C).

    A small uniform mixture ('unimix') keeps every class from ever having
    exactly zero probability, which stabilises the KL and prevents dead
    classes -- the DreamerV3 trick.
    """
    probs = torch.softmax(logits, dim=-1)
    if unimix > 0.0:
        probs = (1 - unimix) * probs + unimix / probs.shape[-1]
    sample = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]), 1
    ).reshape(probs.shape[:-1])
    onehot = F.one_hot(sample, probs.shape[-1]).float()
    onehot = onehot + probs - probs.detach()          # straight-through
    return onehot.reshape(*logits.shape[:-2], -1)


def categorical_kl(post_logits, prior_logits, groups, classes,
                   free_bits, balance, unimix):
    """Balanced KL(post || prior) summed over groups, with free bits."""
    def probs(logits):
        p = torch.softmax(logits, dim=-1)
        if unimix > 0.0:
            p = (1 - unimix) * p + unimix / classes
        return p

    def kl(p_logits, q_logits):
        p = probs(p_logits)
        q = probs(q_logits)
        return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1).sum(-1)

    shape = (*post_logits.shape[:-1], groups, classes)
    post = post_logits.reshape(shape)
    prior = prior_logits.reshape(shape)
    kl_rep = kl(post, prior.detach())                 # trains the encoder
    kl_dyn = kl(post.detach(), prior)                 # trains the dynamics
    kl_rep = torch.clamp(kl_rep, min=free_bits)
    kl_dyn = torch.clamp(kl_dyn, min=free_bits)
    return balance * kl_dyn + (1 - balance) * kl_rep


def gaussian_kl(post_mean, post_std, prior_mean, prior_std,
                free_bits, balance):
    def kl(p_mean, p_std, q_mean, q_std):
        return (torch.log(q_std) - torch.log(p_std)
                + (p_std ** 2 + (p_mean - q_mean) ** 2)
                / (2 * q_std ** 2) - 0.5).sum(-1)

    kl_dyn = kl(post_mean.detach(), post_std.detach(), prior_mean, prior_std)
    kl_rep = kl(post_mean, post_std, prior_mean.detach(), prior_std.detach())
    kl_dyn = torch.clamp(kl_dyn, min=free_bits)
    kl_rep = torch.clamp(kl_rep, min=free_bits)
    return balance * kl_dyn + (1 - balance) * kl_rep
