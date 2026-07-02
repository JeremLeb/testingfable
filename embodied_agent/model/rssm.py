"""Recurrent State-Space Model (RSSM), the Dreamer dynamics core.

State s = (h, z): deterministic recurrent h from a GRU cell, plus a
stochastic latent z. One transition:

    h_t   = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    prior:     p(z_t | h_t)      from h_t alone
    posterior: q(z_t | h_t, x_t) from h_t and the observation embedding

Training uses the posterior (it has seen the observation); imagination uses
the prior. The KL between them is the core prediction-error signal: the
prior is trained to anticipate what the posterior infers from the next
observation.

The stochastic latent is pluggable via `latent_kind`:
  * "discrete" -- a vector of categorical variables with straight-through
    one-hot samples (DreamerV2/V3). Multimodal, collapse-resistant; the
    default for the advanced configs.
  * "gaussian" -- the original diagonal Gaussian, kept for ablation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import (categorical_kl, categorical_sample, gaussian_kl)
from .networks import mlp


class RSSMState:
    """A batch of latent states: deterministic h, stochastic z, and the
    distribution params that produced z (for KL). `params` is a dict holding
    either {'logits'} (discrete) or {'mean','std'} (gaussian)."""

    def __init__(self, h, z, params: dict | None = None):
        self.h = h
        self.z = z
        self.params = params or {}

    def feat(self) -> torch.Tensor:
        return torch.cat([self.h, self.z], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.z.detach(),
                         {k: v.detach() for k, v in self.params.items()})


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int,
                 hidden: int, latent_kind: str = "discrete",
                 stoch_dim: int = 32, groups: int = 16, classes: int = 16,
                 unimix: float = 0.01, min_std: float = 0.1,
                 sparse_latent: bool = False, sparse_frac: float = 0.5):
        super().__init__()
        self.latent_kind = latent_kind
        self.deter_dim = deter_dim
        self.groups, self.classes = groups, classes
        self.unimix, self.min_std = unimix, min_std
        # B4 sparse cortical code: keep only k = ceil(frac * groups) of the
        # discrete groups active (k-winners-take-all), zeroing the rest.
        self.sparse_latent = sparse_latent and latent_kind == "discrete"
        self.k_active = max(1, int(math.ceil(sparse_frac * groups)))

        if latent_kind == "discrete":
            self.stoch_flat = groups * classes
            out = groups * classes
        elif latent_kind == "gaussian":
            self.stoch_dim = stoch_dim
            self.stoch_flat = stoch_dim
            out = 2 * stoch_dim
        else:
            raise ValueError(f"unknown latent_kind: {latent_kind}")
        self.feat_dim = deter_dim + self.stoch_flat

        self.pre_gru = mlp(self.stoch_flat + action_dim, hidden, hidden,
                           layers=1)
        self.gru = nn.GRUCell(hidden, deter_dim)
        self.prior_net = mlp(deter_dim, hidden, out, layers=1)
        self.post_net = mlp(deter_dim + embed_dim, hidden, out, layers=1)

    # ------------------------------------------------------------ states

    def initial(self, batch_size: int, device) -> RSSMState:
        return RSSMState(
            h=torch.zeros(batch_size, self.deter_dim, device=device),
            z=torch.zeros(batch_size, self.stoch_flat, device=device))

    def _latent(self, raw: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Map a network output to (sampled z, distribution params)."""
        if self.latent_kind == "discrete":
            logits = raw.reshape(*raw.shape[:-1], self.groups, self.classes)
            z = categorical_sample(logits, self.unimix)   # (..., G*C) flat
            if self.sparse_latent and self.k_active < self.groups:
                # k-winners over groups: keep the k most confident (peakiest)
                # groups, zero the rest -> a sparse assembly code.
                zc = z.reshape(*z.shape[:-1], self.groups, self.classes)
                conf = torch.softmax(logits, dim=-1).amax(dim=-1)   # (..., G)
                topk = conf.topk(self.k_active, dim=-1).indices
                mask = torch.zeros_like(conf).scatter_(-1, topk, 1.0)
                z = (zc * mask.unsqueeze(-1)).reshape(*z.shape)
            return z, {"logits": raw}
        mean, std = torch.chunk(raw, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        z = mean + std * torch.randn_like(std)
        return z, {"mean": mean, "std": std}

    # ------------------------------------------------------------ transitions

    def _deter(self, prev: RSSMState, prev_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prev.z, prev_action], dim=-1)
        return self.gru(self.pre_gru(x), prev.h)

    def prior_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        h = self._deter(prev, prev_action)
        z, params = self._latent(self.prior_net(h))
        return RSSMState(h, z, params)

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        """One posterior step. Returns (posterior, prior) sharing h. The
        prior carries the posterior's z so both index the same trajectory
        while keeping their own distribution params for the KL."""
        h = self._deter(prev, prev_action)
        _, prior_params = self._latent(self.prior_net(h))
        z, post_params = self._latent(self.post_net(torch.cat([h, embed], -1)))
        post = RSSMState(h, z, post_params)
        prior = RSSMState(h, z, prior_params)
        return post, prior

    # ------------------------------------------------------------ rollouts

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor,
                init: RSSMState | None = None):
        B, T = embeds.shape[:2]
        state = init or self.initial(B, embeds.device)
        posts, priors = [], []
        for t in range(T):
            post, prior = self.obs_step(state, actions[:, t], embeds[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return self._stack(posts), self._stack(priors)

    def imagine(self, policy, start: RSSMState, horizon: int):
        state = start
        states, actions, extras = [], [], []
        for _ in range(horizon):
            action, extra = policy(state.feat())
            state = self.prior_step(state, action)
            states.append(state)
            actions.append(action)
            extras.append(extra)
        return self._stack(states), torch.stack(actions, 1), extras

    @staticmethod
    def _stack(states: list[RSSMState]) -> RSSMState:
        keys = states[0].params.keys()
        return RSSMState(
            torch.stack([s.h for s in states], 1),
            torch.stack([s.z for s in states], 1),
            {k: torch.stack([s.params[k] for s in states], 1) for k in keys})

    # ------------------------------------------------------------ kl

    def kl_loss(self, post: RSSMState, prior: RSSMState, free_bits: float,
                balance: float) -> torch.Tensor:
        if self.latent_kind == "discrete":
            return categorical_kl(
                post.params["logits"], prior.params["logits"],
                self.groups, self.classes, free_bits, balance, self.unimix)
        return gaussian_kl(
            post.params["mean"], post.params["std"],
            prior.params["mean"], prior.params["std"], free_bits, balance)
