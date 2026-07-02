"""Recurrent State-Space Model (RSSM-lite), the Dreamer dynamics core.

State s = (h, z): deterministic recurrent h from a GRU cell, plus a
stochastic Gaussian latent z. One transition:

    h_t   = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    prior:     p(z_t | h_t)      = N(mu_p, sigma_p)     from h_t alone
    posterior: q(z_t | h_t, x_t) = N(mu_q, sigma_q)     from h_t and obs embed

Training uses the posterior (it has seen the observation); imagination uses
the prior (no observations available). The KL between them is the core
prediction-error signal: the prior is trained to anticipate what the
posterior infers from the next observation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import mlp


class RSSMState:
    """A batch of latent states; convenient (h, z) container."""

    def __init__(self, h, z, mean=None, std=None):
        self.h = h
        self.z = z
        self.mean = mean
        self.std = std

    def feat(self) -> torch.Tensor:
        return torch.cat([self.h, self.z], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(
            self.h.detach(), self.z.detach(),
            None if self.mean is None else self.mean.detach(),
            None if self.std is None else self.std.detach())


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int,
                 stoch_dim: int, hidden: int, min_std: float = 0.1):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        self.feat_dim = deter_dim + stoch_dim

        self.pre_gru = mlp(stoch_dim + action_dim, hidden, hidden, layers=1)
        self.gru = nn.GRUCell(hidden, deter_dim)
        self.prior_net = mlp(deter_dim, hidden, 2 * stoch_dim, layers=1)
        self.post_net = mlp(deter_dim + embed_dim, hidden, 2 * stoch_dim,
                            layers=1)

    # ------------------------------------------------------------ states

    def initial(self, batch_size: int, device) -> RSSMState:
        return RSSMState(
            h=torch.zeros(batch_size, self.deter_dim, device=device),
            z=torch.zeros(batch_size, self.stoch_dim, device=device))

    def _dist(self, params):
        mean, std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    @staticmethod
    def _sample(mean, std):
        return mean + std * torch.randn_like(std)

    # ------------------------------------------------------------ transitions

    def _deter(self, prev: RSSMState, prev_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prev.z, prev_action], dim=-1)
        return self.gru(self.pre_gru(x), prev.h)

    def prior_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        h = self._deter(prev, prev_action)
        mean, std = self._dist(self.prior_net(h))
        return RSSMState(h, self._sample(mean, std), mean, std)

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor,
                 embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        """One posterior step. Returns (posterior, prior) sharing h."""
        h = self._deter(prev, prev_action)
        pri_mean, pri_std = self._dist(self.prior_net(h))
        post_in = torch.cat([h, embed], dim=-1)
        post_mean, post_std = self._dist(self.post_net(post_in))
        post = RSSMState(h, self._sample(post_mean, post_std),
                         post_mean, post_std)
        prior = RSSMState(h, post.z, pri_mean, pri_std)
        return post, prior

    # ------------------------------------------------------------ rollouts

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor,
                init: RSSMState | None = None):
        """Filter a batch of sequences with the posterior.

        embeds/actions are (B, T, ...); actions[:, t] is the action that led
        into observation t. Returns stacked posterior and prior states.
        """
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
        """Roll the prior forward under a policy from a flat start state.

        policy maps feat -> (action, extra). Returns stacked states (T) and
        the per-step actions and policy extras (e.g. log-probs, entropy).
        """
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
        return RSSMState(
            torch.stack([s.h for s in states], 1),
            torch.stack([s.z for s in states], 1),
            torch.stack([s.mean for s in states], 1)
            if states[0].mean is not None else None,
            torch.stack([s.std for s in states], 1)
            if states[0].std is not None else None)


def kl_divergence(post: RSSMState, prior: RSSMState,
                  free_bits: float, balance: float) -> torch.Tensor:
    """Balanced KL(post || prior) with free bits, per Dreamer-v2.

    KL balancing trains the prior toward the (stop-grad) posterior more
    strongly than it pulls the posterior toward the prior, so the dynamics
    learn to predict without collapsing the representation.
    """
    def kl(p_mean, p_std, q_mean, q_std):
        return (torch.log(q_std) - torch.log(p_std)
                + (p_std ** 2 + (p_mean - q_mean) ** 2)
                / (2 * q_std ** 2) - 0.5).sum(-1)

    kl_prior = kl(post.mean.detach(), post.std.detach(), prior.mean, prior.std)
    kl_post = kl(post.mean, post.std, prior.mean.detach(), prior.std.detach())
    kl_prior = torch.clamp(kl_prior, min=free_bits)
    kl_post = torch.clamp(kl_post, min=free_bits)
    return balance * kl_prior + (1 - balance) * kl_post
