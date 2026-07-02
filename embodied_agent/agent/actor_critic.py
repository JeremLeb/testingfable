"""Actor-critic learned purely in imagination (Dreamer lineage).

The actor is a reparameterized tanh-Gaussian over the 2D continuous action;
the critic estimates state value with a slow EMA target for stability.
Both are trained on imagined rollouts of the world model's *prior* dynamics
starting from real posterior states, using lambda-returns of predicted
(extrinsic + intrinsic) reward.

Why backprop-through-dynamics rather than a policy-gradient estimator: the
learned RSSM is fully differentiable and the action is reparameterized, so
we can push analytic return gradients straight through the imagined
trajectory into the actor. In this low-dimensional, short-horizon sandbox
that is markedly more sample-efficient than REINFORCE-style estimators; the
cost is reliance on world-model gradient quality, which is acceptable here
because the substrate is validated first (milestone 4). An entropy bonus
keeps the policy from collapsing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .replay import ReplayBuffer  # noqa: F401  (type context)
from ..model.networks import mlp
from ..model.rssm import RSSM, RSSMState


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden: int,
                 min_std: float = 0.15, max_std: float = 1.0):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 2 * action_dim, layers=3)
        self.action_dim = action_dim
        self.min_std, self.max_std = min_std, max_std

    def _dist(self, feat):
        mean, std = torch.chunk(self.net(feat), 2, dim=-1)
        mean = torch.tanh(mean)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std)
        return mean, std

    def forward(self, feat):
        """Reparameterized sample in [-1, 1] plus log-prob and entropy."""
        mean, std = self._dist(feat)
        eps = torch.randn_like(std)
        pre = mean + std * eps                    # pathwise / reparameterized
        action = torch.tanh(pre)
        # tanh-corrected log-prob of the squashed sample
        base = -0.5 * (((pre - mean) / std) ** 2 + 2 * torch.log(std)
                       + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = (base - torch.log(1 - action ** 2 + 1e-6)).sum(-1)
        entropy = (0.5 + 0.5 * torch.log(torch.tensor(2 * torch.pi))
                   + torch.log(std)).sum(-1)
        return action, {"log_prob": log_prob, "entropy": entropy}

    @torch.no_grad()
    def act(self, feat, noise: float = 0.0, deterministic: bool = False):
        mean, std = self._dist(feat)
        if deterministic:
            return torch.tanh(mean)
        pre = mean + (std + noise) * torch.randn_like(std)
        return torch.tanh(pre)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 1, layers=3)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


def lambda_return(reward, value, disc, bootstrap, lam):
    """Time-major lambda-returns.

    reward[t], disc[t] describe the transition from state t to t+1; value[t]
    is V(state t+1); bootstrap is V of the final state. Returns V^lambda for
    each starting state t.
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + disc * next_values * (1 - lam)
    returns, last = [], bootstrap
    for t in reversed(range(reward.shape[0])):
        last = inputs[t] + disc[t] * lam * last
        returns.append(last)
    return torch.stack(list(reversed(returns)), 0)


class ActorCritic:
    def __init__(self, cfg: Config, rssm: RSSM, world_model,
                 action_dim: int = 2, intrinsic=None):
        self.cfg = cfg
        self.ac = cfg.agent
        self.rssm = rssm
        self.wm = world_model
        self.intrinsic = intrinsic
        feat = rssm.feat_dim
        self.actor = Actor(feat, action_dim, cfg.model.hidden)
        self.critic = Critic(feat, cfg.model.hidden)
        self.target_critic = Critic(feat, cfg.model.hidden)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.ac.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.ac.critic_lr)
        # EMA of return scale: decouples the entropy coefficient from the
        # (large, configurable) reward scale so tuning is scale-independent.
        self._ret_scale = 1.0

    def _update_target(self):
        tau = self.ac.critic_ema
        for tp, p in zip(self.target_critic.parameters(),
                         self.critic.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def train_step(self, post: RSSMState) -> dict:
        """One imagination-based actor-critic update from real posterior
        states `post` (shape (B, T, ...)); these are used only as start
        states and are detached from the world-model graph."""
        # flatten (B, T) real states into a batch of imagination starts
        B, T = post.h.shape[:2]
        start = RSSMState(post.h.reshape(B * T, -1).detach(),
                          post.z.reshape(B * T, -1).detach())

        states, actions, extras = self.rssm.imagine(
            self.actor, start, self.ac.horizon)
        # prepend the start state so we have states 0..H
        feats = torch.cat([start.feat()[:, None], states.feat()], dim=1)
        H = self.ac.horizon

        reward = self.wm.predict_reward(feats[:, 1:])          # (N, H)
        cont = self.wm.predict_cont(feats[:, 1:])              # (N, H)
        if self.intrinsic is not None:
            reward = reward + self.intrinsic.reward(feats[:, 1:].detach())

        entropy = torch.stack([e["entropy"] for e in extras], dim=1)  # (N, H)

        # time-major for the return recursion
        value = self.critic(feats)                             # (N, H+1)
        disc = self.ac.gamma * cont
        returns = lambda_return(
            reward.transpose(0, 1), value[:, 1:].transpose(0, 1),
            disc.transpose(0, 1), value[:, -1], self.ac.lam
        ).transpose(0, 1)                                      # (N, H)

        # discount weighting so far-future imagined steps count less
        with torch.no_grad():
            weight = torch.cumprod(
                torch.cat([torch.ones_like(disc[:, :1]), disc[:, :-1]], 1), 1)

        # actor: maximize returns (pathwise) + entropy bonus. Returns are
        # divided by an EMA of their scale so the entropy coefficient is
        # meaningful regardless of the reward scale.
        with torch.no_grad():
            batch_scale = returns.detach().abs().mean().clamp(min=1e-3)
            self._ret_scale = 0.99 * self._ret_scale + 0.01 * float(batch_scale)
        actor_loss = -(weight * returns).mean() / self._ret_scale \
            - self.ac.entropy_scale * (weight * entropy).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                              self.ac.grad_clip)
        self.actor_opt.step()

        # critic: regress value of states 0..H-1 toward lambda-returns
        value_pred = self.critic(feats[:, :-1].detach())
        critic_loss = (weight * 0.5
                       * (value_pred - returns.detach()) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               self.ac.grad_clip)
        self.critic_opt.step()
        self._update_target()

        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "imag_return": float(returns.mean().detach()),
            "imag_reward": float(reward.mean().detach()),
            "actor_entropy": float(entropy.mean().detach()),
            "actor_grad": float(actor_grad),
            "critic_grad": float(critic_grad),
            "return_scale": self._ret_scale,
        }
