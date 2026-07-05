"""Vectorized acting for the colony: run every creature's OWN brain in parallel.

Each creature still has its own separate world model + actor (different weights,
its own memory, its own recurrent state). This module stacks those N separate
parameter sets and runs the whole acting forward pass -- encode senses -> RSSM
step -> actor -> action -- as a *single* batched op via `torch.func.vmap`,
instead of a Python loop over N brains. It is numerically identical to the loop
(verified in tests); it just lets the GPU compute all N at once.

Two implementation choices keep it robust:
  * the GRU cell is written with plain math (vmap has no batching rule for
    aten::gru_cell) using the exact same weights, so nothing falls back to a
    sequential loop;
  * all *random* sampling (the categorical latent, the action noise) is done
    OUTSIDE vmap in ordinary batched torch, so no fragile randomness-under-vmap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap


def _gru(x, h, w_ih, w_hh, b_ih, b_hh):
    """A plain-math GRU cell, numerically identical to nn.GRUCell (which vmap
    cannot batch)."""
    gi = F.linear(x, w_ih, b_ih)
    gh = F.linear(h, w_hh, b_hh)
    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h


class _ObsFeat(nn.Module):
    """Deterministic part of one acting step: obs + (h, z, prev_action) ->
    (new deterministic state h, posterior logits). References a mind's own
    submodules, so stacking N of these stacks N separate brains."""

    def __init__(self, mind):
        super().__init__()
        r = mind.wm.rssm
        self.encoder = mind.wm.encoder
        self.pre_gru = r.pre_gru
        self.gru = r.gru
        self.post_net = r.post_net

    def forward(self, obs, h, z, prev_a):
        x = torch.cat([z, prev_a], -1)
        h_new = _gru(self.pre_gru(x), h, self.gru.weight_ih, self.gru.weight_hh,
                     self.gru.bias_ih, self.gru.bias_hh)
        embed = self.encoder({k: v.unsqueeze(0) for k, v in obs.items()}).squeeze(0)
        logits = self.post_net(torch.cat([h_new, embed], -1))
        return h_new, logits


class _ActorHead(nn.Module):
    """feat -> (mean, std) using a mind's own actor (mirrors Actor._dist)."""

    def __init__(self, mind):
        super().__init__()
        a = mind.ac.actor
        self.net = a.net
        self.register_buffer("action_bias", a.action_bias.clone())
        self.min_std = a.min_std
        self.max_std = a.max_std

    def forward(self, feat):
        mean, std = torch.chunk(self.net(feat), 2, dim=-1)
        mean = torch.tanh(mean + self.action_bias)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std)
        return mean, std


class BatchedActing:
    """Runs the acting forward pass for a whole population in one batched op."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.groups = cfg.model.latent_groups
        self.classes = cfg.model.latent_classes
        self.unimix = cfg.model.unimix
        self._wrap = {}  # id(mind) -> (_ObsFeat, _ActorHead), cached

    def _wrappers(self, minds):
        obs_mods, act_mods = [], []
        for m in minds:
            w = self._wrap.get(id(m))
            if w is None:
                w = (_ObsFeat(m).to(self.device), _ActorHead(m).to(self.device))
                self._wrap[id(m)] = w
            obs_mods.append(w[0])
            act_mods.append(w[1])
        return obs_mods, act_mods

    def _sample_z(self, logits):
        n = logits.shape[0]
        lg = logits.reshape(n, self.groups, self.classes)
        probs = torch.softmax(lg, -1)
        if self.unimix > 0:
            probs = (1 - self.unimix) * probs + self.unimix / self.classes
        idx = torch.multinomial(probs.reshape(n * self.groups, self.classes), 1)
        idx = idx.reshape(n, self.groups)
        oneh = F.one_hot(idx, self.classes).float()
        oneh = oneh + probs - probs.detach()          # straight-through
        return oneh.reshape(n, self.groups * self.classes)

    @torch.no_grad()
    def act(self, minds, obs_stack, h, z, prev_a, bias, noise,
            deterministic=False):
        """minds: list of N Mind; obs_stack: dict of (N, dim); h (N, deter),
        z (N, stoch), prev_a (N, 2), bias (N, 2). Returns (action (N,2),
        h_new (N,deter), z_new (N,stoch))."""
        obs_mods, act_mods = self._wrappers(minds)
        # --- deterministic RSSM step, all brains at once ---
        of_p, of_b = stack_module_state(obs_mods)
        base_of = obs_mods[0]

        def fa(p, b, ob, hh, zz, pa):
            return functional_call(base_of, (p, b), (ob, hh, zz, pa))

        h_new, logits = vmap(fa)(of_p, of_b, obs_stack, h, z, prev_a)
        # --- sample the categorical latent (outside vmap) ---
        z_new = self._sample_z(logits)
        feat = torch.cat([h_new, z_new], -1)
        # --- actor, all brains at once ---
        ac_p, ac_b = stack_module_state(act_mods)
        base_ac = act_mods[0]

        def fb(p, b, ft):
            return functional_call(base_ac, (p, b), (ft,))

        mean, std = vmap(fb)(ac_p, ac_b, feat)
        # --- sample the action (outside vmap), add each genome's instinct ---
        if deterministic:
            action = torch.tanh(mean)
        else:
            action = torch.tanh(mean + (std + noise) * torch.randn_like(std))
        action = torch.clamp(action + bias, -1.0, 1.0)
        return action, h_new, z_new

    def forget(self, mind):
        self._wrap.pop(id(mind), None)
