"""Vectorized *training* for the colony: update every creature's OWN world
model in one batched GPU pass instead of a Python loop over N brains.

This is the training-time twin of ``batched.py`` (which batches *acting*).
Each creature still has its own separate world model -- different weights, its
own replay memory, its own Adam moments. This module stacks those N parameter
sets and computes, for all of them at once, the world-model loss gradient
(``torch.func.vmap`` over ``torch.func.grad``) and a functional Adam step. It is
numerically identical to running each brain's ``WorldModel.train_step`` in a
loop (verified in tests); it just lets the GPU learn all N models in parallel.

Stage 1 batches the *world model* -- the dominant training cost (the largest
network, the full seq_len sequence, and every-modality reconstruction). The
lighter actor-critic + RND updates stay on the existing rotating per-mind budget
(they whiten with a running norm and take two backward passes, so they are
batched separately later).

Three vmap-hostile pieces are replaced by provably-identical plain math:
  * the GRU cell (``batched._gru``) -- vmap has no batching rule for aten::gru_cell;
  * the categorical latent sample -- ``torch.multinomial`` has no batching rule, so
    we use Gumbel-max with externally-drawn noise (distributionally the same
    categorical), keeping the straight-through estimator;
  * the two-hot reward target -- ``searchsorted``/``scatter_`` are not vmap-safe, so
    we use the exact arithmetic two-hot for uniform bins (a triangular kernel).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import (functional_call, grad_and_value, stack_module_state,
                        vmap)

from ..model.distributions import categorical_kl, symlog
from .batched import _gru


def gumbel_like(logits: torch.Tensor) -> torch.Tensor:
    """Standard Gumbel(0,1) noise for Gumbel-max categorical sampling."""
    u = torch.rand_like(logits).clamp_(1e-10, 1.0)
    return -torch.log(-torch.log(u))


class _WMLoss(nn.Module):
    """Wraps one creature's world model and computes its training loss with
    vmap-safe internals. References the mind's real submodules, so stacking N of
    these and routing stacked params via ``functional_call`` runs N separate
    world models. Reproduces ``WorldModel.loss`` exactly (same recon + balanced
    KL + reward + continue), only with the plain-math GRU, Gumbel sampling, and
    arithmetic two-hot substituted in."""

    def __init__(self, mind, cfg):
        super().__init__()
        self.wm = mind.wm                      # submodule -> its params get stacked
        r = mind.wm.rssm
        self.groups, self.classes = r.groups, r.classes
        self.unimix = r.unimix
        self.deter = r.deter_dim
        self.stoch_flat = r.stoch_flat
        mc = cfg.model
        self.recon_scales = dict(mc.recon_scales)
        self.free_bits = mc.free_bits
        self.kl_balance = mc.kl_balance
        self.kl_beta = mc.kl_beta
        self.reward_kind = mc.reward_head

    # -- one posterior step's stochastic sample (Gumbel-max straight-through) --
    def _sample(self, logits, gnoise):
        lg = logits.reshape(*logits.shape[:-1], self.groups, self.classes)
        probs = torch.softmax(lg, -1)
        if self.unimix > 0.0:
            probs = (1 - self.unimix) * probs + self.unimix / self.classes
        idx = torch.argmax(torch.log(probs + 1e-10) + gnoise, dim=-1)   # (.., G)
        # vmap-safe one-hot (F.one_hot is data-dependent under vmap)
        classes = torch.arange(self.classes, device=idx.device)
        onehot = (idx.unsqueeze(-1) == classes).float()
        onehot = onehot + probs - probs.detach()                       # straight-through
        return onehot.reshape(*logits.shape[:-1], self.groups * self.classes)

    def _observe(self, embeds, actions, gnoise):
        B, T = actions.shape[:2]
        r = self.wm.rssm
        h = embeds.new_zeros(B, self.deter)
        z = embeds.new_zeros(B, self.stoch_flat)
        feats, post_lg, prior_lg = [], [], []
        for t in range(T):
            x = torch.cat([z, actions[:, t]], -1)
            h = _gru(r.pre_gru(x), h, r.gru.weight_ih, r.gru.weight_hh,
                     r.gru.bias_ih, r.gru.bias_hh)
            prior_logits = r.prior_net(h)
            post_logits = r.post_net(torch.cat([h, embeds[:, t]], -1))
            z = self._sample(post_logits, gnoise[:, t])
            feats.append(torch.cat([h, z], -1))
            post_lg.append(post_logits)
            prior_lg.append(prior_logits)
        return (torch.stack(feats, 1), torch.stack(post_lg, 1),
                torch.stack(prior_lg, 1))

    def _reward_loss(self, feat, target):
        head = self.wm.reward_head
        if self.reward_kind == "twohot":
            centers = head.centers                       # (bins,) uniform
            d = centers[1] - centers[0]
            t = symlog(target).clamp(centers[0], centers[-1])
            # exact two-hot for uniform bins: triangular kernel over centers
            twohot = F.relu(1 - (t.unsqueeze(-1) - centers).abs() / d).detach()
            logp = F.log_softmax(head.net(feat), dim=-1)
            return -(twohot * logp).sum(-1).mean()
        return F.mse_loss(head(feat).squeeze(-1), target)

    def forward(self, obs, actions, reward, cont, gnoise):
        B, T = actions.shape[:2]
        embeds = self.wm.encoder(obs)
        feat, post_lg, prior_lg = self._observe(embeds, actions, gnoise)

        recon = self.wm.decoder(feat)
        recon_total = feat.new_zeros(())
        for name, target in obs.items():
            err = (recon[name] - target).reshape(B, T, -1)
            recon_total = recon_total + self.recon_scales.get(name, 1.0) \
                * (err ** 2).sum(-1).mean()

        kl = categorical_kl(post_lg, prior_lg, self.groups, self.classes,
                            self.free_bits, self.kl_balance, self.unimix).mean()
        loss = recon_total + self.kl_beta * kl
        loss = loss + self._reward_loss(feat, reward)
        loss = loss + F.binary_cross_entropy_with_logits(
            self.wm.cont_head(feat).squeeze(-1), cont)
        return loss


class BatchedWMTrainer:
    """Batched Adam training of N separate world models in one GPU pass."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.groups = cfg.model.latent_groups
        self.classes = cfg.model.latent_classes
        self.lr = cfg.model.lr
        self.clip = cfg.model.grad_clip
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self._wrap = {}    # id(mind) -> _WMLoss, cached
        self._state = {}   # id(mind) -> {"step": int, "m": {..}, "v": {..}}

    def _wrapper(self, mind):
        w = self._wrap.get(id(mind))
        if w is None:
            w = _WMLoss(mind, self.cfg).to(self.device)
            self._wrap[id(mind)] = w
        return w

    def forget(self, mind):
        self._wrap.pop(id(mind), None)
        self._state.pop(id(mind), None)

    def update(self, minds, batches, gnoise=None):
        """One batched Adam step over `minds` (list of N Mind) using per-mind
        replay `batches` (list of N dicts of (B,T,..) tensors). Writes the new
        weights back into each mind's real world model. `gnoise` (stacked
        (N,B,T,G,C)) can be supplied for reproducibility; otherwise it is drawn
        fresh (Gumbel-max categorical sampling)."""
        wrappers = [self._wrapper(m) for m in minds]
        base = wrappers[0]
        params, buffers = stack_module_state(wrappers)
        names = list(params.keys())

        # --- stack per-mind inputs (leading N dim) ---
        obs_keys = list(batches[0]["obs"].keys())
        obs = {k: torch.stack([b["obs"][k] for b in batches]).to(self.device)
               for k in obs_keys}
        actions = torch.stack([b["prev_action"] for b in batches]).to(self.device)
        reward = torch.stack([b["reward"] for b in batches]).to(self.device)
        cont = torch.stack([b["cont"] for b in batches]).to(self.device)
        N, B, T = actions.shape[:3]
        if gnoise is None:
            gnoise = gumbel_like(
                actions.new_empty(N, B, T, self.groups, self.classes))

        def loss_fn(p, bf, ob, ac, rw, ct, gn):
            return functional_call(base, (p, bf), (ob, ac, rw, ct, gn))

        grads, losses = vmap(grad_and_value(loss_fn),
                             in_dims=(0, 0, 0, 0, 0, 0, 0))(
            params, buffers, obs, actions, reward, cont, gnoise)

        # --- per-mind global grad-norm clip (matches clip_grad_norm_) ---
        sq = torch.zeros(N, device=self.device)
        for name in names:
            g = grads[name]
            sq = sq + (g.reshape(N, -1) ** 2).sum(-1)
        total_norm = torch.sqrt(sq)
        coef = (self.clip / (total_norm + 1e-6)).clamp(max=1.0)   # (N,)

        # --- functional Adam, per mind (elementwise; broadcasts over N) ---
        step = self._stack_step(minds)                # (N,)
        new_params, new_m, new_v = {}, {}, {}
        for name in names:
            g = grads[name] * coef.reshape(N, *([1] * (grads[name].ndim - 1)))
            m, v = self._stack_state(minds, name, params[name])
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            bc = step.reshape(N, *([1] * (m.ndim - 1)))   # per-mind step count
            mhat = m / (1 - self.b1 ** bc)
            vhat = v / (1 - self.b2 ** bc)
            new_params[name] = params[name] - self.lr * mhat / (torch.sqrt(vhat)
                                                                + self.eps)
            new_m[name], new_v[name] = m, v

        self._write_back(wrappers, new_params, names)
        self._save_state(minds, new_m, new_v, names)
        return {"loss": float(losses.mean().detach()),
                "grad_norm": float(total_norm.mean().detach())}

    # -------------------------------------------------------- Adam state helpers

    def _stack_step(self, minds):
        return torch.tensor(
            [self._state.get(id(m), {}).get("step", 0) + 1 for m in minds],
            device=self.device, dtype=torch.float32)

    def _stack_state(self, minds, name, ref):
        ms, vs = [], []
        for m in minds:
            st = self._state.get(id(m))
            if st is None or name not in st["m"]:
                ms.append(torch.zeros_like(ref[0]))
                vs.append(torch.zeros_like(ref[0]))
            else:
                ms.append(st["m"][name]); vs.append(st["v"][name])
        return torch.stack(ms), torch.stack(vs)

    def _write_back(self, wrappers, new_params, names):
        with torch.no_grad():
            for i, w in enumerate(wrappers):
                wp = dict(w.named_parameters())
                for name in names:
                    wp[name].data.copy_(new_params[name][i])

    def _save_state(self, minds, new_m, new_v, names):
        for i, m in enumerate(minds):
            st = self._state.get(id(m))
            if st is None:
                st = {"step": 0, "m": {}, "v": {}}
                self._state[id(m)] = st
            st["step"] += 1
            for name in names:
                st["m"][name] = new_m[name][i].detach()
                st["v"][name] = new_v[name][i].detach()
