"""The world model: encoders + RSSM + decoders + reward/continue heads.

Trains on real sequences from the replay buffer. Loss = per-modality
reconstruction + balanced KL (with free bits) + reward + continue. This is
the prediction-error substrate: with the reward/continue heads ablated it
still learns a usable model of the sensory stream (the --no-reward ablation
in scripts/train_world_model.py checks exactly this).

Advanced (Tier 1) options, selected from config:
  * discrete categorical latents in the RSSM (vs the original Gaussian)
  * a symlog two-hot reward head (vs plain MSE), robust to the wide reward
    range this environment produces
  * image modalities (the pixel retina) reconstructed by a CNN decoder
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .distributions import TwoHotHead
from .networks import MultiDecoder, MultiEncoder, mlp
from .rssm import RSSM, RSSMState


class WorldModel(nn.Module):
    def __init__(self, cfg: Config, obs_spaces: dict, action_dim: int = 2):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        self.obs_spaces = obs_spaces
        self.recon_scales = mc.recon_scales

        self.encoder = MultiEncoder(obs_spaces, mc.embed_dim, mc.hidden,
                                    mc.cnn_depth)
        self.rssm = RSSM(
            action_dim, mc.embed_dim, mc.deter_dim, mc.hidden,
            latent_kind=mc.latent_kind, stoch_dim=mc.stoch_dim,
            groups=mc.latent_groups, classes=mc.latent_classes,
            unimix=mc.unimix, sparse_latent=mc.sparse_latent,
            sparse_frac=mc.sparse_frac)
        feat = self.rssm.feat_dim
        self.decoder = MultiDecoder(obs_spaces, feat, mc.hidden, mc.cnn_depth)

        self.reward_kind = mc.reward_head
        if self.reward_kind == "twohot":
            self.reward_head = TwoHotHead(feat, mc.hidden, mc.reward_bins,
                                          mc.reward_low, mc.reward_high)
        else:
            self.reward_head = mlp(feat, mc.hidden, 1, layers=2)
        self.cont_head = mlp(feat, mc.hidden, 1, layers=2)

        self.opt = torch.optim.Adam(self.parameters(), lr=mc.lr)

    # ------------------------------------------------------------ prediction

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.reward_kind == "twohot":
            return self.reward_head.mean(feat)
        return self.reward_head(feat).squeeze(-1)

    def predict_cont(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cont_head(feat)).squeeze(-1)

    def _reward_loss(self, feat, target):
        if self.reward_kind == "twohot":
            return self.reward_head.loss(feat, target).mean()
        return F.mse_loss(self.reward_head(feat).squeeze(-1), target)

    # ------------------------------------------------------------ loss

    def loss(self, batch: dict, use_reward: bool = True):
        obs = batch["obs"]
        actions = batch["prev_action"]
        B, T = actions.shape[:2]

        embeds = self.encoder(obs)
        post, prior = self.rssm.observe(embeds, actions)
        feat = post.feat()

        recon = self.decoder(feat)
        recon_losses, recon_total = {}, feat.new_zeros(())
        for name, target in obs.items():
            err = (recon[name] - target).reshape(B, T, -1)
            per = (err ** 2).sum(-1)                     # sum over features
            scaled = self.recon_scales.get(name, 1.0) * per.mean()
            recon_losses[name] = scaled
            recon_total = recon_total + scaled

        kl = self.rssm.kl_loss(post, prior, self.cfg.model.free_bits,
                               self.cfg.model.kl_balance).mean()

        loss = recon_total + self.cfg.model.kl_beta * kl
        s = lambda x: float(x.detach())  # noqa: E731
        metrics = {"recon": s(recon_total), "kl": s(kl)}
        for name, v in recon_losses.items():
            metrics[f"recon_{name}"] = s(v)

        if use_reward:
            rew_loss = self._reward_loss(feat, batch["reward"])
            cont_loss = F.binary_cross_entropy_with_logits(
                self.cont_head(feat).squeeze(-1), batch["cont"])
            loss = loss + rew_loss + cont_loss
            metrics["reward"] = s(rew_loss)
            metrics["cont"] = s(cont_loss)

        metrics["loss"] = s(loss)
        return loss, post, metrics

    def train_step(self, batch: dict, use_reward: bool = True):
        loss, post, metrics = self.loss(batch, use_reward=use_reward)
        self.opt.zero_grad()
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.parameters(),
                                        self.cfg.model.grad_clip)
        self.opt.step()
        metrics["grad_norm"] = float(grad)
        return post.detach(), metrics

    # ------------------------------------------------------------ evaluation

    @torch.no_grad()
    def open_loop_error(self, batch: dict, context: int) -> dict:
        """Multi-step prediction error: filter `context` steps with the
        posterior, then roll the prior open-loop for the remainder and
        measure per-step reconstruction MSE against the real observations."""
        obs = batch["obs"]
        actions = batch["prev_action"]
        T = actions.shape[1]
        embeds = self.encoder(obs)
        post, _ = self.rssm.observe(embeds[:, :context], actions[:, :context])
        state = RSSMState(post.h[:, -1], post.z[:, -1])

        errors = []
        for t in range(context, T):
            state = self.rssm.prior_step(state, actions[:, t])
            recon = self.decoder(state.feat())
            step_err = sum(
                F.mse_loss(recon[k], obs[k][:, t]) for k in obs
            ) / len(obs)
            errors.append(float(step_err))
        return {"open_loop_mse": sum(errors) / max(len(errors), 1),
                "per_step": errors}
