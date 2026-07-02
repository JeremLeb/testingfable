"""The world model: encoders + RSSM + decoders + reward/continue heads.

Trains on real sequences from the replay buffer. Loss = per-modality
reconstruction (Gaussian NLL / MSE) + balanced KL with free bits + reward
regression + continue (Bernoulli) prediction. This is the prediction-error
substrate: with the reward/continue heads ablated it still learns a usable
model of the sensory stream (a milestone-8 ablation checks exactly this).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .networks import MultiDecoder, MultiEncoder, mlp
from .rssm import RSSM, RSSMState, kl_divergence


class WorldModel(nn.Module):
    def __init__(self, cfg: Config, obs_spaces: dict[str, int],
                 action_dim: int = 2):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        self.obs_spaces = obs_spaces
        self.recon_scales = mc.recon_scales

        self.encoder = MultiEncoder(obs_spaces, mc.embed_dim, mc.hidden)
        self.rssm = RSSM(action_dim, mc.embed_dim, mc.deter_dim,
                         mc.stoch_dim, mc.hidden)
        feat = self.rssm.feat_dim
        self.decoder = MultiDecoder(obs_spaces, feat, mc.hidden)
        self.reward_head = mlp(feat, mc.hidden, 1, layers=2)
        self.cont_head = mlp(feat, mc.hidden, 1, layers=2)

        self.opt = torch.optim.Adam(self.parameters(), lr=mc.lr)

    # ------------------------------------------------------------ prediction

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reward_head(feat).squeeze(-1)

    def predict_cont(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cont_head(feat)).squeeze(-1)

    # ------------------------------------------------------------ loss

    def loss(self, batch: dict, use_reward: bool = True):
        obs = batch["obs"]
        actions = batch["prev_action"]
        B, T = actions.shape[:2]

        embeds = self.encoder({k: v for k, v in obs.items()})
        post, prior = self.rssm.observe(embeds, actions)
        feat = post.feat()

        recon = self.decoder(feat)
        recon_losses = {}
        recon_total = feat.new_zeros(())
        for name, target in obs.items():
            per = F.mse_loss(recon[name], target, reduction="none").sum(-1)
            scaled = self.recon_scales.get(name, 1.0) * per.mean()
            recon_losses[name] = scaled
            recon_total = recon_total + scaled

        kl = kl_divergence(post, prior, self.cfg.model.free_bits,
                           self.cfg.model.kl_balance).mean()

        loss = recon_total + self.cfg.model.kl_beta * kl
        s = lambda x: float(x.detach())  # noqa: E731
        metrics = {"recon": s(recon_total), "kl": s(kl)}
        for name, v in recon_losses.items():
            metrics[f"recon_{name}"] = s(v)

        if use_reward:
            rew_pred = self.predict_reward(feat)
            rew_loss = F.mse_loss(rew_pred, batch["reward"])
            cont_pred = self.cont_head(feat).squeeze(-1)
            cont_loss = F.binary_cross_entropy_with_logits(
                cont_pred, batch["cont"])
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
        measure per-step reconstruction MSE against the real observations.
        Decreasing error here is the milestone-4 success signal.
        """
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
