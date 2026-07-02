"""Online agent: carries a live RSSM posterior state through real env steps
and picks actions with the actor. The shield (milestone 7) wraps the action
here, above the policy."""
from __future__ import annotations

import numpy as np
import torch

from ..model.rssm import RSSMState


class DreamerAgent:
    def __init__(self, world_model, actor_critic, device="cpu", shield=None):
        self.wm = world_model
        self.ac = actor_critic
        self.device = device
        self.shield = shield
        self.reset_state()

    def reset_state(self):
        self._state = self.wm.rssm.initial(1, self.device)
        self._prev_action = torch.zeros(1, 2, device=self.device)
        self.last_intervened = False

    @torch.no_grad()
    def _encode(self, obs):
        batch = {k: torch.as_tensor(v, device=self.device)[None].float()
                 for k, v in obs.items()}
        embed = self.wm.encoder(batch)
        post, _ = self.wm.rssm.obs_step(self._state, self._prev_action, embed)
        self._state = post
        return post.feat()

    @torch.no_grad()
    def act(self, obs, env=None, noise: float = 0.0,
            deterministic: bool = False) -> np.ndarray:
        feat = self._encode(obs)
        action = self.ac.actor.act(feat, noise=noise,
                                   deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        self.last_intervened = False
        if self.shield is not None and env is not None:
            safe, intervened = self.shield.filter(env, action_np)
            self.last_intervened = intervened
            action_np = safe
        self._prev_action = torch.as_tensor(
            action_np, device=self.device)[None].float()
        return action_np
