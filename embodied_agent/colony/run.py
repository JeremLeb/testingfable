"""Run a living colony where **each creature is its own individual**: its own
senses, its own body, its own genome, AND its own mind -- a private world model
+ actor + memory that learns only from its own life. Newborns inherit a copy of
their parent's brain (nature), then diverge through their own experience
(nurture); when a creature dies, its mind dies with it.

(Set ``colony.shared_brain: true`` for the alternative pooled "species brain".)
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..agent.actor_critic import ActorCritic
from ..agent.agent import DreamerAgent
from ..agent.replay import Episode, ReplayBuffer
from ..config import load_config
from ..env.sensors import SensorSuite
from ..intrinsic import build_intrinsic
from ..model.world_model import WorldModel
from ..utils.seeding import seed_everything
from .world import ColonyEnv


class Mind:
    """One creature's private brain: world model + actor-critic + memory."""

    def __init__(self, cfg, spaces, device):
        self.wm = WorldModel(cfg, spaces).to(device)
        self.intrinsic = build_intrinsic(cfg, self.wm.rssm.feat_dim, device)
        self.ac = ActorCritic(cfg, self.wm.rssm, self.wm,
                              intrinsic=self.intrinsic)
        self.ac.actor.to(device)
        self.ac.critic.to(device)
        self.ac.target_critic.to(device)
        self.buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    def inherit_from(self, other: "Mind"):
        """Nature: be born knowing what a parent knew (a copy of its weights)."""
        self.wm.load_state_dict(other.wm.state_dict())
        self.ac.actor.load_state_dict(other.ac.actor.state_dict())
        self.ac.critic.load_state_dict(other.ac.critic.state_dict())
        self.ac.target_critic.load_state_dict(other.ac.target_critic.state_dict())


def run_colony(cfg, verbose: bool = True, on_step=None, should_stop=None) -> dict:
    seed = cfg.train.seed
    seed_everything(seed)
    device = "cuda" if (cfg.train.device == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    log = print if verbose else (lambda *a, **k: None)
    shared = cfg.colony.shared_brain

    spaces = SensorSuite(cfg.sensor, cfg.env,
                         np.random.default_rng(seed)).spaces
    shared_mind = Mind(cfg, spaces, device) if shared else None
    env = ColonyEnv(cfg, seed=seed)
    seg = max(2 * cfg.train.seq_len, 96)
    rng = np.random.default_rng(seed)

    def attach(c):
        if shared:
            c.mind = shared_mind
        else:
            c.mind = Mind(cfg, spaces, device)
            if cfg.colony.brain_inherit and c.parent is not None \
                    and c.parent.mind is not None:
                c.mind.inherit_from(c.parent.mind)
        c.agent = DreamerAgent(c.mind.wm, c.mind.ac, device=device)
        c.episode = Episode()
        c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)

    for c in env.reset():
        attach(c)

    gui_every = cfg.train.gui_every or 8
    live = {"metrics": {}}
    t0 = time.time()
    ptr = 0  # round-robin training pointer over living creatures
    n_params = sum(p.numel() for p in env.living[0].mind.wm.parameters())
    log(f"colony | device={device} | {'shared' if shared else 'individual'} "
        f"minds | brain {n_params/1e3:.0f}k params each | start pop "
        f"{len(env.living)}")

    for step in range(1, cfg.train.total_steps + 1):
        if should_stop is not None and should_stop():
            log("stop requested"); break

        acting = env.living
        actions = {c.id: c.agent.act(c.obs, bias=c.action_bias,
                                     noise=cfg.agent.expl_noise)
                   for c in acting}
        results, births, deaths = env.step(actions)

        for c in acting:
            r = results.get(c.id)
            if r is None or c.episode is None:
                continue
            obs2, reward, done, _ = r
            c.episode.add(obs2, actions[c.id], reward, 0.0 if done else 1.0)
            if not done and len(c.episode) >= seg:
                c.mind.buffer.ingest(c.episode)
                c.episode = Episode()
                c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)
        for d in deaths:
            # a shared brain keeps the memory; a private mind dies with the body
            if shared and d.episode is not None:
                shared_mind.buffer.ingest(d.episode)
            d.agent = d.episode = d.mind = None
        for b in births:
            attach(b)

        # train a rotating handful of minds (bounds real-time cost); only on
        # every train_every-th step so heavy per-brain training doesn't dominate.
        living = env.living
        trained, seen, unique = 0, 0, set()
        while step % max(cfg.colony.train_every, 1) == 0 and living \
                and trained < cfg.colony.max_trains_per_step \
                and seen < len(living):
            c = living[ptr % len(living)]
            ptr += 1
            seen += 1
            mid = id(c.mind)
            if mid in unique:
                continue
            unique.add(mid)
            if c.mind.buffer.can_sample(cfg.train.seq_len):
                batch = c.mind.buffer.sample(cfg.train.batch_size,
                                             cfg.train.seq_len, rng)
                post, wm_m = c.mind.wm.train_step(batch)
                ac_m = c.mind.ac.train_step(post)
                if c.mind.intrinsic is not None:
                    c.mind.intrinsic.train_step(
                        post.feat().reshape(-1, c.mind.wm.rssm.feat_dim).detach())
                live["metrics"] = {**wm_m, **ac_m}
                trained += 1

        if step % max(cfg.train.log_every, 1) == 0:
            s = env.stats()
            sps = step / (time.time() - t0)
            log(f"step {step:6d} | pop {s['population']:3d} | births "
                f"{s['births']:4d} deaths {s['deaths']:4d} | maxgen "
                f"{s['max_generation']:2d} | mean E {s['mean_energy']:.2f} | "
                f"wm {live['metrics'].get('loss', 0):.2f} | {sps:.0f} it/s")

        if on_step is not None and step % gui_every == 0:
            on_step({"step": step, "env": env, "stats": env.stats(),
                     "metrics": live["metrics"], "sps": step / (time.time() - t0)})

    stats = env.stats()
    log(f"\nFINAL | pop {stats['population']} | total births {stats['births']} "
        f"| deaths {stats['deaths']} | max generation {stats['max_generation']}")
    return {"stats": stats, "shared_brain": shared}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="colony")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shared-brain", action="store_true",
                    help="one pooled brain instead of individual minds")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.shared_brain:
        cfg.colony.shared_brain = True
    return run_colony(cfg)


if __name__ == "__main__":
    main()
