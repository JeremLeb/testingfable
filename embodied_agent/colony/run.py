"""Run a living colony: one shared "species brain" (world model + actor) learns
from the pooled experience of every creature, while each creature keeps its own
body, recurrent state and genome. Births attach a fresh brain-state + memory
stream; deaths flush that stream into the shared buffer.
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


def run_colony(cfg, verbose: bool = True, on_step=None, should_stop=None) -> dict:
    seed = cfg.train.seed
    seed_everything(seed)
    device = "cuda" if (cfg.train.device == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    log = print if verbose else (lambda *a, **k: None)

    spaces = SensorSuite(cfg.sensor, cfg.env,
                         np.random.default_rng(seed)).spaces
    wm = WorldModel(cfg, spaces).to(device)
    intrinsic = build_intrinsic(cfg, wm.rssm.feat_dim, device)
    ac = ActorCritic(cfg, wm.rssm, wm, intrinsic=intrinsic)
    ac.actor.to(device); ac.critic.to(device); ac.target_critic.to(device)
    buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    env = ColonyEnv(cfg, seed=seed)
    seg = max(2 * cfg.train.seq_len, 96)   # memory-segment length per creature
    rng = np.random.default_rng(seed)

    def attach(c):
        c.agent = DreamerAgent(wm, ac, device=device)
        c.episode = Episode()
        c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)

    for c in env.reset():
        attach(c)

    gui_every = cfg.train.gui_every or 8
    live = {"metrics": {}}
    t0 = time.time()
    log(f"colony | device={device} | wm params "
        f"{sum(p.numel() for p in wm.parameters())/1e3:.0f}k | "
        f"start pop {len(env.living)}")

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
            if not done and len(c.episode) >= seg:   # segment long lives
                buffer.ingest(c.episode)
                c.episode = Episode()
                c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)
        for d in deaths:
            if d.episode is not None:
                buffer.ingest(d.episode)
            d.agent = None; d.episode = None
        for b in births:
            attach(b)

        if step % cfg.train.train_every == 0 and \
                buffer.can_sample(cfg.train.seq_len):
            for _ in range(cfg.train.updates_per_train):
                batch = buffer.sample(cfg.train.batch_size,
                                      cfg.train.seq_len, rng)
                post, wm_metrics = wm.train_step(batch)
                ac_metrics = ac.train_step(post)
                if intrinsic is not None:
                    intrinsic.train_step(
                        post.feat().reshape(-1, wm.rssm.feat_dim).detach())
                live["metrics"] = {**wm_metrics, **ac_metrics}

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

    for c in env.living:  # flush surviving memories
        if c.episode is not None:
            buffer.ingest(c.episode)
    stats = env.stats()
    log(f"\nFINAL | pop {stats['population']} | total births {stats['births']} "
        f"| deaths {stats['deaths']} | max generation {stats['max_generation']}")
    return {"stats": stats, "buffer_steps": buffer.num_steps}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="colony")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    return run_colony(cfg)


if __name__ == "__main__":
    main()
