"""Evaluate a trained checkpoint: run episodes, report homeostatic survival
and reward vs the random baseline, and optionally save a rollout GIF.

  python -m embodied_agent.evaluate --config cpu_small \
      --checkpoint runs/cpu_small/checkpoint.pt --gif
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from .agent.actor_critic import ActorCritic
from .agent.agent import DreamerAgent
from .config import load_config
from .env import make_env
from .intrinsic import build_intrinsic
from .model.world_model import WorldModel
from .safety import build_shield
from .train import random_baseline, save_eval_gif
from .utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shield", default=None, choices=["on", "off"])
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.shield:
        cfg.shield.enabled = args.shield == "on"
    seed_everything(args.seed)
    device = "cpu"

    env = make_env(cfg, seed=args.seed)
    wm = WorldModel(cfg, env.sensors.spaces).to(device)
    intrinsic = build_intrinsic(cfg, wm.rssm.feat_dim, device)
    ac = ActorCritic(cfg, wm.rssm, wm, intrinsic=intrinsic)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    wm.load_state_dict(ckpt["wm"])
    ac.actor.load_state_dict(ckpt["actor"])
    ac.critic.load_state_dict(ckpt["critic"])
    shield = build_shield(cfg, env)
    agent = DreamerAgent(wm, ac, device=device, shield=shield)

    rewards, foods, lengths, covers, interventions = [], [], [], [], []
    cell = cfg.env.arena_size / 12
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + 100 + ep)
        agent.reset_state()
        total, food, steps, visited, interv = 0.0, 0, 0, set(), 0
        while True:
            action = agent.act(obs, env=env,
                               deterministic=not args.stochastic)
            interv += int(agent.last_intervened)
            obs, r, term, trunc, info = env.step(action)
            total += r
            food += info["food_eaten"]
            steps += 1
            visited.add((int(info["pos"][0] / cell),
                         int(info["pos"][1] / cell)))
            if term or trunc:
                break
        rewards.append(total)
        foods.append(food)
        lengths.append(steps)
        covers.append(len(visited))
        interventions.append(interv)

    base_r, base_cov = random_baseline(cfg, args.seed, episodes=args.episodes)
    print(f"evaluated {args.episodes} episodes "
          f"(shield={'on' if shield else 'off'})")
    print(f"  reward     {np.mean(rewards):8.1f} +- {np.std(rewards):.1f} "
          f"(random {base_r:.1f})")
    print(f"  food eaten {np.mean(foods):8.2f}")
    print(f"  ep length  {np.mean(lengths):8.1f} / {cfg.env.max_episode_steps}")
    print(f"  coverage   {np.mean(covers):8.1f} (random {base_cov:.1f})")
    if shield:
        print(f"  shield interventions/ep {np.mean(interventions):.1f}")

    if args.gif:
        out = pathlib.Path(args.checkpoint).parent / "evaluate_rollout.gif"
        save_eval_gif(agent, cfg, args.seed, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
