"""Milestone-1 demo: watch a random agent wander the arena.

Usage: python -m embodied_agent.scripts.random_rollout --config cpu_small
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from ..config import load_config
from ..env import make_env
from ..viz.render import ArenaRenderer, save_gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/random_rollout")
    ap.add_argument("--render-every", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = make_env(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    renderer = ArenaRenderer(env)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset(seed=args.seed)
    print("observation shapes:",
          {k: v.shape for k, v in obs.items()})
    frames, collisions, eaten, deaths = [], 0, 0, 0
    # smoothed random walk so the agent visibly explores instead of jittering
    action = np.zeros(2)
    for t in range(args.steps):
        action = 0.8 * action + 0.2 * rng.uniform(-1, 1, 2)
        obs, reward, terminated, truncated, info = env.step(action)
        collisions += int(info["collision"])
        eaten += info["food_eaten"]
        if t % args.render_every == 0:
            frames.append(renderer.render(info))
        if terminated or truncated:
            deaths += int(terminated)
            obs, info = env.reset()

    gif_path = out / "random_rollout.gif"
    save_gif(frames, str(gif_path))
    renderer.close()
    print(f"steps={args.steps} collisions={collisions} "
          f"food_eaten={eaten} deaths={deaths}")
    print(f"final intero: energy={info.get('energy', float('nan')):.3f} "
          f"temp={info.get('temp', float('nan')):.3f} "
          f"integrity={info.get('integrity', float('nan')):.3f}")
    print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
