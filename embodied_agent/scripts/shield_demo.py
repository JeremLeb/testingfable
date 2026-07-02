"""Milestone-7 demo: show the hard-constraint shield is respected and logged.

An adversarial policy drives straight at the forbidden zone. With the shield
off the agent enters it; with the shield on entry count should be ~0 while
interventions are logged. A GIF is saved with the zone drawn.

  python -m embodied_agent.scripts.shield_demo --config cpu_small
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from ..config import load_config
from ..env import make_env
from ..safety import build_shield
from ..viz.render import ArenaRenderer, save_gif


def adversarial_action(env, zone):
    """Full thrust straight toward the zone center."""
    to_zone = zone[:2] - env.pos
    desired = np.arctan2(to_zone[1], to_zone[0])
    turn = np.clip((desired - env.heading + np.pi) % (2 * np.pi) - np.pi, -1, 1)
    return np.array([1.0, turn])


def run(cfg, shield_on, steps, seed, save_path=None):
    cfg.shield.enabled = True  # always build to define the zone for measurement
    env = make_env(cfg, seed=seed)
    shield = build_shield(cfg, env)
    zone = np.asarray(env.forbidden_zones[0])
    env.reset(seed=seed)
    renderer = ArenaRenderer(env) if save_path else None
    frames, entries, interventions = [], 0, 0
    r = cfg.env.agent_radius
    for t in range(steps):
        action = adversarial_action(env, zone)
        if shield_on:
            action, intervened = shield.filter(env, action)
            interventions += int(intervened)
        env.step(action)
        if np.linalg.norm(env.pos - zone[:2]) < zone[2] + r:
            entries += 1
        if renderer and t % 2 == 0:
            frames.append(renderer.render(env._info(0)))
    if renderer:
        save_gif(frames, str(save_path))
        renderer.close()
    return {"entries": entries, "interventions": interventions,
            "min_dist": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/shield_demo")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    off = run(load_config(args.config), False, args.steps, args.seed)
    on = run(load_config(args.config), True, args.steps, args.seed,
             out / "shield_on.gif")
    print(f"adversarial policy driving at the forbidden zone "
          f"({args.steps} steps):")
    print(f"  shield OFF: zone entries={off['entries']:3d}  "
          f"interventions={off['interventions']}")
    print(f"  shield ON : zone entries={on['entries']:3d}  "
          f"interventions={on['interventions']}")
    assert on["entries"] == 0, "shield failed: agent entered forbidden zone"
    print("  OK: shield respected (0 entries) and interventions logged")
    print(f"wrote {out/'shield_on.gif'}")


if __name__ == "__main__":
    main()
