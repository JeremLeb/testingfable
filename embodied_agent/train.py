"""End-to-end training: collect real experience with the actor, train the
world model on replayed sequences, and train the actor-critic in
imagination. Milestone 5 onward.

  python -m embodied_agent.train --config cpu_small
  python -m embodied_agent.train --config cpu_small --intrinsic none --shield off

Intrinsic motivation (milestone 6) and the safety shield (milestone 7) are
wired in here behind config flags so later milestones only flip switches.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from .agent.actor_critic import ActorCritic
from .agent.agent import DreamerAgent
from .agent.collect import collect_random
from .agent.replay import ReplayBuffer
from .config import load_config
from .env import make_env
from .intrinsic import build_intrinsic
from .model.world_model import WorldModel
from .safety import build_shield
from .utils.logger import CSVLogger
from .utils.seeding import seed_everything


def evaluate(agent, cfg, seed, episodes=3, deterministic=True):
    """Run the current policy and return mean episode reward + diagnostics."""
    env = make_env(cfg, seed=seed + 777)
    totals, lengths, eaten, coverage = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 777 + ep)
        agent.reset_state()
        total, steps, food = 0.0, 0, 0
        visited = set()
        cell = cfg.env.arena_size / 12
        while True:
            action = agent.act(obs, env=env, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            steps += 1
            food += info["food_eaten"]
            visited.add((int(info["pos"][0] / cell), int(info["pos"][1] / cell)))
            if term or trunc:
                break
        totals.append(total)
        lengths.append(steps)
        eaten.append(food)
        coverage.append(len(visited))
    return {
        "eval_reward": float(np.mean(totals)),
        "eval_length": float(np.mean(lengths)),
        "eval_food": float(np.mean(eaten)),
        "eval_coverage": float(np.mean(coverage)),
    }


def random_baseline(cfg, seed, episodes=3):
    env = make_env(cfg, seed=seed + 999)
    rng = np.random.default_rng(seed + 999)
    totals, cover = [], []
    cell = cfg.env.arena_size / 12
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 999 + ep)
        a = np.zeros(2)
        total, visited = 0.0, set()
        while True:
            a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
            obs, reward, term, trunc, info = env.step(a)
            total += reward
            visited.add((int(info["pos"][0] / cell), int(info["pos"][1] / cell)))
            if term or trunc:
                break
        totals.append(total)
        cover.append(len(visited))
    return float(np.mean(totals)), float(np.mean(cover))


def save_eval_gif(agent, cfg, seed, path):
    from .viz.render import ArenaRenderer, save_gif
    env = make_env(cfg, seed=seed + 5)
    obs, _ = env.reset(seed=seed + 5)
    agent.reset_state()
    renderer = ArenaRenderer(env)
    frames = []
    for t in range(cfg.env.max_episode_steps):
        action = agent.act(obs, env=env, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if t % 3 == 0:
            frames.append(renderer.render(info))
        if term or trunc:
            break
    save_gif(frames, str(path))
    renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--intrinsic", default=None,
                    help="override intrinsic method: none|rnd|disagreement")
    ap.add_argument("--shield", default=None, choices=["on", "off"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.out:
        cfg.train.out_dir = args.out
    if args.intrinsic:
        cfg.intrinsic.method = args.intrinsic
    if args.shield:
        cfg.shield.enabled = args.shield == "on"

    seed = cfg.train.seed
    seed_everything(seed)
    device = "cuda" if (cfg.train.device == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    out = pathlib.Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "metrics.csv")

    env = make_env(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    wm = WorldModel(cfg, env.sensors.spaces).to(device)
    intrinsic = build_intrinsic(cfg, wm.rssm.feat_dim, device)
    ac = ActorCritic(cfg, wm.rssm, wm, intrinsic=intrinsic)
    ac.actor.to(device)
    ac.critic.to(device)
    ac.target_critic.to(device)
    shield = build_shield(cfg, env)
    agent = DreamerAgent(wm, ac, device=device, shield=shield)

    print(f"device={device} | wm params "
          f"{sum(p.numel() for p in wm.parameters())/1e3:.0f}k | "
          f"intrinsic={cfg.intrinsic.method} | "
          f"shield={'on' if shield else 'off'}")

    # warmup with random policy to seed the buffer and the world model
    print(f"warmup: {cfg.train.warmup_steps} random steps")
    collect_random(env, buffer, cfg.train.warmup_steps, rng)

    base_reward, base_cover = random_baseline(cfg, seed)
    print(f"random baseline: reward={base_reward:.1f} coverage={base_cover:.1f}")

    obs, _ = env.reset(seed=seed)
    agent.reset_state()
    buffer.start_episode()
    buffer.add(obs, np.zeros(2), 0.0, 1.0)
    ep_reward, ep_len, interventions = 0.0, 0, 0
    t0 = time.time()

    for step in range(1, cfg.train.total_steps + 1):
        action = agent.act(obs, env=env, noise=cfg.agent.expl_noise)
        interventions += int(agent.last_intervened)
        obs, reward, term, trunc, info = env.step(action)
        buffer.add(obs, action, reward, 0.0 if term else 1.0)
        ep_reward += reward
        ep_len += 1
        logger.add(reward=reward, energy=info["energy"], temp=info["temp"],
                   integrity=info["integrity"],
                   drive_energy=info["drive_energy"],
                   drive_thermal=info["drive_thermal"],
                   drive_integrity=info["drive_integrity"])
        for k in ("reward_energy", "reward_thermal", "reward_integrity"):
            logger.add(**{k: info.get(k, 0.0)})

        if term or trunc:
            logger.add(ep_reward=ep_reward, ep_len=ep_len,
                       interventions=interventions)
            buffer.end_episode()
            obs, _ = env.reset()
            agent.reset_state()
            buffer.start_episode()
            buffer.add(obs, np.zeros(2), 0.0, 1.0)
            ep_reward, ep_len, interventions = 0.0, 0, 0

        if step % cfg.train.train_every == 0 and \
                buffer.can_sample(cfg.train.seq_len):
            for _ in range(cfg.train.updates_per_train):
                batch = buffer.sample(cfg.train.batch_size,
                                      cfg.train.seq_len, rng)
                post, wm_metrics = wm.train_step(batch)
                ac_metrics = ac.train_step(post)
                if intrinsic is not None:
                    intr_metrics = intrinsic.train_step(
                        post.feat().reshape(-1, wm.rssm.feat_dim).detach())
                    logger.add(**intr_metrics)
                logger.add(**wm_metrics, **ac_metrics)

        if step % cfg.train.log_every == 0:
            row = logger.flush(step)
            sps = step / (time.time() - t0)
            print(f"step {step:6d} | ep_reward {row.get('ep_reward', 0):7.1f} "
                  f"| wm_loss {row.get('loss', 0):6.2f} "
                  f"| imag_ret {row.get('imag_return', 0):6.2f} "
                  f"| entropy {row.get('actor_entropy', 0):5.2f} "
                  f"| {sps:.0f} steps/s")

        if step % cfg.train.eval_every == 0:
            ev = evaluate(agent, cfg, seed)
            ev["baseline_reward"] = base_reward
            ev["baseline_coverage"] = base_cover
            logger.add(**ev)
            print(f"  [eval] reward {ev['eval_reward']:.1f} "
                  f"(random {base_reward:.1f}) | food {ev['eval_food']:.1f} "
                  f"| coverage {ev['eval_coverage']:.1f} "
                  f"(random {base_cover:.1f})")

        if step % cfg.train.gif_every == 0:
            save_eval_gif(agent, cfg, seed, out / f"rollout_{step}.gif")

    torch.save({"wm": wm.state_dict(), "actor": ac.actor.state_dict(),
                "critic": ac.critic.state_dict()}, out / "checkpoint.pt")
    final = evaluate(agent, cfg, seed, episodes=5)
    print(f"\nFINAL eval reward {final['eval_reward']:.1f} "
          f"(random {base_reward:.1f}) | food {final['eval_food']:.1f} "
          f"| coverage {final['eval_coverage']:.1f} (random {base_cover:.1f})")
    logger.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
