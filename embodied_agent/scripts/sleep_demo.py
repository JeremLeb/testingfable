"""B2 demo: sleep, consolidation & dreaming.

Shows the three claims of the module, each ON vs its ablation:

  (1) Prioritized ("emotional") replay preferentially samples *salient*
      memories -- episodes with big reward spikes (feeding, collisions,
      death) -- whereas the baseline replays uniformly.
  (2) A consolidation curve: during a single simulated night, world-model
      error on a held-fixed probe batch drops as salient experience is
      replayed; prioritized replay consolidates faster than uniform.
  (3, with --ablate) End-to-end: a short life with sleep + prioritized replay
      vs uniform replay -> sample efficiency (final world-model loss).

  python -m embodied_agent.scripts.sleep_demo --config cpu_small
  python -m embodied_agent.scripts.sleep_demo --config cpu_small --ablate
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from ..config import load_config
from ..agent.collect import collect_random
from ..agent.replay import ReplayBuffer
from ..model.world_model import WorldModel
from ..env import make_env


def _populate(cfg, seed, steps):
    """A buffer of random-agent life, with the world model to train on it."""
    env = make_env(cfg, seed=seed)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)
    collect_random(env, buffer, steps, np.random.default_rng(seed))
    wm = WorldModel(cfg, env.sensors.spaces)
    return buffer, wm


def _replay_salience(buffer, cfg):
    """Empirical: draw many episodes prioritized vs uniform; return the peak
    salience of each drawn episode for both regimes."""
    seq = cfg.train.seq_len
    rng = np.random.default_rng(0)
    sal_p, prob_p = buffer.episode_priorities(seq, cfg.sleep)
    sal_u, prob_u = buffer.episode_priorities(seq, None)
    draws = 4000
    idx_p = rng.choice(len(sal_p), size=draws, p=prob_p)
    idx_u = rng.choice(len(sal_u), size=draws, p=prob_u)
    return sal_p, prob_p, sal_p[idx_p], sal_u[idx_u]


def _consolidation_curve(cfg, buffer, wm, updates, prioritized):
    """One night: snapshot a fixed *salient* probe batch (the memories that
    matter -- feeding/collision/near-death), then run `updates` consolidation
    steps and record probe loss after each. Prioritized replay should drive
    error on these salient memories down faster than uniform replay."""
    rng = np.random.default_rng(1)
    probe = buffer.sample(cfg.train.batch_size, cfg.train.seq_len, rng,
                          sleep_cfg=cfg.sleep)

    @torch.no_grad()
    def probe_loss():
        return float(wm.loss(probe)[0].detach())

    curve = [probe_loss()]
    for _ in range(updates):
        batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len, rng,
                              sleep_cfg=cfg.sleep if prioritized else None)
        wm.train_step(batch)
        curve.append(probe_loss())
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collect", type=int, default=4000)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--out", default="runs/sleep_demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.sleep.enabled = True

    # (1) prioritized vs uniform replay salience
    buffer, wm = _populate(cfg, args.seed, args.collect)
    sal, prob, drawn_p, drawn_u = _replay_salience(buffer, cfg)

    # (2) consolidation curve, prioritized vs uniform (fresh model each)
    _, wm_p = _populate(cfg, args.seed, args.collect)
    _, wm_u = _populate(cfg, args.seed, args.collect)
    curve_p = _consolidation_curve(cfg, buffer, wm_p, args.updates, True)
    curve_u = _consolidation_curve(cfg, buffer, wm_u, args.updates, False)

    ablation = _end_to_end(args) if args.ablate else None

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(sal, prob, drawn_p, drawn_u, curve_p, curve_u, ablation,
          out / "sleep_demo.png")

    print("=== B2 sleep, consolidation & dreaming ===")
    hi = sal > np.median(sal)
    print(f"prioritized replay draws high-salience episodes "
          f"{drawn_p.mean() / max(drawn_u.mean(), 1e-6):.1f}x more (mean peak "
          f"|reward| {drawn_p.mean():.2f} vs uniform {drawn_u.mean():.2f})")
    print(f"consolidation over {args.updates} updates: probe loss "
          f"{curve_p[0]:.2f} -> {curve_p[-1]:.2f} (prioritized), "
          f"{curve_u[0]:.2f} -> {curve_u[-1]:.2f} (uniform)")
    if ablation:
        print(f"end-to-end final wm loss: prioritized {ablation['on']:.2f} "
              f"vs uniform {ablation['off']:.2f}")
    print(f"wrote {out / 'sleep_demo.png'}")


def _end_to_end(args) -> dict:
    from ..train import run
    results = {}
    for label, prioritized in (("on", True), ("off", False)):
        cfg = load_config(args.config)
        cfg.sleep.enabled = True
        cfg.sleep.prioritized = prioritized
        cfg.train.total_steps = args.train_steps
        cfg.train.out_dir = f"{args.out}/train_{label}"
        cfg.train.seed = args.seed
        summary = run(cfg, verbose=False)
        results[label] = summary["final"]["eval_reward"]
    return results


def _plot(sal, prob, drawn_p, drawn_u, curve_p, curve_u, ablation, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 3 if ablation else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    ax = axes[0]
    bins = np.linspace(0, max(drawn_p.max(), drawn_u.max()) + 1e-6, 24)
    ax.hist(drawn_u, bins=bins, alpha=0.6, color="gray", label="uniform")
    ax.hist(drawn_p, bins=bins, alpha=0.6, color="tab:red",
            label="prioritized")
    ax.set_title("Replayed-episode salience\n(emotional memory)")
    ax.set_xlabel("peak |reward| of drawn episode")
    ax.set_ylabel("count (of 4000 draws)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(curve_u, color="gray", label="uniform")
    ax.plot(curve_p, color="tab:blue", label="prioritized")
    ax.set_title("Consolidation curve\n(loss on salient memories)")
    ax.set_xlabel("consolidation update")
    ax.set_ylabel("world-model probe loss")
    ax.legend(fontsize=8)

    if ablation:
        ax = axes[2]
        ax.bar(["prioritized", "uniform"], [ablation["on"], ablation["off"]],
               color=["tab:blue", "gray"])
        ax.set_title("End-to-end eval reward")
        ax.set_ylabel("mean episode reward")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
