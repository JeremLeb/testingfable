"""Milestone-4 demo: train the world model offline on random trajectories
and show reconstruction + multi-step (open-loop) prediction improving.

This validates the prediction-error substrate *before* any policy exists.

Usage: python -m embodied_agent.scripts.train_world_model --config cpu_small
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from ..agent.collect import collect_random
from ..agent.replay import ReplayBuffer
from ..config import load_config
from ..env import make_env
from ..model.world_model import WorldModel
from ..utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--collect-steps", type=int, default=8000)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/world_model")
    ap.add_argument("--no-reward", action="store_true",
                    help="ablate reward/continue heads (substrate test): the "
                         "model trains on prediction only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    device = cfg.train.device if torch.cuda.is_available() \
        or cfg.train.device == "cpu" else "cpu"

    env = make_env(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    print(f"collecting {args.collect_steps} random steps...")
    stats = collect_random(env, buffer, args.collect_steps, rng)
    print(f"  buffer steps={buffer.num_steps} episodes={len(buffer.episodes)} "
          f"food_eaten={stats['food_eaten']} deaths={stats['deaths']}")

    wm = WorldModel(cfg, env.sensors.spaces).to(device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"world model params: {n_params/1e3:.1f}k")

    seq_len = cfg.train.seq_len
    context = seq_len // 2
    history = []
    for step in range(1, args.updates + 1):
        batch = buffer.sample(cfg.train.batch_size, seq_len, rng)
        _, metrics = wm.train_step(batch, use_reward=not args.no_reward)
        if step % 100 == 0 or step == 1:
            eval_batch = buffer.sample(cfg.train.batch_size, seq_len, rng)
            ol = wm.open_loop_error(eval_batch, context)
            metrics["open_loop_mse"] = ol["open_loop_mse"]
            history.append((step, metrics))
            print(f"step {step:5d} | loss {metrics['loss']:7.3f} "
                  f"| recon {metrics['recon']:7.3f} | kl {metrics['kl']:6.3f} "
                  f"| reward {metrics.get('reward', 0):.3f} "
                  f"| open-loop MSE {ol['open_loop_mse']:.4f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(history, out / "world_model_learning.png")
    torch.save(wm.state_dict(), out / "world_model.pt")

    first = history[0][1]["open_loop_mse"]
    last = history[-1][1]["open_loop_mse"]
    print(f"\nopen-loop MSE: {first:.4f} -> {last:.4f} "
          f"({100*(1-last/first):.1f}% reduction)")
    print(f"wrote {out/'world_model_learning.png'}")


def _plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(steps, [h[1]["recon"] for h in history])
    axes[0].set_title("reconstruction loss")
    axes[1].plot(steps, [h[1]["kl"] for h in history], color="tab:orange")
    axes[1].set_title("KL(post || prior)")
    axes[2].plot(steps, [h[1]["open_loop_mse"] for h in history],
                 color="tab:green")
    axes[2].set_title("open-loop multi-step MSE")
    for ax in axes:
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
