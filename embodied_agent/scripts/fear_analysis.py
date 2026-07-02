"""Demonstrate that anticipatory avoidance ('fear') emerges from prediction.

Trains (or loads) an agent, then measures three signatures of learned fear
and writes fear_analysis.png:

  1. the world model's imagined integrity-drop rising *before* real collisions
  2. the agent braking before contact, unlike a random policy
  3. a linear latent probe reading 'collision imminent' above chance

  python -m embodied_agent.scripts.fear_analysis --config cpu_small \
      --checkpoint runs/cpu_small/checkpoint.pt
  # or let it train a short model first:
  python -m embodied_agent.scripts.fear_analysis --config cpu_small --train-steps 12000
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from ..agent.actor_critic import ActorCritic
from ..analysis import fear
from ..config import load_config
from ..env import make_env
from ..intrinsic import build_intrinsic
from ..model.world_model import WorldModel
from ..utils.seeding import seed_everything


def _load_or_train(cfg, args):
    env = make_env(cfg, seed=args.seed)
    wm = WorldModel(cfg, env.sensors.spaces)
    intrinsic = build_intrinsic(cfg, wm.rssm.feat_dim, "cpu")
    ac = ActorCritic(cfg, wm.rssm, wm, intrinsic=intrinsic)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=True)
        wm.load_state_dict(ckpt["wm"])
        ac.actor.load_state_dict(ckpt["actor"])
        ac.critic.load_state_dict(ckpt["critic"])
        print(f"loaded checkpoint {args.checkpoint}")
    else:
        from ..train import run
        cfg.train.total_steps = args.train_steps
        cfg.train.out_dir = args.out
        print(f"no checkpoint given; training {args.train_steps} steps first")
        run(cfg, verbose=True)
        ckpt = torch.load(pathlib.Path(args.out) / "checkpoint.pt",
                          map_location="cpu", weights_only=True)
        wm.load_state_dict(ckpt["wm"])
        ac.actor.load_state_dict(ckpt["actor"])
        ac.critic.load_state_dict(ckpt["critic"])
    return wm, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--train-steps", type=int, default=12000)
    ap.add_argument("--trace-steps", type=int, default=5000)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/fear_analysis")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    wm, ac = _load_or_train(cfg, args)
    wm.eval()

    print("collecting trained-policy trace...")
    tr = fear.collect_trace(cfg, wm, ac, args.seed + 1, args.trace_steps,
                            horizon=args.horizon)
    print("collecting random-policy control...")
    rnd = fear.collect_random_trace(cfg, args.seed + 1, args.trace_steps)

    probe = fear.fit_latent_probe(tr["feat"], tr["collision"], args.horizon)

    W = args.window
    et_drop = fear.event_triggered(tr["pred_drop"], tr["collision"], W)
    et_prob = fear.event_triggered(probe["prob"], tr["collision"], W)
    coll_rate_tr = float(tr["collision"].mean())
    coll_rate_rnd = float(rnd["collision"].mean())

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(W, et_drop, et_prob, probe, coll_rate_tr, coll_rate_rnd,
          args.horizon, out / "fear_analysis.png")

    # a compact quantitative summary printed for the record
    n_coll = int(((tr["collision"] == 1)
                  & (np.concatenate([[0], tr["collision"][:-1]]) == 0)).sum())
    print("\n=== emergent-fear summary ===")
    print(f"collision onsets analyzed: {n_coll}")
    if et_drop is not None:
        pre = et_drop[W - 5:W].mean()       # 5 steps just before contact
        far = et_drop[:5].mean()            # far before contact
        print(f"counterfactual danger (model 'if I kept going'): far={far:.4f}"
              f" -> just-before-contact={pre:.4f} "
              f"(rises = model foresees the hit)")
    print(f"latent probe AUC (collision within {args.horizon} steps): "
          f"{probe['auc']:.3f}  [base rate {probe['base_rate']:.2f}, "
          f"chance 0.5]; probe P peaks at lag "
          f"{np.arange(-W, W + 1)[int(np.argmax(et_prob))] if et_prob is not None else 'NA'}")
    print(f"collision rate: trained {coll_rate_tr:.3f}/step vs "
          f"random {coll_rate_rnd:.3f}/step "
          f"({coll_rate_rnd / max(coll_rate_tr, 1e-6):.1f}x fewer = learned avoidance)")
    print(f"wrote {out / 'fear_analysis.png'}")


def _plot(W, et_drop, et_prob, probe, coll_tr, coll_rnd, horizon, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags = np.arange(-W, W + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    if et_drop is not None:
        ax.plot(lags, et_drop, color="tab:red", marker="o", ms=3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title("Model danger foresight:\ncounterfactual 'if I kept going'")
    ax.set_xlabel("steps relative to collision")
    ax.set_ylabel("predicted integrity drop")

    ax = axes[1]
    if et_prob is not None:
        ax.plot(lags, et_prob, color="tab:purple", marker="o", ms=3)
    ax.axhline(probe["base_rate"], color="gray", ls=":",
               label=f"base rate {probe['base_rate']:.2f}")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title(f"Latent probe P(collision within {horizon})\n"
                 f"held-out AUC = {probe['auc']:.2f}")
    ax.set_xlabel("steps relative to collision")
    ax.set_ylabel("probe probability")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(["trained\n(learned)", "random"], [coll_tr, coll_rnd],
           color=["tab:blue", "gray"])
    ax.set_title("Learned avoidance:\ncollision rate")
    ax.set_ylabel("collisions per step")

    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
