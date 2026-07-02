"""Ablation harness. Runs a set of training conditions and compares final
eval reward and exploration coverage, saving a summary bar chart + CSV.

  # curiosity: does RND improve exploration coverage?
  python -m embodied_agent.scripts.ablation --set curiosity --config cpu_small

  # shield: how often does the safety layer intervene?
  python -m embodied_agent.scripts.ablation --set shield --config cpu_small

(The reward-head / substrate ablation is a world-model question and lives in
scripts/train_world_model.py via --no-reward.)
"""
from __future__ import annotations

import argparse
import copy
import csv
import pathlib

from ..config import load_config
from ..train import run

ABLATION_SETS = {
    # (label, mutations applied to the base config)
    "curiosity": [
        ("no_curiosity", {"intrinsic.method": "none", "shield.enabled": False}),
        ("rnd", {"intrinsic.method": "rnd", "shield.enabled": False}),
        ("disagreement",
         {"intrinsic.method": "disagreement", "shield.enabled": False}),
    ],
    "shield": [
        ("shield_off", {"shield.enabled": False}),
        ("shield_on", {"shield.enabled": True}),
    ],
}


def _mutate(cfg, mutations):
    for key, val in mutations.items():
        section, field = key.split(".")
        setattr(getattr(cfg, section), field, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="curiosity", choices=list(ABLATION_SETS))
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    conditions = ABLATION_SETS[args.set]
    out = pathlib.Path(args.out) / args.set
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for label, mutations in conditions:
        for seed in range(args.seeds):
            cfg = load_config(args.config)
            if args.steps:
                cfg.train.total_steps = args.steps
            cfg.train.seed = seed
            cfg.train.out_dir = str(out / f"{label}_s{seed}")
            cfg.train.gif_every = 10 ** 9  # skip gifs during ablation
            _mutate(cfg, mutations)
            print(f"\n=== {args.set}: {label} (seed {seed}) ===")
            summary = run(cfg, verbose=True)
            results.append({
                "label": label, "seed": seed,
                "eval_reward": summary["final"]["eval_reward"],
                "eval_coverage": summary["final"]["eval_coverage"],
                "eval_food": summary["final"]["eval_food"],
                "baseline_reward": summary["baseline_reward"],
                "baseline_coverage": summary["baseline_coverage"],
            })

    _write_csv(results, out / "summary.csv")
    _plot(results, args.set, out / "summary.png")
    _print_table(results)
    print(f"\nwrote {out}/summary.csv and summary.png")


def _write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


def _agg(results):
    labels = []
    for r in results:
        if r["label"] not in labels:
            labels.append(r["label"])
    agg = {}
    for lab in labels:
        rs = [r for r in results if r["label"] == lab]
        agg[lab] = {
            "reward": sum(r["eval_reward"] for r in rs) / len(rs),
            "coverage": sum(r["eval_coverage"] for r in rs) / len(rs),
            "food": sum(r["eval_food"] for r in rs) / len(rs),
        }
    return labels, agg


def _print_table(results):
    labels, agg = _agg(results)
    base_cov = results[0]["baseline_coverage"]
    base_rew = results[0]["baseline_reward"]
    print(f"\n{'condition':<16}{'reward':>10}{'coverage':>10}{'food':>8}")
    print(f"{'random':<16}{base_rew:>10.1f}{base_cov:>10.1f}{'-':>8}")
    for lab in labels:
        a = agg[lab]
        print(f"{lab:<16}{a['reward']:>10.1f}{a['coverage']:>10.1f}"
              f"{a['food']:>8.1f}")


def _plot(results, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, agg = _agg(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [agg[l]["coverage"] for l in labels], color="tab:green")
    axes[0].axhline(results[0]["baseline_coverage"], ls="--", color="gray",
                    label="random")
    axes[0].set_title(f"{name}: exploration coverage")
    axes[0].legend()
    axes[1].bar(labels, [agg[l]["reward"] for l in labels], color="tab:blue")
    axes[1].axhline(results[0]["baseline_reward"], ls="--", color="gray",
                    label="random")
    axes[1].set_title(f"{name}: eval reward")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
