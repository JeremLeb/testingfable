"""B6 demo: active inference (one objective instead of reward + curiosity).

Expected free energy (EFE) is a single quantity that unites reaching preferred
homeostatic outcomes (the *pragmatic* term) and information gain (the
*epistemic* term). This demo makes two honest points:

  (1) Optimizing EFE -- with NO reward head and NO separately tuned curiosity
      weight -- reproduces food-seeking comparable to the reward-based deep-RL
      baseline. Value emerges from homeostatic *preferences*, not an engineered
      reward signal.
  (2) The single objective genuinely decomposes into the two drives: the
      epistemic (information-gain) term is largest early in training (drive to
      explore an unknown world) and recedes as the model becomes confident,
      while the pragmatic term is present throughout (drive toward preferred
      body states).

  python -m embodied_agent.scripts.active_inference_demo --config cpu_small

Note: the ensemble-based epistemic term is a Plan2Explore-style estimator; its
net *behavioural* exploration benefit is expected to grow with scale/compute
beyond this small CPU preset -- here we show the objective's structure, not a
large-scale exploration win.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np


def _run(config, objective, steps, seed, out):
    from ..config import load_config
    from ..train import run
    cfg = load_config(config)
    cfg.agent.objective = objective
    cfg.intrinsic.method = "disagreement"   # supplies the epistemic term
    cfg.train.total_steps = steps
    cfg.train.warmup_steps = min(600, steps // 4)
    cfg.train.log_every = 250
    cfg.train.seed = seed
    cfg.train.out_dir = out
    f = run(cfg, verbose=False)["final"]
    return {"food": f["eval_food"], "coverage": f["eval_coverage"],
            "reward": f["eval_reward"], "csv": pathlib.Path(out) / "metrics.csv"}


def _series(csv_path, *keys):
    rows = list(csv.DictReader(open(csv_path)))
    out = {k: [] for k in ("step", *keys)}
    for r in rows:
        if all(r.get(k) not in (None, "") for k in keys):
            out["step"].append(float(r["step"]))
            for k in keys:
                out[k].append(float(r[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/active_inference_demo")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    efe = _run(args.config, "expected_free_energy", args.steps, args.seed,
               str(out / "efe"))
    base = _run(args.config, "return", args.steps, args.seed, str(out / "base"))
    decomp = _series(efe["csv"], "efe_pragmatic", "efe_epistemic")

    out.mkdir(parents=True, exist_ok=True)
    _plot(efe, base, decomp, out / "active_inference_demo.png")

    print("=== B6 active inference (expected free energy) ===")
    print(f"food eaten: EFE {efe['food']:.2f} vs return-baseline "
          f"{base['food']:.2f} (EFE forages with no reward head / curiosity "
          f"weight)")
    print(f"coverage: EFE {efe['coverage']:.1f} vs baseline "
          f"{base['coverage']:.1f}")
    if decomp["step"]:
        e = decomp["efe_epistemic"]
        print(f"epistemic term: early {np.mean(e[:3]):.2f} -> late "
              f"{np.mean(e[-3:]):.2f} (explore-then-exploit)")
    print(f"wrote {out / 'active_inference_demo.png'}")


def _plot(efe, base, decomp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    x = ["EFE\n(no reward head)", "return\n(baseline)"]
    ax.bar(x, [efe["food"], base["food"]], color=["tab:blue", "gray"])
    ax.set_title("Food-seeking emerges from\npreferences alone")
    ax.set_ylabel("mean food / episode")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    if decomp["step"]:
        ax.plot(decomp["step"], decomp["efe_pragmatic"], color="tab:green",
                label="pragmatic (reach setpoints)")
        ax.plot(decomp["step"], decomp["efe_epistemic"], color="tab:red",
                label="epistemic (information gain)")
    ax.set_title("One objective, two drives\n(EFE decomposition over training)")
    ax.set_xlabel("training step")
    ax.set_ylabel("value component")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
