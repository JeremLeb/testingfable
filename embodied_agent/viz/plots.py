"""Time-series plots from a training metrics.csv (observability)."""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k is None or isinstance(v, list) or v in ("", None):
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def _series(rows, key):
    xs, ys = [], []
    for r in rows:
        if r.get(key) is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def plot_metrics(csv_path: str, out_path: str):
    rows = _read(csv_path)
    panels = [
        ("homeostasis", [("energy", "energy"), ("temp", "temperature"),
                         ("integrity", "integrity")]),
        ("drive reduction (reward terms)",
         [("reward_energy", "energy"), ("reward_thermal", "thermal"),
          ("reward_integrity", "integrity")]),
        ("episode reward", [("ep_reward", "ep_reward")]),
        ("world-model losses", [("recon", "recon"), ("kl", "kl"),
                                ("reward", "reward head")]),
        ("actor-critic", [("imag_return", "imag return"),
                          ("actor_entropy", "entropy")]),
        ("eval vs random", [("eval_reward", "policy"),
                            ("baseline_reward", "random")]),
        ("exploration coverage", [("eval_coverage", "policy"),
                                  ("baseline_coverage", "random")]),
        ("intrinsic / shield", [("intrinsic_reward", "intrinsic"),
                                ("interventions", "shield")]),
    ]
    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow))
    axes = axes.flatten()
    for ax, (title, keys) in zip(axes, panels):
        drew = False
        for key, label in keys:
            xs, ys = _series(rows, key)
            if xs:
                ax.plot(xs, ys, label=label)
                drew = True
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if drew:
            ax.legend(fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "runs/cpu_small/metrics.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/cpu_small/metrics.png"
    print("wrote", plot_metrics(csv_path, out))
