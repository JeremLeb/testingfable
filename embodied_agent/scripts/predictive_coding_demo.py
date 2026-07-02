"""B7 demo: local learning (predictive coding) vs backprop.

The project's substrate is next-observation prediction. Here two networks learn
exactly that -- predict o_{t+1} from (o_t, a_t) drawn from a random life -- one
trained by **backprop** (global loss + autograd) and one by **predictive
coding** (local error neurons + Hebbian updates, no backward pass). The claim
(Whittington & Bogacz): the biologically-local rule reaches comparable open-loop
MSE, and its weight update points the same way as the backprop gradient.

  python -m embodied_agent.scripts.predictive_coding_demo --config cpu_small
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def _collect(cfg, n, seed):
    """(o_t, a_t) -> o_{t+1} transitions from a random-agent life."""
    from ..env import make_env
    env = make_env(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    def flat(o):
        return np.concatenate([np.asarray(o[k], np.float32).ravel()
                               for k in sorted(o)])

    X, Y = [], []
    a = np.zeros(2)
    prev = flat(obs)
    for _ in range(n):
        a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
        obs, _, term, trunc, _ = env.step(a)
        cur = flat(obs)
        X.append(np.concatenate([prev, a]))
        Y.append(cur)
        prev = cur
        if term or trunc:
            obs, _ = env.reset(); prev = flat(obs); a = np.zeros(2)
    X, Y = np.array(X), np.array(Y)
    # standardise targets so MSE is comparable across modalities
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    return X.astype(np.float32), ((Y - mu) / sd).astype(np.float32)


def _backprop_curve(X, Y, Xte, Yte, hidden, epochs, seed):
    # per-sample (online) SGD so the update budget matches predictive coding's
    # per-sample local updates -- a fair epoch-for-epoch comparison.
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.SGD(net.parameters(), lr=3e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    Xv, Yv = torch.tensor(Xte), torch.tensor(Yte)
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            opt.zero_grad()
            loss = ((net(Xt[i]) - Yt[i]) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            curve.append(float(((net(Xv) - Yv) ** 2).mean()))
    return curve


def _pc_curve(X, Y, Xte, Yte, hidden, epochs, cfg, seed):
    from ..model.predictive_coding import PredictiveCodingNet
    pc = PredictiveCodingNet(
        [X.shape[1], hidden, Y.shape[1]], seed=seed,
        infer_steps=cfg.model.pc_inference_steps,
        infer_lr=cfg.model.pc_inference_lr,
        weight_lr=cfg.model.pc_weight_lr)
    rng = np.random.default_rng(seed)
    curve = []
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        pc.learn_epoch(X[idx], Y[idx])
        curve.append(float(np.mean((pc.predict_batch(Xte) - Yte) ** 2)))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/predictive_coding_demo")
    args = ap.parse_args()

    from ..config import load_config
    cfg = load_config(args.config)
    X, Y = _collect(cfg, args.samples, args.seed)
    Xte, Yte = _collect(cfg, args.samples // 2, args.seed + 100)

    bp = _backprop_curve(X, Y, Xte, Yte, args.hidden, args.epochs, args.seed)
    pc = _pc_curve(X, Y, Xte, Yte, args.hidden, args.epochs, cfg, args.seed)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(bp, pc, out / "predictive_coding_demo.png")

    print("=== B7 predictive coding vs backprop ===")
    print(f"open-loop next-obs test MSE: backprop {bp[-1]:.3f} vs "
          f"predictive coding {pc[-1]:.3f} (local Hebbian, no backward pass)")
    print(f"wrote {out / 'predictive_coding_demo.png'}")


def _plot(bp, pc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bp, color="gray", label="backprop (global loss)")
    ax.plot(pc, color="tab:red", label="predictive coding (local)")
    ax.set_title("Same substrate, biological rule:\nopen-loop next-observation prediction")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test MSE")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
