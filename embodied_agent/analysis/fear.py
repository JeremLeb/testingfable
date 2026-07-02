"""Measuring emergent fear: does anticipatory avoidance arise from the world
model predicting incoming damage before it lands?

Nothing in this project hardcodes avoidance. Interoceptive pain (integrity)
is just another modality the world model predicts. If "fear" is real, three
things should hold for a trained agent and *not* for a random one:

  1. Model anticipation -- imagining forward under the current policy, the
     model's predicted integrity drops in the steps *before* a real
     collision (it sees damage coming).
  2. Behavioral anticipation -- the agent slows down before contact
     (braking), which a random policy does not.
  3. Linear readability -- a linear probe on the latent state predicts
     "collision within K steps" well above chance, i.e. imminent
     nociception is explicitly encoded in the representation.

This module collects the traces and fits the probe; scripts/fear_analysis.py
turns them into a figure.
"""
from __future__ import annotations

import numpy as np
import torch

from ..env import make_env


@torch.no_grad()
def _counterfactual_danger(wm, state, horizon: int, device: str) -> float:
    """Model-predicted damage under a *reckless* counterfactual: 'if I kept
    driving straight forward from here, how much integrity would I lose?'

    This probes the world model's danger foresight directly, decoupled from
    the policy's avoidance. It should spike as a wall approaches -- the model
    seeing incoming nociception before it lands, which is exactly the
    substrate from which anticipatory avoidance emerges."""
    forward = torch.tensor([[1.0, 0.0]], device=device)

    def policy(feat):
        return forward.expand(feat.shape[0], 2), {}

    states, _, _ = wm.rssm.imagine(policy, state, horizon)
    integ_now = wm.decoder(state.feat())["intero"][..., 2]
    integ_future = wm.decoder(states.feat())["intero"][..., 2]  # (1, H)
    drop = integ_now - integ_future.min(dim=1).values
    return float(drop.clamp(min=0.0).squeeze())


def collect_trace(cfg, wm, ac, seed: int, steps: int, horizon: int = 8,
                  device: str = "cpu", explore_noise: float = 0.3) -> dict:
    """Run the trained policy (shield off, so avoidance is purely learned)
    and record per-step latent, speed, collisions, and the model's imagined
    integrity drop.

    We add exploration noise so the agent is occasionally perturbed toward
    walls: a fully-trained deterministic policy avoids collisions so well
    that there are no events to align to. The noise creates risky approaches;
    the question is whether the model *anticipates* the resulting damage and
    the policy *brakes* before it -- both learned, neither coded."""
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    state = wm.rssm.initial(1, device)
    prev_action = torch.zeros(1, 2, device=device)

    feats, speeds, collisions, pred_drop, integ = [], [], [], [], []
    for _ in range(steps):
        batch = {k: torch.as_tensor(v, device=device)[None].float()
                 for k, v in obs.items()}
        with torch.no_grad():
            embed = wm.encoder(batch)
            state, _ = wm.rssm.obs_step(state, prev_action, embed)
            feat = state.feat()
            action = ac.actor.act(feat, noise=explore_noise,
                                  deterministic=False)
        pred_drop.append(_counterfactual_danger(wm, state, horizon, device))
        feats.append(feat.squeeze(0).cpu().numpy())
        prev_action = action
        obs, r, term, trunc, info = env.step(action.squeeze(0).cpu().numpy())
        speeds.append(abs(info["speed"]))
        collisions.append(int(info["collision"]))
        integ.append(info["integrity"])
        if term or trunc:
            obs, _ = env.reset()
            state = wm.rssm.initial(1, device)
            prev_action = torch.zeros(1, 2, device=device)
    return {
        "feat": np.array(feats), "speed": np.array(speeds),
        "collision": np.array(collisions), "pred_drop": np.array(pred_drop),
        "integrity": np.array(integ),
    }


def collect_random_trace(cfg, seed: int, steps: int) -> dict:
    """Matching trace for a smoothed-random policy (the control)."""
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    a = np.zeros(2)
    speeds, collisions = [], []
    for _ in range(steps):
        a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
        obs, r, term, trunc, info = env.step(a)
        speeds.append(abs(info["speed"]))
        collisions.append(int(info["collision"]))
        if term or trunc:
            obs, _ = env.reset()
            a = np.zeros(2)
    return {"speed": np.array(speeds), "collision": np.array(collisions)}


def event_triggered(signal: np.ndarray, collisions: np.ndarray,
                    window: int) -> np.ndarray | None:
    """Average `signal` in a [-window, +window] window around collision
    onsets (rising edges of the collision flag). Returns None if no events."""
    onsets = np.where((collisions == 1)
                      & (np.concatenate([[0], collisions[:-1]]) == 0))[0]
    segs = []
    for t in onsets:
        if t - window < 0 or t + window + 1 > len(signal):
            continue
        segs.append(signal[t - window: t + window + 1])
    if not segs:
        return None
    return np.mean(segs, axis=0)


def fit_latent_probe(feat: np.ndarray, collision: np.ndarray, horizon: int,
                     epochs: int = 400) -> dict:
    """Logistic probe: from latent feat_t predict 'a collision occurs within
    the next `horizon` steps'. Reports held-out AUC and per-step probability."""
    n = len(feat)
    label = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if collision[t + 1: t + 1 + horizon].any():
            label[t] = 1.0

    x = torch.as_tensor(feat, dtype=torch.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.as_tensor(label)
    idx = torch.randperm(n)
    split = int(0.7 * n)
    tr, te = idx[:split], idx[split:]

    probe = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(probe(x[tr]).squeeze(-1), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(probe(x).squeeze(-1)).numpy()
    auc = _auc(prob[te.numpy()], label[te.numpy()])
    return {"prob": prob, "label": label, "auc": auc,
            "base_rate": float(label.mean())}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic (ties handled by avg rank)."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)     # 1-based ranks
    # average ranks within tied score groups
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
