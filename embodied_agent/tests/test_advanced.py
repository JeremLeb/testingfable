"""Tests for the Tier-1 upgrades: symlog/two-hot, discrete latents, the CNN
retina path, and the fear-analysis primitives."""
import numpy as np
import torch

from embodied_agent.config import load_config
from embodied_agent.env import make_env
from embodied_agent.model.distributions import (TwoHotHead, categorical_sample,
                                                symexp, symlog)
from embodied_agent.model.world_model import WorldModel
from embodied_agent.agent.replay import ReplayBuffer
from embodied_agent.agent.actor_critic import ActorCritic
from embodied_agent.analysis import fear


def test_symlog_roundtrip():
    x = torch.tensor([-1234.0, -1.0, 0.0, 0.5, 987.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-3)


def test_twohot_head_recovers_target():
    torch.manual_seed(0)
    head = TwoHotHead(8, 32, bins=51, low=-8, high=8)
    feat = torch.randn(256, 8)
    target = torch.randn(256) * 40          # wide range incl. outliers
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        head.loss(feat, target).mean().backward()
        opt.step()
    pred = head.mean(feat)
    assert torch.corrcoef(torch.stack([pred, target]))[0, 1] > 0.9


def test_categorical_sample_is_straight_through_onehot():
    logits = torch.randn(4, 6, 8, requires_grad=True)   # (B, G, C)
    z = categorical_sample(logits, unimix=0.01)
    assert z.shape == (4, 48)
    groups = z.reshape(4, 6, 8)
    # forward pass is one-hot per group (up to float rounding)
    assert torch.allclose(groups.sum(-1), torch.ones(4, 6), atol=1e-4)
    near01 = (groups.abs() < 1e-4) | ((groups - 1).abs() < 1e-4)
    assert near01.all()
    assert (groups > 0.5).sum().item() == 4 * 6      # exactly one hot per group
    # gradient flows via the straight-through estimator
    z.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_discrete_world_model_learns_and_kl_positive():
    cfg = load_config("cpu_small")
    assert cfg.model.latent_kind == "discrete"
    env = make_env(cfg, seed=0)
    buf = ReplayBuffer(100000)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    buf.start_episode()
    buf.add(obs, np.zeros(2), 0.0, 1.0)
    a = np.zeros(2)
    for _ in range(400):
        a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
        obs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, 0.0 if term else 1.0)
        if term or trunc:
            buf.end_episode(); obs, _ = env.reset()
            buf.start_episode(); buf.add(obs, np.zeros(2), 0.0, 1.0)
    buf.end_episode()
    wm = WorldModel(cfg, env.sensors.spaces)
    losses = []
    for _ in range(60):
        _, m = wm.train_step(buf.sample(8, 16, rng))
        losses.append(m["loss"])
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    assert m["kl"] >= 0.0


def test_pixel_retina_path_shapes_and_recon():
    cfg = load_config("cpu_pixels")
    env = make_env(cfg, seed=0)
    spaces = env.sensors.spaces
    assert spaces["retina"].kind == "image"
    obs, _ = env.reset(seed=0)
    assert obs["retina"].shape == (3, cfg.sensor.retina_res, cfg.sensor.retina_res)
    assert obs["retina"].min() >= 0.0 and obs["retina"].max() <= 1.0
    wm = WorldModel(cfg, spaces)
    batch = {k: torch.as_tensor(v)[None, None].float() for k, v in obs.items()}
    embed = wm.encoder(batch)
    assert embed.shape == (1, 1, cfg.model.embed_dim)
    recon = wm.decoder(torch.zeros(1, 1, wm.rssm.feat_dim))
    assert recon["retina"].shape == (1, 1, 3, cfg.sensor.retina_res,
                                     cfg.sensor.retina_res)


def test_auc_and_probe_direction():
    # perfectly separable scores -> AUC 1
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert abs(fear._auc(scores, labels) - 1.0) < 1e-9
    # reversed -> AUC 0
    assert abs(fear._auc(-scores, labels) - 0.0) < 1e-9


def test_fear_probe_beats_chance_on_synthetic_signal():
    # construct a trace where the latent at t predicts a collision that lands
    # a couple of steps later -- exactly the imminent-nociception structure
    # the real probe reads out.
    rng = np.random.default_rng(0)
    n = 800
    feat = rng.normal(size=(n, 16)).astype(np.float32)
    collision = np.zeros(n, dtype=int)
    for t in range(n - 3):
        if feat[t, 0] > 0.7:            # danger cue now -> impact soon
            collision[t + 2] = 1
    out = fear.fit_latent_probe(feat, collision, horizon=3, epochs=250)
    assert out["auc"] > 0.7
