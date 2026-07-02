import numpy as np
import torch

from embodied_agent.config import load_config
from embodied_agent.env import make_env
from embodied_agent.model.world_model import WorldModel
from embodied_agent.agent.replay import ReplayBuffer
from embodied_agent.agent.actor_critic import ActorCritic, lambda_return


def _fill_buffer(env, buf, steps=200):
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    buf.start_episode()
    buf.add(obs, np.zeros(2), 0.0, 1.0)
    a = np.zeros(2)
    for _ in range(steps):
        a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
        obs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, 0.0 if term else 1.0)
        if term or trunc:
            buf.end_episode()
            obs, _ = env.reset()
            buf.start_episode()
            buf.add(obs, np.zeros(2), 0.0, 1.0)
    buf.end_episode()


def test_world_model_train_step_reduces_loss():
    cfg = load_config("cpu_small")
    env = make_env(cfg, seed=0)
    buf = ReplayBuffer(100000)
    _fill_buffer(env, buf, 400)
    wm = WorldModel(cfg, env.sensors.spaces)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(60):
        batch = buf.sample(8, 16, rng)
        _, m = wm.train_step(batch)
        losses.append(m["loss"])
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_lambda_return_constant_reward():
    # constant reward r, gamma*cont = d, infinite-horizon value = r/(1-d)
    T = 40
    r = torch.ones(T)
    d = torch.full((T,), 0.9)
    v = torch.zeros(T)
    ret = lambda_return(r, v, d, bootstrap=torch.tensor(0.0), lam=1.0)
    assert abs(ret[0].item() - (1 - 0.9 ** T) / (1 - 0.9)) < 1e-3


def test_actor_critic_step_runs_and_actions_bounded():
    cfg = load_config("cpu_small")
    env = make_env(cfg, seed=0)
    buf = ReplayBuffer(100000)
    _fill_buffer(env, buf, 300)
    wm = WorldModel(cfg, env.sensors.spaces)
    ac = ActorCritic(cfg, wm.rssm, wm)
    rng = np.random.default_rng(0)
    batch = buf.sample(8, 16, rng)
    post, _ = wm.train_step(batch)
    metrics = ac.train_step(post)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    feat = post.feat().reshape(-1, wm.rssm.feat_dim)
    action = ac.actor.act(feat, deterministic=True)
    assert action.abs().max().item() <= 1.0
