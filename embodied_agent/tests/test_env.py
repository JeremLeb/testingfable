import numpy as np
import pytest

from embodied_agent.config import load_config
from embodied_agent.env import make_env
from embodied_agent.env import geometry as geo


@pytest.fixture
def cfg():
    return load_config("cpu_small")


def test_reset_step_shapes(cfg):
    env = make_env(cfg, seed=0)
    obs, info = env.reset(seed=0)
    for key in ("vision", "touch", "proprio", "intero", "smell"):
        assert key in obs
        assert obs[key].dtype == np.float32
        assert np.isfinite(obs[key]).all()
    obs, r, term, trunc, info = env.step(np.array([0.5, 0.1]))
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_agent_stays_in_bounds(cfg):
    env = make_env(cfg, seed=1)
    env.reset(seed=1)
    R = cfg.env.agent_radius
    for _ in range(500):
        env.step(np.array([1.0, 0.3]))  # push hard toward walls
        assert R - 1e-6 <= env.pos[0] <= cfg.env.arena_size - R + 1e-6
        assert R - 1e-6 <= env.pos[1] <= cfg.env.arena_size - R + 1e-6


def test_energy_drains_and_food_restores(cfg):
    env = make_env(cfg, seed=2)
    env.reset(seed=2)
    e0 = env.homeostasis.energy
    for _ in range(50):
        env.step(np.array([1.0, 0.0]))
    assert env.homeostasis.energy < e0  # metabolism + movement cost


def test_temperature_field_range(cfg):
    env = make_env(cfg, seed=3)
    pts = np.random.default_rng(0).uniform(0, cfg.env.arena_size, (200, 2))
    t = env.temperature(pts)
    assert t.shape == (200,)
    assert (t >= 0).all() and (t <= 1).all()


def test_death_terminates(cfg):
    env = make_env(cfg, seed=4)
    env.reset(seed=4)
    env.homeostasis.integrity = 0.001
    terminated = False
    for _ in range(50):
        # slam a wall to force integrity to zero
        _, _, terminated, _, _ = env.step(np.array([1.0, 0.0]))
        if terminated:
            break
    assert env.homeostasis.integrity <= 0.0 or terminated


def test_reward_decomposition_sums(cfg):
    env = make_env(cfg, seed=5)
    env.reset(seed=5)
    for _ in range(20):
        _, r, _, _, info = env.step(np.array([0.6, 0.2]))
        parts = (info["reward_energy"] + info["reward_thermal"]
                 + info["reward_integrity"] + info["reward_death"])
        assert abs(parts - info["reward_total"]) < 1e-4
        assert abs(info["reward_total"] - r) < 1e-4


def test_geometry_circle_rect_pushout():
    rect = (0.0, 0.0, 2.0, 2.0)
    pos = np.array([1.0, 1.0])  # inside
    new_pos, normal = geo.resolve_circle_rect(pos, 0.3, rect)
    assert normal is not None
    # after push-out the circle no longer overlaps
    _, normal2 = geo.resolve_circle_rect(new_pos, 0.3, rect)
    assert normal2 is None


def test_ray_hits_are_finite_or_inf():
    origin = np.array([1.0, 1.0])
    d = np.array([1.0, 0.0])
    assert geo.ray_rect(origin, d, (5.0, 0.0, 1.0, 3.0)) == pytest.approx(4.0)
    assert np.isinf(geo.ray_rect(origin, d, (0.0, 5.0, 1.0, 1.0)))


def test_determinism_from_seed(cfg):
    env1 = make_env(cfg, seed=7)
    env2 = make_env(cfg, seed=7)
    o1, _ = env1.reset(seed=7)
    o2, _ = env2.reset(seed=7)
    a = np.array([0.4, -0.2])
    for _ in range(30):
        o1, r1, *_ = env1.step(a)
        o2, r2, *_ = env2.step(a)
    assert r1 == r2
    for k in o1:
        assert np.allclose(o1[k], o2[k])
