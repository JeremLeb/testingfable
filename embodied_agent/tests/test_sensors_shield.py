import numpy as np

from embodied_agent.config import load_config
from embodied_agent.env import make_env
from embodied_agent.safety import build_shield


def test_sensor_ranges():
    # test the sensor encodings' intrinsic ranges; observation noise (added
    # after squashing) is a separate concern, so disable it here
    cfg = load_config("cpu_small")
    for k in cfg.sensor.noise:
        cfg.sensor.noise[k] = 0.0
    env = make_env(cfg, seed=0)
    obs, _ = env.reset(seed=0)
    # vision distances (every 4th value) are normalized to [0, 1]
    vis = obs["vision"].reshape(-1, 4)
    assert (vis[:, 0] >= -1e-4).all() and (vis[:, 0] <= 1.0001).all()
    # intero variables in [0, 1]
    assert (obs["intero"] >= 0).all() and (obs["intero"] <= 1.0001).all()
    # smell is tanh-squashed to [-1, 1]
    assert (np.abs(obs["smell"]) <= 1.0001).all()


def test_touch_fires_on_collision():
    cfg = load_config("cpu_small")
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    touched = False
    # drive straight ahead: the agent must reach a wall regardless of heading
    for _ in range(400):
        obs, *_ = env.step(np.array([1.0, 0.0]))
        if obs["touch"].max() > 0.5:
            touched = True
            break
    assert touched  # driving into a wall must register contact pressure


def test_smell_updates_at_lower_rate():
    # disable smell noise so held values are exactly repeated between refreshes
    cfg = load_config("cpu_small")
    cfg.sensor.smell_period = 5
    cfg.sensor.noise["smell"] = 0.0
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    vals = []
    for _ in range(10):
        obs, *_ = env.step(np.array([0.5, 0.1]))
        vals.append(obs["smell"].copy())
    # with period 5 over 10 steps only a couple of refreshes occur, so far
    # fewer than 9 changes (which is what a per-step refresh would give)
    changes = sum(not np.allclose(vals[i], vals[i - 1])
                  for i in range(1, len(vals)))
    assert changes <= 4


def test_shield_blocks_forbidden_zone():
    cfg = load_config("cpu_small")
    cfg.shield.enabled = True
    env = make_env(cfg, seed=1)
    shield = build_shield(cfg, env)
    zone = np.asarray(env.forbidden_zones[0])
    env.reset(seed=1)
    entered = 0
    for _ in range(300):
        to_zone = zone[:2] - env.pos
        desired = np.arctan2(to_zone[1], to_zone[0])
        turn = np.clip((desired - env.heading + np.pi) % (2 * np.pi) - np.pi,
                       -1, 1)
        action, _ = shield.filter(env, np.array([1.0, turn]))
        env.step(action)
        if np.linalg.norm(env.pos - zone[:2]) < zone[2] + cfg.env.agent_radius:
            entered += 1
    assert entered == 0
