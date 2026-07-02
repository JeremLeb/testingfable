"""Tests for the biological-plausibility modules (Phase 1)."""
import numpy as np

from embodied_agent.agent.development import Development
from embodied_agent.agent.replay import ReplayBuffer
from embodied_agent.agent.sleep import SleepController
from embodied_agent.config import load_config
from embodied_agent.env import make_env
from embodied_agent.env.neuromod import Neuromodulators, allostatic_weights


# ------------------------------------------------------------------ B1

def test_allostatic_weights_scale_with_deficit():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = True
    rc, nm = cfg.reward, cfg.neuromod
    low = allostatic_weights({"energy": 0.1, "thermal": 0, "integrity": 0},
                             rc, nm)["energy"]
    high = allostatic_weights({"energy": 0.9, "thermal": 0, "integrity": 0},
                              rc, nm)["energy"]
    assert high > low > 0
    assert abs(low - rc.w_energy) < abs(high - rc.w_energy)  # super-linear


def test_allostatic_disabled_returns_base_weights():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = False
    rc, nm = cfg.reward, cfg.neuromod
    w = allostatic_weights({"energy": 0.9, "thermal": 0.9, "integrity": 0.9},
                           rc, nm)
    assert w["energy"] == rc.w_energy
    assert w["integrity"] == rc.w_integrity


def test_near_death_integrity_dominates():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = True
    w = allostatic_weights({"energy": 0.3, "thermal": 0.3, "integrity": 0.95},
                           cfg.reward, cfg.neuromod)
    assert w["integrity"] > w["energy"] and w["integrity"] > w["thermal"]


def test_ne_gain_rises_with_surprise_and_is_bounded():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = True
    n = Neuromodulators(cfg.neuromod)
    for _ in range(20):
        g_calm = n.update_surprise(1.0)
    g_spike = n.update_surprise(10.0)
    assert abs(g_calm - 1.0) < 0.05          # steady state ~ no boost
    assert 1.0 < g_spike <= cfg.neuromod.ne_gain + 1e-6


def test_neuromod_disabled_is_identity():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = False
    n = Neuromodulators(cfg.neuromod)
    assert n.update_surprise(100.0) == 1.0
    assert n.actor_lr_gain() == 1.0


def test_reward_logs_instantaneous_weights_when_enabled():
    cfg = load_config("cpu_small")
    cfg.neuromod.enabled = True
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5, 0.0]))
    assert "weight_energy" in info and "weight_integrity" in info
    assert info["weight_energy"] > 0


# ------------------------------------------------------------------ B2

def _buffer_with_saliences(saliences, length=30):
    """A buffer with one episode per requested peak-|reward| salience."""
    buf = ReplayBuffer(capacity=100000)
    obs_dim = 4
    for s in saliences:
        buf.start_episode()
        for t in range(length):
            obs = {"proprio": np.zeros(obs_dim, dtype=np.float32)}
            r = s if t == length // 2 else 0.0  # a single reward spike
            buf.add(obs, np.zeros(2), r, 1.0)
        buf.end_episode()
    return buf


def test_prioritized_replay_favors_salient_episodes():
    cfg = load_config("cpu_small")
    cfg.sleep.enabled = True
    buf = _buffer_with_saliences([0.0, 0.0, 0.0, 5.0])  # one salient memory
    sal, prob = buf.episode_priorities(seq_len=10, sleep_cfg=cfg.sleep)
    salient = int(np.argmax(sal))
    assert prob[salient] == prob.max()
    assert prob[salient] > 1.0 / len(prob)  # above uniform


def test_uniform_replay_is_flat():
    buf = _buffer_with_saliences([0.0, 1.0, 5.0])
    _, prob = buf.episode_priorities(seq_len=10, sleep_cfg=None)
    assert np.allclose(prob, prob[0])


def test_prioritized_sampling_draws_salient_more_often():
    cfg = load_config("cpu_small")
    cfg.sleep.enabled = True
    buf = _buffer_with_saliences([0.0, 0.0, 0.0, 0.0, 8.0])
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        batch = buf.sample(4, 10, rng, sleep_cfg=cfg.sleep)
        hits += int(batch["reward"].abs().max() > 4.0)
    # the lone salient episode is 1/5 of memory but drawn far more than 20%
    assert hits > 80


def test_sleep_controller_wake_sleep_phase():
    cfg = load_config("cpu_small")
    cfg.sleep.day_steps = 100
    cfg.sleep.wake_frac = 0.8
    ctl = SleepController(cfg.sleep)
    assert ctl.is_awake(1) and ctl.is_awake(79)
    assert not ctl.is_awake(80) and not ctl.is_awake(99)
    assert ctl.just_fell_asleep(80)
    assert ctl.just_woke(100)  # next dawn
    assert 0.0 <= ctl.phase(50) < 1.0


def test_sleep_disabled_leaves_baseline_sampling_uniform():
    # with no SleepConfig passed, sampling ignores salience entirely
    buf = _buffer_with_saliences([0.0, 9.0])
    _, prob = buf.episode_priorities(seq_len=10, sleep_cfg=None)
    assert np.allclose(prob, 0.5)


# ------------------------------------------------------------------ B3

def test_plasticity_anneals_with_age():
    cfg = load_config("cpu_small")
    cfg.dev.enabled = True
    cfg.dev.young_gain = 3.0
    cfg.dev.floor_gain = 0.5
    cfg.dev.critical_period_steps = 1000
    dev = Development(cfg.dev)
    assert abs(dev.plasticity_gain(0) - 3.0) < 1e-6            # young = max
    assert dev.plasticity_gain(0) > dev.plasticity_gain(1000) \
        > dev.plasticity_gain(5000)                            # monotone decay
    assert abs(dev.plasticity_gain(10 ** 6) - 0.5) < 1e-3      # -> floor


def test_development_disabled_is_identity():
    cfg = load_config("cpu_small")
    cfg.dev.enabled = False
    dev = Development(cfg.dev)
    assert dev.plasticity_gain(0) == 1.0
    assert dev.plasticity_gain(10000) == 1.0


def test_continual_life_survives_truncation(tmp_path):
    # a continual life keeps ageing across episode-truncation boundaries; the
    # logged age must exceed max_episode_steps, proving the body was not reset.
    from embodied_agent.train import run
    import csv
    cfg = load_config("cpu_small")
    cfg.dev.enabled = True
    cfg.env.max_episode_steps = 40
    cfg.train.total_steps = 200
    cfg.train.warmup_steps = 40
    cfg.train.log_every = 40
    cfg.train.eval_every = 10 ** 9   # skip eval/gif during the test
    cfg.train.gif_every = 10 ** 9
    cfg.train.out_dir = str(tmp_path / "life")
    run(cfg, verbose=False)
    rows = list(csv.DictReader(open(tmp_path / "life" / "metrics.csv")))
    ages = [float(r["age"]) for r in rows if r.get("age")]
    assert max(ages) > cfg.env.max_episode_steps
