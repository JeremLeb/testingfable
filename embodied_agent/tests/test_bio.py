"""Tests for the biological-plausibility modules (Phase 1)."""
import numpy as np

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
