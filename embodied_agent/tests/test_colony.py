"""Tests for the multi-agent living colony."""
import numpy as np

from embodied_agent.config import load_config
from embodied_agent.colony.world import ColonyEnv


def _tiny(cfg):
    cfg.model.embed_dim = cfg.model.deter_dim = cfg.model.hidden = 64
    cfg.model.latent_groups = cfg.model.latent_classes = 8
    return cfg


def test_colony_resets_with_initial_population():
    cfg = load_config("colony")
    cfg.colony.n_init = 7
    env = ColonyEnv(cfg, seed=0)
    living = env.reset()
    assert len(living) == 7
    # every creature perceives the shared world with the shared body plan
    obs = living[0].observe()
    assert set(obs) == set(living[0].sensors.spaces)
    # distinct genomes -> generally distinct set-points
    setpoints = {c.cfg.temp_setpoint for c in living}
    assert len(setpoints) > 1


def test_colony_step_runs_and_shares_the_world():
    cfg = load_config("colony")
    cfg.colony.n_init = 6
    env = ColonyEnv(cfg, seed=1)
    living = env.reset()
    actions = {c.id: np.array([1.0, 0.2]) for c in living}
    results, births, deaths = env.step(actions)
    assert set(results) == {c.id for c in living}
    # all creatures see the SAME food list object (one shared world)
    assert living[0].foods is living[1].foods is env.foods


def test_reproduction_grows_the_population():
    cfg = load_config("colony")
    cfg.colony.n_init = 4
    cfg.colony.n_max = 12
    cfg.colony.repro_energy_threshold = 0.0   # always fertile
    cfg.colony.repro_rate = 0.5               # ready in ~2 steps
    env = ColonyEnv(cfg, seed=2)
    env.reset()
    for _ in range(20):
        env.step({c.id: np.zeros(2) for c in env.living})
    assert env.births > 0
    assert len(env.living) > 4
    # offspring carry a later generation than the founders
    assert env.stats()["max_generation"] >= 1


def test_death_and_immigration_floor():
    cfg = load_config("colony")
    cfg.colony.n_init = 8
    cfg.colony.n_min = 3
    cfg.env.base_metabolism = 0.05            # starve fast
    cfg.env.n_food = 2
    env = ColonyEnv(cfg, seed=3)
    env.reset()
    pops = []
    for _ in range(120):
        env.step({c.id: np.zeros(2) for c in env.living})
        pops.append(len(env.living))
    assert env.deaths > 0
    assert min(pops) >= cfg.colony.n_min       # immigration holds the floor


def test_run_colony_trains_shared_brain():
    from embodied_agent.colony.run import run_colony
    cfg = _tiny(load_config("colony"))
    cfg.colony.n_init = 6
    cfg.colony.n_max = 10
    cfg.train.total_steps = 200
    cfg.train.seq_len = 12
    cfg.train.batch_size = 8
    cfg.train.log_every = 100
    out = run_colony(cfg, verbose=False)
    assert out["stats"]["population"] >= cfg.colony.n_min
    assert out["buffer_steps"] > 0             # pooled experience was stored
