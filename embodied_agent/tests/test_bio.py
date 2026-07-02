"""Tests for the biological-plausibility modules (Phase 1)."""
import numpy as np

from embodied_agent.agent.development import Development
from embodied_agent.agent.metabolism import Metabolism, Sensorimotor
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


# ------------------------------------------------------------------ B4

def test_bounded_planning_scales_horizon_with_energy():
    cfg = load_config("cpu_small")
    cfg.metab.enabled = True
    cfg.metab.min_horizon = 3
    m = Metabolism(cfg.metab)
    base = cfg.agent.horizon
    assert m.planning_horizon(1.0, base) == base            # satiated -> full
    assert m.planning_horizon(0.0, base) == cfg.metab.min_horizon
    assert m.planning_horizon(1.0, base) > m.planning_horizon(0.4, base) \
        >= m.planning_horizon(0.0, base)


def test_metabolism_disabled_is_free_and_full():
    cfg = load_config("cpu_small")
    cfg.metab.enabled = False
    m = Metabolism(cfg.metab)
    assert m.planning_horizon(0.0, cfg.agent.horizon) == cfg.agent.horizon
    assert m.cognition_cost(10_000) == 0.0


def test_cognition_cost_is_proportional():
    cfg = load_config("cpu_small")
    cfg.metab.enabled = True
    m = Metabolism(cfg.metab)
    assert m.cognition_cost(0) == 0.0
    assert m.cognition_cost(2000) == 2 * m.cognition_cost(1000) > 0


def test_sensorimotor_delays_perception_and_action():
    cfg = load_config("cpu_small")
    cfg.metab.enabled = True
    cfg.metab.obs_delay = 2
    cfg.metab.action_delay = 1
    cfg.metab.motor_noise = 0.0
    smr = Sensorimotor(cfg.metab)
    o0 = {"x": np.zeros(1)}
    smr.reset(o0)
    # obs_delay=2: the first two fresh observations are still the stale o0
    assert smr.perceive({"x": np.ones(1)})["x"][0] == 0.0
    assert smr.perceive({"x": np.full(1, 2.0)})["x"][0] == 0.0
    assert smr.perceive({"x": np.full(1, 3.0)})["x"][0] == 1.0
    # action_delay=1: the world receives the previous action
    assert smr.execute(np.array([0.9, -0.9]))[0] == 0.0
    assert abs(smr.execute(np.array([0.1, 0.1]))[0] - 0.9) < 1e-9


def test_sensorimotor_disabled_is_identity():
    cfg = load_config("cpu_small")
    cfg.metab.enabled = False
    smr = Sensorimotor(cfg.metab)
    smr.reset({"x": np.zeros(1)})
    obs = {"x": np.full(1, 5.0)}
    assert smr.perceive(obs) is obs
    a = np.array([0.5, -0.3])
    assert np.array_equal(smr.execute(a), a)


def test_spend_energy_debits_and_clips():
    cfg = load_config("cpu_small")
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    e0 = env.homeostasis.energy
    env.homeostasis.spend_energy(0.1)
    assert abs(env.homeostasis.energy - (e0 - 0.1)) < 1e-6
    env.homeostasis.spend_energy(10.0)          # cannot go below zero
    assert env.homeostasis.energy == 0.0
    assert env.homeostasis.dead


def test_sparse_latent_fires_fewer_groups():
    cfg = load_config("cpu_small")
    cfg.model.sparse_latent = True
    cfg.model.sparse_frac = 0.5
    env = make_env(cfg, seed=0)
    from embodied_agent.model.world_model import WorldModel
    from embodied_agent.agent.replay import ReplayBuffer
    from embodied_agent.agent.collect import collect_random
    import torch
    buf = ReplayBuffer(cfg.train.buffer_capacity)
    collect_random(env, buf, 400, np.random.default_rng(0))
    wm = WorldModel(cfg, env.sensors.spaces)
    batch = buf.sample(4, cfg.train.seq_len, np.random.default_rng(0))
    with torch.no_grad():
        post, _ = wm.rssm.observe(wm.encoder(batch["obs"]),
                                  batch["prev_action"])
        z = post.z.reshape(*post.z.shape[:-1], wm.rssm.groups, wm.rssm.classes)
        frac = float((z.abs().amax(-1) > 0).float().mean())
    assert abs(frac - 0.5) < 0.1  # ~half the groups fire


# ------------------------------------------------------------------ B5

def test_genome_random_within_bounds_and_expresses():
    from embodied_agent.evolution import Genome, GENE_BOUNDS
    g = Genome.random(np.random.default_rng(0))
    for name, (lo, hi, is_int) in GENE_BOUNDS.items():
        assert lo <= g.genes[name] <= hi
        if is_int:
            assert float(g.genes[name]).is_integer()
    cfg = g.to_config(load_config("cpu_small"))
    assert cfg.sensor.n_rays == int(g.genes["n_rays"])
    assert cfg.env.temp_setpoint == g.genes["temp_setpoint"]
    assert cfg.agent.action_bias == [g.genes["action_bias_thrust"],
                                     g.genes["action_bias_turn"]]


def test_genome_mutation_stays_in_bounds():
    from embodied_agent.evolution import Genome, GENE_BOUNDS
    rng = np.random.default_rng(1)
    g = Genome.random(rng)
    for _ in range(50):
        g = g.mutate(rng, rate=0.5, prob=1.0)
    for name, (lo, hi, _) in GENE_BOUNDS.items():
        assert lo <= g.genes[name] <= hi


def test_crossover_inherits_from_parents():
    from embodied_agent.evolution import Genome, GENE_BOUNDS
    rng = np.random.default_rng(2)
    a, b = Genome.random(rng), Genome.random(rng)
    c = Genome.crossover(a, b, rng)
    for name in GENE_BOUNDS:
        assert c.genes[name] in (a.genes[name], b.genes[name])


def test_action_bias_shifts_newborn_behaviour():
    import torch
    from embodied_agent.agent.actor_critic import Actor
    torch.manual_seed(0)
    unbiased = Actor(8, 2, 16, action_bias=[0.0, 0.0])
    torch.manual_seed(0)
    forward = Actor(8, 2, 16, action_bias=[3.0, 0.0])
    feat = torch.zeros(1, 8)
    m0 = unbiased.act(feat, deterministic=True)
    m1 = forward.act(feat, deterministic=True)
    assert m1[0, 0] > m0[0, 0]  # innate forward instinct, before any learning


def test_evolution_improves_fitness_in_harsh_arena():
    from embodied_agent.evolution import Population, innate_fitness
    cfg = load_config("cpu_small")
    e = cfg.env
    e.fixed_layout = True
    e.n_hot, e.n_cold, e.temp_amp, e.temp_sigma_frac = 0, 4, 0.5, 0.35
    e.temp_danger, e.temp_damage_rate = 0.2, 0.02
    e.n_food, e.base_metabolism = 4, 0.0015
    e.max_episode_steps = 100000
    cfg.evolution.population = 8
    cfg.evolution.generations = 5
    pop = Population(cfg.evolution, cfg, seed=3)
    hist = pop.evolve(lambda g, s: innate_fitness(g.to_config(cfg), s, 800))
    assert len(hist) == 5
    assert hist[-1]["fitness_mean"] > hist[0]["fitness_mean"]      # adapts
    # cold arena selects a lower innate thermal set-point
    assert hist[-1]["gene_mean"]["temp_setpoint"] < \
        hist[0]["gene_mean"]["temp_setpoint"]


def test_reproduction_drive_bears_offspring_when_enabled():
    cfg = load_config("cpu_small")
    cfg.evolution.reproduction = True
    cfg.evolution.repro_energy_threshold = 0.0
    cfg.evolution.repro_rate = 0.6
    cfg.evolution.repro_cost = 0.1
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros(2))
    assert info["offspring"] >= 1


def test_reproduction_disabled_bears_none():
    cfg = load_config("cpu_small")
    cfg.evolution.reproduction = False
    env = make_env(cfg, seed=0)
    env.reset(seed=0)
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros(2))
    assert info["offspring"] == 0
