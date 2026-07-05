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


def _spaces(cfg):
    from embodied_agent.env.sensors import SensorSuite
    return SensorSuite(cfg.sensor, cfg.env, np.random.default_rng(0)).spaces


def test_each_creature_gets_its_own_mind():
    import torch
    from embodied_agent.colony.run import Mind
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    a, b = Mind(cfg, sp, "cpu"), Mind(cfg, sp, "cpu")
    assert a.wm is not b.wm and a.buffer is not b.buffer   # separate individuals
    # training one mind must not touch another's weights (independent learning)
    before = next(b.wm.parameters()).detach().clone()
    opt = a.wm.opt
    x = next(a.wm.parameters()); opt.zero_grad(); (x.sum()).backward(); opt.step()
    assert torch.allclose(before, next(b.wm.parameters()).detach())


def test_newborn_inherits_parent_brain():
    import torch
    from embodied_agent.colony.run import Mind
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    parent = Mind(cfg, sp, "cpu")
    child = Mind(cfg, sp, "cpu")
    child.inherit_from(parent)
    for p, c in zip(parent.wm.parameters(), child.wm.parameters()):
        assert torch.allclose(p.detach(), c.detach())


def test_lineage_gene_means_and_tree():
    from embodied_agent.colony.tree import TreeRenderer
    cfg = load_config("colony")
    cfg.colony.n_init = 5
    cfg.colony.n_max = 14
    cfg.colony.repro_energy_threshold = 0.0
    cfg.colony.repro_rate = 0.05
    env = ColonyEnv(cfg, seed=1)
    env.reset()
    founders = {c.founder for c in env.living}
    assert founders == {c.id for c in env.living}   # founders are themselves
    for _ in range(60):
        env.step({c.id: np.zeros(2) for c in env.living})
    # offspring inherit the parent's bloodline (founder) and a later generation
    kids = [c for c in env.living if c.generation > 0]
    assert kids and all(c.founder in founders for c in kids)
    # gene means expose the heritable innate traits
    gm = env.gene_means()
    assert {"temp_setpoint", "action_bias_thrust", "w_energy"} <= set(gm)
    # lineage counts sum to the living population
    assert sum(n for _, n in env.lineages()) == len(env.living)
    # the family tree renders
    png_arr = TreeRenderer().render(env, focus_id=env.living[0].id)
    assert png_arr.ndim == 3 and png_arr.shape[2] == 3


def test_batched_acting_matches_individual_brains():
    # the vectorized engine must be numerically identical to running each
    # creature's own brain in a loop (same separate brains, computed in parallel)
    import torch
    from torch.func import stack_module_state, functional_call, vmap
    from embodied_agent.colony.run import Mind
    from embodied_agent.colony.batched import _ObsFeat
    from embodied_agent.model.rssm import RSSMState
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    N = 3
    minds = [Mind(cfg, sp, "cpu") for _ in range(N)]
    for m in minds:
        m.wm.eval()
    d = lambda v: v if isinstance(v, int) else int(np.prod(v.shape))
    obs = [{k: torch.randn(d(v)) for k, v in sp.items()} for _ in range(N)]
    deter = cfg.model.deter_dim
    stoch = minds[0].wm.rssm.stoch_flat
    h = torch.randn(N, deter); z = torch.randn(N, stoch); pa = torch.randn(N, 2)
    of = [_ObsFeat(m) for m in minds]
    with torch.no_grad():
        # (a) matches the real RSSM obs_step (deterministic h + posterior logits)
        r = minds[0].wm.rssm
        embed = minds[0].wm.encoder({k: v[None] for k, v in obs[0].items()})
        post, _ = r.obs_step(RSSMState(h[0:1], z[0:1]), pa[0:1], embed)
        hh, lg = of[0](obs[0], h[0], z[0], pa[0])
        assert torch.allclose(post.h.squeeze(0), hh, atol=1e-4)
        assert torch.allclose(post.params["logits"].squeeze(0), lg, atol=1e-4)
        # (b) vmap over N stacked brains == the per-brain loop
        hl = torch.stack([of[i](obs[i], h[i], z[i], pa[i])[0] for i in range(N)])
        p, b = stack_module_state(of)
        base = of[0]
        ob = {k: torch.stack([obs[i][k] for i in range(N)]) for k in sp}
        hv, _ = vmap(lambda p, b, o, x, y, w:
                     functional_call(base, (p, b), (o, x, y, w)))(p, b, ob, h, z, pa)
        assert torch.allclose(hv, hl, atol=1e-4)


def test_batched_wm_twohot_matches_reference():
    # the vmap-safe arithmetic two-hot must equal the searchsorted/scatter one
    import torch, torch.nn.functional as F
    from embodied_agent.model.distributions import TwoHotHead, symlog
    head = TwoHotHead(8, 16, bins=51, low=-8.0, high=8.0)
    target = torch.randn(4, 6) * 5.0
    t = symlog(target)
    ref = head._twohot(t)                                   # scatter/searchsorted
    centers = head.centers
    d = centers[1] - centers[0]
    tc = t.clamp(centers[0], centers[-1])
    mine = F.relu(1 - (tc.unsqueeze(-1) - centers).abs() / d)
    assert torch.allclose(ref, mine, atol=1e-5)


def test_batched_wm_observe_matches_rssm():
    # the plain-math GRU observe step must match the real RSSM obs_step
    import torch
    from embodied_agent.colony.run import Mind
    from embodied_agent.colony.batched_train import _WMLoss
    from embodied_agent.model.rssm import RSSMState
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    m = Mind(cfg, sp, "cpu"); m.wm.eval()
    wl = _WMLoss(m, cfg)
    d = lambda v: v if isinstance(v, int) else int(np.prod(v.shape))
    obs = {k: torch.randn(1, 1, d(v)) for k, v in sp.items()}
    action = torch.randn(1, 1, 2)
    gnoise = torch.zeros(1, 1, cfg.model.latent_groups, cfg.model.latent_classes)
    with torch.no_grad():
        feat, post_lg, prior_lg = wl._observe(
            wl.wm.encoder(obs), action, gnoise)
        embed = m.wm.encoder(obs)[:, 0]
        r = m.wm.rssm
        post, prior = r.obs_step(r.initial(1, "cpu"), action[:, 0], embed)
        assert torch.allclose(post.h, feat[:, 0, :r.deter_dim], atol=1e-5)
        assert torch.allclose(post.params["logits"], post_lg[:, 0], atol=1e-5)
        assert torch.allclose(prior.params["logits"], prior_lg[:, 0], atol=1e-5)


def _wm_batch(sp, B=4, T=6):
    import torch
    d = lambda v: v if isinstance(v, int) else int(np.prod(v.shape))
    return {"obs": {k: torch.randn(B, T, d(v)) for k, v in sp.items()},
            "prev_action": torch.randn(B, T, 2),
            "reward": torch.randn(B, T),
            "cont": torch.rand(B, T)}


def test_batched_wm_grad_vmap_matches_loop():
    # vmap(grad) over N stacked world models == the per-brain grad loop
    import torch
    from torch.func import functional_call, grad, stack_module_state, vmap
    from embodied_agent.colony.run import Mind
    from embodied_agent.colony.batched_train import _WMLoss
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    N, B, T = 3, 4, 6
    wl = [_WMLoss(Mind(cfg, sp, "cpu"), cfg) for _ in range(N)]
    for w in wl:
        w.eval()
    batches = [_wm_batch(sp, B, T) for _ in range(N)]
    G, C = cfg.model.latent_groups, cfg.model.latent_classes
    gnoise = torch.randn(N, B, T, G, C)
    obs = {k: torch.stack([b["obs"][k] for b in batches]) for k in sp}
    act = torch.stack([b["prev_action"] for b in batches])
    rew = torch.stack([b["reward"] for b in batches])
    cont = torch.stack([b["cont"] for b in batches])
    params, buffers = stack_module_state(wl)
    base = wl[0]

    def loss_fn(p, bf, ob, ac, rw, ct, gn):
        return functional_call(base, (p, bf), (ob, ac, rw, ct, gn))

    gv = vmap(grad(loss_fn), in_dims=(0, 0, 0, 0, 0, 0, 0))(
        params, buffers, obs, act, rew, cont, gnoise)
    for i in range(N):
        pi = {k: v[i] for k, v in params.items()}
        bi = {k: v[i] for k, v in buffers.items()}
        gi = grad(loss_fn)(pi, bi, {k: obs[k][i] for k in sp}, act[i], rew[i],
                           cont[i], gnoise[i])
        for name in params:
            assert torch.allclose(gv[name][i], gi[name], atol=1e-4), name


def test_batched_wm_update_matches_perbrain_adam():
    # the full batched update (grad + per-mind clip + Adam + write-back) must
    # equal a per-brain functional Adam step given the same sampling noise
    import torch
    from torch.func import functional_call, grad
    from embodied_agent.colony.run import Mind
    from embodied_agent.colony.batched_train import _WMLoss, BatchedWMTrainer
    cfg = _tiny(load_config("colony"))
    sp = _spaces(cfg)
    N, B, T = 3, 4, 6
    minds = [Mind(cfg, sp, "cpu") for _ in range(N)]
    batches = [_wm_batch(sp, B, T) for _ in range(N)]
    G, C = cfg.model.latent_groups, cfg.model.latent_classes
    gnoise = torch.randn(N, B, T, G, C)
    lr, clip = cfg.model.lr, cfg.model.grad_clip
    b1, b2, eps = 0.9, 0.999, 1e-8

    # reference: one functional Adam step per brain, from zero state
    expected = []
    for i, m in enumerate(minds):
        wl = _WMLoss(m, cfg)
        p0 = {n: v.detach().clone() for n, v in wl.named_parameters()}

        def loss_fn(p, ob, ac, rw, ct, gn, _wl=wl):
            bf = dict(_wl.named_buffers())
            return functional_call(_wl, (p, bf), (ob, ac, rw, ct, gn))

        g = grad(loss_fn)(p0, batches[i]["obs"], batches[i]["prev_action"],
                          batches[i]["reward"], batches[i]["cont"], gnoise[i])
        norm = torch.sqrt(sum((gg ** 2).sum() for gg in g.values()))
        coef = min(1.0, clip / (float(norm) + 1e-6))
        exp = {}
        for n in p0:
            gc = g[n] * coef
            mm = (1 - b1) * gc
            vv = (1 - b2) * gc * gc
            mhat = mm / (1 - b1)
            vhat = vv / (1 - b2)
            exp[n] = p0[n] - lr * mhat / (torch.sqrt(vhat) + eps)
        expected.append(exp)

    trainer = BatchedWMTrainer(cfg, "cpu")
    trainer.update(minds, batches, gnoise=gnoise)
    for i, m in enumerate(minds):
        got = dict(_WMLoss(m, cfg).named_parameters())
        for n, exp in expected[i].items():
            assert torch.allclose(got[n], exp, atol=1e-5), n


def test_run_colony_batched_train_runs():
    from embodied_agent.colony.run import run_colony
    cfg = _tiny(load_config("colony"))
    cfg.colony.batched = True
    cfg.colony.batched_train = True
    cfg.colony.n_init = 5
    cfg.colony.n_max = 8
    cfg.train.total_steps = 150
    cfg.train.seq_len = 12
    cfg.train.batch_size = 8
    cfg.train.log_every = 75
    out = run_colony(cfg, verbose=False)
    assert out["stats"]["population"] >= cfg.colony.n_min


def test_run_colony_batched_runs():
    from embodied_agent.colony.run import run_colony
    cfg = _tiny(load_config("colony"))
    cfg.colony.batched = True
    cfg.colony.n_init = 5
    cfg.colony.n_max = 8
    cfg.train.total_steps = 120
    cfg.train.seq_len = 12
    cfg.train.batch_size = 8
    cfg.train.log_every = 60
    out = run_colony(cfg, verbose=False)
    assert out["stats"]["population"] >= cfg.colony.n_min


def test_run_colony_individual_and_shared():
    from embodied_agent.colony.run import run_colony
    for shared in (False, True):
        cfg = _tiny(load_config("colony"))
        cfg.colony.n_init = 5
        cfg.colony.n_max = 8
        cfg.colony.shared_brain = shared
        cfg.colony.max_trains_per_step = 3
        cfg.train.total_steps = 150
        cfg.train.seq_len = 12
        cfg.train.batch_size = 8
        cfg.train.log_every = 100
        out = run_colony(cfg, verbose=False)
        assert out["shared_brain"] is shared
        assert out["stats"]["population"] >= cfg.colony.n_min
