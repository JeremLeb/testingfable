"""Smoke tests: the headline scripts run a few hundred steps without error."""
from embodied_agent.config import load_config
from embodied_agent.train import run


def _tiny(cfg):
    cfg.train.total_steps = 200
    cfg.train.warmup_steps = 60
    cfg.train.eval_every = 200
    cfg.train.gif_every = 10 ** 9
    cfg.train.log_every = 100
    return cfg


def test_train_smoke_no_intrinsic(tmp_path):
    cfg = _tiny(load_config("cpu_small"))
    cfg.intrinsic.method = "none"
    cfg.shield.enabled = False
    cfg.train.out_dir = str(tmp_path / "run")
    summary = run(cfg, verbose=False)
    assert "final" in summary


def test_train_smoke_rnd_and_shield(tmp_path):
    cfg = _tiny(load_config("cpu_small"))
    cfg.intrinsic.method = "rnd"
    cfg.shield.enabled = True
    cfg.train.out_dir = str(tmp_path / "run")
    summary = run(cfg, verbose=False)
    assert summary["final"]["eval_length"] > 0
