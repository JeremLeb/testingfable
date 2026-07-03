"""Browser-free tests for the observation GUI and the run() live hooks."""
import json

import numpy as np

from embodied_agent.config import load_config
from embodied_agent.gui.server import SCENARIOS, SWITCHES, GuiState, _placeholder_png


def test_scenario_presets_all_load():
    # every scenario the UI offers must resolve to a real, loadable preset
    for name, (preset, desc) in SCENARIOS.items():
        cfg = load_config(preset)
        assert cfg.train.total_steps > 0
        assert isinstance(desc, str) and desc


def test_switches_map_to_real_config_groups():
    cfg = load_config("cpu_small")
    for key in SWITCHES:
        assert hasattr(getattr(cfg, key), "enabled")


def test_placeholder_frame_is_png():
    png = _placeholder_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_state_json_is_valid_and_complete():
    d = json.loads(GuiState().state_json())
    for key in ("running", "device", "status", "history", "scenarios",
                "switches"):
        assert key in d
    assert d["running"] is False
    assert set(d["scenarios"]) == set(SCENARIOS)


def test_senses_renderer_draws_rays_and_retina():
    from embodied_agent.gui.senses import SensesRenderer
    cfg = load_config("colony")
    n = cfg.sensor.n_rays
    vision = np.zeros(n * 4, dtype=np.float32)
    vision[1] = 1.0  # first ray sees food
    obs = {"vision": vision, "touch": np.zeros(cfg.sensor.touch_sectors),
           "smell": np.array([0.6, 0.3, -0.2]), "intero": np.array([0.8, 0.5, 0.9])}
    sr = SensesRenderer(cfg.sensor)
    png = sr.png(obs)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # pixel-retina path also renders
    png2 = sr.png({"retina": np.zeros((3, 8, 8), dtype=np.float32)})
    assert png2[:8] == b"\x89PNG\r\n\x1a\n"


def test_focus_click_selects_nearest_creature():
    from embodied_agent.gui.server import GuiState
    gs = GuiState()
    gs._snap = {"S": 10.0, "pts": [(5, 1.0, 1.0), (9, 8.0, 8.0)]}
    gs.focus_at(0.8, 0.2)   # -> world (8, 8): nearest is creature 9
    assert gs.focused_id == 9
    gs.focus_at(0.1, 0.9)   # -> world (1, 1): nearest is creature 5
    assert gs.focused_id == 5


def test_run_live_hooks_fire_and_stop_early():
    from embodied_agent.train import run
    cfg = load_config("cpu_small")
    cfg.env.max_episode_steps = 50          # keep the baseline rollout short
    cfg.train.warmup_steps = 60
    cfg.train.total_steps = 100000
    cfg.train.gui_every = 5
    cfg.train.out_dir = "runs/test_gui"
    seen = {"n": 0, "last_step": 0}

    def on_step(state):
        seen["n"] += 1
        seen["last_step"] = state["step"]
        assert "energy" in state["info"]

    summary = run(cfg, verbose=False, on_step=on_step,
                  should_stop=lambda: seen["n"] >= 4)
    assert seen["n"] >= 4
    assert seen["last_step"] < cfg.train.total_steps   # stopped early
    assert "final" in summary
