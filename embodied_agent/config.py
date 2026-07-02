"""Configuration: nested dataclasses with YAML preset overrides.

All rates are *per simulation step* (dt is only used for kinematic
integration), which keeps the homeostatic bookkeeping easy to reason about.
"""
from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass, field, fields, is_dataclass

import yaml

CONFIG_DIR = pathlib.Path(__file__).parent / "configs"


@dataclass
class EnvConfig:
    arena_size: float = 20.0
    dt: float = 0.1
    max_episode_steps: int = 1000
    fixed_layout: bool = False
    # agent body / kinematics (unicycle model)
    agent_radius: float = 0.4
    accel: float = 6.0            # forward acceleration at |thrust|=1 (units/s^2)
    v_max: float = 3.0            # max forward speed (units/s)
    reverse_frac: float = 0.3     # max reverse speed as fraction of v_max
    turn_rate: float = 2.5        # angular speed at |turn|=1 (rad/s)
    drag: float = 0.08            # per-step fractional velocity decay
    # food
    n_food: int = 6
    food_radius: float = 0.35
    food_energy: float = 0.35
    food_respawn_steps: int = 250
    # temperature field: mixture of hot/cold Gaussian blobs over a 0.5 baseline
    n_hot: int = 2
    n_cold: int = 2
    temp_amp: float = 0.45
    temp_sigma_frac: float = 0.15  # blob sigma as fraction of arena_size
    # obstacles (axis-aligned rectangles)
    n_obstacles: int = 3
    obstacle_min: float = 1.2
    obstacle_max: float = 4.0
    # homeostasis (per-step rates; all viability variables live in [0, 1])
    base_metabolism: float = 0.0009
    move_cost: float = 0.0012      # extra energy per step at |thrust|=1
    temp_coupling: float = 0.02    # internal temp relaxation toward ambient
    temp_setpoint: float = 0.5
    temp_danger: float = 0.35      # |temp - setpoint| beyond this burns integrity
    temp_damage_rate: float = 0.004
    collision_damage: float = 0.04  # integrity loss per unit impact speed
    integrity_regen: float = 0.0002


@dataclass
class SensorConfig:
    # vision: egocentric ray-casts returning distance + hit channel
    n_rays: int = 12
    fov_deg: float = 140.0
    ray_max_dist: float = 8.0
    # touch: contact pressure per body sector
    touch_sectors: int = 8
    touch_decay: float = 0.5      # per-step decay of touch activation
    # smell: slow chemical gradient toward food (different rate from vision)
    smell_sigma_frac: float = 0.25  # plume sigma as fraction of arena_size
    smell_period: int = 4           # smell only refreshes every N steps
    # per-modality observation noise (std of additive Gaussian noise)
    noise: dict = field(default_factory=lambda: {
        "vision": 0.01, "touch": 0.0, "proprio": 0.005,
        "intero": 0.0, "smell": 0.02,
    })


@dataclass
class ModelConfig:
    embed_dim: int = 128          # fused embedding fed to the RSSM posterior
    deter_dim: int = 128          # GRU deterministic state h
    stoch_dim: int = 24           # Gaussian stochastic latent z
    hidden: int = 128             # MLP hidden width
    lr: float = 3e-4
    kl_beta: float = 1.0
    kl_balance: float = 0.8       # weight on training the prior toward posterior
    free_bits: float = 1.0        # nats of KL below which no gradient flows
    grad_clip: float = 100.0
    recon_scales: dict = field(default_factory=lambda: {
        "vision": 1.0, "touch": 1.0, "proprio": 1.0,
        "intero": 10.0, "smell": 1.0,
    })


@dataclass
class AgentConfig:
    horizon: int = 12             # imagination rollout length
    gamma: float = 0.98
    lam: float = 0.95
    actor_lr: float = 8e-5
    critic_lr: float = 2e-4
    entropy_scale: float = 3e-3   # bonus vs scale-normalized returns
    critic_ema: float = 0.98      # EMA rate for the target critic
    grad_clip: float = 100.0
    expl_noise: float = 0.2       # action noise when collecting real steps


@dataclass
class IntrinsicConfig:
    method: str = "rnd"           # "rnd" | "disagreement" | "none"
    scale: float = 0.05
    lr: float = 1e-4
    hidden: int = 64
    out_dim: int = 32
    ensemble: int = 4             # for disagreement


@dataclass
class ShieldConfig:
    enabled: bool = True
    # circular no-go zones: list of [cx, cy, radius]; empty -> auto-place one
    forbidden_zones: list = field(default_factory=list)
    lookahead_steps: int = 3      # kinematic lookahead for the override check
    brake_thrust: float = -1.0


@dataclass
class RewardConfig:
    # drive weights for the homeostatic reward (arbitration is explicit here)
    w_energy: float = 1.0
    w_thermal: float = 0.5
    w_integrity: float = 1.5
    death_penalty: float = 2.0
    reward_scale: float = 100.0   # drives move slowly; rescale to O(1) rewards


@dataclass
class TrainConfig:
    total_steps: int = 30000
    warmup_steps: int = 1500      # random-policy steps before training starts
    seq_len: int = 24
    batch_size: int = 16
    buffer_capacity: int = 100000
    train_every: int = 8          # env steps between gradient updates
    updates_per_train: int = 1
    log_every: int = 500
    eval_every: int = 5000
    gif_every: int = 10000
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/default"
    headless: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _apply(dc, overrides: dict, path: str = ""):
    names = {f.name for f in fields(dc)}
    for key, val in overrides.items():
        if key not in names:
            raise KeyError(f"unknown config key: {path}{key}")
        cur = getattr(dc, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _apply(cur, val, path=f"{path}{key}.")
        elif isinstance(cur, dict) and isinstance(val, dict):
            cur.update(val)
        else:
            setattr(dc, key, val)


def load_config(name_or_path: str) -> Config:
    """Load a preset by name (e.g. 'cpu_small') or a YAML path."""
    path = pathlib.Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = Config()
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    _apply(cfg, copy.deepcopy(overrides))
    return cfg
