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
    # vision mode: "rays" (vector ray-casts) or "pixels" (egocentric retina)
    vision_mode: str = "rays"
    # vision: egocentric ray-casts returning distance + hit channel
    n_rays: int = 12
    fov_deg: float = 140.0
    ray_max_dist: float = 8.0
    # retina (pixels): egocentric rasterized RGB patch in front of the agent
    retina_res: int = 16          # H = W (keep a power of two for the CNN)
    retina_range: float = 7.0     # world units the patch spans forward
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
    hidden: int = 128             # MLP hidden width
    cnn_depth: int = 16           # base channel count for image encoder/decoder
    lr: float = 3e-4
    kl_beta: float = 1.0
    kl_balance: float = 0.8       # weight on training the prior toward posterior
    free_bits: float = 1.0        # nats of KL below which no gradient flows
    grad_clip: float = 100.0
    # stochastic latent: "discrete" (categorical, DreamerV3) or "gaussian"
    latent_kind: str = "discrete"
    latent_groups: int = 16       # categorical variables (discrete)
    latent_classes: int = 16      # classes per variable (discrete)
    unimix: float = 0.01          # uniform mixture on categorical probs
    stoch_dim: int = 24           # Gaussian latent size (gaussian only)
    # reward head: "twohot" (symlog two-hot classification) or "mse"
    reward_head: str = "twohot"
    reward_bins: int = 51
    reward_low: float = -8.0      # symlog-space bin range (symexp(8) ~ 2980)
    reward_high: float = 8.0
    # B4 sparse cortical code: keep only a fraction of latent groups active
    # (k-winners-take-all over the discrete groups). Off -> dense (baseline).
    sparse_latent: bool = False
    sparse_frac: float = 0.5      # fraction of groups allowed to fire
    # B7 learning rule for the predictive substrate. "backprop" is the default
    # (BPTT + global loss); "predictive_coding" uses local error-neuron /
    # Hebbian updates (no autograd through the loss). See model/predictive_coding.
    learning_rule: str = "backprop"
    pc_inference_steps: int = 20  # relaxation iterations per predictive-coding step
    pc_inference_lr: float = 0.1  # step size of the latent relaxation
    pc_weight_lr: float = 0.02    # local Hebbian weight-update rate
    recon_scales: dict = field(default_factory=lambda: {
        "vision": 1.0, "retina": 1.0, "touch": 1.0, "proprio": 1.0,
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
    # B5 innate action bias (a heritable behavioural prior): added to the
    # actor's pre-tanh mean, so a newborn acts on instinct before it learns.
    action_bias: list = field(default_factory=lambda: [0.0, 0.0])
    # B6 active inference. "return" -> maximize lambda-returns of reward (+ a
    # hand-scaled curiosity bonus); "expected_free_energy" -> minimize EFE, a
    # single quantity uniting preference-seeking (pragmatic) and information
    # gain (epistemic) with no separately tuned curiosity weight.
    objective: str = "return"
    efe_precision: float = 5.0    # precision of the preferred-outcome prior C
    efe_epistemic: float = 1.0    # weight on the epistemic term (1 = pure EFE)


@dataclass
class IntrinsicConfig:
    method: str = "rnd"           # "rnd" | "disagreement" | "none"
    # bonus magnitude; sized to be comparable to the typical per-step
    # extrinsic (homeostatic) reward so curiosity actually influences the
    # imagined return during ordinary wandering, while extrinsic spikes
    # (food/collision) still dominate when something urgent happens
    scale: float = 0.5
    lr: float = 1e-4
    hidden: int = 64
    out_dim: int = 32
    ensemble: int = 4             # for disagreement


@dataclass
class ShieldConfig:
    enabled: bool = True
    # circular no-go zones: list of [cx, cy, radius]; empty -> auto-place one
    forbidden_zones: list = field(default_factory=list)
    lookahead_steps: int = 6      # kinematic lookahead for the override check
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
class NeuromodConfig:
    # B1 -- neuromodulation & allostasis. Off by default so the deep-RL
    # baseline (fixed drive weights, fixed learning rate) is preserved.
    enabled: bool = False
    # allostasis: interoceptive deficits amplify their drive's weight
    # super-linearly, so the most-threatened variable dominates arbitration.
    allostatic: bool = True
    energy_gain: float = 4.0       # how strongly low energy up-weights feeding
    integrity_gain: float = 6.0    # near-death integrity dominates
    thermal_gain: float = 3.0
    urgency_power: float = 2.0     # super-linear deficit -> urgency
    # norepinephrine: prediction-error (surprise) raises effective plasticity
    ne_enabled: bool = True
    ne_gain: float = 1.5           # max multiplicative boost to the LR
    ne_ema: float = 0.99           # EMA horizon for the surprise baseline
    # dopamine: |RPE| tone, logged and (optionally) gates the actor LR
    da_enabled: bool = True
    da_gain: float = 0.5
    da_ema: float = 0.99


@dataclass
class SleepConfig:
    # B2 -- sleep, consolidation & dreaming. Off by default so the baseline
    # (interleaved uniform-replay updates, no circadian structure) is preserved.
    enabled: bool = False
    # circadian clock: one day = day_steps env steps; the last (1 - wake_frac)
    # of each day is sleep. Phase is exposed as an interoceptive-style signal.
    day_steps: int = 1000
    wake_frac: float = 0.8         # fraction of a day spent awake and foraging
    # while awake the agent does only *light* fast adaptation; the bulk of
    # world-model consolidation is deferred to sleep (complementary systems).
    wake_updates: int = 1          # updates per train_every during wake
    sleep_updates: int = 40        # consolidation updates per sleep phase
    # prioritized ("emotional") replay: salient episodes -- big |reward| spikes
    # (feeding, collisions, death) -- are replayed preferentially during sleep.
    prioritized: bool = True
    priority_exponent: float = 0.8  # alpha: 0 -> uniform, 1 -> full priority
    priority_eps: float = 0.02
    reward_salience: float = 1.0    # weight of peak |reward| in episode salience
    surprise_salience: float = 1.0  # weight of reward variability (eventfulness)
    # dreaming: extra actor-critic imagination updates from salient sleep
    # batches -- self-generated behavioural training on recombined experience.
    dream: bool = True
    dream_updates: int = 2         # extra AC imagination passes per sleep batch


@dataclass
class DevelopmentConfig:
    # B3 -- continual single life, development & critical periods. Off by
    # default so the baseline (episodic resets, constant plasticity) stands.
    enabled: bool = False
    # continual life: truncation no longer wipes the body -- it only segments
    # memory for replay. Only death (terminated) ends a life; a new individual
    # then begins (the hook B5 will populate with a fresh genome).
    continual: bool = True
    # critical period: plasticity is high at birth and anneals toward a mature
    # floor with a fixed time constant, so early experience imprints strongly.
    critical_period: bool = True
    young_gain: float = 3.0            # LR multiplier at birth (age 0)
    floor_gain: float = 0.7           # mature plasticity floor
    critical_period_steps: int = 8000  # decay time constant (steps)


@dataclass
class MetabolismConfig:
    # B4 -- metabolic cost of cognition & sensorimotor realism. Off by default
    # so the baseline (free imagination, instantaneous noiseless sensing) holds.
    enabled: bool = False
    # cognition costs energy: every imagined step (planning / dreaming) debits
    # the body's energy, so thinking is not free.
    cognition_cost: bool = True
    imagination_energy_cost: float = 6e-7  # energy per imagined step
    # bounded planning: a low-energy body cannot afford to think far ahead --
    # the imagination horizon shrinks toward min_horizon as energy falls.
    bounded_planning: bool = True
    min_horizon: int = 3
    # sensorimotor realism: latency (steps) and motor noise on the closed loop.
    obs_delay: int = 0
    action_delay: int = 0
    motor_noise: float = 0.0


@dataclass
class EvolutionConfig:
    # B5 -- evolutionary outer loop & reproduction. Off by default (a single
    # lifetime is the baseline); the evolution loop is opt-in via scripts.
    enabled: bool = False
    population: int = 8
    generations: int = 6
    elite_frac: float = 0.25      # top fraction copied forward unmutated
    tournament: int = 3           # tournament size for parent selection
    mutation_rate: float = 0.15   # gene std as a fraction of its range
    mutation_prob: float = 0.9    # per-gene chance of mutating
    life_steps: int = 1500        # env steps per fitness evaluation (full life)
    # reproduction as a homeostatic drive: sustained energy surplus accrues
    # reproductive readiness; crossing the threshold spawns an offspring.
    reproduction: bool = False
    repro_energy_threshold: float = 0.7   # energy above which readiness accrues
    repro_rate: float = 0.002             # readiness gained per surplus step
    repro_cost: float = 0.25              # energy spent bearing one offspring


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
    gui_every: int = 0            # steps between live-GUI callbacks (0 = off)
    seed: int = 0
    device: str = "cuda"          # "cuda" auto-falls back to CPU if unavailable
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
    neuromod: NeuromodConfig = field(default_factory=NeuromodConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    dev: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    metab: MetabolismConfig = field(default_factory=MetabolismConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
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
