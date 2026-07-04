# Operating Manual — Embodied Multisensory Learning Agent

This is the complete reference for operating the sandbox: what it can do,
every entry point and flag, the full configuration reference, how to read
the outputs, and how to extend it. For the short overview and design
philosophy, see the top-level [README](../README.md).

---

## 1. What this program can do

The system simulates a small embodied creature in a continuous 2D arena and
lets it **teach itself to survive** with no labels, no demonstrations, and no
hand-coded behavior. Concretely, it can:

| Capability | Entry point |
|---|---|
| Simulate the arena with a random agent and render GIFs | `scripts.random_rollout` |
| Learn a predictive world model of the agent's five senses from raw experience | `scripts.train_world_model` |
| Train a full agent (world model + actor-critic in imagination) end-to-end | `train` |
| Evaluate a trained checkpoint against a random baseline | `evaluate` |
| Compare intrinsic-motivation variants and measure their effect on exploration | `scripts.ablation` |
| Prove the safety shield blocks forbidden actions even under an adversarial policy | `scripts.shield_demo` |
| Re-render the observability dashboard from any run's CSV | `viz.plots` |

What the trained agent *learns to do* (emergent, not coded): seek food when
energy is low, drift toward its thermal comfort zone, avoid walls (including
braking before contact), and explore the arena far more thoroughly than a
random walker.

What it deliberately **cannot** do: no self-preservation-against-shutdown,
no self-replication, no status-seeking. The shield structurally prevents
entering forbidden zones regardless of what the policy wants.

## 2. Installation

Requires Python 3.10+. CPU-only PyTorch is sufficient for `cpu_small`.

```bash
pip install -r requirements.txt
# or explicitly with the CPU wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pyyaml imageio pytest
```

All commands are run from the repository root as modules (`python -m ...`).
Everything is headless-safe (matplotlib Agg backend); no display is needed.

## 3. Quickstart (5 minutes on a laptop CPU)

**No-code path:** run the installer (`install.sh` / `install.bat`), then the
launcher (`run.sh` / `run.bat`) to open the live dashboard at
http://localhost:8000. `python -m embodied_agent.doctor` reports your Python,
packages, and GPU; `python -m embodied_agent.gui` launches the dashboard
directly (`--port`, `--host`, `--no-browser` flags). See
[QUICKSTART.md](../QUICKSTART.md).

**Command-line path:**

```bash
# 1. sanity: watch a random agent wander (writes a GIF)
python -m embodied_agent.scripts.random_rollout --config cpu_small

# 2. validate the learning substrate: world model on random data (~1 min)
python -m embodied_agent.scripts.train_world_model --config cpu_small

# 3. full training (~5 min): food-seeking + exploration emerge
python -m embodied_agent.train --config cpu_small

# 4. inspect results
python -m embodied_agent.evaluate --config cpu_small \
    --checkpoint runs/cpu_small/checkpoint.pt --gif
open runs/cpu_small/metrics.png   # observability dashboard
```

Expected outcome of step 3 on `cpu_small`: final eval reward around −50
versus around −80 for random; ~5 food eaten per episode; coverage ~4× random.
Exact numbers vary by seed; the direction is the point.

## 4. Command reference

### 4.1 `python -m embodied_agent.train` — end-to-end training

The main loop: collects real steps with the actor (+ exploration noise),
trains the world model on replayed sequences, trains the actor-critic on
imagined rollouts, evaluates periodically against a random baseline, and
writes all observability artifacts.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `cpu_small` | preset name in `embodied_agent/configs/` or a YAML path |
| `--steps` | from config | override `train.total_steps` |
| `--seed` | from config | override `train.seed` |
| `--out` | from config | override `train.out_dir` |
| `--intrinsic` | from config | `none` \| `rnd` \| `disagreement` |
| `--shield` | from config | `on` \| `off` |

Console output per log interval: `ep_reward` (last finished episode),
`wm_loss` (world model total), `imag_ret` (mean imagined λ-return),
`entropy` (actor), steps/s. Eval lines compare reward/food/coverage to the
random baseline measured at startup.

Programmatic use: `from embodied_agent.train import run; summary = run(cfg)`
returns `{"final": {...}, "baseline_reward", "baseline_coverage", "out_dir"}`.

### 4.2 `python -m embodied_agent.evaluate` — checkpoint evaluation

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `cpu_small` | must match the checkpoint's architecture sizes |
| `--checkpoint` | (required) | path to `checkpoint.pt` from a training run |
| `--episodes` | 10 | evaluation episodes |
| `--seed` | 0 | eval seed (episodes use seed+100+i) |
| `--shield` | from config | force shield `on`/`off` at eval time |
| `--gif` | off | save `evaluate_rollout.gif` next to the checkpoint |
| `--stochastic` | off | sample the actor instead of using its mean |

Reports mean ± std reward, food eaten, episode length (early end = death),
coverage, and shield interventions per episode, plus the random baseline.

### 4.3 `python -m embodied_agent.scripts.random_rollout` — env demo

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `cpu_small` | preset |
| `--steps` | 400 | env steps |
| `--seed` | 0 | seed |
| `--out` | `runs/random_rollout` | output dir |
| `--render-every` | 2 | frame subsampling for the GIF |

Prints observation shapes, collision/food/death counts, final interoception,
and the summed per-drive reward decomposition.

### 4.4 `python -m embodied_agent.scripts.train_world_model` — substrate

Offline world-model training on smoothed-random trajectories. No policy is
involved: this validates that the prediction substrate learns before any
behavior exists.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `cpu_small` | preset |
| `--collect-steps` | 8000 | random env steps to gather first |
| `--updates` | 2000 | gradient updates |
| `--seed` | 0 | seed |
| `--out` | `runs/world_model` | output dir |
| `--no-reward` | off | ablate reward/continue heads (prediction-only) |

Watch the **open-loop MSE** column: the model filters half a sequence with
observations, then predicts the rest blind; falling error means the dynamics
are real, not just per-frame reconstruction. Writes
`world_model_learning.png` (reconstruction, KL, open-loop MSE curves) and
`world_model.pt`. With `--no-reward` the open-loop error should fall roughly
as much as with reward heads on — proof that reward is a thin layer.

### 4.5 `python -m embodied_agent.scripts.ablation` — condition comparison

| Flag | Default | Meaning |
|---|---|---|
| `--set` | `curiosity` | `curiosity` (none/rnd/disagreement) or `shield` (off/on) |
| `--config` | `cpu_small` | preset |
| `--steps` | from config | shorten runs for quick comparisons |
| `--seeds` | 1 | seeds per condition (means are reported across seeds) |
| `--out` | `runs/ablation` | output root; runs land in `<out>/<set>/<label>_s<seed>/` |

Writes `summary.csv` and `summary.png` (coverage and reward bars vs the
random baseline) and prints a table. Budget guidance: one condition-seed at
8k steps is ~2–3 CPU-minutes; `--set curiosity --seeds 2` is 6 runs.

### 4.6 `python -m embodied_agent.scripts.shield_demo` — safety proof

Drives an adversarial policy (full thrust at the forbidden zone) with the
shield off, then on. Asserts zero zone entries with the shield on and prints
both entry counts and the number of logged interventions. Saves
`shield_on.gif` with the zone drawn as a dashed circle.

Flags: `--config`, `--steps` (400), `--seed` (1), `--out` (`runs/shield_demo`).

### 4.7 `python -m embodied_agent.scripts.fear_analysis` — emergent-fear proof

Demonstrates that anticipatory avoidance is *learned*, not coded. Loads a
trained checkpoint (or trains a short model first if none is given), runs a
mildly-exploratory trace, and writes `fear_analysis.png` with three
signatures: (1) the world model's **counterfactual danger** — imagined
integrity drop under a "keep driving forward" rollout — rising before
collisions; (2) a **linear latent probe** for "collision within K steps",
reporting held-out AUC and showing its probability peak a few steps *before*
contact; (3) the learned policy's **collision rate vs a random policy**. Also
prints a quantitative summary.

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `cpu_small` | preset |
| `--checkpoint` | none | trained checkpoint; if omitted, trains `--train-steps` first |
| `--train-steps` | 12000 | steps to train when no checkpoint is given |
| `--trace-steps` | 5000 | analysis trace length |
| `--horizon` | 8 | imagination horizon / probe look-ahead window |
| `--window` | 12 | ± steps for the event-triggered averages |
| `--seed` | 0 | seed |
| `--out` | `runs/fear_analysis` | output dir |

The signature is clearest on a reasonably trained agent (≥ ~15k `cpu_small`
steps): probe AUC ≈ 0.87, probe probability peaking around lag −4, and ~4×
fewer collisions than random.

### 4.8 `python -m embodied_agent.viz.plots` — re-render dashboard

```bash
python -m embodied_agent.viz.plots runs/cpu_small/metrics.csv out.png
```

### 4.9 Biological-mechanism demos (Phase 1)

Each writes a figure showing the mechanism **on vs its ablation**; several take
`--ablate` for the end-to-end comparison.

```bash
python -m embodied_agent.scripts.neuromod_demo --config cpu_small       # B1
python -m embodied_agent.scripts.sleep_demo --config cpu_small          # B2
python -m embodied_agent.scripts.development_demo --config cpu_small     # B3
python -m embodied_agent.scripts.metabolism_demo --config cpu_small      # B4
python -m embodied_agent.train --config cpu_bio                          # Phase 1 on
python -m embodied_agent.scripts.evolution_demo --config cpu_small       # B5 (Phase 2)
python -m embodied_agent.scripts.active_inference_demo --config cpu_small # B6 (Phase 3)
python -m embodied_agent.scripts.predictive_coding_demo --config cpu_small # B7 (Phase 3)
python -m embodied_agent.colony.run --config colony                     # living colony
```

**Living colony** (`embodied_agent/colony/`): many creatures in one bigger
shared world, each an **individual** with its own body, senses, genome, and its
own **mind** (`Mind` = a private world model + actor + replay buffer, learning
only from its own life); newborns inherit a copy of a parent's brain
(`brain_inherit`) then diverge, and a mind dies with its body. They contest
food, collide, reproduce into live offspring, die, and are refilled by
immigration. `ColonyConfig` sets `n_init` / `n_max` / `n_min`, `shared_brain`
(one pooled brain instead of individuals), `brain_inherit`,
`max_trains_per_step` (rotating per-step training budget), reproduction
(`repro_energy_threshold`, `repro_rate`, `repro_cost`), `mutation_rate`, and
`creature_collision_damage`; the `colony` preset also enlarges the arena/food
and keeps each per-creature brain small. Run headless with `python -m
embodied_agent.colony.run --config colony` (`--shared-brain` to pool), or pick
**Colony** in the GUI. `run_colony(cfg, on_step=…, should_stop=…)` mirrors
`train.run`'s live hooks.

**B6 active inference** is `agent.objective` = `return` (baseline: λ-returns of
reward + curiosity) or `expected_free_energy` (minimize EFE = pragmatic
log-preference over predicted intero setpoints + epistemic ensemble
disagreement); tune with `agent.efe_precision` and `agent.efe_epistemic`. EFE
mode needs `intrinsic.method: disagreement` for the epistemic term and logs
`efe_pragmatic` / `efe_epistemic`.

**B7 predictive coding** is `model.learning_rule` = `backprop` (default) or
`predictive_coding` (local error-neuron / Hebbian rule,
`model/predictive_coding.py`, with `pc_inference_steps`, `pc_inference_lr`,
`pc_weight_lr`). The recurrent `WorldModel` is BPTT and rejects the switch; the
predictive-coding rule applies to the feedforward prediction substrate (its
demo shows open-loop MSE parity with backprop).

### 4.10 Tests

```bash
python -m pytest embodied_agent/tests/ -q          # full suite
python -m pytest embodied_agent/tests/ -q -k "not smoke"   # fast subset
```

Covers env dynamics (bounds, energy drain, temperature range, death,
reward-decomposition consistency, seed determinism), geometry, sensor
ranges/rates, replay buffer, world-model loss decrease, λ-return math,
actor-critic step, shield blocking, the Tier-1 upgrades (symlog/two-hot,
discrete straight-through latents, the CNN retina path, AUC + latent probe),
the biological modules (`tests/test_bio.py`: allostatic weighting,
prioritized-replay ordering, wake/sleep phase, plasticity annealing, continual
life across truncation, bounded planning, sensorimotor delay, sparse latent,
the B5 genome/mutation/crossover, action-bias instinct, fitness adaptation
in a harsh arena, and the reproduction drive; the B6 EFE decomposition and
scale-free epistemic term; and the B7 predictive-coding learning and its
gradient-alignment with backprop), and end-to-end training smoke tests (incl.
the pixel retina).

## 5. Configuration reference

Presets live in `embodied_agent/configs/`: `cpu_small.yaml` (fast, discrete
latents + two-hot reward + ray vision), `cpu_pixels.yaml` (CPU smoke of the
pixel retina + CNN), `cpu_bio.yaml` (cpu_small with all Phase-1 biological
mechanisms on), `gpu_laptop.yaml` (biological stack, mid-size model, tuned for
a laptop RTX 4060 Ti / 8 GB — `device: cuda` auto-falls back to CPU),
`gpu_default.yaml` (full pixel-retina stack for a desktop GPU), and
`colony.yaml` (the multi-agent living colony, individual minds, CPU-friendly),
and `colony_gpu.yaml` (a bigger colony for a GPU: larger per-creature brains and
population, same individual-minds design; the GUI's Colony scenario auto-picks
it when CUDA is present). A YAML preset overrides the dataclass defaults in
`embodied_agent/config.py`;
unknown keys raise an error. Pass either a preset name or a YAML path to
`--config`. Rates are **per step** unless noted; the arena uses abstract
length units; all viability variables live in [0, 1].

### `env` — world physics and homeostasis

| Key | Default | Meaning |
|---|---|---|
| `arena_size` | 20.0 | side length of the square arena |
| `dt` | 0.1 | integration timestep (kinematics only) |
| `max_episode_steps` | 1000 | truncation horizon |
| `fixed_layout` | false | deterministic arena for reproducible eval |
| `agent_radius` | 0.4 | body radius |
| `accel` | 6.0 | forward acceleration at full thrust (units/s²) |
| `v_max` | 3.0 | max forward speed (units/s) |
| `reverse_frac` | 0.3 | max reverse speed as a fraction of `v_max` |
| `turn_rate` | 2.5 | angular speed at full turn (rad/s) |
| `drag` | 0.08 | per-step fractional velocity decay |
| `n_food` | 6 | food items |
| `food_radius` | 0.35 | pickup radius (contact = agent radius + this) |
| `food_energy` | 0.35 | energy restored per item |
| `food_respawn_steps` | 250 | respawn delay (new location unless fixed layout) |
| `n_hot`, `n_cold` | 2, 2 | Gaussian temperature blobs |
| `temp_amp` | 0.45 | blob amplitude around the 0.5 baseline |
| `temp_sigma_frac` | 0.15 | blob width as fraction of `arena_size` |
| `n_obstacles` | 3 | rectangular obstacles |
| `obstacle_min/max` | 1.2 / 4.0 | obstacle side range |
| `base_metabolism` | 0.0009 | passive energy drain per step |
| `move_cost` | 0.0012 | extra drain per step at full thrust |
| `temp_coupling` | 0.02 | internal-temp relaxation toward ambient |
| `temp_setpoint` | 0.5 | thermal comfort point |
| `temp_danger` | 0.35 | \|temp − setpoint\| beyond this burns integrity |
| `temp_damage_rate` | 0.004 | integrity loss rate in the danger zone |
| `collision_damage` | 0.04 | integrity loss per unit impact speed |
| `integrity_regen` | 0.0002 | slow self-repair |

Death = energy ≤ 0 or integrity ≤ 0; the episode terminates.

### `sensor` — the modalities

| Key | Default | Meaning |
|---|---|---|
| `vision_mode` | `rays` | `rays` (vector ray-casts) or `pixels` (egocentric RGB retina) |
| `n_rays` | 12 | vision rays across the FOV (rays mode) |
| `fov_deg` | 140 | field of view (rays mode) |
| `ray_max_dist` | 8.0 | vision range, distances normalized by this (rays mode) |
| `retina_res` | 16 | retina H = W in pixels; keep a power of two for the CNN (pixels mode) |
| `retina_range` | 7.0 | world units the retina patch spans forward (pixels mode) |
| `touch_sectors` | 8 | contact-pressure sectors around the body |
| `touch_decay` | 0.5 | per-step decay of touch activation |
| `smell_sigma_frac` | 0.25 | food-plume width fraction |
| `smell_period` | 4 | smell refresh period (holds value between; the multi-rate stressor) |
| `noise` | per-modality dict | additive Gaussian std applied after encoding |

Observation dict (float32). Vision is **either** `vision (n_rays*4)` — per
ray, normalized distance + one-hot food/wall/none (rays mode) — **or**
`retina (3, retina_res, retina_res)` — an egocentric RGB patch with
wall / food / temperature channels in [0,1] (pixels mode). The rest are
always present: `touch (sectors)`; `proprio (6)` — speed, sin/cos heading,
efference copy of last thrust/turn, contact flag; `intero (3)` — energy,
temperature, integrity; `smell (3)` — tanh concentration + egocentric
gradient (forward, lateral). Action: `[thrust, turn]` in [−1, 1]².

### `model` — world model (RSSM)

| Key | Default | Meaning |
|---|---|---|
| `embed_dim` | 128 | fused observation embedding |
| `deter_dim` | 128 | GRU deterministic state h |
| `hidden` | 128 | MLP width |
| `cnn_depth` | 16 | base channel count for the retina CNN encoder/decoder |
| `latent_kind` | `discrete` | stochastic latent: `discrete` (categorical, DreamerV3) or `gaussian` |
| `latent_groups` | 16 | categorical variables (discrete mode); stoch size = groups × classes |
| `latent_classes` | 16 | classes per categorical variable (discrete mode) |
| `unimix` | 0.01 | uniform mixture on categorical probs (prevents dead classes) |
| `stoch_dim` | 24 | Gaussian latent size (gaussian mode only) |
| `reward_head` | `twohot` | `twohot` (symlog two-hot classification) or `mse` (scalar regression) |
| `reward_bins` | 51 | number of symlog bins (twohot mode) |
| `reward_low` / `reward_high` | −8 / 8 | symlog-space bin range; symexp(8) ≈ 2980 |
| `lr` | 3e-4 | world-model learning rate |
| `kl_beta` | 1.0 | KL weight |
| `kl_balance` | 0.8 | fraction of KL gradient training prior→posterior (dynamics vs representation) |
| `free_bits` | 1.0 | nats of KL below which no gradient flows |
| `grad_clip` | 100.0 | gradient norm clip |
| `recon_scales` | intero=10, rest=1 | per-modality reconstruction weights (intero is tiny but is the reward source, so it is upweighted; `retina` defaults to 1) |

### `agent` — actor-critic in imagination

| Key | Default | Meaning |
|---|---|---|
| `horizon` | 12 | imagination rollout length |
| `gamma`, `lam` | 0.98, 0.95 | discount and λ-return mixing |
| `actor_lr`, `critic_lr` | 8e-5, 2e-4 | learning rates |
| `entropy_scale` | 3e-3 | entropy bonus (vs scale-normalized returns) |
| `critic_ema` | 0.98 | target-critic EMA rate |
| `expl_noise` | 0.2 | extra action noise when collecting real steps |

### `intrinsic` — curiosity

| Key | Default | Meaning |
|---|---|---|
| `method` | `rnd` | `rnd` \| `disagreement` \| `none` |
| `scale` | 0.5 | bonus size; **must be comparable to typical per-step extrinsic reward** or it is swamped |
| `lr`, `hidden`, `out_dim` | 1e-4, 64, 32 | predictor nets |
| `ensemble` | 4 | members for disagreement |

### `shield` — hard constraints

| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | build and apply the shield |
| `forbidden_zones` | [] | list of `[cx, cy, radius]`; empty auto-places one disc |
| `lookahead_steps` | 6 | kinematic lookahead for the violation check |
| `brake_thrust` | −1.0 | fallback braking command |

### `reward` — drive arbitration (explicit and inspectable)

| Key | Default | Meaning |
|---|---|---|
| `w_energy` | 1.0 | energy-drive weight |
| `w_thermal` | 0.5 | thermal-drive weight |
| `w_integrity` | 1.5 | integrity-drive weight |
| `death_penalty` | 2.0 | one-off penalty on death |
| `reward_scale` | 100.0 | drives move slowly; rescales to O(1) rewards |

Reward per step = Σ w · (drive_before − drive_after) · scale − death penalty.
These weights are the hand-set arbitration (the deep-RL baseline); with B1
(`neuromod`) enabled they instead become the *base* weights that the body's
interoceptive state re-weights dynamically.

### Biological mechanisms (Phase 1) — `neuromod`, `sleep`, `dev`, `metab`

All four groups are **off by default** so the deep-RL baseline is preserved;
`configs/cpu_bio.yaml` turns them all on. Each is a clean ablation switch.

**`neuromod` — B1 neuromodulation & allostasis**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | master switch |
| `allostatic` | true | interoceptive deficits re-weight their drive super-linearly |
| `energy_gain`, `integrity_gain`, `thermal_gain` | 4, 6, 3 | how strongly each deficit up-weights its drive |
| `urgency_power` | 2.0 | super-linear exponent on the deficit |
| `ne_enabled`, `ne_gain`, `ne_ema` | true, 1.5, 0.99 | norepinephrine: surprise (prediction error) raises the LR up to ×`ne_gain` |
| `da_enabled`, `da_gain`, `da_ema` | true, 0.5, 0.99 | dopamine: |TD-error| tone gates the actor LR |

**`sleep` — B2 sleep, consolidation & dreaming**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | master switch (wake/sleep loop) |
| `day_steps`, `wake_frac` | 1000, 0.8 | circadian period; fraction awake |
| `wake_updates`, `sleep_updates` | 1, 40 | light adaptation while awake; bulk consolidation per night |
| `prioritized`, `priority_exponent`, `priority_eps` | true, 0.8, 0.02 | salience-weighted ("emotional") replay |
| `reward_salience`, `surprise_salience` | 1.0, 1.0 | salience = peak |reward| + reward variability |
| `dream`, `dream_updates` | true, 2 | extra actor-critic imagination passes during sleep |

**`dev` — B3 continual life & critical periods**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | master switch |
| `continual` | true | truncation only segments memory; only death starts a new life |
| `critical_period` | true | age-annealed plasticity |
| `young_gain`, `floor_gain` | 3.0, 0.7 | LR multiplier at birth; mature floor |
| `critical_period_steps` | 8000 | decay time constant of plasticity |

**`metab` — B4 metabolic cost of cognition & sensorimotor realism**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | master switch |
| `cognition_cost`, `imagination_energy_cost` | true, 6e-7 | energy debited per imagined step |
| `bounded_planning`, `min_horizon` | true, 3 | imagination horizon shrinks with low energy |
| `obs_delay`, `action_delay`, `motor_noise` | 0, 0, 0 | sensory/motor latency and motor noise |

The sparse **k-winners latent** is a model option: `model.sparse_latent`
(bool) and `model.sparse_frac` (fraction of latent groups allowed to fire).

**`evolution` — B5 evolutionary outer loop & reproduction**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | (single lifetime is the baseline; the loop is script-driven) |
| `population`, `generations` | 8, 6 | population size; generations to run |
| `elite_frac`, `tournament` | 0.25, 3 | elitism fraction; tournament size |
| `mutation_rate`, `mutation_prob` | 0.15, 0.9 | gene std (frac of range); per-gene chance |
| `life_steps` | 1500 | env steps per full-life fitness evaluation |
| `reproduction` | false | reproduction as a homeostatic drive |
| `repro_energy_threshold`, `repro_rate`, `repro_cost` | 0.7, 0.002, 0.25 | surplus threshold; readiness/step; energy per offspring |

The heritable **genome** (`evolution/genome.py`) sets innate `temp_setpoint`,
base drive weights, sensor morphology (`n_rays`, `fov_deg`), early plasticity
(`young_gain`), and an innate `action_bias`; `Genome.to_config` expresses it
onto a `Config`. `Population` (`evolution/population.py`) runs selection +
mutation; `innate_fitness` scores instinct-only survival, while
`train.live_one_life` scores a full life with lifetime learning. When
reproduction is on, `metrics.csv` / `info` carry `offspring` and
`repro_readiness`.

Extra `metrics.csv` columns when these are on: `weight_energy/thermal/integrity`
(B1 instantaneous drive weights), `ne_gain`/`da_tone` (B1), `circadian_phase`,
`asleep`, `sleep_consolidation`, `sleep_probe_loss` (B2), `age`,
`plasticity_gain`, `lifespan` (B3), `plan_horizon` (B4).

### `train` — loop and logging

| Key | Default | Meaning |
|---|---|---|
| `total_steps` | 30000 | real env steps |
| `warmup_steps` | 1500 | random-policy steps to seed the buffer |
| `seq_len`, `batch_size` | 24, 16 | replay sequence sampling |
| `buffer_capacity` | 100000 | replay steps kept |
| `train_every` | 8 | env steps between gradient updates |
| `updates_per_train` | 1 | updates per training event |
| `log_every`, `eval_every`, `gif_every` | 500, 5000, 10000 | cadences |
| `seed` | 0 | master seed |
| `device` | `cpu` | `cpu`/`cuda` (falls back to cpu if cuda absent) |
| `out_dir` | `runs/default` | artifact directory |

## 6. Reading the outputs

Each training run writes to its `out_dir`:

- **`metrics.csv`** — one row per `log_every` steps, averaged between
  flushes. Key columns: interoception (`energy`, `temp`, `integrity`),
  drives (`drive_*`), reward decomposition (`reward_energy`,
  `reward_thermal`, `reward_integrity`) — *this answers "which drive is
  winning right now"* — episode stats (`ep_reward`, `ep_len`,
  `interventions`), world model (`loss`, `recon`, `recon_<modality>`, `kl`,
  `reward`, `cont`, `grad_norm`), actor-critic (`actor_loss`, `critic_loss`,
  `imag_return`, `actor_entropy`, `return_scale`), intrinsic
  (`intrinsic_reward`, `rnd_loss`/`disagreement_loss`), and eval columns
  (`eval_reward`, `eval_food`, `eval_coverage`, `eval_length`,
  `baseline_reward`, `baseline_coverage`).
- **`metrics.png`** — 8-panel dashboard of the above.
- **`rollout_<step>.gif`** — deterministic-policy rollouts during training.
- **`checkpoint.pt`** — dict with `wm`, `actor`, `critic` state dicts.

Health signs: `recon` falls then plateaus; `kl` settles in single digits
(collapse to ~0 or explosion are both bad); `actor_entropy` should not pin at
its minimum early; `eval_reward` and `eval_coverage` should separate from the
baselines after a few thousand steps on `cpu_small`.

In GIFs: red/blue background = hot/cold field, gray = obstacles, green =
food, blue circle = agent (white tick = heading, turns red on contact),
dashed circle = forbidden zone.

## 7. Extending the sandbox

- **New sensor**: add a method + entry in `SensorSuite.observe()`
  (`env/sensors.py`) and its spec in `spaces` — a bare `int` for a vector
  modality, or `ModalitySpec((C, H, W), "image")` for an image (see
  `model/spaces.py`). `MultiEncoder`/`MultiDecoder` dispatch on the spec kind
  (MLP vs CNN) automatically; add a `recon_scales` entry if it needs
  weighting.
- **New latent or reward head**: the RSSM latent (`model/rssm.py`) and reward
  head (`model/world_model.py`) are selected by `model.latent_kind` /
  `model.reward_head`; add a branch and a config value to introduce another.
- **New drive**: add the variable + dynamics in `env/homeostasis.py`
  (`update`, `drives`, `info`), a weight in `RewardConfig`, and the term in
  `_reward()`.
- **New intrinsic method**: implement `reward(feat) -> tensor` and
  `train_step(feat) -> dict` (see `intrinsic/rnd.py`), register it in
  `intrinsic/__init__.py`. Learning-progress curiosity would slot in here.
- **New shield rule**: extend `Shield._violates()` / `filter()`
  (`safety/shield.py`); rules are analytic and auditable by design.
- **New preset**: copy a YAML in `configs/`; only override what differs.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Agent never eats during warmup | Random walkers rarely cross food; raise `n_food`/`food_radius`, or lengthen warmup. World model needs *some* eating events to learn energy dynamics. |
| Actor entropy collapses to the floor early | `entropy_scale` too small relative to normalized returns; raise it (returns are already scale-normalized, so values ~3e-3 are meaningful). |
| Curiosity has no visible effect | `intrinsic.scale` swamped by extrinsic reward; size it near the typical per-step extrinsic magnitude. |
| KL collapses toward 0 | Latent unused; lower `free_bits` or check encoders. |
| Checkpoint fails to load | `--config` must match the sizes used at training time. |
| Everything is slow | Both training and tests are CPU-hungry; do not run two training processes concurrently on a small machine. |
| Eval intero prints 1.0 after a rollout script run | The episode truncated and reset before the final print; it is the post-reset state, not a bug. |

## 9. Reproducibility

`train.seed` seeds Python/NumPy/torch (`utils/seeding.py`); env layout,
food respawns, sensor noise, and warmup policy all derive from it. Two runs
with the same config and seed produce identical trajectories on the same
hardware/PyTorch build (bitwise GPU determinism is not guaranteed by torch).
Use `env.fixed_layout: true` for a stable arena across eval episodes.
