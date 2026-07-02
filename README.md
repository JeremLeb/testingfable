# Embodied Multisensory Learning Agent

A from-scratch (PyTorch) research sandbox for an **embodied agent that learns
almost entirely by predicting its own multisensory stream**. Reward is a thin
layer defined by **homeostasis** (staying alive); exploration is driven by
**curiosity**; behavior is learned **in imagination** with a world model. No
pretrained corpus, no human labels, no hand-coded policy — the world is
self-labeling because the agent's actions change its senses in predictable
ways and the senses cross-predict one another.

Runs on a laptop CPU in the `cpu_small` preset and shows visible learning in
minutes; scales up with `gpu_default`.

## Design philosophy

1. **Prediction error is the substrate.** ~99% of the learning signal is
   self-supervised prediction of the agent's own future sensory input. Reward
   is a thin layer on top, not the main driver.
2. **The loop is closed.** Actions produce sensory consequences the agent must
   predict; learning is continuous and online-capable.
3. **Cross-modal self-supervision.** All modalities fuse into a single latent
   state forced to reconstruct every sense — that is where the free labels
   come from.
4. **Reward = drive reduction (homeostasis).** Internal viability variables
   (energy, temperature, integrity) have setpoints; reward is the weighted
   reduction in distance from them. Pain/pleasure are derived from real
   interoceptive channels, not invented outside.
5. **Fear is learned, not coded.** Interoception (including damage) is just
   another modality the world model predicts, so anticipatory avoidance
   emerges from the model foreseeing incoming damage — nothing hardcodes it.
6. **Learn in imagination.** The policy is trained mostly on rollouts inside
   the world model, because a real body cannot be crashed a million times.
7. **Hard constraints are separate from reward.** A shield/override layer sits
   *above* the policy and reward; no predicted return can buy through it.
8. **Deliberately omitted:** self-preservation-against-shutdown,
   self-replication, and status/reputation maximization. This is a
   perception–action–homeostasis sandbox, not an autonomy experiment.

## Architecture

- **Environment** (`env/`) — a continuous 2D "petri arena" with unicycle
  kinematics, respawning food, a smooth hot/cold temperature field, and
  blocking obstacles. Gymnasium-style `reset`/`step`, no physics-engine
  dependency.
- **Senses** (`env/sensors.py`) — multi-rate, egocentric, correlated:
  - *vision* — egocentric ray-casts returning normalized distance + a
    food/wall/none channel,
  - *touch* — decaying contact pressure per body sector,
  - *proprioception* — speed, heading, and efference copy of the last motor
    command,
  - *interoception* — energy, internal temperature, integrity (also the
    source of reward),
  - *smell* — a slow chemical gradient toward food that refreshes at a lower
    rate than vision, forcing heterogeneous-rate fusion.
- **World model** (`model/`) — an RSSM-lite (Dreamer lineage): per-modality
  encoders fuse into one embedding; a GRU deterministic state `h` plus a
  Gaussian stochastic latent `z`; per-modality decoders and reward/continue
  heads. Loss = reconstruction + balanced KL (free bits) + reward + continue.
- **Actor-critic in imagination** (`agent/`) — a reparameterized
  tanh-Gaussian actor and a value critic (EMA target) trained on imagined
  rollouts of the prior dynamics from real posterior start states, using
  lambda-returns. Actor uses backprop-through-dynamics (justified below).
- **Intrinsic motivation** (`intrinsic/`) — Random Network Distillation
  (default) or ensemble disagreement, as a novelty bonus.
- **Safety shield** (`safety/`) — an analytic override above the policy.
- **Replay buffer** (`agent/replay.py`) of real trajectories for the world
  model.

## Install

```bash
pip install -r requirements.txt   # torch CPU wheel is fine
```

## Run — what to look for

```bash
# M1: watch a random agent wander (renders a GIF)
python -m embodied_agent.scripts.random_rollout --config cpu_small

# M4: train the world model offline on random data; open-loop multi-step
#     prediction MSE should drop sharply (the substrate learns before any policy)
python -m embodied_agent.scripts.train_world_model --config cpu_small

# M4 substrate ablation: prediction-only world model (reward heads off)
python -m embodied_agent.scripts.train_world_model --config cpu_small --no-reward

# M5+: full training. Watch eval reward and coverage overtake the random
#      baseline; food-seeking and wall-avoidance emerge with no coded policy
python -m embodied_agent.train --config cpu_small

# M6: does curiosity improve exploration coverage?
python -m embodied_agent.scripts.ablation --set curiosity --config cpu_small

# M7: the shield is respected (0 forbidden-zone entries) and logged
python -m embodied_agent.scripts.shield_demo --config cpu_small

# evaluate a checkpoint
python -m embodied_agent.evaluate --config cpu_small \
    --checkpoint runs/cpu_small/checkpoint.pt --gif
```

Outputs land in `runs/<name>/`: `metrics.csv`, `metrics.png` (the
observability dashboard: homeostasis, drive decomposition, world-model
losses, KL, actor-critic, eval-vs-random, coverage, intrinsic/shield),
rollout GIFs, and a checkpoint.

### Representative `cpu_small` results (~20k steps, ~4 min CPU)

| metric | learned policy | random baseline |
|---|---|---|
| eval reward | ≈ −48 | ≈ −79 |
| food eaten / episode | ≈ 5 | ≈ 1 |
| exploration coverage | ≈ 40 cells | ≈ 11 cells |
| world-model open-loop MSE | ≈ 0.037 (from 0.42) | — |

(Exact numbers vary by seed; the *direction* — policy beats random on reward,
food, and coverage; open-loop prediction error falls — is the point.)

## Arbitration (which drive is winning)

Reward is an explicit weighted sum of energy, thermal, and integrity drive
reduction plus intrinsic curiosity. The weights live in `RewardConfig` and
every drive's instantaneous contribution is logged, so "which drive is
winning right now" is answerable from `metrics.csv` / `metrics.png`. Biology
solves this arbitration with neuromodulation; **hand-set weights are the known
weak point here.**

## Why backprop-through-dynamics for the actor

The learned RSSM is fully differentiable and the tanh-Gaussian action is
reparameterized, so analytic return gradients flow straight through the
imagined trajectory into the actor. In this low-dimensional, short-horizon
sandbox that is markedly more sample-efficient than a REINFORCE-style
estimator; the cost is reliance on world-model gradient quality, which is
acceptable because the substrate is validated first (milestone 4). Returns
are divided by an EMA of their scale so the entropy coefficient is
independent of the (large, configurable) reward scale.

## Curiosity ablation (cpu_small, 8k steps, 2 seeds)

| condition | coverage | food | eval reward |
|---|---|---|---|
| no curiosity | 13.5 | 0.9 | −81.7 |
| RND | 15.8 | 1.7 | −105.7 |
| disagreement | 15.5 | 1.2 | −93.7 |

Removing curiosity reduces exploration coverage and food discovery; the
bonus costs some extrinsic reward (explore/exploit trade). Be honest about
the error bars: at this tiny scale the effect is modest and seed variance is
large — the intrinsic `scale` must be sized comparable to the typical
per-step extrinsic reward or the bonus is swamped entirely (that constant
lives in `IntrinsicConfig` with a comment). Longer runs and more seeds
sharpen the separation.

## Known hard parts & honest limitations

- **Sample efficiency in a non-resettable body.** A real body cannot be
  crashed a million times; imagination helps but the world model must be good
  first, and early exploration is fragile.
- **Multi-rate sensor fusion.** Smell updates slower than vision on purpose;
  the RSSM has to integrate heterogeneous rates, and mis-set rates hurt.
- **Drive arbitration.** Fixed reward weights are brittle — a slightly
  different weighting changes whether the agent prioritizes food, warmth, or
  safety. Neuromodulatory, context-dependent weighting is the right answer and
  is not implemented.
- **Reward scale sensitivity.** Drives move slowly, so rewards are rescaled;
  the critic target magnitude matters and is mitigated (not solved) by return
  normalization.
- **Not SOTA, not distributed.** Single machine, small nets, correctness and
  observability over performance.
- **Emergent avoidance is qualitative.** Anticipatory braking before walls and
  drift toward the thermal comfort zone appear, but this sandbox measures them
  through logs/behavior rather than a formal causal test.

## Future work

- **Learning-progress curiosity** (Oudeyer-style): reward the *rate of
  decrease* of prediction error rather than raw novelty/error. It would slot
  into `intrinsic/` behind the same `reward(feat)` / `train_step(feat)`
  interface as RND and disagreement.
- **Forward-dynamics ensemble disagreement**: condition each ensemble member
  on the action and predict the next latent; current disagreement uses a
  shared random target for simplicity.
- **Hierarchical prediction**: a slow latent predicting long-horizon
  structure above the fast latent, with errors propagating up only when the
  lower level fails to explain them.

## Layout

```
embodied_agent/
  env/         arena, sensors, homeostasis, geometry
  model/       encoders/decoders, rssm, world model + heads
  agent/       actor-critic, imagination, replay, online agent
  intrinsic/   RND, ensemble disagreement
  safety/      hard-constraint shield
  configs/     cpu_small.yaml, gpu_default.yaml
  viz/         arena renderer, metrics plots
  scripts/     random_rollout, train_world_model, ablation, shield_demo
  tests/       env, sensors, replay, model, shield, smoke
  train.py     end-to-end training (run(cfg))
  evaluate.py  load a checkpoint and report vs random
```

## Reproducibility

Everything is seeded (`utils/seeding.py`); `env.fixed_layout` gives a
deterministic arena for eval. Metrics are written to CSV and rendered to PNG.
