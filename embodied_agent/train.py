"""End-to-end training: collect real experience with the actor, train the
world model on replayed sequences, and train the actor-critic in
imagination. Milestone 5 onward.

  python -m embodied_agent.train --config cpu_small
  python -m embodied_agent.train --config cpu_small --intrinsic none --shield off

Intrinsic motivation (milestone 6) and the safety shield (milestone 7) are
wired in here behind config flags so later milestones only flip switches.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from .agent.actor_critic import ActorCritic
from .agent.agent import DreamerAgent
from .agent.collect import collect_random
from .agent.development import Development
from .agent.metabolism import Metabolism, Sensorimotor
from .agent.replay import ReplayBuffer
from .agent.sleep import SleepController
from .config import load_config
from .env import make_env
from .env.neuromod import Neuromodulators
from .intrinsic import build_intrinsic
from .model.world_model import WorldModel
from .safety import build_shield
from .utils.logger import CSVLogger
from .utils.seeding import seed_everything


def evaluate(agent, cfg, seed, episodes=3, deterministic=True):
    """Run the current policy and return mean episode reward + diagnostics."""
    env = make_env(cfg, seed=seed + 777)
    # B4: sensorimotor latency/noise is part of the world, so evaluation faces
    # it too (this is what makes the robustness measurement meaningful).
    smr = Sensorimotor(cfg.metab, seed=seed) if cfg.metab.enabled else None
    totals, lengths, eaten, coverage = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 777 + ep)
        agent.reset_state()
        if smr is not None:
            smr.reset(obs)
        total, steps, food = 0.0, 0, 0
        visited = set()
        cell = cfg.env.arena_size / 12
        while True:
            percept = smr.perceive(obs) if smr is not None else obs
            action = agent.act(percept, env=env, deterministic=deterministic)
            motor = smr.execute(action) if smr is not None else action
            obs, reward, term, trunc, info = env.step(motor)
            total += reward
            steps += 1
            food += info["food_eaten"]
            visited.add((int(info["pos"][0] / cell), int(info["pos"][1] / cell)))
            if term or trunc:
                break
        totals.append(total)
        lengths.append(steps)
        eaten.append(food)
        coverage.append(len(visited))
    return {
        "eval_reward": float(np.mean(totals)),
        "eval_length": float(np.mean(lengths)),
        "eval_food": float(np.mean(eaten)),
        "eval_coverage": float(np.mean(coverage)),
    }


def random_baseline(cfg, seed, episodes=3):
    env = make_env(cfg, seed=seed + 999)
    rng = np.random.default_rng(seed + 999)
    totals, cover = [], []
    cell = cfg.env.arena_size / 12
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 999 + ep)
        a = np.zeros(2)
        total, visited = 0.0, set()
        while True:
            a = 0.8 * a + 0.2 * rng.uniform(-1, 1, 2)
            obs, reward, term, trunc, info = env.step(a)
            total += reward
            visited.add((int(info["pos"][0] / cell), int(info["pos"][1] / cell)))
            if term or trunc:
                break
        totals.append(total)
        cover.append(len(visited))
    return float(np.mean(totals)), float(np.mean(cover))


def save_eval_gif(agent, cfg, seed, path):
    from .viz.render import ArenaRenderer, save_gif
    env = make_env(cfg, seed=seed + 5)
    obs, _ = env.reset(seed=seed + 5)
    agent.reset_state()
    renderer = ArenaRenderer(env)
    frames = []
    for t in range(cfg.env.max_episode_steps):
        action = agent.act(obs, env=env, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if t % 3 == 0:
            frames.append(renderer.render(info))
        if term or trunc:
            break
    save_gif(frames, str(path))
    renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--intrinsic", default=None,
                    help="override intrinsic method: none|rnd|disagreement")
    ap.add_argument("--shield", default=None, choices=["on", "off"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.out:
        cfg.train.out_dir = args.out
    if args.intrinsic:
        cfg.intrinsic.method = args.intrinsic
    if args.shield:
        cfg.shield.enabled = args.shield == "on"
    return run(cfg)


def run(cfg, verbose: bool = True, on_step=None, should_stop=None) -> dict:
    """Train end-to-end from a Config. Returns a summary dict; also writes
    metrics.csv, an observability plot, rollout GIFs, and a checkpoint.

    on_step(state): optional callback invoked every ``cfg.train.gui_every``
        steps with a live-state dict (step, info, env, metrics, eval, phase) --
        used by the observation GUI. should_stop(): optional predicate; when it
        returns True the run halts early and still finalises cleanly."""
    seed = cfg.train.seed
    seed_everything(seed)
    device = "cuda" if (cfg.train.device == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    out = pathlib.Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "metrics.csv")

    env = make_env(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    wm = WorldModel(cfg, env.sensors.spaces).to(device)
    intrinsic = build_intrinsic(cfg, wm.rssm.feat_dim, device)
    ac = ActorCritic(cfg, wm.rssm, wm, intrinsic=intrinsic)
    ac.actor.to(device)
    ac.critic.to(device)
    ac.target_critic.to(device)
    shield = build_shield(cfg, env)
    agent = DreamerAgent(wm, ac, device=device, shield=shield)

    # B1: neuromodulators gate learning rates (no-ops unless neuromod.enabled)
    neuromod = Neuromodulators(cfg.neuromod)
    base_wm_lr, base_actor_lr = cfg.model.lr, cfg.agent.actor_lr

    # B2: circadian wake/sleep consolidation (None -> baseline interleaved
    # uniform-replay updates, unchanged).
    sleep_ctl = SleepController(cfg.sleep) if cfg.sleep.enabled else None

    # B3: continual life + age-dependent plasticity (None -> episodic resets,
    # constant plasticity). `life` tracks the current individual's age.
    development = Development(cfg.dev) if cfg.dev.enabled else None
    continual = development is not None and cfg.dev.continual
    life = {"age": 0, "births": 0}

    # B4: metabolic cost of cognition + energy-gated planning, and sensorimotor
    # latency/noise (both None/no-op unless metab.enabled).
    metabolism = Metabolism(cfg.metab) if cfg.metab.enabled else None
    smr = Sensorimotor(cfg.metab, seed=seed) if cfg.metab.enabled else None
    body = {"energy": 1.0}  # most recent body energy, gates planning depth

    def gradient_update(prioritized: bool = False, dream: bool = False):
        """One consolidation step: world-model + actor-critic update on a
        replayed sequence, with neuromodulatory LR gating (B1). During sleep,
        `prioritized` draws salient memories and `dream` adds extra imagination
        passes (self-generated behavioural training)."""
        batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len, rng,
                              sleep_cfg=cfg.sleep if prioritized else None)
        post, wm_metrics = wm.train_step(batch)
        # B4: energy-gated planning depth (bounded planning) + metabolic cost.
        H = metabolism.planning_horizon(body["energy"], cfg.agent.horizon) \
            if metabolism else cfg.agent.horizon
        ac_metrics = ac.train_step(post, horizon=H)
        passes = 1
        if dream:
            for _ in range(cfg.sleep.dream_updates):
                ac.train_step(post, horizon=H)
                passes += 1
        if metabolism is not None:
            imagined = H * cfg.train.batch_size * cfg.train.seq_len * passes
            env.homeostasis.spend_energy(metabolism.cognition_cost(imagined))
            logger.add(plan_horizon=H)
        if intrinsic is not None:
            intr_metrics = intrinsic.train_step(
                post.feat().reshape(-1, wm.rssm.feat_dim).detach())
            logger.add(**intr_metrics)
        # B1: surprise (prediction error) -> NE plasticity gain on the
        # world-model LR; |TD error| -> dopamine tone on the actor LR.
        ne_gain = neuromod.update_surprise(wm_metrics["recon"] + wm_metrics["kl"])
        neuromod.update_rpe(ac_metrics["critic_loss"] ** 0.5)
        # B3: age-dependent plasticity (critical period) scales the LR on top of
        # the B1 neuromodulatory gain -- high early in life, annealing to a floor.
        plast = development.plasticity_gain(life["age"]) if development else 1.0
        for g in wm.opt.param_groups:
            g["lr"] = base_wm_lr * ne_gain * plast
        for g in ac.actor_opt.param_groups:
            g["lr"] = base_actor_lr * neuromod.actor_lr_gain() * plast
        logger.add(**wm_metrics, **ac_metrics, **neuromod.metrics())
        if development is not None:
            logger.add(**development.metrics(life["age"]))
        return wm_metrics

    @torch.no_grad()
    def probe_loss(batch) -> float:
        return float(wm.loss(batch)[0].detach())

    night: dict = {}  # holds a fixed probe batch + pre-sleep error per night

    log = print if verbose else (lambda *a, **k: None)
    log(f"device={device} | wm params "
        f"{sum(p.numel() for p in wm.parameters())/1e3:.0f}k | "
        f"intrinsic={cfg.intrinsic.method} | "
        f"shield={'on' if shield else 'off'}")

    # warmup with random policy to seed the buffer and the world model
    log(f"warmup: {cfg.train.warmup_steps} random steps")
    collect_random(env, buffer, cfg.train.warmup_steps, rng)

    base_reward, base_cover = random_baseline(cfg, seed)
    log(f"random baseline: reward={base_reward:.1f} coverage={base_cover:.1f}")

    obs, _ = env.reset(seed=seed)
    agent.reset_state()
    if smr is not None:
        smr.reset(obs)
    buffer.start_episode()
    buffer.add(obs, np.zeros(2), 0.0, 1.0)
    ep_reward, ep_len, interventions = 0.0, 0, 0
    t0 = time.time()
    live = {"metrics": {}, "eval": {}}  # latest values for the observation GUI
    gui_every = cfg.train.gui_every or 10

    for step in range(1, cfg.train.total_steps + 1):
        if should_stop is not None and should_stop():
            log("stop requested"); break
        awake = sleep_ctl is None or sleep_ctl.is_awake(step)
        # hunger arousal: a starving body wakes itself to forage rather than
        # sleeping to death (biological drive overrides the circadian clock).
        if (not awake and cfg.sleep.wake_energy > 0
                and body["energy"] < cfg.sleep.wake_energy):
            awake = True
        # B4: the agent acts on a (possibly delayed) perception, and the world
        # receives a (possibly delayed, noisy) action.
        percept = smr.perceive(obs) if smr is not None else obs
        if awake:
            action = agent.act(percept, env=env, noise=cfg.agent.expl_noise)
            interventions += int(agent.last_intervened)
        else:
            action = np.zeros(2, dtype=np.float32)  # asleep: rest, don't forage
            agent.last_intervened = False
        motor = smr.execute(action) if smr is not None else action
        obs, reward, term, trunc, info = env.step(motor)
        buffer.add(obs, motor, reward, 0.0 if term else 1.0)
        body["energy"] = info["energy"]
        if on_step is not None and step % gui_every == 0:
            on_step({"step": step, "info": info, "env": env, "obs": obs,
                     "metrics": live["metrics"], "eval": live["eval"],
                     "asleep": not awake, "sps": step / (time.time() - t0)})
        ep_reward += reward
        ep_len += 1
        life["age"] += 1
        logger.add(reward=reward, energy=info["energy"], temp=info["temp"],
                   integrity=info["integrity"],
                   drive_energy=info["drive_energy"],
                   drive_thermal=info["drive_thermal"],
                   drive_integrity=info["drive_integrity"])
        for k in ("reward_energy", "reward_thermal", "reward_integrity",
                  "weight_energy", "weight_thermal", "weight_integrity"):
            if k in info:
                logger.add(**{k: info[k]})
        if sleep_ctl is not None:
            logger.add(**sleep_ctl.metrics(step))

        # B3 continual life: truncation only *segments* stored memory -- the
        # same individual keeps living (body, recurrent state, age preserved),
        # so learning is one non-stationary stream. Only death ends a life; a
        # new individual is then born (age reset). Baseline: any term/trunc
        # resets the episode as before.
        if continual and trunc and not term:
            logger.add(ep_reward=ep_reward, ep_len=ep_len,
                       interventions=interventions)
            buffer.end_episode()
            buffer.start_episode()
            buffer.add(obs, np.zeros(2), 0.0, 1.0)
            env.step_count = 0  # restart the truncation clock, keep the body
            ep_reward, ep_len, interventions = 0.0, 0, 0
        elif term or trunc:
            if continual:  # a death: log the completed lifespan, begin anew
                logger.add(lifespan=life["age"])
                life["births"] += 1
            logger.add(ep_reward=ep_reward, ep_len=ep_len,
                       interventions=interventions)
            buffer.end_episode()
            obs, _ = env.reset()
            agent.reset_state()
            if smr is not None:
                smr.reset(obs)
            buffer.start_episode()
            buffer.add(obs, np.zeros(2), 0.0, 1.0)
            ep_reward, ep_len, interventions = 0.0, 0, 0
            life["age"] = 0

        can_train = buffer.can_sample(cfg.train.seq_len)
        if sleep_ctl is None:
            # baseline: interleaved uniform-replay updates every train_every.
            if step % cfg.train.train_every == 0 and can_train:
                for _ in range(cfg.train.updates_per_train):
                    gradient_update()
        elif awake:
            # wake: only light, uniform fast adaptation.
            if step % cfg.train.train_every == 0 and can_train:
                for _ in range(cfg.sleep.wake_updates):
                    gradient_update()
        else:
            # sleep: consolidate the day's experience -- prioritized ("emotional")
            # replay of salient memories, plus dreaming. A fixed probe batch
            # sampled at dusk tracks the per-night consolidation curve.
            if sleep_ctl.just_fell_asleep(step) and can_train:
                night["batch"] = buffer.sample(cfg.train.batch_size,
                                               cfg.train.seq_len, rng)
                night["pre"] = probe_loss(night["batch"])
            if can_train:
                for _ in range(sleep_ctl.updates_per_sleep_step):
                    gradient_update(prioritized=cfg.sleep.prioritized,
                                    dream=cfg.sleep.dream)
                if "batch" in night:
                    cur = probe_loss(night["batch"])
                    logger.add(sleep_probe_loss=cur,
                               sleep_consolidation=night["pre"] - cur)

        if step % cfg.train.log_every == 0:
            row = logger.flush(step)
            live["metrics"] = row
            sps = step / (time.time() - t0)
            log(f"step {step:6d} | ep_reward {row.get('ep_reward', 0):7.1f} "
                f"| wm_loss {row.get('loss', 0):6.2f} "
                f"| imag_ret {row.get('imag_return', 0):6.2f} "
                f"| entropy {row.get('actor_entropy', 0):5.2f} "
                f"| {sps:.0f} steps/s")

        if step % cfg.train.eval_every == 0:
            ev = evaluate(agent, cfg, seed)
            ev["baseline_reward"] = base_reward
            ev["baseline_coverage"] = base_cover
            live["eval"] = ev
            logger.add(**ev)
            log(f"  [eval] reward {ev['eval_reward']:.1f} "
                f"(random {base_reward:.1f}) | food {ev['eval_food']:.1f} "
                f"| coverage {ev['eval_coverage']:.1f} "
                f"(random {base_cover:.1f})")

        if step % cfg.train.gif_every == 0:
            save_eval_gif(agent, cfg, seed, out / f"rollout_{step}.gif")

    torch.save({"wm": wm.state_dict(), "actor": ac.actor.state_dict(),
                "critic": ac.critic.state_dict()}, out / "checkpoint.pt")
    final = evaluate(agent, cfg, seed, episodes=5)
    log(f"\nFINAL eval reward {final['eval_reward']:.1f} "
        f"(random {base_reward:.1f}) | food {final['eval_food']:.1f} "
        f"| coverage {final['eval_coverage']:.1f} (random {base_cover:.1f})")
    logger.close()
    try:
        from .viz.plots import plot_metrics
        plot_metrics(str(out / "metrics.csv"), str(out / "metrics.png"))
    except Exception as e:  # plotting is best-effort observability
        log(f"(plot skipped: {e})")
    log(f"wrote {out}")
    return {
        "final": final, "baseline_reward": base_reward,
        "baseline_coverage": base_cover, "out_dir": str(out),
        "births": life["births"], "offspring": int(env.homeostasis.offspring),
    }


def live_one_life(cfg, verbose: bool = False) -> dict:
    """Run one individual's whole life (B1-B4 lifetime learning included) and
    return its *endogenous* fitness -- what the body achieved, not a designed
    objective. Used as the full-life fitness for the B5 evolutionary loop.

    `cfg.train.total_steps` sets the lifespan budget; the phenotype (setpoints,
    morphology, instinct, plasticity) comes from the genome via to_config."""
    summary = run(cfg, verbose=verbose)
    f = summary["final"]
    return {
        "fitness": f["eval_reward"] + 8.0 * summary["offspring"],
        "reward": f["eval_reward"], "lifespan": f["eval_length"],
        "food": f["eval_food"], "offspring": summary["offspring"],
    }


if __name__ == "__main__":
    main()
