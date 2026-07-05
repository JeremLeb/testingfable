"""Run a living colony where **each creature is its own individual**: its own
senses, its own body, its own genome, AND its own mind -- a private world model
+ actor + memory that learns only from its own life. Newborns inherit a copy of
their parent's brain (nature), then diverge through their own experience
(nurture); when a creature dies, its mind dies with it.

(Set ``colony.shared_brain: true`` for the alternative pooled "species brain".)
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..agent.actor_critic import ActorCritic
from ..agent.agent import DreamerAgent
from ..agent.replay import Episode, ReplayBuffer
from ..config import load_config
from ..env.sensors import SensorSuite
from ..intrinsic import build_intrinsic
from ..model.world_model import WorldModel
from ..utils.seeding import seed_everything
from .batched import BatchedActing
from .batched_train import BatchedWMTrainer
from .world import ColonyEnv


class Mind:
    """One creature's private brain: world model + actor-critic + memory."""

    def __init__(self, cfg, spaces, device):
        self.wm = WorldModel(cfg, spaces).to(device)
        self.intrinsic = build_intrinsic(cfg, self.wm.rssm.feat_dim, device)
        self.ac = ActorCritic(cfg, self.wm.rssm, self.wm,
                              intrinsic=self.intrinsic)
        self.ac.actor.to(device)
        self.ac.critic.to(device)
        self.ac.target_critic.to(device)
        self.buffer = ReplayBuffer(cfg.train.buffer_capacity, device=device)

    def inherit_from(self, other: "Mind"):
        """Nature: be born knowing what a parent knew (a copy of its weights)."""
        self.wm.load_state_dict(other.wm.state_dict())
        self.ac.actor.load_state_dict(other.ac.actor.state_dict())
        self.ac.critic.load_state_dict(other.ac.critic.state_dict())
        self.ac.target_critic.load_state_dict(other.ac.target_critic.state_dict())


def run_colony(cfg, verbose: bool = True, on_step=None, should_stop=None,
               resume=None) -> dict:
    seed = cfg.train.seed
    seed_everything(seed)
    device = "cuda" if (cfg.train.device == "cuda"
                        and torch.cuda.is_available()) else "cpu"
    log = print if verbose else (lambda *a, **k: None)
    shared = cfg.colony.shared_brain

    spaces = SensorSuite(cfg.sensor, cfg.env,
                         np.random.default_rng(seed)).spaces
    seg = max(2 * cfg.train.seq_len, 96)
    rng = np.random.default_rng(seed)
    # batched acting: run every individual mind's forward pass as one vmapped
    # GPU op (same separate brains, computed in parallel). Individual minds only.
    batched = cfg.colony.batched and not shared
    batcher = BatchedActing(cfg, device) if batched else None
    # batched training: update every mind's world model as one vmapped+grad GPU
    # pass (the dominant training cost). Actor-critic stays on the rotating budget.
    batched_train = cfg.colony.batched_train and not shared
    wm_trainer = BatchedWMTrainer(cfg, device) if batched_train else None
    deter = cfg.model.deter_dim
    obs_keys = sorted(spaces)

    def attach(c, new_mind=True):
        if new_mind:
            if shared:
                c.mind = shared_mind
            else:
                c.mind = Mind(cfg, spaces, device)
                if cfg.colony.brain_inherit and c.parent is not None \
                        and c.parent.mind is not None:
                    c.mind.inherit_from(c.parent.mind)
        c.episode = Episode()
        c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)
        if batched:                      # each creature keeps its own state
            if getattr(c, "h", None) is None:   # keep any restored state
                stoch = c.mind.wm.rssm.stoch_flat
                c.h = torch.zeros(deter, device=device)
                c.z = torch.zeros(stoch, device=device)
                c.prev_a = torch.zeros(2, device=device)
        else:
            c.agent = DreamerAgent(c.mind.wm, c.mind.ac, device=device)

    # resume an evolved colony from a snapshot, or start a fresh one
    if resume is not None:
        from .persistence import load_colony
        env, device = load_colony(resume, device=device)
        shared_mind = env.living[0].mind if (shared and env.living) else None
        for c in env.living:
            attach(c, new_mind=False)
        start_step = env.steps + 1
        log(f"resumed colony from {resume} at step {env.steps} | pop "
            f"{len(env.living)} | max gen {env.stats()['max_generation']}")
    else:
        shared_mind = Mind(cfg, spaces, device) if shared else None
        env = ColonyEnv(cfg, seed=seed)
        for c in env.reset():
            attach(c)
        start_step = 1

    # where to auto-save (so a long evolved run survives a Stop / crash)
    import pathlib
    save_path = pathlib.Path(cfg.train.out_dir) / "colony.pt"
    save_every = max(int(cfg.train.save_every), 0)

    def save(reason=""):
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            from .persistence import save_colony
            save_colony(env, save_path)
            log(f"saved colony -> {save_path}{reason}")
        except Exception as e:                       # never crash a run on save
            log(f"warning: colony save failed: {e}")

    gui_every = cfg.train.gui_every or 8
    live = {"metrics": {}}
    t0 = time.time()
    ptr = 0       # round-robin pointer for actor-critic updates
    wm_ptr = 0    # round-robin pointer for the batched world-model passes
    n_params = sum(p.numel() for p in env.living[0].mind.wm.parameters())
    log(f"colony | device={device} | {'shared' if shared else 'individual'} "
        f"minds | brain {n_params/1e3:.0f}k params each | start pop "
        f"{len(env.living)}")

    for step in range(start_step, cfg.train.total_steps + 1):
        if should_stop is not None and should_stop():
            log("stop requested")
            if save_every:
                save(" (on stop)")
            break

        acting = env.living
        if batched and acting:
            # one vmapped forward for the whole population (each its own brain)
            obs_stack = {k: torch.as_tensor(
                np.stack([c.obs[k] for c in acting]), device=device).float()
                for k in obs_keys}
            h = torch.stack([c.h for c in acting])
            z = torch.stack([c.z for c in acting])
            pa = torch.stack([c.prev_a for c in acting])
            bias = torch.as_tensor(
                np.stack([c.action_bias for c in acting]), device=device).float()
            act, h_new, z_new = batcher.act(
                [c.mind for c in acting], obs_stack, h, z, pa, bias,
                cfg.agent.expl_noise)
            act_np = act.cpu().numpy()          # one GPU->CPU sync for all N
            actions = {c.id: act_np[i] for i, c in enumerate(acting)}
            for i, c in enumerate(acting):
                c.h, c.z, c.prev_a = h_new[i], z_new[i], act[i]
        else:
            actions = {c.id: c.agent.act(c.obs, bias=c.action_bias,
                                         noise=cfg.agent.expl_noise)
                       for c in acting}
        results, births, deaths = env.step(actions)

        for c in acting:
            r = results.get(c.id)
            if r is None or c.episode is None:
                continue
            obs2, reward, done, _ = r
            c.episode.add(obs2, actions[c.id], reward, 0.0 if done else 1.0)
            if not done and len(c.episode) >= seg:
                c.mind.buffer.ingest(c.episode)
                c.episode = Episode()
                c.episode.add(c.obs, np.zeros(2), 0.0, 1.0)
        for d in deaths:
            # a shared brain keeps the memory; a private mind dies with the body
            if shared and d.episode is not None:
                shared_mind.buffer.ingest(d.episode)
            if batcher is not None and d.mind is not None:
                batcher.forget(d.mind)
            if wm_trainer is not None and d.mind is not None:
                wm_trainer.forget(d.mind)
            d.agent = d.episode = d.mind = None
        for b in births:
            attach(b)

        # train minds (bounds real-time cost); only on every train_every-th step
        # so heavy training doesn't dominate the frame rate.
        living = env.living
        do_train = step % max(cfg.colony.train_every, 1) == 0 and living
        seq, bs = cfg.train.seq_len, cfg.train.batch_size

        if do_train and wm_trainer is not None:
            # (1) batched WORLD-MODEL update: many minds in one GPU pass. Capped
            # per pass and rotated, so the per-step cost stays bounded as the
            # population grows (CPU replay sampling + param stacking scale with N).
            cap = cfg.colony.max_wm_batch or len(living)
            start = wm_ptr % max(len(living), 1)
            ordered = living[start:] + living[:start]
            ready, batches, seen_mid = [], [], set()
            for c in ordered:
                if len(ready) >= cap:
                    break
                mid = id(c.mind)
                if mid in seen_mid or not c.mind.buffer.can_sample(seq):
                    continue
                seen_mid.add(mid)
                ready.append(c.mind)
                batches.append(c.mind.buffer.sample(bs, seq, rng))
            wm_ptr += max(len(ready), 1)
            if ready:
                live["metrics"] = wm_trainer.update(ready, batches)
            # (2) actor-critic + intrinsic: rotating budget, per mind (lighter).
            trained, seen, unique = 0, 0, set()
            while trained < cfg.colony.max_trains_per_step and seen < len(living):
                c = living[ptr % len(living)]
                ptr += 1
                seen += 1
                mid = id(c.mind)
                if mid in unique or not c.mind.buffer.can_sample(seq):
                    continue
                unique.add(mid)
                batch = c.mind.buffer.sample(bs, seq, rng)
                with torch.no_grad():   # posterior = imagination starts (encoder +
                    wm = c.mind.wm      # RSSM only; skip the decoder/heads we'd drop)
                    post, _ = wm.rssm.observe(wm.encoder(batch["obs"]),
                                              batch["prev_action"])
                ac_m = c.mind.ac.train_step(post)
                if c.mind.intrinsic is not None:
                    c.mind.intrinsic.train_step(
                        post.feat().reshape(-1, c.mind.wm.rssm.feat_dim).detach())
                live["metrics"] = {**live["metrics"], **ac_m}
                trained += 1
        elif do_train:
            trained, seen, unique = 0, 0, set()
            while trained < cfg.colony.max_trains_per_step \
                    and seen < len(living):
                c = living[ptr % len(living)]
                ptr += 1
                seen += 1
                mid = id(c.mind)
                if mid in unique:
                    continue
                unique.add(mid)
                if c.mind.buffer.can_sample(seq):
                    batch = c.mind.buffer.sample(bs, seq, rng)
                    post, wm_m = c.mind.wm.train_step(batch)
                    ac_m = c.mind.ac.train_step(post)
                    if c.mind.intrinsic is not None:
                        c.mind.intrinsic.train_step(
                            post.feat().reshape(
                                -1, c.mind.wm.rssm.feat_dim).detach())
                    live["metrics"] = {**wm_m, **ac_m}
                    trained += 1

        if step % max(cfg.train.log_every, 1) == 0:
            s = env.stats()
            sps = step / (time.time() - t0)
            log(f"step {step:6d} | pop {s['population']:3d} | births "
                f"{s['births']:4d} deaths {s['deaths']:4d} | maxgen "
                f"{s['max_generation']:2d} | mean E {s['mean_energy']:.2f} | "
                f"wm {live['metrics'].get('loss', 0):.2f} | {sps:.0f} it/s")

        if on_step is not None and step % gui_every == 0:
            on_step({"step": step, "env": env, "stats": env.stats(),
                     "metrics": live["metrics"], "sps": step / (time.time() - t0)})

        if save_every and step % save_every == 0:
            save()

    stats = env.stats()
    if save_every:
        save(" (final)")
    log(f"\nFINAL | pop {stats['population']} | total births {stats['births']} "
        f"| deaths {stats['deaths']} | max generation {stats['max_generation']}")
    return {"stats": stats, "shared_brain": shared, "save_path": str(save_path)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="colony")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shared-brain", action="store_true",
                    help="one pooled brain instead of individual minds")
    ap.add_argument("--resume", default=None,
                    help="resume an evolved colony from a saved snapshot (.pt)")
    ap.add_argument("--save-every", type=int, default=None,
                    help="auto-save the colony every N steps (0 = off)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.shared_brain:
        cfg.colony.shared_brain = True
    if args.save_every is not None:
        cfg.train.save_every = args.save_every
    elif cfg.train.save_every == 0:
        cfg.train.save_every = 2000        # sensible default from the CLI
    return run_colony(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
