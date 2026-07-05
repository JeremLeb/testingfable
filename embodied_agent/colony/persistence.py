"""Save and restore a living colony -- the whole evolved run, not just weights.

A snapshot captures everything needed to resume exactly where you stopped:
  * the config and the world (food positions/timers; walls and the temperature
    field are seed-deterministic, so they are rebuilt from the stored seed);
  * every living creature's body (position, heading, velocity, age), its
    interoception (energy / temperature / integrity / reproduction), its genome
    (the heritable innate traits), and its lineage (id, founder bloodline,
    parent, generation, birth step);
  * every creature's OWN mind -- the world model + actor + critic weights -- and
    its recurrent state, so a resumed creature keeps thinking mid-thought.

Optimizer moments are not stored (Adam re-warms in a few dozen steps) and dead
ancestors are not persisted (the living family tree, grouped by founder, is).
"""
from __future__ import annotations

import numpy as np
import torch

from ..env.sensors import SensorSuite
from ..evolution.genome import Genome
from .creature import Creature
from .world import ColonyEnv

FORMAT = 1


def _mind_state(mind) -> dict:
    return {
        "wm": {k: v.cpu() for k, v in mind.wm.state_dict().items()},
        "actor": {k: v.cpu() for k, v in mind.ac.actor.state_dict().items()},
        "critic": {k: v.cpu() for k, v in mind.ac.critic.state_dict().items()},
        "target_critic": {k: v.cpu()
                          for k, v in mind.ac.target_critic.state_dict().items()},
    }


def _load_mind_state(mind, st):
    mind.wm.load_state_dict(st["wm"])
    mind.ac.actor.load_state_dict(st["actor"])
    mind.ac.critic.load_state_dict(st["critic"])
    mind.ac.target_critic.load_state_dict(st["target_critic"])


def _recur(c):
    if hasattr(c, "h") and c.h is not None:
        return {"h": c.h.detach().cpu(), "z": c.z.detach().cpu(),
                "prev_a": c.prev_a.detach().cpu()}
    return None


def _creature_snap(c, shared: bool) -> dict:
    h = c.homeostasis
    return {
        "id": c.id, "generation": c.generation, "founder": c.founder,
        "birth_step": c.birth_step,
        "parent_id": c.parent.id if c.parent is not None else None,
        "genes": dict(c.genome.genes),
        "pos": np.asarray(c.pos).tolist(), "heading": float(c.heading),
        "v": float(c.v), "last_action": np.asarray(c.last_action).tolist(),
        "age": int(c.age), "repro_readiness": float(c.repro_readiness),
        "impact_speed": float(c.impact_speed),
        "homeo": {"energy": h.energy, "temp": h.temp, "integrity": h.integrity,
                  "repro_readiness": h.repro_readiness, "offspring": h.offspring,
                  "dead": h.dead},
        "mind": None if shared else _mind_state(c.mind),
        "recur": _recur(c),
    }


def save_colony(env: ColonyEnv, path) -> None:
    """Write a full snapshot of the colony to `path`."""
    shared = env.full_cfg.colony.shared_brain
    living = env.living
    snap = {
        "format": FORMAT,
        "cfg": env.full_cfg,
        "seed": env._seed,
        "steps": env.steps, "births": env.births, "deaths": env.deaths,
        "next_id": Creature._next_id,
        "rng": env.rng.bit_generator.state,
        "foods": [{"pos": np.asarray(f.pos).tolist(), "active": bool(f.active),
                   "timer": int(f.timer)} for f in env.foods],
        "shared_brain": shared,
        "shared_mind": _mind_state(living[0].mind) if (shared and living)
        else None,
        "creatures": [_creature_snap(c, shared) for c in living],
    }
    tmp = str(path) + ".tmp"
    torch.save(snap, tmp)
    import os
    os.replace(tmp, path)          # atomic: never leave a half-written save


def load_colony(path, device: str | None = None):
    """Rebuild a colony from a snapshot. Returns (env, device) with every
    creature restored, its mind loaded, and its recurrent state stashed on the
    creature (as c.h/c.z/c.prev_a) for the batched runner to pick up."""
    from .run import Mind
    snap = torch.load(path, map_location="cpu", weights_only=False)
    cfg = snap["cfg"]
    device = device or ("cuda" if (cfg.train.device == "cuda"
                                   and torch.cuda.is_available()) else "cpu")
    shared = snap.get("shared_brain", False)

    env = ColonyEnv(cfg, seed=snap["seed"])
    env.reset()                                    # deterministic walls + field
    env.creatures = []
    env.steps, env.births, env.deaths = snap["steps"], snap["births"], snap["deaths"]
    env.rng.bit_generator.state = snap["rng"]
    for f, fs in zip(env.foods, snap["foods"]):
        f.pos = np.asarray(fs["pos"], dtype=np.float64)
        f.active = fs["active"]
        f.timer = fs["timer"]

    spaces = SensorSuite(cfg.sensor, cfg.env,
                         np.random.default_rng(snap["seed"])).spaces
    shared_mind = None
    if shared:
        shared_mind = Mind(cfg, spaces, device)
        if snap["shared_mind"] is not None:
            _load_mind_state(shared_mind, snap["shared_mind"])

    id2c = {}
    for cs in snap["creatures"]:
        c = Creature(env, Genome(dict(cs["genes"])), cs["pos"], cs["heading"],
                     seed=int(env.rng.integers(1 << 31)),
                     generation=cs["generation"])
        c.id, c.founder, c.birth_step = cs["id"], cs["founder"], cs["birth_step"]
        c.v = cs["v"]
        c.last_action = np.asarray(cs["last_action"], dtype=np.float64)
        c.age, c.repro_readiness = cs["age"], cs["repro_readiness"]
        c.impact_speed = cs["impact_speed"]
        h, hs = c.homeostasis, cs["homeo"]
        h.energy, h.temp, h.integrity = hs["energy"], hs["temp"], hs["integrity"]
        h.repro_readiness, h.offspring, h.dead = (hs["repro_readiness"],
                                                  hs["offspring"], hs["dead"])
        mind = shared_mind if shared else Mind(cfg, spaces, device)
        if not shared and cs["mind"] is not None:
            _load_mind_state(mind, cs["mind"])
        c.mind = mind
        r = cs["recur"]
        if r is not None:
            c.h = r["h"].to(device)
            c.z = r["z"].to(device)
            c.prev_a = r["prev_a"].to(device)
        c.obs = c.observe()
        env.creatures.append(c)
        id2c[c.id] = c

    for cs in snap["creatures"]:              # relink living parent pointers
        if cs["parent_id"] is not None:
            id2c[cs["id"]].parent = id2c.get(cs["parent_id"])
    Creature._next_id = snap["next_id"]
    return env, device
