"""A living colony: many embodied creatures sharing one world and one species
brain, foraging, colliding, reproducing into live offspring, and evolving in
place. Built on the single-agent substrate (same senses, homeostasis, world
model) but multi-body.
"""
from .world import ColonyEnv
from .creature import Creature

__all__ = ["ColonyEnv", "Creature"]
