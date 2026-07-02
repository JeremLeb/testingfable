"""Evolutionary outer loop & reproduction (B5).

Three nested loops of adaptation: evolution sets innate priors over
generations, development wires them (B3), lifetime learning (B1-B4) tunes them.
What is reliably learned can become innate (the Baldwin effect).
"""
from .genome import Genome, GENE_BOUNDS
from .population import Population, innate_fitness

__all__ = ["Genome", "GENE_BOUNDS", "Population", "innate_fitness"]
