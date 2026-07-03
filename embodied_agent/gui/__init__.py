"""Live observation GUI: watch the agent live its life and learn, in a browser.

A tiny, dependency-free (Python stdlib only) local web app. `python -m
embodied_agent.gui` starts training in a background thread and serves a
dashboard at http://localhost:8000 -- the arena with the agent moving, its body
state (energy / temperature / integrity), live learning curves, and a
plain-language readout of what each biological mechanism is doing.
"""
from .server import serve

__all__ = ["serve"]
