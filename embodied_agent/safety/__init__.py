"""Hard-constraint shield factory. Full implementation lands in milestone 7."""
from __future__ import annotations

from ..config import Config


def build_shield(cfg: Config, env=None):
    if not cfg.shield.enabled:
        return None
    from .shield import Shield
    return Shield(cfg, env)
