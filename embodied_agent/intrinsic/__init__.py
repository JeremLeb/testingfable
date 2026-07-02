"""Intrinsic motivation factory. RND and ensemble disagreement land in
milestone 6; until then only "none" is available."""
from __future__ import annotations

from ..config import Config


def build_intrinsic(cfg: Config, feat_dim: int, device: str = "cpu"):
    method = cfg.intrinsic.method
    if method == "none":
        return None
    if method == "rnd":
        from .rnd import RND
        return RND(cfg, feat_dim, device)
    if method == "disagreement":
        from .disagreement import Disagreement
        return Disagreement(cfg, feat_dim, device)
    raise ValueError(f"unknown intrinsic method: {method}")
