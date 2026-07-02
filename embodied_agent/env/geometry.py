"""Analytic 2D geometry: circle-rect collision resolution and ray casting."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def resolve_circle_rect(pos: np.ndarray, radius: float, rect) -> tuple:
    """Push a circle out of an axis-aligned rect (x, y, w, h).

    Returns (new_pos, normal) where normal is the unit push-out direction,
    or (pos, None) if there is no overlap.
    """
    x, y, w, h = rect
    closest = np.array([np.clip(pos[0], x, x + w), np.clip(pos[1], y, y + h)])
    delta = pos - closest
    dist = np.linalg.norm(delta)
    if dist >= radius:
        return pos, None
    if dist > EPS:
        normal = delta / dist
        return closest + normal * radius, normal
    # center is inside the rect: push out through the nearest face
    gaps = np.array([pos[0] - x, x + w - pos[0], pos[1] - y, y + h - pos[1]])
    face = int(np.argmin(gaps))
    normal = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])[face]
    targets = [x - radius, x + w + radius, y - radius, y + h + radius]
    new_pos = pos.copy()
    new_pos[face // 2] = targets[face]
    return new_pos, normal


def ray_rect(origin: np.ndarray, direction: np.ndarray, rect) -> float:
    """Slab-method ray vs axis-aligned rect. Returns hit distance or inf."""
    x, y, w, h = rect
    lo, hi = np.array([x, y]), np.array([x + w, y + h])
    inv = 1.0 / np.where(np.abs(direction) < EPS, EPS, direction)
    t1, t2 = (lo - origin) * inv, (hi - origin) * inv
    tmin = np.max(np.minimum(t1, t2))
    tmax = np.min(np.maximum(t1, t2))
    if tmax < max(tmin, 0.0):
        return np.inf
    return tmin if tmin > 0.0 else (tmax if tmax > 0.0 else np.inf)


def ray_circle(origin: np.ndarray, direction: np.ndarray,
               center: np.ndarray, radius: float) -> float:
    """Ray vs circle. Returns hit distance or inf."""
    oc = origin - center
    b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return np.inf
    sq = np.sqrt(disc)
    t = -b - sq
    if t > 0.0:
        return t
    t = -b + sq
    return t if t > 0.0 else np.inf


def ray_bounds(origin: np.ndarray, direction: np.ndarray, size: float) -> float:
    """Ray vs the inside of the [0, size]^2 arena walls."""
    best = np.inf
    for axis in (0, 1):
        if abs(direction[axis]) < EPS:
            continue
        for wall in (0.0, size):
            t = (wall - origin[axis]) / direction[axis]
            if t > 0.0:
                other = origin[1 - axis] + t * direction[1 - axis]
                if -EPS <= other <= size + EPS:
                    best = min(best, t)
    return best
