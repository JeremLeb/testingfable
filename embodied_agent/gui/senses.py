"""Render an agent's *egocentric* senses -- what a single creature perceives
right now -- from its observation dict. Ray-vision is drawn as a first-person
fan (forward = up); the pixel retina is shown directly; touch, smell and the
interoceptive body state are drawn around it.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Wedge, FancyArrow  # noqa: E402

_HIT = {"food": "#2ca02c", "wall": "#888888", "none": "#31527a"}


class SensesRenderer:
    def __init__(self, sensor_cfg, figsize: float = 4.0):
        self.cfg = sensor_cfg
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize), dpi=80)

    def png(self, obs: dict) -> bytes:
        arr = self.render(obs)
        buf = io.BytesIO()
        mpimg.imsave(buf, arr, format="png")
        return buf.getvalue()

    def render(self, obs: dict) -> np.ndarray:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#0e1116")
        if "retina" in obs:                      # pixel vision: show it directly
            img = np.asarray(obs["retina"])
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            ax.imshow(np.clip(img, 0, 1), origin="upper")
            ax.set_title("retina (what its eye sees)", color="#e6edf3",
                         fontsize=10)
        else:
            self._draw_rays(ax, obs)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.fig.tight_layout(pad=0.2)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _draw_rays(self, ax, obs):
        cfg = self.cfg
        n = cfg.n_rays
        vis = np.asarray(obs.get("vision", np.zeros(n * 4))).reshape(n, 4)
        half = np.deg2rad(cfg.fov_deg) / 2
        offsets = np.linspace(-half, half, n)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        # field-of-view wedge (forward = up = 90 deg)
        ax.add_patch(Wedge((0, 0), 1.05, 90 - np.rad2deg(half),
                           90 + np.rad2deg(half), color="#1e242d", zorder=0))
        for off, (dist, food, wall, none) in zip(offsets, vis):
            kind = ("food" if food > wall and food > none
                    else "wall" if wall > none else "none")
            d = float(np.clip(dist, 0.02, 1.0))
            # forward is +y; a positive offset turns to the agent's left
            x, y = -np.sin(off) * d, np.cos(off) * d
            ax.plot([0, x], [0, y], color=_HIT[kind],
                    lw=2.2 if kind != "none" else 1.0,
                    alpha=0.95 if kind != "none" else 0.5, zorder=2)
            if kind != "none":
                ax.plot(x, y, "o", color=_HIT[kind], ms=5, zorder=3)
        # the creature at the centre, facing up
        ax.add_patch(Circle((0, 0), 0.07, color="#4c8dff", zorder=4))
        ax.plot([0, 0], [0, 0.16], color="white", lw=2, zorder=5)
        # smell: an arrow toward the food gradient (forward, lateral)
        smell = np.asarray(obs.get("smell", np.zeros(3)))
        conc, fwd, lat = (float(smell[0]), float(smell[1]), float(smell[2])) \
            if len(smell) >= 3 else (0.0, 0.0, 0.0)
        mag = min(1.0, abs(conc)) * 0.8
        if mag > 0.03:
            ax.add_patch(FancyArrow(0, 0, -lat * mag, fwd * mag, width=0.02,
                                    color="#b980ff", alpha=0.8, zorder=3,
                                    length_includes_head=True))
        # touch: red arcs where the body is being pressed
        touch = np.asarray(obs.get("touch", []))
        for i, t in enumerate(touch):
            if t > 0.05:
                a = 360.0 * i / len(touch)
                ax.add_patch(Wedge((0, 0), 0.16, a - 18, a + 18,
                                   width=0.05, color="#ff5c5c",
                                   alpha=float(min(1.0, t)), zorder=4))
        # interoception: three tiny corner bars (energy / temp / integrity)
        intero = np.asarray(obs.get("intero", [1, 0.5, 1]))
        labels = [("#39d98a", intero[0]), ("#4c8dff", intero[1]),
                  ("#ffab4c", intero[2] if len(intero) > 2 else 1.0)]
        for i, (col, val) in enumerate(labels):
            x0 = -1.1 + i * 0.12
            ax.add_patch(plt.Rectangle((x0, -1.1), 0.08,
                                       0.4 * float(np.clip(val, 0, 1)),
                                       color=col))
        ax.set_title("what it senses (vision fan · smell · touch)",
                     color="#e6edf3", fontsize=9)

    def close(self):
        plt.close(self.fig)
