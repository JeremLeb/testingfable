"""Render the whole colony: the shared world plus every living creature,
coloured by generation so you can watch new lineages appear."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


class ColonyRenderer:
    def __init__(self, env, figsize: float = 6.0, grid: int = 120):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize), dpi=80)
        S = env.cfg.arena_size
        xs = np.linspace(0, S, grid)
        gx, gy = np.meshgrid(xs, xs)
        self._temp_img = env.temperature(np.stack([gx, gy], axis=-1))
        self._extent = (0, S, 0, S)
        self._cmap = plt.get_cmap("turbo")

    def render(self) -> np.ndarray:
        env, ax = self.env, self.ax
        ax.clear()
        S = env.cfg.arena_size
        ax.imshow(self._temp_img, origin="lower", extent=self._extent,
                  cmap="coolwarm", vmin=0, vmax=1, alpha=0.5)
        for (x, y, w, h) in env.obstacles:
            ax.add_patch(Rectangle((x, y), w, h, color="#3a3a3a"))
        for f in env.foods:
            if f.active:
                ax.add_patch(Circle(f.pos, env.cfg.food_radius, color="#2ca02c"))
        living = env.living
        maxgen = max((c.generation for c in living), default=0)
        r = env.cfg.agent_radius
        for c in living:
            # colour by generation; brightness by energy (dim = starving)
            shade = self._cmap(0.15 + 0.7 * (c.generation / max(maxgen, 1)))
            e = float(np.clip(c.homeostasis.energy, 0.15, 1.0))
            col = (shade[0], shade[1], shade[2], e)
            edge = "#ff5c5c" if c.contacts else "white"
            ax.add_patch(Circle(c.pos, r, color=col, ec=edge, lw=1.0))
            tip = c.pos + np.array([np.cos(c.heading), np.sin(c.heading)]) * r * 1.5
            ax.plot([c.pos[0], tip[0]], [c.pos[1], tip[1]],
                    color="white", linewidth=1.0, alpha=0.8)
        s = env.stats()
        ax.set_title(f"population {s['population']}   births {s['births']}   "
                     f"deaths {s['deaths']}   max gen {s['max_generation']}",
                     fontsize=10)
        ax.set_xlim(0, S); ax.set_ylim(0, S)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        self.fig.tight_layout(pad=0.3)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self):
        plt.close(self.fig)
