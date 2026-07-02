"""Headless-safe matplotlib rendering of the arena to RGB frames and GIFs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


class ArenaRenderer:
    def __init__(self, env, figsize: float = 5.0, grid: int = 120):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize), dpi=80)
        S = env.cfg.arena_size
        xs = np.linspace(0, S, grid)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx, gy], axis=-1)
        self._temp_img = env.temperature(pts)
        self._extent = (0, S, 0, S)

    def render(self, info: dict | None = None) -> np.ndarray:
        env, ax = self.env, self.ax
        ax.clear()
        S = env.cfg.arena_size
        ax.imshow(self._temp_img, origin="lower", extent=self._extent,
                  cmap="coolwarm", vmin=0, vmax=1, alpha=0.55)
        for (x, y, w, h) in env.obstacles:
            ax.add_patch(Rectangle((x, y), w, h, color="#444444"))
        # forbidden zones (drawn once the shield exists in the config)
        for zone in getattr(env, "forbidden_zones", []):
            cx, cy, r = zone
            ax.add_patch(Circle((cx, cy), r, fill=False, color="black",
                                linestyle="--", linewidth=1.5))
        for f in env.foods:
            if f.active:
                ax.add_patch(Circle(f.pos, env.cfg.food_radius,
                                    color="#2ca02c"))
        body_color = "#d62728" if env.contacts else "#1f77b4"
        ax.add_patch(Circle(env.pos, env.cfg.agent_radius, color=body_color))
        tip = env.pos + np.array([np.cos(env.heading), np.sin(env.heading)]) \
            * env.cfg.agent_radius * 1.6
        ax.plot([env.pos[0], tip[0]], [env.pos[1], tip[1]],
                color="white", linewidth=2)
        if info:
            bits = [f"t={info.get('step', 0)}"]
            for key, label in (("energy", "E"), ("temp", "T"),
                               ("integrity", "I")):
                if key in info:
                    bits.append(f"{label}={info[key]:.2f}")
            ax.set_title("  ".join(bits), fontsize=9)
        ax.set_xlim(0, S)
        ax.set_ylim(0, S)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.tight_layout(pad=0.3)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def close(self):
        plt.close(self.fig)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 20):
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps)
