"""Render the colony's family tree: who descended from whom. Generations run
top (founders) to bottom (newest); living creatures are bright, dead ancestors
faint; colour tracks generation. Lines link parent to child.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


class TreeRenderer:
    def __init__(self, figsize: float = 6.0):
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize * 0.6), dpi=80)
        self._cmap = plt.get_cmap("turbo")

    def render(self, env, focus_id=None) -> np.ndarray:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#0e1116")
        nodes = env.tree_creatures()
        if not nodes:
            ax.text(0.5, 0.5, "no lineage yet", color="#9aa7b4",
                    ha="center", va="center")
            return self._finish()
        maxgen = max(c.generation for c in nodes) or 1
        living = {c.id for c in env.living}
        # x position: spread each generation's nodes evenly across [0, 1]
        by_gen: dict[int, list] = {}
        for c in nodes:
            by_gen.setdefault(c.generation, []).append(c)
        pos = {}
        for g, cs in by_gen.items():
            cs.sort(key=lambda c: c.id)
            for i, c in enumerate(cs):
                x = (i + 1) / (len(cs) + 1)
                pos[c.id] = (x, -g)

        for c in nodes:                       # descent lines
            if c.parent is not None and c.parent.id in pos:
                x0, y0 = pos[c.parent.id]
                x1, y1 = pos[c.id]
                ax.plot([x0, x1], [y0, y1], color="#39425180", lw=0.8, zorder=1)
        for c in nodes:                       # nodes
            x, y = pos[c.id]
            col = self._cmap(0.15 + 0.7 * (c.generation / maxgen))
            if c.id in living:
                ax.scatter([x], [y], s=90, color=col, edgecolors="white",
                           linewidths=1.0, zorder=3)
                if focus_id is not None and c.id == focus_id:
                    ax.scatter([x], [y], s=230, facecolors="none",
                               edgecolors="#ffd24c", linewidths=2.0, zorder=4)
            else:
                ax.scatter([x], [y], s=22, color=col, alpha=0.35, zorder=2)

        ax.set_title(f"family tree  ·  {len(living)} alive  ·  "
                     f"{len(nodes)} in the tree  ·  {maxgen + 1} generations",
                     color="#e6edf3", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-maxgen - 0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks(range(-maxgen, 1))
        ax.set_yticklabels([f"gen {g + 1}" for g in range(maxgen, -1, -1)],
                           color="#9aa7b4", fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return self._finish()

    def _finish(self):
        self.fig.tight_layout(pad=0.3)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)
