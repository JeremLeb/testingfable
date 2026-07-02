"""Running scalar normalizer (Welford) for intrinsic-reward whitening."""
from __future__ import annotations


class RunningNorm:
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        # x: 1D iterable of floats (a flat batch of bonuses)
        import numpy as np
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = (m_a + m_b + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    @property
    def std(self):
        return max(self.var ** 0.5, 1e-6)
