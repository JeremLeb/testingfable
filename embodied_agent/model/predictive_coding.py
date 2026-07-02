"""Predictive coding: a local, biologically-plausible learning rule (B7).

The rest of the model learns by backprop through a global loss (BPTT). Cortex
almost certainly does not: the dominant account (Rao & Ballard 1999) is
*predictive coding* -- each layer holds value neurons and *error neurons*; the
error neurons carry the mismatch between a layer's state and the top-down
prediction of it; inference relaxes the value neurons to minimise local
prediction error, and weights change by a purely **local Hebbian** rule
(post-synaptic error x pre-synaptic activity). No global loss, no backward
pass. Whittington & Bogacz (2017) showed this update *approximates* the
backprop gradient at the inference equilibrium -- so it is the project's own
thesis (prediction error IS the substrate) rendered as actual local
computation.

This module implements a predictive-coding MLP with numpy (deliberately no
autograd, to make the point that no backprop is used). It is the
`learning_rule: predictive_coding` alternative to a backprop predictor; the
demo shows it reaches comparable open-loop (next-observation) MSE.
"""
from __future__ import annotations

import numpy as np


def _tanh(x):
    return np.tanh(x)


def _dtanh(x):
    t = np.tanh(x)
    return 1.0 - t * t


class PredictiveCodingNet:
    """A feedforward predictor trained by predictive coding.

    Layers x_0..x_L with weights W_l : x_{l-1} -> x_l and biases b_l. The
    top-down prediction of layer l is mu_l = W_l f(x_{l-1}) + b_l and the local
    error is e_l = x_l - mu_l. Inference relaxes the hidden x_l; learning is the
    local Hebbian rule dW_l ~ e_l (f(x_{l-1}))^T.
    """

    def __init__(self, sizes, seed: int = 0, infer_steps: int = 20,
                 infer_lr: float = 0.1, weight_lr: float = 0.02):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.L = len(sizes) - 1
        # small random init (Xavier-ish)
        self.W = [rng.normal(0, np.sqrt(1.0 / sizes[l]), (sizes[l + 1], sizes[l]))
                  for l in range(self.L)]
        self.b = [np.zeros(sizes[l + 1]) for l in range(self.L)]
        self.infer_steps = infer_steps
        self.infer_lr = infer_lr
        self.weight_lr = weight_lr

    # ------------------------------------------------------------ prediction

    def _feedforward(self, x0):
        """Top-down predictions layer by layer (also the test-time output)."""
        acts = [x0]
        for l in range(self.L):
            pre = acts[l] if l == 0 else _tanh(acts[l])
            acts.append(self.W[l] @ pre + self.b[l])
        return acts

    def predict(self, x0: np.ndarray) -> np.ndarray:
        """Inference with only the input clamped == the feedforward pass."""
        return self._feedforward(np.asarray(x0, dtype=float))[-1]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.predict(x) for x in X])

    # ------------------------------------------------------------ learning

    def learn(self, x0: np.ndarray, target: np.ndarray) -> float:
        """One predictive-coding learning step on a single example.

        Clamp input and output, relax the hidden value neurons to reduce local
        error, then apply the local Hebbian weight update. Returns the output
        prediction error (MSE) *before* the update, for logging."""
        x0 = np.asarray(x0, dtype=float)
        target = np.asarray(target, dtype=float)
        x = self._feedforward(x0)     # initialise nodes at feedforward values
        x[-1] = target                # clamp the output layer to the target

        pre_err = float(np.mean((x[-1] - self._feedforward(x0)[-1]) ** 2))

        # relax hidden nodes x_1..x_{L-1} toward local-error equilibrium
        for _ in range(self.infer_steps):
            mu = self._predictions(x)
            e = [x[l + 1] - mu[l] for l in range(self.L)]  # e[l] is error of layer l+1
            for l in range(1, self.L):                     # hidden layers only
                # top-down pull from own error + bottom-up push through W^T
                grad = -e[l - 1] + _dtanh(x[l]) * (self.W[l].T @ e[l])
                x[l] = x[l] + self.infer_lr * grad

        # local Hebbian weight update: dW_l ~ e_l (f(x_{l-1}))^T  (no backprop)
        mu = self._predictions(x)
        e = [x[l + 1] - mu[l] for l in range(self.L)]
        for l in range(self.L):
            pre = x[l] if l == 0 else _tanh(x[l])
            self.W[l] += self.weight_lr * np.outer(e[l], pre)
            self.b[l] += self.weight_lr * e[l]
        return pre_err

    def _predictions(self, x):
        """mu_l = W_l f(x_{l-1}) + b_l for each layer."""
        mu = []
        for l in range(self.L):
            pre = x[l] if l == 0 else _tanh(x[l])
            mu.append(self.W[l] @ pre + self.b[l])
        return mu

    def learn_epoch(self, X: np.ndarray, Y: np.ndarray) -> float:
        errs = [self.learn(x, y) for x, y in zip(X, Y)]
        return float(np.mean(errs))

    # ------------------------------------------------------------ analysis

    def local_weight_update(self, x0, target):
        """The dW the local rule would apply for one example (for comparing its
        direction to the backprop gradient -- the Whittington-Bogacz result)."""
        x0 = np.asarray(x0, dtype=float)
        x = self._feedforward(x0)
        x[-1] = np.asarray(target, dtype=float)
        for _ in range(self.infer_steps):
            mu = self._predictions(x)
            e = [x[l + 1] - mu[l] for l in range(self.L)]
            for l in range(1, self.L):
                grad = -e[l - 1] + _dtanh(x[l]) * (self.W[l].T @ e[l])
                x[l] = x[l] + self.infer_lr * grad
        mu = self._predictions(x)
        e = [x[l + 1] - mu[l] for l in range(self.L)]
        dW = []
        for l in range(self.L):
            pre = x[l] if l == 0 else _tanh(x[l])
            dW.append(np.outer(e[l], pre))
        return dW
