from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from config import Config


class IT2Gaussian:
    def __init__(self, c: float, sigma: float, delta: float):
        if sigma - delta <= 0:
            raise ValueError("sigma - delta must be > 0")
        self.c, self.s_up, self.s_lo = c, sigma + delta, sigma - delta

    def upper(self, x):
        return np.exp(-((x - self.c) ** 2) / (2 * self.s_up ** 2))

    def lower(self, x):
        return np.exp(-((x - self.c) ** 2) / (2 * self.s_lo ** 2))


def km_center_of_sets(f_lo: np.ndarray, f_up: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(y, kind="stable")
    y = y[order]; fl = f_lo[:, order]; fu = f_up[:, order]
    N = len(y)
    B = fl.shape[0]

    k = np.arange(N + 1)[:, None]
    left = (np.arange(N)[None, :] < k)

    wl = np.where(left[None], fu[:, None, :], fl[:, None, :])

    wr = np.where(left[None], fl[:, None, :], fu[:, None, :])
    eps = 1e-300
    yl_all = (wl * y).sum(-1) / np.maximum(wl.sum(-1), eps)
    yr_all = (wr * y).sum(-1) / np.maximum(wr.sum(-1), eps)
    return yl_all.min(axis=1), yr_all.max(axis=1)


class IT2FLS:
    def __init__(self, cfg: Config):
        self.mf_snr = [IT2Gaussian(*p) for p in cfg.it2_mf_snr]
        self.mf_unc = [IT2Gaussian(*p) for p in cfg.it2_mf_unc]
        centers = np.asarray(cfg.it2_out_centers, dtype=float)
        rules = np.asarray(cfg.it2_rules, dtype=int)

        self.rule_idx = [(i, j) for i in range(3) for j in range(3)]
        self.y = np.array([centers[rules[i, j]] for i, j in self.rule_idx])

    def firing(self, z1, z2):
        z1 = np.atleast_1d(np.asarray(z1, dtype=float))
        z2 = np.atleast_1d(np.asarray(z2, dtype=float))
        su = np.stack([m.upper(z1) for m in self.mf_snr], -1)
        sl = np.stack([m.lower(z1) for m in self.mf_snr], -1)
        uu = np.stack([m.upper(z2) for m in self.mf_unc], -1)
        ul = np.stack([m.lower(z2) for m in self.mf_unc], -1)
        f_up = np.stack([uu[:, i] * su[:, j] for i, j in self.rule_idx], -1)
        f_lo = np.stack([ul[:, i] * sl[:, j] for i, j in self.rule_idx], -1)
        return f_lo, f_up

    def interval(self, z1, z2):
        f_lo, f_up = self.firing(z1, z2)
        return km_center_of_sets(f_lo, f_up, self.y)

    def confidence(self, z1, z2):
        scalar = np.isscalar(z1) and np.isscalar(z2)
        yl, yr = self.interval(z1, z2)
        c = 0.5 * (yl + yr)
        return float(c[0]) if scalar else c


def fit_uncertainty_scale(e_values: Sequence[float], percentile: float = 99.0) -> float:
    v = float(np.percentile(np.abs(np.asarray(e_values, dtype=float)), percentile))
    return max(v, 1e-6)


def normalize_uncertainty(e, e_scale: float):
    return np.minimum(np.abs(np.asarray(e, dtype=float)) / e_scale, 1.0)
