from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.special import erfc, log_ndtr
from scipy.stats import gamma as gamma_dist, norm

from config import Config

DB_FLOOR = -300.0


def qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / math.sqrt(2.0))


def ber_mpam(snr_lin, M: int):
    snr_lin = np.maximum(np.asarray(snr_lin, dtype=float), 0.0)
    ps = 2.0 * (M - 1) / M * qfunc(np.sqrt(snr_lin) / (M - 1))
    return np.minimum(ps / math.log2(M), 0.5)


def ber_mpam_scalar(snr_lin: float, M: int) -> float:
    if snr_lin <= 0.0:
        return 0.5
    ps = 2.0 * (M - 1) / M * 0.5 * math.erfc(math.sqrt(snr_lin) / (M - 1) / math.sqrt(2.0))
    return min(ps / math.log2(M), 0.5)


def log_ber_mpam(snr_lin: float, M: int) -> float:
    x = math.sqrt(max(snr_lin, 0.0)) / (M - 1)
    return math.log(2.0 * (M - 1) / M / math.log2(M)) + float(log_ndtr(-x))


@lru_cache(maxsize=None)
def _threshold_db(M: int, ber_target: float) -> float:
    f = lambda x_db: log_ber_mpam(10 ** (x_db / 10.0), M) - math.log(ber_target)
    return brentq(f, -10.0, 80.0, xtol=1e-12)


def snr_thresholds_db(cfg: Config) -> Dict[int, float]:
    return {M: _threshold_db(M, cfg.ber_max) for M in cfg.mod_orders}


def snr_thresholds_lin(cfg: Config) -> Dict[int, float]:
    return {M: 10 ** (v / 10.0) for M, v in snr_thresholds_db(cfg).items()}


def ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n)
    z0 = rng.standard_normal()
    s = math.sqrt(1.0 - rho * rho)
    z, _ = lfilter([s], [1.0, -rho], w, zi=[rho * z0])
    return z


def gamma_from_gauss(z: np.ndarray, shape: float) -> np.ndarray:
    out = np.empty_like(z)
    lo = z < 0
    out[lo] = gamma_dist.ppf(norm.cdf(z[lo]), shape, scale=1.0 / shape)
    out[~lo] = gamma_dist.isf(norm.sf(z[~lo]), shape, scale=1.0 / shape)
    return out


def gauss_from_gamma(x: np.ndarray, shape: float) -> np.ndarray:
    return norm.ppf(gamma_dist.cdf(x, shape, scale=1.0 / shape))


class ChannelGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        c, d = self.cfg, self.cfg.derived
        zx = ar1(n, d["rho_c"], rng)
        zy = ar1(n, d["rho_c"], rng)
        X = gamma_from_gauss(zx, c.gg_alpha)
        Y = gamma_from_gauss(zy, c.gg_beta)
        h_at = X * Y

        sx = d["sigma_s_m"] * ar1(n, d["rho_pe"], rng)
        sy = d["sigma_s_m"] * ar1(n, d["rho_pe"], rng)
        ax = c.sigma_aoa_rad * ar1(n, d["rho_aoa"], rng)
        ay = c.sigma_aoa_rad * ar1(n, d["rho_aoa"], rng)
        if c.hover_vib_enable:
            zf = ar1(n, math.exp(-c.slot_s * 1e3 / c.hover_vib_tau_f_ms), rng)
            f = c.hover_vib_freq_hz * (1.0 + c.hover_vib_freq_dev * zf)
            phase = rng.uniform(0, 2 * math.pi) + 2 * math.pi * np.cumsum(f) * c.slot_s
            sx = sx + c.hover_vib_amp_urad * 1e-6 * d["link_m"] * np.sin(phase)
        r2 = sx * sx + sy * sy
        h_pe = d["A0"] * np.exp(-2.0 * r2 / d["wzeq2"])

        theta = np.hypot(ax, ay)
        h_aoa = (theta <= c.fov_rad).astype(float)

        h = d["h_l"] * h_at * h_pe * h_aoa
        return dict(h=h, h_at=h_at, X=X, Y=Y, zx=zx, zy=zy,
                    h_pe=h_pe, h_aoa=h_aoa, theta_aoa=theta)


def snr_lin(cfg: Config, P, h):
    d = cfg.derived
    return d["snr_ref_lin"] * (np.asarray(P) / cfg.p_max_w) ** 2 * (np.asarray(h) / d["h_ref"]) ** 2


def snr_ref_db(cfg: Config, h):
    g = snr_lin(cfg, cfg.p_max_w, h)
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(g)
    return np.maximum(out, DB_FLOOR)


def normalize_snr_db(cfg: Config, snr_db):
    lo, hi = cfg.snr_norm_min_db, cfg.snr_norm_max_db
    return np.clip((np.asarray(snr_db) - lo) / (hi - lo), 0.0, 1.0)


def gg_cdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    f = lambda s: gamma_dist.pdf(s, a, scale=1 / a) * gamma_dist.cdf(x / s, b, scale=1 / b)
    val, _ = integrate.quad(f, 0, np.inf, limit=400, epsabs=1e-12, epsrel=1e-9)
    return min(max(val, 0.0), 1.0)


def outage_analytic(cfg: Config, P: float, M: int) -> float:
    d = cfg.derived
    g_th = snr_thresholds_lin(cfg)[M]

    h_th = d["h_ref"] * math.sqrt(g_th / d["snr_ref_lin"]) * (cfg.p_max_w / P)
    t = h_th / d["h_l"]
    xi2, A0 = d["xi2"], d["A0"]

    inner = lambda w: gg_cdf(t / (A0 * w ** (1.0 / xi2)), cfg.gg_alpha, cfg.gg_beta)
    p_pe_at, _ = integrate.quad(inner, 0.0, 1.0, limit=200, epsabs=1e-10)
    p_aoa_fail = math.exp(-cfg.fov_rad ** 2 / (2 * cfg.sigma_aoa_rad ** 2))
    return p_aoa_fail + (1.0 - p_aoa_fail) * p_pe_at
