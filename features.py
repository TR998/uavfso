from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from channel import normalize_snr_db, snr_thresholds_db
from config import Config
from it2fls import IT2FLS, fit_uncertainty_scale, normalize_uncertainty

SCHEMES = ("sac_only", "sac_lstm", "sac_it2", "original_design", "perfect_sac", "proposed_beta099", "proposed_hist", "proposed")
USES_PRED = {"sac_lstm", "original_design", "proposed_beta099", "proposed_hist", "proposed"}
USES_IT2 = {"sac_it2", "original_design", "proposed_beta099", "proposed_hist", "proposed"}


BASE = {"proposed_beta099": "original_design", "proposed_hist": "original_design", "proposed": "original_design"}


def raw_uncertainty(scheme: str, X: np.ndarray, pred: Optional[np.ndarray]) -> np.ndarray:
    scheme = BASE.get(scheme, scheme)
    if scheme == "original_design":
        return np.abs(pred - X[:, -1])
    if scheme == "sac_it2":
        return np.abs(X[:, -1] - X[:, -2])
    raise ValueError(scheme)


def fit_e_scale(cfg: Config, scheme: str, X_val: np.ndarray, predictor=None) -> float:
    pred = predictor.predict_batch(X_val) if BASE.get(scheme, scheme) == "original_design" else None
    return fit_uncertainty_scale(raw_uncertainty(scheme, X_val, pred), cfg.it2_e_percentile)


class StateBuilder:
    def __init__(self, cfg: Config, scheme: str, predictor=None, e_scale: Optional[float] = None):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if scheme in USES_PRED and predictor is None:
            raise ValueError(f"scheme {scheme} requires a predictor")
        if scheme in USES_IT2 and e_scale is None:
            raise ValueError(f"scheme {scheme} requires e_scale (fit_e_scale on validation data)")
        self.cfg, self.scheme, self.predictor, self.e_scale = cfg, scheme, predictor, e_scale
        self.it2 = IT2FLS(cfg) if scheme in USES_IT2 else None
        self.append_conf = (cfg.state_append_conf and scheme in USES_IT2) or scheme in ("proposed_beta099", "proposed_hist", "proposed")
        self.append_hist = scheme == "proposed_hist"
        self.dim = 3 + int(self.append_conf) + (cfg.lstm_window if self.append_hist else 0)
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def _snr_feature_batch(self, X: np.ndarray, true_norm: Optional[np.ndarray] = None):
        s = BASE.get(self.scheme, self.scheme)
        n = len(X)
        if s == "perfect_sac":
            return true_norm, np.ones(n)
        if s == "sac_only":
            return X[:, -1], np.ones(n)
        pred = self.predictor.predict_batch(X) if s in USES_PRED else None
        if s == "sac_lstm":
            return pred, np.ones(n)
        z1 = pred if s == "original_design" else X[:, -1]
        z2 = normalize_uncertainty(raw_uncertainty(s, X, pred), self.e_scale)
        c = self.it2.confidence(z1, z2)
        return c * z1, c

    def on_reset(self, env) -> None:
        c = self.cfg
        D, L, T, H = c.csi_delay, c.lstm_window, c.episode_slots, env.hist_len
        norm = normalize_snr_db(c, env.snr_db)
        i = np.arange(T) + H
        idx = (i - D)[:, None] + np.arange(-L + 1, 1)[None, :]
        X = norm[idx]
        feat, conf = self._snr_feature_batch(X, true_norm=norm[i])
        self._cache = dict(feat=np.asarray(feat, dtype=np.float64), conf=np.asarray(conf, dtype=np.float64), hist=X)

    def build(self, obs: Dict) -> np.ndarray:
        c = self.cfg
        if self._cache is not None:
            feat, conf = self._cache["feat"][obs["t"]], self._cache["conf"][obs["t"]]
            hist = self._cache["hist"][obs["t"]]
        else:
            X = normalize_snr_db(c, obs["snr_hist_db"])[None, :]
            true = np.array([normalize_snr_db(c, obs["snr_true_db"])])
            f, cf = self._snr_feature_batch(X, true_norm=true)
            feat, conf = float(np.asarray(f)[0]), float(np.asarray(cf)[0])
            hist = X[0]
        s = [feat, obs["queue_bits"] / c.q_max_bits, obs["battery_frac"]]
        if self.append_conf:
            s.append(conf)
        if self.append_hist:
            s.extend(np.asarray(hist, dtype=float).tolist())
        return np.asarray(s, dtype=np.float32)

    def confidence(self, obs: Dict) -> float:
        return float(self._cache["conf"][obs["t"]]) if self._cache is not None else 1.0


def modulation_mask(cfg: Config, state: np.ndarray, margin_db: float = 0.0) -> np.ndarray:
    st = np.atleast_2d(state)
    snr_db = cfg.snr_norm_min_db + st[:, 0] * (cfg.snr_norm_max_db - cfg.snr_norm_min_db)
    thr = snr_thresholds_db(cfg)
    m = np.stack([(snr_db - margin_db >= thr[M]) for M in cfg.mod_orders], -1)
    m[:, 0] = True
    return m if np.ndim(state) == 2 else m[0]
