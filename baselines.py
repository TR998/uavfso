from __future__ import annotations

import math

from channel import snr_thresholds_db
from config import Config


def _highest_feasible(snr_db: float, thr_db: dict, mods, margin_db: float = 0.0) -> int:
    best = mods[0]
    for M in mods:
        if snr_db - margin_db >= thr_db[M]:
            best = M
    return best


def _queue_downgrade(M: int, q_bits: float, cfg: Config) -> int:
    for m in cfg.mod_orders:
        if m > M:
            break
        if cfg.n_symbols_per_slot * math.log2(m) >= q_bits:
            return m
    return M


class HeuristicPolicy:
    name = "Heuristic"

    def __init__(self, cfg: Config, power_w: float = 0.55):
        self.cfg, self.P = cfg, power_w
        self.thr = snr_thresholds_db(cfg)

    def act(self, obs):
        snr_at_P = obs["snr_delayed_db"] + 20 * math.log10(self.P / self.cfg.p_max_w)
        return self.P, _highest_feasible(snr_at_P, self.thr, self.cfg.mod_orders)


class ThresholdPolicy:
    name = "Threshold"

    def __init__(self, cfg: Config, margin_db: float = 3.0):
        self.cfg, self.margin = cfg, margin_db
        self.thr = snr_thresholds_db(cfg)

    def act(self, obs):
        M = _highest_feasible(obs["snr_delayed_db"], self.thr, self.cfg.mod_orders, self.margin)
        return self.cfg.p_max_w, _queue_downgrade(M, obs["queue_bits"], self.cfg)


class OraclePolicy:
    name = "Perfect-CSI"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.thr = snr_thresholds_db(cfg)

    def act(self, obs):
        M = _highest_feasible(obs["snr_true_db"], self.thr, self.cfg.mod_orders)
        return self.cfg.p_max_w, _queue_downgrade(M, obs["queue_bits"], self.cfg)


ALL_BASELINES = (HeuristicPolicy, ThresholdPolicy, OraclePolicy)
