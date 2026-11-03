from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from baselines import OraclePolicy, ThresholdPolicy, _queue_downgrade
from channel import normalize_snr_db, snr_thresholds_db
from config import get_config, with_overrides
from env import rollout_policy
from predictors_np import LinearARPredictor
from traces import build_lstm_dataset

SCENARIOS = {"default": {}, "hover_vib": dict(hover_vib_enable=True)}
METRICS = ["EE_kbitJ", "outage", "BER_deliv", "BER_all", "backlog_bits", "delay_slots", "drop_rate"]


class UAThresholdPolicy:
    def __init__(self, cfg, m0, k, use_pred, ar):
        self.cfg, self.m0, self.k, self.use_pred, self.ar = cfg, m0, k, use_pred, ar
        self.thr = snr_thresholds_db(cfg)
        self.lo, self.hi = cfg.snr_norm_min_db, cfg.snr_norm_max_db

    def act(self, obs):
        g_hat = obs["snr_delayed_db"]
        x = normalize_snr_db(self.cfg, obs["snr_hist_db"])
        pred = self.lo + float(x @ self.ar.w[:-1] + self.ar.w[-1]) * (self.hi - self.lo)
        f = pred if self.use_pred else g_hat
        margin = self.m0 + self.k * abs(pred - max(g_hat, self.lo))
        M = self.cfg.mod_orders[0]
        for m in self.cfg.mod_orders:
            if f - margin >= self.thr[m]:
                M = m
        return self.cfg.p_max_w, _queue_downgrade(M, obs["queue_bits"], self.cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="default", choices=list(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = with_overrides(get_config("medium"), **SCENARIOS[a.scenario])
    ds = build_lstm_dataset(with_overrides(cfg, trace_slots=200_000), seed=123)
    ar = LinearARPredictor().fit(ds["X_train"], ds["y_train"])
    rows = []
    for name, pol in (("Perfect-CSI", OraclePolicy(cfg)), ("Threshold", ThresholdPolicy(cfg))):
        m, _, _ = rollout_policy(cfg, pol, a.episodes)
        rows.append(dict(policy=name, **{k: m[k] for k in METRICS}))
    for use_pred, m0, k in itertools.product((False, True), np.arange(0, 9, 1.0), (0.0, 0.5, 1.0, 2.0)):
        m, _, _ = rollout_policy(cfg, UAThresholdPolicy(cfg, m0, k, use_pred, ar), a.episodes)
        rows.append(dict(policy="UA-threshold", feature="AR" if use_pred else "persistence", m0_dB=m0, k=k,
                         **{kk: m[kk] for kk in METRICS}))
    df = pd.DataFrame(rows)[["policy", "feature", "m0_dB", "k", *METRICS]]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ua_threshold_{a.scenario}.csv")
    df.to_csv(path, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
