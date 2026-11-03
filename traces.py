from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

from channel import ChannelGenerator, normalize_snr_db, snr_ref_db
from config import Config, get_config, with_overrides


def make_snr_traces(cfg: Config, n: int, seed: int) -> Dict[str, np.ndarray]:
    ch = ChannelGenerator(cfg).generate(n, np.random.default_rng(seed))
    db = snr_ref_db(cfg, ch["h"])
    return dict(snr_db=db, snr_norm=normalize_snr_db(cfg, db))


def split_70_15_15(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    i1, i2 = int(0.70 * n), int(0.85 * n)
    return x[:i1], x[i1:i2], x[i2:]


def make_windows(trace: np.ndarray, L: int, D: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(trace) - L - D + 1
    if n <= 0:
        raise ValueError("trace too short")
    idx = np.arange(L)[None, :] + np.arange(n)[:, None]
    X = trace[idx].astype(np.float32)
    y = trace[np.arange(n) + L - 1 + D].astype(np.float32)
    return X, y


def build_lstm_dataset(cfg: Config, seed: int = 123, n: int | None = None) -> Dict[str, np.ndarray]:
    n = n or cfg.trace_slots
    tr = make_snr_traces(cfg, n, seed)["snr_norm"]
    out = {}
    for name, part in zip(("train", "val", "test"), split_70_15_15(tr)):
        X, y = make_windows(part, cfg.lstm_window, cfg.csi_delay)
        out[f"X_{name}"], out[f"y_{name}"] = X, y
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick")
    ap.add_argument("--delay", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = get_config(a.mode)
    if a.delay is not None:
        cfg = with_overrides(cfg, csi_delay=a.delay)
    ds = build_lstm_dataset(cfg, seed=a.seed)
    out = a.out or f"traces_D{cfg.csi_delay}_{cfg.mode}.npz"
    np.savez_compressed(out, **ds)
    for k, v in ds.items():
        print(f"{k:8s} {v.shape}")
    print("disimpan ke", out)
