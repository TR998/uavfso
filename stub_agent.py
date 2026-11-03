from __future__ import annotations

import pickle

import numpy as np

from channel import snr_thresholds_db
from config import Config


class StubAgent:
    def __init__(self, cfg: Config, state_dim: int, seed: int = 0, total_steps: int | None = None, **_):
        self.cfg, self.dim = cfg, state_dim
        self.rng = np.random.default_rng(seed)
        self.thr = [snr_thresholds_db(cfg)[M] for M in cfg.mod_orders]
        self.margin = 3.0
        self.steps = self.updates = 0
        self.total_steps = total_steps or cfg.train_episodes * cfg.episode_slots

    def p_watts(self, p_norm):
        c = self.cfg
        return c.p_min_w + float(p_norm) * (c.p_max_w - c.p_min_w)

    def _greedy(self, state):
        c = self.cfg
        snr = c.snr_norm_min_db + float(state[0]) * (c.snr_norm_max_db - c.snr_norm_min_db)
        m = 0
        for k, t in enumerate(self.thr):
            if snr - self.margin >= t:
                m = k
        return 1.0, m

    def act(self, state, deterministic=False, mask=None):
        if not deterministic and self.rng.uniform() < 0.1:
            return float(self.rng.uniform()), int(self.rng.integers(len(self.thr)))
        return self._greedy(state)

    def observe(self, s, p_norm, m_idx, r, s2, done, mask2=None):
        self.steps += 1
        if self.steps < self.cfg.sac_warmup_steps:
            return None
        self.updates += 1
        p = np.full(len(self.thr), 0.1 / len(self.thr)); p[self._greedy(s)[1]] += 0.9
        return dict(loss_q=np.nan, loss_pi=np.nan, loss_alpha=np.nan, alpha_c=np.nan, alpha_d=np.nan,
                    H_c=np.nan, H_d=float(-(p * np.log(p)).sum()), q_mean=np.nan, gumbel_tau=np.nan)

    def n_params(self):
        return dict(actor=0, critic_each=0)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(steps=self.steps, updates=self.updates, rng=self.rng.bit_generator.state), f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.steps, self.updates = d["steps"], d["updates"]
        self.rng.bit_generator.state = d["rng"]

    save_full_ = save
    load_full_ = load
