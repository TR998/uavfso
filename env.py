from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from channel import ChannelGenerator, DB_FLOOR, ber_mpam_scalar, snr_ref_db, snr_thresholds_lin
from config import Config


class UAVFSOEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gen = ChannelGenerator(cfg)
        self.g_th = snr_thresholds_lin(cfg)
        self.mods = tuple(cfg.mod_orders)
        self.hist_len = cfg.lstm_window + cfg.csi_delay
        self.t = 0
        self._ready = False

    def reset(self, seed: Optional[int] = None, channel: Optional[Dict[str, np.ndarray]] = None):
        c = self.cfg
        ss = np.random.SeedSequence(seed)
        ch_ss, arr_ss, bat_ss = ss.spawn(3)
        n = c.episode_slots + self.hist_len
        if channel is None:
            channel = self.gen.generate(n, np.random.default_rng(ch_ss))
        self.h = channel["h"][:n]
        self.snr_db = snr_ref_db(c, self.h)
        self.arrivals = np.random.default_rng(arr_ss).poisson(c.arrival_bits_per_slot, c.episode_slots)
        lo, hi = c.battery_init_frac
        self.e_rem = float(np.random.default_rng(bat_ss).uniform(lo, hi)) * c.battery_j
        self.q = float(c.q0_bits)
        self.t = 0
        self._ready = True
        return self._obs(), {}

    def _idx(self, t):
        return t + self.hist_len

    def _obs(self) -> Dict:
        c = self.cfg
        i = self._idx(self.t)
        D, L = c.csi_delay, c.lstm_window
        return dict(
            snr_delayed_db=float(self.snr_db[i - D]),
            snr_hist_db=self.snr_db[i - D - L + 1: i - D + 1].copy(),
            queue_bits=self.q,
            energy_rem_j=self.e_rem,
            battery_frac=self.e_rem / c.battery_j,
            snr_true_db=float(self.snr_db[i]),
            t=self.t,
        )

    def step(self, P: float, M: int):
        assert self._ready, "call reset() first"
        c, d = self.cfg, self.cfg.derived
        if M not in self.g_th:
            raise ValueError(f"M={M} not in {self.mods}")
        P = float(min(max(P, c.p_min_w), c.p_max_w))
        i = self._idx(self.t)

        h = self.h[i]
        gamma = d["snr_ref_lin"] * (P / c.p_max_w) ** 2 * (h / d["h_ref"]) ** 2
        outage = gamma < self.g_th[M]
        ber = ber_mpam_scalar(gamma, M)

        q_t = self.q
        cap = 0.0 if outage else c.n_symbols_per_slot * math.log2(M)
        mu = min(q_t, cap)
        a = float(self.arrivals[self.t])
        q_raw = q_t - mu + a
        drop = max(q_raw - c.q_max_bits, 0.0)
        self.q = min(q_raw, c.q_max_bits)

        e = (P + c.p_circuit_w + c.p_hover_w) * c.slot_s
        e_comm = (P + c.p_circuit_w) * c.slot_s
        self.e_rem -= e

        r_ee = c.w_ee * (mu / (e + c.eps_reward)) / d["eta_ref"]
        r_out = c.w_out * float(outage)
        if c.reward_ber_delivered_only and outage:
            r_ber = 0.0
        else:
            r_ber = c.w_ber * min(math.log1p(ber / c.ber_max), c.ber_term_clip)
        r_q = c.w_queue * max(q_t - c.q_th_bits, 0.0) / c.q_max_bits
        reward = r_ee - r_out - r_ber - r_q

        info = dict(P=P, M=M, gamma_lin=gamma,
                    snr_db=10 * math.log10(gamma) if gamma > 0 else DB_FLOOR,
                    outage=bool(outage), ber=ber, capacity=cap, mu=mu,
                    arrival=a, drop=drop, queue_before=q_t, queue_after=self.q,
                    energy=e, energy_nohover=e_comm,
                    r_ee=r_ee, r_out=r_out, r_ber=r_ber, r_queue=r_q)

        self.t += 1
        truncated = self.t >= c.episode_slots
        terminated = False
        if truncated:
            self._ready = False
            obs = None
        else:
            obs = self._obs()
        return obs, reward, terminated, truncated, info


class MetricsAccumulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keys = ("mu", "energy", "energy_nohover", "outage", "ber", "drop", "arrival", "queue_after")
        self.data = {k: [] for k in self.keys}
        self.ber_deliv = []
        self.rewards = []

    def add(self, t: int, info: Dict, reward: float):
        self.rewards.append(reward)
        if t < self.cfg.transient_slots:
            return
        for k in self.keys:
            self.data[k].append(float(info[k]))
        if not info["outage"]:
            self.ber_deliv.append(info["ber"])

    def summary(self) -> Dict[str, float]:
        c = self.cfg
        a = {k: np.asarray(v) for k, v in self.data.items()}
        backlog = a["queue_after"].mean()
        return dict(
            EE_kbitJ=a["mu"].sum() / a["energy"].sum() / 1e3,
            EE_nohover_kbitJ=a["mu"].sum() / a["energy_nohover"].sum() / 1e3,
            outage=a["outage"].mean(),
            BER_deliv=float(np.mean(self.ber_deliv)) if self.ber_deliv else float("nan"),
            BER_all=a["ber"].mean(),
            backlog_bits=backlog,
            delay_slots=backlog / c.arrival_bits_per_slot,
            drop_rate=a["drop"].sum() / a["arrival"].sum(),
            throughput_bits=a["mu"].mean(),
            episode_return=float(np.sum(self.rewards)),
        )


def ee_upper_bound_kbitJ(cfg: Config) -> float:
    return cfg.derived["eta_ref"] / 1e3


def rollout_policy(cfg: Config, policy, n_episodes: int, seed_base: Optional[int] = None):
    seed_base = cfg.eval_seed_base if seed_base is None else seed_base
    env = UAVFSOEnv(cfg)
    per_ep = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + k)
        acc = MetricsAccumulator(cfg)
        t = 0
        while True:
            P, M = policy.act(obs)
            obs, r, term, trunc, info = env.step(P, M)
            acc.add(t, info, r)
            t += 1
            if term or trunc:
                break
        per_ep.append(acc.summary())
    keys = per_ep[0].keys()
    mean = {k: float(np.nanmean([e[k] for e in per_ep])) for k in keys}
    std = {k: float(np.nanstd([e[k] for e in per_ep])) for k in keys}
    return mean, std, per_ep
