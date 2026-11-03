from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from config import Config
from env import MetricsAccumulator, UAVFSOEnv
from features import StateBuilder, modulation_mask


class AgentPolicy:
    def __init__(self, cfg: Config, builder: StateBuilder, agent, name: str = "SAC"):
        self.cfg, self.builder, self.agent, self.name = cfg, builder, agent, name

    def on_reset(self, env):
        self.builder.on_reset(env)

    def act(self, obs):
        s = self.builder.build(obs)
        mask = modulation_mask(self.cfg, s, self.cfg.sac_mask_margin_db) if self.cfg.sac_mod_masking else None
        p_norm, m_idx = self.agent.act(s, deterministic=True, mask=mask)
        return self.agent.p_watts(p_norm), self.cfg.mod_orders[m_idx]


def run_training_episode(cfg: Config, env: UAVFSOEnv, builder: StateBuilder, agent, seed: int,
                         max_slots: Optional[int] = None) -> Dict:
    obs, _ = env.reset(seed=seed)
    builder.on_reset(env)
    s = builder.build(obs)
    mask = modulation_mask(cfg, s, cfg.sac_mask_margin_db) if cfg.sac_mod_masking else None
    acc = MetricsAccumulator(cfg)
    logs: List[Dict] = []
    t = 0
    while True:
        p_norm, m_idx = agent.act(s, deterministic=False, mask=mask)
        obs2, r, term, trunc, info = env.step(agent.p_watts(p_norm), cfg.mod_orders[m_idx])
        acc.add(t, info, r)
        if trunc and not term:
            t += 1
            break
        s2 = builder.build(obs2)
        mask2 = modulation_mask(cfg, s2, cfg.sac_mask_margin_db) if cfg.sac_mod_masking else None
        out = agent.observe(s, p_norm, m_idx, r, s2, term, mask2)
        if out is not None:
            logs.append(out)
        t += 1
        if term or trunc or (max_slots and t >= max_slots):
            break
        s, obs, mask = s2, obs2, mask2
    return dict(summary=acc.summary(), updates=logs, slots=t)


def evaluate_policy(cfg: Config, policy, n_episodes: int, seed_base: Optional[int] = None):
    seed_base = cfg.eval_seed_base if seed_base is None else seed_base
    env = UAVFSOEnv(cfg)
    per = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + k)
        if hasattr(policy, "on_reset"):
            policy.on_reset(env)
        acc, t = MetricsAccumulator(cfg), 0
        while True:
            P, M = policy.act(obs)
            obs, r, te, tr, info = env.step(P, M)
            acc.add(t, info, r); t += 1
            if te or tr:
                break
        per.append(acc.summary())
    keys = per[0].keys()
    mean = {k: float(np.nanmean([e[k] for e in per])) for k in keys}
    std = {k: float(np.nanstd([e[k] for e in per])) for k in keys}
    return mean, std, per
