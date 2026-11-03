from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

NEG_INF = -1e9


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, n_mod: int, hidden=(128, 64), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.body = mlp([state_dim, *hidden], out_act=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], 1)
        self.logvar = nn.Linear(hidden[-1], 1)
        self.logits = nn.Linear(hidden[-1], n_mod)
        self.lo, self.hi = log_std_bounds

    def forward(self, s, mask: Optional[torch.Tensor] = None):
        h = self.body(s)
        mu = self.mu(h)
        log_std = torch.clamp(0.5 * self.logvar(h), self.lo, self.hi)
        logits = self.logits(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, NEG_INF)
        return mu, log_std, logits

    def sample(self, s, gumbel_tau: float, mask=None):
        mu, log_std, logits = self(s, mask)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp_c = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp_c = logp_c - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        p_norm = (a + 1) / 2
        logp_all = F.log_softmax(logits, -1)
        probs = logp_all.exp()
        onehot = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True)
        return dict(p_norm=p_norm, logp_c=logp_c, onehot=onehot, probs=probs, logp_all=logp_all)

    @torch.no_grad()
    def deterministic(self, s, mask=None):
        mu, _, logits = self(s, mask)
        return (torch.tanh(mu) + 1) / 2, logits.argmax(-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, n_mod: int, hidden=(128, 128)):
        super().__init__()
        self.q = mlp([state_dim + 1 + n_mod, *hidden, 1])

    def forward(self, s, p_norm, onehot):
        return self.q(torch.cat([s, p_norm, onehot], -1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, state_dim: int, n_mod: int, capacity: int, seed: int = 0):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.p = np.zeros((capacity, 1), np.float32)
        self.m = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.mask2 = np.ones((capacity, n_mod), bool)
        self.rng = np.random.default_rng(seed)

    def add(self, s, p_norm, m_idx, r, s2, done, mask2=None):
        j = self.i
        self.s[j], self.p[j, 0], self.m[j], self.r[j], self.s2[j], self.d[j] = s, p_norm, m_idx, r, s2, float(done)
        if mask2 is not None:
            self.mask2[j] = mask2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, device):
        idx = self.rng.integers(0, self.n, batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.s), t(self.p), t(self.m), t(self.r), t(self.s2), t(self.d), t(self.mask2)


class HybridSAC:
    def __init__(self, cfg: Config, state_dim: int, seed: int = 0, device: str = "cpu",
                 total_steps: Optional[int] = None):
        torch.manual_seed(seed)
        self.cfg, self.device = cfg, torch.device(device)
        self.mods = tuple(cfg.mod_orders)
        n = len(self.mods)
        self.actor = Actor(state_dim, n, cfg.sac_actor_hidden, cfg.sac_log_std_bounds).to(self.device)
        self.q1 = Critic(state_dim, n, cfg.sac_critic_hidden).to(self.device)
        self.q2 = Critic(state_dim, n, cfg.sac_critic_hidden).to(self.device)
        self.q1_t = Critic(state_dim, n, cfg.sac_critic_hidden).to(self.device)
        self.q2_t = Critic(state_dim, n, cfg.sac_critic_hidden).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict()); self.q2_t.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.log_alpha_c = torch.tensor(math.log(cfg.sac_init_alpha), device=self.device, requires_grad=True)
        self.log_alpha_d = torch.tensor(math.log(cfg.sac_init_alpha), device=self.device, requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac_lr_actor)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.sac_lr_critic)
        self.opt_alpha = torch.optim.Adam([self.log_alpha_c, self.log_alpha_d], lr=cfg.sac_lr_alpha)
        self.log_alpha_max = math.log(cfg.sac_alpha_max)
        self.h_c_target = cfg.sac_target_entropy_cont
        self.h_d_target = cfg.derived["target_entropy_disc"]
        self.buffer = ReplayBuffer(state_dim, n, cfg.sac_buffer, seed)
        self.total_steps = total_steps or cfg.train_episodes * cfg.episode_slots
        self.steps = 0
        self.updates = 0
        self.rng = np.random.default_rng(seed + 1)

    def gumbel_tau(self) -> float:
        t0, t1 = self.cfg.sac_gumbel_tau
        frac = min(self.steps / max(self.cfg.sac_gumbel_anneal_frac * self.total_steps, 1), 1.0)
        return t0 + (t1 - t0) * frac

    def p_watts(self, p_norm: float) -> float:
        c = self.cfg
        return c.p_min_w + float(p_norm) * (c.p_max_w - c.p_min_w)

    def _mask_t(self, mask):
        if mask is None:
            return None
        return torch.as_tensor(np.atleast_2d(mask), dtype=torch.bool, device=self.device)

    def act(self, state: np.ndarray, deterministic: bool = False, mask: Optional[np.ndarray] = None):
        if not deterministic and self.steps < self.cfg.sac_warmup_steps:
            allowed = np.flatnonzero(mask) if mask is not None else np.arange(len(self.mods))
            return float(self.rng.uniform()), int(self.rng.choice(allowed))
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        with torch.no_grad():
            if deterministic:
                p, m = self.actor.deterministic(s, self._mask_t(mask))
                return float(p[0, 0]), int(m[0])
            out = self.actor.sample(s, self.gumbel_tau(), self._mask_t(mask))
            return float(out["p_norm"][0, 0]), int(out["onehot"][0].argmax())

    def _q_all(self, critic, s, p_norm):
        B, n = s.shape[0], len(self.mods)
        eye = torch.eye(n, device=s.device)
        s_r = s.repeat_interleave(n, 0)
        p_r = p_norm.repeat_interleave(n, 0)
        oh = eye.repeat(B, 1)
        return critic(s_r, p_r, oh).view(B, n)

    def update(self) -> Dict[str, float]:
        c = self.cfg
        s, p, m, r, s2, d, mask2 = self.buffer.sample(c.sac_batch, self.device)
        mask2 = mask2 if c.sac_mod_masking else None
        onehot = F.one_hot(m, len(self.mods)).float()
        alpha_c, alpha_d = self.log_alpha_c.exp().detach(), self.log_alpha_d.exp().detach()
        tau_g = self.gumbel_tau()

        expected = c.sac_discrete_mode == "expected"

        with torch.no_grad():
            nxt = self.actor.sample(s2, tau_g, mask2)
            ent_d = (nxt["probs"] * nxt["logp_all"]).sum(-1)
            if expected:
                qn = (nxt["probs"] * torch.min(self._q_all(self.q1_t, s2, nxt["p_norm"]),
                                               self._q_all(self.q2_t, s2, nxt["p_norm"]))).sum(-1)
            else:
                qn = torch.min(self.q1_t(s2, nxt["p_norm"], nxt["onehot"]),
                               self.q2_t(s2, nxt["p_norm"], nxt["onehot"]))
            v = qn - alpha_c * nxt["logp_c"] - alpha_d * ent_d
            y = r + c.sac_gamma * (1 - d) * v
        q1, q2 = self.q1(s, p, onehot), self.q2(s, p, onehot)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        mask1 = None
        if c.sac_mod_masking:
            from features import modulation_mask
            mask1 = torch.as_tensor(modulation_mask(c, s.cpu().numpy(), c.sac_mask_margin_db), device=self.device)
        cur = self.actor.sample(s, tau_g, mask1)
        if expected:
            qpi = (cur["probs"] * torch.min(self._q_all(self.q1, s, cur["p_norm"]),
                                            self._q_all(self.q2, s, cur["p_norm"]))).sum(-1)
        else:
            qpi = torch.min(self.q1(s, cur["p_norm"], cur["onehot"]), self.q2(s, cur["p_norm"], cur["onehot"]))
        neg_h_d = (cur["probs"] * cur["logp_all"]).sum(-1)
        loss_pi = (alpha_c * cur["logp_c"] + alpha_d * neg_h_d - qpi).mean()
        self.opt_actor.zero_grad(); loss_pi.backward(); self.opt_actor.step()

        h_c = -cur["logp_c"].detach()
        h_d = -neg_h_d.detach()
        loss_alpha = (self.log_alpha_c * (h_c - self.h_c_target)).mean() + \
                     (self.log_alpha_d * (h_d - self.h_d_target)).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha_c.clamp_(max=self.log_alpha_max)
            self.log_alpha_d.clamp_(max=self.log_alpha_max)

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for pp, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - c.sac_tau).add_(c.sac_tau * pp)
        self.updates += 1
        return dict(loss_q=loss_q.item(), loss_pi=loss_pi.item(), loss_alpha=loss_alpha.item(),
                    alpha_c=float(alpha_c), alpha_d=float(alpha_d), H_c=float(h_c.mean()),
                    H_d=float(h_d.mean()), q_mean=float(q1.mean()), gumbel_tau=tau_g)

    def observe(self, s, p_norm, m_idx, r, s2, done, mask2=None) -> Optional[Dict[str, float]]:
        self.buffer.add(s, p_norm, m_idx, r, s2, done, mask2)
        self.steps += 1
        out = None
        if self.steps >= self.cfg.sac_warmup_steps and self.buffer.n >= self.cfg.sac_batch:
            for _ in range(self.cfg.sac_updates_per_step):
                out = self.update()
        return out

    def n_params(self) -> Dict[str, int]:
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        return dict(actor=cnt(self.actor), critic_each=cnt(self.q1))

    def save(self, path: str):
        torch.save(dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                        log_alpha_c=self.log_alpha_c.detach(), log_alpha_d=self.log_alpha_d.detach(),
                        steps=self.steps), path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"]); self.q1.load_state_dict(ck["q1"]); self.q2.load_state_dict(ck["q2"])
        self.q1_t.load_state_dict(ck["q1"]); self.q2_t.load_state_dict(ck["q2"])
        with torch.no_grad():
            self.log_alpha_c.copy_(ck["log_alpha_c"]); self.log_alpha_d.copy_(ck["log_alpha_d"])
        self.steps = ck["steps"]


def save_full(agent: "HybridSAC", path: str) -> None:
    b = agent.buffer
    n = b.n
    torch.save(dict(
        actor=agent.actor.state_dict(), q1=agent.q1.state_dict(), q2=agent.q2.state_dict(),
        q1_t=agent.q1_t.state_dict(), q2_t=agent.q2_t.state_dict(),
        opt_actor=agent.opt_actor.state_dict(), opt_q=agent.opt_q.state_dict(),
        opt_alpha=agent.opt_alpha.state_dict(),
        log_alpha_c=agent.log_alpha_c.detach().cpu(), log_alpha_d=agent.log_alpha_d.detach().cpu(),
        steps=agent.steps, updates=agent.updates, total_steps=agent.total_steps,
        buf=dict(n=n, i=b.i, s=torch.from_numpy(b.s[:n].copy()), s2=torch.from_numpy(b.s2[:n].copy()),
                 p=torch.from_numpy(b.p[:n].copy()), m=torch.from_numpy(b.m[:n].copy()),
                 r=torch.from_numpy(b.r[:n].copy()), d=torch.from_numpy(b.d[:n].copy()),
                 mask2=torch.from_numpy(b.mask2[:n].copy())),
        np_rng=agent.rng.bit_generator.state, buf_rng=b.rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
    ), path)


def load_full(agent: "HybridSAC", path: str) -> None:
    ck = torch.load(path, map_location=agent.device, weights_only=False)
    for name in ("actor", "q1", "q2", "q1_t", "q2_t"):
        getattr(agent, name).load_state_dict(ck[name])
    agent.opt_actor.load_state_dict(ck["opt_actor"]); agent.opt_q.load_state_dict(ck["opt_q"])
    agent.opt_alpha.load_state_dict(ck["opt_alpha"])
    with torch.no_grad():
        agent.log_alpha_c.copy_(ck["log_alpha_c"].to(agent.device))
        agent.log_alpha_d.copy_(ck["log_alpha_d"].to(agent.device))
    agent.steps, agent.updates, agent.total_steps = ck["steps"], ck["updates"], ck["total_steps"]
    bf, b = ck["buf"], agent.buffer
    n = bf["n"]
    b.s[:n], b.s2[:n], b.p[:n] = bf["s"].numpy(), bf["s2"].numpy(), bf["p"].numpy()
    b.m[:n], b.r[:n], b.d[:n], b.mask2[:n] = bf["m"].numpy(), bf["r"].numpy(), bf["d"].numpy(), bf["mask2"].numpy()
    b.n, b.i = n, bf["i"]
    agent.rng.bit_generator.state = ck["np_rng"]
    b.rng.bit_generator.state = ck["buf_rng"]
    torch.set_rng_state(ck["torch_rng"])
