from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from baselines import HeuristicPolicy, OraclePolicy, ThresholdPolicy
from config import Config, with_overrides
from env import MetricsAccumulator, UAVFSOEnv, ee_upper_bound_kbitJ
from features import StateBuilder, fit_e_scale, modulation_mask
from predictors_np import LinearARPredictor, PersistencePredictor, mse
from traces import build_lstm_dataset

LEARNED = ("sac_only", "sac_lstm", "sac_it2", "original_design")
OPTIONAL_LEARNED = ("perfect_sac", "proposed_beta099", "proposed_hist", "proposed")
FINAL_SCHEMES = ("sac_only", "sac_lstm", "sac_it2", "original_design", "proposed")

SCHEME_CFG = {"proposed": dict(sac_gamma=0.95)}

SCENARIOS = {"default": {}, "hover_vib": dict(hover_vib_enable=True)}
BASELINES = {"Heuristic": HeuristicPolicy, "Threshold": ThresholdPolicy, "Perfect-CSI": OraclePolicy}
DISPLAY = {"sac_only": "SAC only", "sac_lstm": "SAC + LSTM", "sac_it2": "SAC + IT2FLS",
           "original_design": "Original design", "perfect_sac": "Perfect-CSI (SAC)",
           "proposed_beta099": "Proposed state (beta = 0.99)",
           "proposed_hist": "Proposed state + CSI history",
           "proposed": "Proposed (LSTM + IT2FLS + SAC)",
           "Heuristic": "Heuristic", "Threshold": "Threshold-based", "Perfect-CSI": "Perfect-CSI"}
DATASET_SEED = 123


class Paths:
    def __init__(self, root: str):
        self.root = Path(root)
        for d in ("lstm", "agents", "eval", "tables", "figures"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, k):
        return self.root / k

    def agent_dir(self, scheme, seed):
        p = self.root / "agents" / f"{scheme}_s{seed}"
        p.mkdir(parents=True, exist_ok=True)
        return p


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def get_predictor(cfg: Config, paths: Paths, backend: str):
    ds = build_lstm_dataset(cfg, seed=DATASET_SEED)
    tag = f"D{cfg.csi_delay}"
    if cfg.csi_delay == 0:
        pp = PersistencePredictor()
        m0 = mse(pp.predict_batch(ds["X_test"]), ds["y_test"])
        meta = dict(backend="persistence (D=0)", compare=dict(
            mse_mean=mse(ds["y_test"] * 0 + ds["y_train"].mean(), ds["y_test"]), mse_persistence=m0,
            mse_linear_AR=float("nan"), mse_lstm=m0, lstm_gain_vs_persistence_pct=float("nan"),
            lstm_gain_vs_linear_pct=float("nan")))
        (paths.lstm / f"lstm_{tag}.json").write_text(json.dumps(meta, indent=1))
        log(f"LSTM {tag}: not trained, persistence is exact (test MSE {m0:.2e})")
        return pp, ds, meta
    if backend == "stub":
        return LinearARPredictor().fit(ds["X_train"], ds["y_train"]), ds, dict(backend="stub-linearAR")
    from lstm_predictor import LSTMPredictor, compare_predictors
    pt, js = paths.lstm / f"lstm_{tag}.pt", paths.lstm / f"lstm_{tag}.json"
    m = LSTMPredictor(cfg, seed=0)
    if pt.exists() and js.exists():
        m.load(str(pt))
        meta = json.loads(js.read_text())
        log(f"LSTM {tag}: loaded from {pt}")
    else:
        log(f"LSTM {tag}: training ({len(ds['y_train'])} samples, max {cfg.lstm_epochs} epoch)")
        info = m.fit(ds, verbose=True)
        meta = dict(info=info, compare=compare_predictors(ds, m), history=m.history, n_params=m.n_params())
        m.save(str(pt) + ".tmp"); os.replace(str(pt) + ".tmp", pt)
        js.write_text(json.dumps(meta, indent=1))
        c = meta["compare"]
        log(f"LSTM {tag}: test MSE persistence {c['mse_persistence']:.5f} | AR {c['mse_linear_AR']:.5f} | "
            f"LSTM {c['mse_lstm']:.5f}")
    return m, ds, meta


def make_builder(cfg: Config, scheme: str, predictor, ds) -> StateBuilder:
    pred = predictor if scheme in ("sac_lstm", "original_design", "proposed_beta099", "proposed_hist", "proposed") else None
    e = fit_e_scale(cfg, scheme, ds["X_val"], predictor) if scheme in ("sac_it2", "original_design", "proposed_beta099", "proposed_hist", "proposed") else None
    return StateBuilder(cfg, scheme, pred, e)


def make_agent(cfg: Config, backend: str, dim: int, seed: int, scheme: Optional[str] = None):
    if scheme in SCHEME_CFG:
        cfg = with_overrides(cfg, **SCHEME_CFG[scheme])
    total = cfg.train_episodes * cfg.episode_slots
    if backend == "stub":
        from stub_agent import StubAgent
        return StubAgent(cfg, dim, seed=seed, total_steps=total)
    from sac_hybrid import HybridSAC
    return HybridSAC(cfg, dim, seed=seed, total_steps=total)


def _save_full(agent, backend, path):
    tmp = str(path) + ".tmp"
    if backend == "stub":
        agent.save(tmp)
    else:
        from sac_hybrid import save_full
        save_full(agent, tmp)
    os.replace(tmp, path)


def _load_full(agent, backend, path):
    if backend == "stub":
        agent.load(str(path))
    else:
        from sac_hybrid import load_full
        load_full(agent, str(path))


TRAIN_COLS = ["episode", "return", "EE_kbitJ", "outage", "backlog_bits", "drop_rate", "P_mean",
              "share_M2", "share_M4", "share_M8", "share_M16", "H_d", "H_c", "alpha_c", "alpha_d",
              "loss_q", "q_mean", "steps", "updates", "seconds"]


def _train_episode(cfg, env, builder, agent, seed):
    obs, _ = env.reset(seed=seed)
    builder.on_reset(env)
    s = builder.build(obs)
    use_mask = cfg.sac_mod_masking
    mask = modulation_mask(cfg, s, cfg.sac_mask_margin_db) if use_mask else None
    acc = MetricsAccumulator(cfg)
    counts = np.zeros(len(cfg.mod_orders)); psum = 0.0
    upd = []
    t = 0
    while True:
        p_norm, m_idx = agent.act(s, deterministic=False, mask=mask)
        P = agent.p_watts(p_norm)
        obs2, r, term, trunc, info = env.step(P, cfg.mod_orders[m_idx])
        acc.add(t, info, r); counts[m_idx] += 1; psum += P
        t += 1
        if trunc and not term:
            break
        s2 = builder.build(obs2)
        mask2 = modulation_mask(cfg, s2, cfg.sac_mask_margin_db) if use_mask else None
        out = agent.observe(s, p_norm, m_idx, r, s2, term, mask2)
        if out is not None:
            upd.append(out)
        if term:
            break
        s, mask = s2, mask2
    sm = acc.summary()
    row = dict(**{k: sm[k] for k in ("EE_kbitJ", "outage", "backlog_bits", "drop_rate")},
               P_mean=psum / t, **{f"share_M{M}": counts[i] / t for i, M in enumerate(cfg.mod_orders)})
    row["return"] = sm["episode_return"]
    for k in ("H_d", "H_c", "alpha_c", "alpha_d", "loss_q", "q_mean"):
        row[k] = float(np.nanmean([u[k] for u in upd])) if upd and not all(np.isnan(u[k]) for u in upd) else float("nan")
    return row


def _read_log(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_log(path: Path, rows: List[Dict]):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_COLS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAIN_COLS})
    os.replace(tmp, path)


def train_scheme(cfg: Config, scheme: str, seed: int, paths: Paths, backend: str, predictor, ds,
                 deadline: float = float("inf")) -> str:
    d = paths.agent_dir(scheme, seed)
    if (d / "DONE").exists():
        log(f"{scheme} s{seed}: already done, skipped"); return "done"
    builder = make_builder(cfg, scheme, predictor, ds)
    agent = make_agent(cfg, backend, builder.dim, seed, scheme)
    rows = _read_log(d / "train_log.csv")
    meta_p = d / "progress.json"
    start = 0
    if (d / "ckpt.pt").exists() and meta_p.exists():
        _load_full(agent, backend, d / "ckpt.pt")
        start = json.loads(meta_p.read_text())["episodes_done"]
        rows = rows[:start]
        log(f"{scheme} s{seed}: resuming from episode {start}")
    else:
        rows = []
    env = UAVFSOEnv(cfg)
    t0 = time.time()
    for ep in range(start, cfg.train_episodes):
        te = time.time()
        row = _train_episode(cfg, env, builder, agent, seed=cfg.train_seed_base + 10_000 * seed + ep)
        row.update(episode=ep + 1, steps=agent.steps, updates=agent.updates, seconds=round(time.time() - te, 2))
        rows.append(row)
        last = ep + 1 == cfg.train_episodes
        if (ep + 1) % cfg.ckpt_every == 0 or last or time.time() > deadline:
            _save_full(agent, backend, d / "ckpt.pt")
            _write_log(d / "train_log.csv", rows)
            meta_p.write_text(json.dumps(dict(episodes_done=ep + 1)))
            log(f"{scheme} s{seed}: ep {ep+1}/{cfg.train_episodes} | return {row['return']:.1f} | "
                f"EE {row['EE_kbitJ']:.3f} | out {100*row['outage']:.1f}% | 2-PAM {100*row['share_M2']:.0f}% "
                f"| H_d {row['H_d']:.3f} | a_d {row['alpha_d']:.3f} | {time.time()-t0:.0f}s")
            if not last and time.time() > deadline:
                log(f"{scheme} s{seed}: time budget reached -> paused (re-run the same command to resume)")
                return "paused"
    agent.save(str(d / "policy.pt"))
    (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return "done"


def load_policy_agent(cfg: Config, scheme: str, seed: int, paths: Paths, backend: str, dim: int):
    agent = make_agent(cfg, backend, dim, seed, scheme)
    agent.load(str(paths.agent_dir(scheme, seed) / "policy.pt"))
    return agent


class LearnedPolicy:
    def __init__(self, cfg, builder, agent):
        self.cfg, self.builder, self.agent = cfg, builder, agent

    def on_reset(self, env):
        self.builder.on_reset(env)

    def act(self, obs):
        s = self.builder.build(obs)
        mask = modulation_mask(self.cfg, s, self.cfg.sac_mask_margin_db) if self.cfg.sac_mod_masking else None
        p, m = self.agent.act(s, deterministic=True, mask=mask)
        return self.agent.p_watts(p), self.cfg.mod_orders[m]


def evaluate(cfg: Config, policy, n_episodes: int, trace: bool = False) -> Dict:
    env = UAVFSOEnv(cfg)
    per, q_trace = [], None
    counts = np.zeros(len(cfg.mod_orders)); psum = 0.0; nslots = 0
    for k in range(n_episodes):
        obs, _ = env.reset(seed=cfg.eval_seed_base + k)
        if hasattr(policy, "on_reset"):
            policy.on_reset(env)
        acc, t, qs = MetricsAccumulator(cfg), 0, []
        while True:
            P, M = policy.act(obs)
            obs, r, te, tr, info = env.step(P, M)
            acc.add(t, info, r); t += 1
            counts[cfg.mod_orders.index(M)] += 1; psum += P; nslots += 1
            if trace and k == 0:
                qs.append(info["queue_after"])
            if te or tr:
                break
        per.append(acc.summary())
        if trace and k == 0:
            q_trace = qs
    out = {k: float(np.nanmean([e[k] for e in per])) for k in per[0]}
    out["delay_slots"] = out["backlog_bits"] / max(out["admitted_bits"], 1e-12)
    out.update({f"{k}_std": float(np.nanstd([e[k] for e in per])) for k in ("EE_kbitJ", "outage", "backlog_bits")})
    out.update({f"share_M{M}": counts[i] / nslots for i, M in enumerate(cfg.mod_orders)})
    out["P_mean"] = psum / nslots
    out["n_episodes"] = n_episodes
    if trace:
        out["_trace"] = q_trace
    return out


def _append_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


def run_main_eval(cfg: Config, paths: Paths, backend: str, schemes, seeds, predictor, ds) -> List[Dict]:
    rows, traces = [], {}
    for name, P in BASELINES.items():
        res = evaluate(cfg, P(cfg), cfg.eval_episodes, trace=True)
        traces[name] = res.pop("_trace")
        rows.append(dict(scheme=name, seed="-", **res))
        log(f"eval {name:12s}: EE {res['EE_kbitJ']:.3f} | out {100*res['outage']:.1f}% | backlog {res['backlog_bits']:.0f}")
    for sch in schemes:
        b = make_builder(cfg, sch, predictor, ds)
        for sd in seeds:
            if not (paths.agent_dir(sch, sd) / "DONE").exists():
                log(f"eval {sch} s{sd}: training not finished -> skipped"); continue
            ag = load_policy_agent(cfg, sch, sd, paths, backend, b.dim)
            res = evaluate(cfg, LearnedPolicy(cfg, b, ag), cfg.eval_episodes, trace=(sd == seeds[0]))
            tr = res.pop("_trace", None)
            if tr is not None:
                traces[sch] = tr
            rows.append(dict(scheme=sch, seed=sd, **res))
            log(f"eval {sch:12s} s{sd}: EE {res['EE_kbitJ']:.3f} | out {100*res['outage']:.1f}% | "
                f"backlog {res['backlog_bits']:.0f} | 2-PAM {100*res['share_M2']:.0f}%")
    _append_csv(paths.eval / "main.csv", rows)
    tr_rows = [dict(slot=t + 1, **{k: v[t] for k, v in traces.items()}) for t in range(cfg.episode_slots)]
    _append_csv(paths.eval / "backlog_trace.csv", tr_rows)
    return rows


def convergence_episode(returns, window: int, frac: float) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 * window:
        return float("nan")
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    final = r[-window:].mean()
    lo = ma.min()
    if final <= lo:
        return float("nan")
    ok = ma >= lo + frac * (final - lo)
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= window:
            return float(i - window + 1 + window)
    return float("nan")


def run_sensitivity(cfg: Config, paths: Paths, backend: str, seeds, predictor, ds, schemes=("sac_only", "original_design")):
    rows = []
    schemes = list(schemes) + [s for s in ("proposed_beta099", "sac_lstm", "proposed") if s not in schemes and any((paths.agent_dir(s, d) / "DONE").exists() for d in seeds)]
    conds = [("turbulence", lab, dict(gg_alpha=a, gg_beta=b)) for lab, a, b in cfg.sens_turbulence]
    conds += [("jitter_urad", j, dict(jitter_urad=j)) for j in cfg.sens_jitter_urad]
    conds += [("delay_slots", D, dict(csi_delay=D)) for D in cfg.sens_delay]
    for axis, level, ov in conds:
        c2 = with_overrides(cfg, **ov)
        if axis == "delay_slots":
            pred2, ds2, _ = get_predictor(c2, paths, backend)
        else:
            pred2, ds2 = predictor, ds
        for name in ("Threshold", "Perfect-CSI"):
            res = evaluate(c2, BASELINES[name](c2), cfg.sens_episodes)
            rows.append(dict(axis=axis, level=level, scheme=name, seed="-", **res))
        for sch in schemes:
            b = make_builder(c2, sch, pred2, ds2)
            for sd in seeds:
                if not (paths.agent_dir(sch, sd) / "DONE").exists():
                    continue
                ag = load_policy_agent(cfg, sch, sd, paths, backend, b.dim)
                res = evaluate(c2, LearnedPolicy(c2, b, ag), cfg.sens_episodes)
                rows.append(dict(axis=axis, level=level, scheme=sch, seed=sd, **res))
        log(f"sensitivity {axis}={level}: done")
    _append_csv(paths.eval / "sensitivity.csv", rows)
    return rows


def measure_overhead(cfg: Config, paths: Paths, backend: str, predictor, ds, seed: int = 0):
    if backend != "torch":
        return None
    b = make_builder(cfg, "original_design", predictor, ds)
    if not (paths.agent_dir("original_design", seed) / "DONE").exists():
        return None
    ag = load_policy_agent(cfg, "original_design", seed, paths, backend, b.dim)
    env = UAVFSOEnv(cfg); obs, _ = env.reset(seed=cfg.eval_seed_base)
    from channel import normalize_snr_db
    from it2fls import normalize_uncertainty
    X = normalize_snr_db(cfg, obs["snr_hist_db"])[None, :]
    n = 500

    def tm(f):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        return (time.perf_counter() - t0) / n * 1e3
    s = b.build(obs)
    res = dict(lstm_ms=tm(lambda: predictor.predict_batch(X)),
               it2fls_ms=tm(lambda: b.it2.confidence(0.5, 0.2)),
               actor_ms=tm(lambda: ag.act(s, deterministic=True)),
               )
    env.reset(seed=1)
    res["env_step_ms"] = tm(lambda: env.step(1.0, 2))
    res["online_total_ms"] = res["lstm_ms"] + res["it2fls_ms"] + res["actor_ms"]
    res["actor_params"] = ag.n_params()["actor"]
    (paths.eval / "overhead.json").write_text(json.dumps(res, indent=1))
    log("overhead (ms/slot): " + ", ".join(f"{k}={v:.3f}" for k, v in res.items() if k.endswith("ms")))
    return res


def ee_bound(cfg):
    return ee_upper_bound_kbitJ(cfg)
