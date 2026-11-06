from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from experiment import DISPLAY, convergence_episode, ee_bound

ORDER = ["Heuristic", "Threshold", "sac_only", "sac_it2", "sac_lstm", "original_design",
         "proposed_beta099", "proposed_hist", "proposed", "perfect_sac", "Perfect-CSI"]
T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _ms(m, s, nd=3, k=1.0):
    if s is None or np.isnan(s) or s == 0:
        return f"{m * k:.{nd}f}"
    return f"{m * k:.{nd}f} ± {s * k:.{nd}f}"


def aggregate(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sch, g in main.groupby("scheme", sort=False):
        r = {"scheme": sch, "n_seeds": len(g)}
        for c in ("EE_kbitJ", "EE_nohover_kbitJ", "outage", "BER_deliv", "BER_all", "backlog_bits",
                  "delay_slots", "drop_rate", "share_M2", "P_mean"):
            r[c] = g[c].mean()
            r[c + "_std"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(r)
    a = pd.DataFrame(rows).set_index("scheme")
    return a.reindex([s for s in ORDER if s in a.index])


def main_table(A: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Scheme": DISPLAY.get(s, s), "n": int(a.n_seeds),
        "EE (kbits/J)": _ms(a.EE_kbitJ, a.EE_kbitJ_std),
        "Outage (%)": _ms(a.outage, a.outage_std, 2, 100),
        "BER_deliv": f"{a.BER_deliv:.1e}", "BER_all": f"{a.BER_all:.1e}",
        "Backlog (bits)": _ms(a.backlog_bits, a.backlog_bits_std, 0),
        "Delay (slots)": f"{a.delay_slots:.2f}", "Drop (%)": f"{100 * a.drop_rate:.1f}",
        "EE without hover (kbits/J)": f"{a.EE_nohover_kbitJ:.1f}"} for s, a in A.iterrows()])


def paired_table(main: pd.DataFrame, ref: str = "sac_only") -> pd.DataFrame:
    m = main[main.seed.astype(str) != "-"].copy()
    m["seed"] = m.seed.astype(int)
    r = m[m.scheme == ref].set_index("seed")
    rows = []
    for sch, g in m.groupby("scheme", sort=False):
        if sch == ref:
            continue
        g = g.set_index("seed")
        common = [s for s in g.index if s in r.index]
        if len(common) < 2:
            continue
        dee = (g.loc[common, "EE_kbitJ"] - r.loc[common, "EE_kbitJ"]).values
        dout = 100 * (g.loc[common, "outage"] - r.loc[common, "outage"]).values
        n = len(common)
        sd = dout.std(ddof=1)
        t = dout.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
        rows.append({"Scheme": DISPLAY.get(sch, sch), "n": n, "seeds": " ".join(map(str, common)),
                     "dEE (kbits/J)": f"{dee.mean():+.3f} ± {dee.std(ddof=1):.3f}",
                     "dOutage (points)": f"{dout.mean():+.2f} ± {sd:.2f}",
                     "lower outage": f"{(dout < 0).sum()}/{n}", "t": round(t, 2),
                     "significant (alpha=0.05)": bool(abs(t) > T_CRIT.get(n - 1, 1.96))})
    return pd.DataFrame(rows)


def convergence_table(root: Path, cfg: Config) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(root / "agents" / "*" / "train_log.csv"))):
        sch, sd = Path(f).parent.name.rsplit("_s", 1)
        df = pd.read_csv(f)
        if len(df):
            rows.append(dict(scheme=sch, seed=int(sd), episodes=len(df),
                             convergence_episode=convergence_episode(df["return"].values, cfg.convergence_window,
                                                                     cfg.convergence_frac)))
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    return c.groupby("scheme", sort=False)["convergence_episode"].agg(["mean", "count"]).reset_index()


def sensitivity_table(sens: pd.DataFrame) -> pd.DataFrame:
    g = sens.groupby(["axis", "level", "scheme"], sort=False).agg(
        EE_kbitJ=("EE_kbitJ", "mean"), outage=("outage", "mean"), n_seeds=("seed", "nunique")).reset_index()
    g["outage_pct"] = (100 * g.outage).round(2)
    g["scheme"] = g.scheme.map(lambda s: DISPLAY.get(s, s))
    return g.drop(columns="outage")


def figures(cfg: Config, root: Path, logs: dict, trace, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = root / "figures"
    F.mkdir(exist_ok=True)
    if logs:
        w = max(1, cfg.convergence_window // 2)
        sm = {}
        for sch, lst in logs.items():
            L = min(len(x) for x in lst)
            sm[sch] = np.array([np.convolve(x[:L], np.ones(w) / w, mode="valid") for x in lst])
        lo = min(v.min() for v in sm.values()); hi = max(v.max() for v in sm.values())
        plt.figure(figsize=(6, 4))
        for sch, v in sm.items():
            nm = 2 * (v - lo) / (hi - lo + 1e-12) - 1
            mu, sd = nm.mean(0), nm.std(0, ddof=1)
            x = np.arange(len(mu)) + w
            plt.plot(x, mu, lw=1.2, label=DISPLAY.get(sch, sch)); plt.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        plt.xlabel("Training episode"); plt.ylabel("Normalized return"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(F / "learning_curves.png", dpi=300); plt.close()
    if trace is not None and not trace.empty:
        plt.figure(figsize=(6, 4))
        for c in [c for c in trace.columns if c != "slot"]:
            plt.plot(trace["slot"], trace[c], lw=1, label=DISPLAY.get(c, c))
        plt.xlabel("Time slot"); plt.ylabel("Backlog (bits)"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(F / "backlog_trace.png", dpi=300); plt.close()
    if sens is not None and not sens.empty:
        for axn, xl in (("turbulence", "Turbulence regime"), ("jitter_urad", "Pointing jitter (µrad)"),
                        ("delay_slots", "CSI delay (slots)")):
            sub = sens[sens.axis == axn]
            if sub.empty:
                continue
            plt.figure(figsize=(6, 4))
            for sch, s in sub.groupby("scheme", sort=False):
                s = s.groupby("level", sort=False).outage.mean()
                plt.plot([str(v) for v in s.index], 100 * s.values, marker="o", label=DISPLAY.get(sch, sch))
            plt.xlabel(xl); plt.ylabel("Outage probability (%)"); plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(F / f"sensitivity_{axn}.png", dpi=300); plt.close()


def make_report(cfg: Config, out: str, backend: str):
    root = Path(out)
    mp = root / "eval" / "main.csv"
    if not mp.exists():
        print("eval/main.csv not found; run the eval step first")
        return None
    T = root / "tables"
    T.mkdir(exist_ok=True)
    main = pd.read_csv(mp)
    A = aggregate(main)
    A.to_csv(T / "main_aggregated.csv")
    mt = main_table(A); mt.to_csv(T / "main_table.csv", index=False)
    pt = paired_table(main); pt.to_csv(T / "paired_vs_sac_only.csv", index=False)
    conv = convergence_table(root, cfg); conv.to_csv(T / "convergence.csv", index=False)
    sp = root / "eval" / "sensitivity.csv"
    sens = pd.read_csv(sp) if sp.exists() else None
    if sens is not None:
        sensitivity_table(sens).to_csv(T / "sensitivity.csv", index=False)
    tp = root / "eval" / "backlog_trace.csv"
    trace = pd.read_csv(tp) if tp.exists() else None
    logs = {}
    for f in sorted(glob.glob(str(root / "agents" / "*" / "train_log.csv"))):
        df = pd.read_csv(f)
        if len(df):
            logs.setdefault(Path(f).parent.name.rsplit("_s", 1)[0], []).append(df["return"].values)
    figures(cfg, root, logs, trace, sens)
    with open(root / "RESULTS.md", "w") as f:
        if backend == "stub":
            f.write("> Stub backend: placeholder agents for pipeline testing only; numbers are not results.\n\n")
        f.write(f"# Results (mode {cfg.mode})\n\nEE upper bound: {ee_bound(cfg):.4f} kbits/J\n\n")
        f.write("## Main comparison (mean ± std over training seeds)\n\n" + mt.to_markdown(index=False) + "\n\n")
        if not pt.empty:
            f.write("## Paired seed-level differences relative to SAC only\n\n" + pt.to_markdown(index=False) + "\n\n")
        if not conv.empty:
            f.write("## Convergence episode (mean over seeds)\n\n" + conv.round(1).to_markdown(index=False) + "\n\n")
        for p in sorted(glob.glob(str(root / "lstm" / "*.json"))):
            c = json.load(open(p)).get("compare")
            if c:
                f.write(f"- {Path(p).stem}: test MSE persistence {c['mse_persistence']:.5f}, "
                        f"linear AR {c['mse_linear_AR']:.5f}, LSTM {c['mse_lstm']:.5f}\n")
        oh = root / "eval" / "overhead.json"
        if oh.exists():
            d = json.load(open(oh))
            f.write("\nOnline inference (ms/slot): " + ", ".join(f"{k} = {v:.3f}" for k, v in d.items()
                                                              if k.endswith("ms")) + "\n")
    return mt
