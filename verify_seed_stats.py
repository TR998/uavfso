from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

METRICS = (("EE_kbitJ", "EE (kbits/J)", 3), ("outage_pct", "Outage (%)", 2), ("backlog_bits", "Backlog (bits)", 0))


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.seed.astype(str) != "-"].copy()
    d["seed"] = d.seed.astype(int)
    d["outage_pct"] = 100 * d.outage
    return d.drop_duplicates(["scheme", "seed"], keep="last")


def marginal_table(d: pd.DataFrame, max_seed: int) -> pd.DataFrame:
    rows = []
    for s, g in d[d.seed < max_seed].groupby("scheme"):
        r = {"scheme": s, "n": len(g)}
        for c, lab, k in METRICS:
            r[lab] = f"{g[c].mean():.{k}f} ± {g[c].std(ddof=1):.{k}f}"
        rows.append(r)
    return pd.DataFrame(rows)


def paired_table(d: pd.DataFrame, ref: str = "sac_only") -> pd.DataFrame:
    base = d[d.scheme == ref].set_index("seed")
    rows = []
    for s, g in d[d.scheme != ref].groupby("scheme"):
        g = g.set_index("seed")
        common = sorted(set(g.index) & set(base.index))
        for idx in sorted({tuple(c for c in common if c < 3), tuple(common)}, key=len):
            if len(idx) < 2:
                continue
            idx = list(idx)
            r = {"scheme": s, "n": len(idx)}
            for c, lab, k in METRICS[:2]:
                a, b = g.loc[idx, c].to_numpy(), base.loc[idx, c].to_numpy()
                dd = a - b
                t, p = stats.ttest_rel(a, b)
                r[f"d {lab}"] = f"{dd.mean():+.{k}f} ± {dd.std(ddof=1):.{k}f}"
                r[f"t, p ({lab})"] = f"{t:+.2f}, {p:.3f}"
                r[f"SD bound ({lab})"] = f"{dd.std(ddof=1):.4f} <= {a.std(ddof=1) + b.std(ddof=1):.4f}"
            r["lower outage"] = f"{int((g.loc[idx, 'outage_pct'] < base.loc[idx, 'outage_pct']).sum())}/{len(idx)}"
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="results/default/main_per_seed.csv")
    ap.add_argument("--max-seed", type=int, default=3)
    a = ap.parse_args()
    d = load(a.csv)
    pd.set_option("display.width", 250)
    print("Mean ± sample standard deviation over training seeds")
    print(marginal_table(d, a.max_seed).to_string(index=False))
    print("\nPaired differences relative to SAC only")
    print(paired_table(d).T.to_string(header=False))


if __name__ == "__main__":
    main()
