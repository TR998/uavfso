from __future__ import annotations

import argparse
import os
import subprocess
import sys

from experiment import FINAL_SCHEMES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=["quick", "medium", "full"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="torch", choices=["torch", "stub"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--schemes", nargs="*", default=list(FINAL_SCHEMES))
    ap.add_argument("--scenario", default="default", choices=["default", "hover_vib"])
    ap.add_argument("--eval-only", action="store_true", help="skip training; use the models in <out>/agents")
    ap.add_argument("--no-sens", action="store_true", help="skip the sensitivity analysis")
    ap.add_argument("--time-budget-min", type=float, default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "runs", f"{a.scenario}_{a.mode}_{a.backend}")
    steps = ["lstm"] + ([] if a.eval_only else ["train"]) + ["eval"] + ([] if a.no_sens else ["sens"]) + ["overhead", "report"]
    cmd = [sys.executable, os.path.join(HERE, "run_all.py"), "--mode", a.mode, "--out", out, "--backend", a.backend,
           "--scenario", a.scenario, "--schemes", *a.schemes, "--steps", *steps]
    if a.seeds is not None:
        cmd += ["--seeds", *map(str, a.seeds)]
    if a.time_budget_min:
        cmd += ["--time-budget-min", str(a.time_budget_min)]
    if a.eval_only:
        cmd += ["--allow-partial"]
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit("run_all.py failed")
    if os.path.exists(os.path.join(out, "RESULTS.md")):
        print(f"\nReport: {out}/RESULTS.md")


if __name__ == "__main__":
    main()
