from __future__ import annotations

import argparse
import json
import sys
import time

from config import get_config
import experiment as X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=["quick", "medium", "full"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="torch", choices=["torch", "stub"])
    ap.add_argument("--steps", nargs="*", default=["lstm", "train", "eval", "sens", "overhead", "report"])
    ap.add_argument("--schemes", nargs="*", default=list(X.LEARNED))
    ap.add_argument("--with-perfect-sac", action="store_true", help="also train SAC with perfect CSI")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: 0..n_seeds-1 of the mode")
    ap.add_argument("--time-budget-min", type=float, default=None, help="pause training after N minutes (resume later)")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                    help="agent-side config override, e.g. sac_gamma=0.95")
    ap.add_argument("--scenario", default="default", choices=list(X.SCENARIOS),
                    help="environment scenario")
    ap.add_argument("--tag", default="", help="job label")
    ap.add_argument("--allow-partial", action="store_true", help="evaluate even if some training runs are unfinished")
    a = ap.parse_args()

    import ast
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            ov[k] = v
    cfg = get_config(a.mode, **ov, **X.SCENARIOS[a.scenario])
    out = a.out or f"runs/{a.mode}_{a.backend}"
    paths = X.Paths(out)
    seeds = a.seeds if a.seeds is not None else list(range(cfg.n_seeds))
    schemes = list(a.schemes) + (["perfect_sac"] if a.with_perfect_sac else [])
    deadline = time.time() + 60 * a.time_budget_min if a.time_budget_min else float("inf")
    if a.backend == "torch":
        try:
            import torch
            X.log(f"torch {torch.__version__} | threads {torch.get_num_threads()} | CUDA {torch.cuda.is_available()}")
        except ImportError:
            sys.exit("PyTorch not found. Install torch, or use --backend stub to test the pipeline only.")
    X.log(f"mode={cfg.mode} backend={a.backend} out={out} schemes={schemes} seeds={seeds} "
          f"episode={cfg.train_episodes} eval={cfg.eval_episodes} sens={cfg.sens_episodes}")
    (paths.root / "run_config.json").write_text(json.dumps(dict(mode=cfg.mode, backend=a.backend, schemes=schemes,
                                                                seeds=seeds, train_episodes=cfg.train_episodes,
                                                                reward_ber_delivered_only=cfg.reward_ber_delivered_only,
                                                                tau_c_ms=cfg.tau_c_ms, n_symbols_per_slot=cfg.n_symbols_per_slot,
                                                                overrides=ov, scenario=a.scenario),
                                                           indent=1))
    t0 = time.time()
    predictor, ds, meta = X.get_predictor(cfg, paths, a.backend)

    if "train" in a.steps:
        status = {}
        for sd in seeds:
            for sch in schemes:
                status[(sch, sd)] = X.train_scheme(cfg, sch, sd, paths, a.backend, predictor, ds, deadline)
                if status[(sch, sd)] == "paused":
                    break
            if "paused" in status.values():
                break
        pending = [k for k in [(s, d) for d in seeds for s in schemes]
                   if not (paths.agent_dir(*k) / "DONE").exists()]
        if pending:
            X.log(f"UNFINISHED: {pending}. Re-run the same command to resume.")
            if not a.allow_partial:
                X.log(f"Stopped before evaluation ({(time.time()-t0)/60:.1f} min)."); return
    if "eval" in a.steps:
        X.run_main_eval(cfg, paths, a.backend, schemes, seeds, predictor, ds)
    if "sens" in a.steps:
        X.run_sensitivity(cfg, paths, a.backend, seeds, predictor, ds)
    if "overhead" in a.steps:
        X.measure_overhead(cfg, paths, a.backend, predictor, ds, seed=seeds[0])
    if "report" in a.steps:
        from report import make_report
        cmp = make_report(cfg, out, a.backend)
        if cmp is not None:
            print("\n" + cmp.to_string(index=False))
            X.log(f"Report: {paths.root / 'RESULTS.md'} | figures: {paths.figures}")
    X.log(f"Finished in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
