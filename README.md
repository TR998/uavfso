# UAV-FSO link adaptation with LSTM, IT2FLS and hybrid SAC

Simulation code for energy-efficient link adaptation of a hovering UAV-assisted free-space optical (FSO) link under delayed CSI. The controller combines an LSTM SNR predictor, an interval type-2 fuzzy logic system (IT2FLS) and a hybrid-action soft actor-critic (SAC) agent (continuous transmit power, discrete PAM order).

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

A CPU is sufficient.

## Reproducing the results

The `medium` mode is the protocol used in the paper: 500 training episodes of 1000 slots per seed, training seeds 0-2, 50 deterministic test episodes (seeds 10000-10049) shared by all schemes, and 20 test episodes per condition in the sensitivity analysis.

| Paper item | Command |
|---|---|
| Tables 8-11, Fig. 3, Fig. 4 | `python main.py --mode medium --out runs/default` |
| Proposed state with beta = 0.99 (Tables 8-12, Figs. 5-7) | `python main.py --mode medium --out runs/default --schemes proposed_beta099` |
| Five-seed rows of Table 12 | `python main.py --mode medium --out runs/default --seeds 0 1 2 3 4 --schemes sac_only sac_lstm proposed_beta099` |
| Figs. 5-7 (held-out conditions) | produced by the commands above (sensitivity step) |
| Table 13 (rotor-vibration family) | `python main.py --mode medium --scenario hover_vib --out runs/hover_vib --no-sens` |
| Tuned UA-threshold (72 configurations) | `python ua_threshold.py --scenario default` and `python ua_threshold.py --scenario hover_vib` |

Training is resumed automatically from the last checkpoint when a command is repeated. Training can be split over several processes, for example one process per seed with `--seeds 0`, `--seeds 1` and `--seeds 2`, followed by the same command with `--eval-only`.

`python main.py --backend stub` runs the complete pipeline with placeholder agents in a few seconds; it only tests the code path.

## Schemes

| Name | Description |
|---|---|
| `proposed` | LSTM + IT2FLS + hybrid SAC, state [c·γ̃, c, Q/Qmax, E/Ebat], discount factor 0.95 |
| `proposed_beta099` | same state, discount factor 0.99 |
| `original_design` | state [c·γ̃, Q/Qmax, E/Ebat] (confidence not observable) |
| `sac_lstm` | SAC with the LSTM prediction |
| `sac_it2` | SAC with the confidence-weighted delayed SNR |
| `sac_only` | SAC with the delayed SNR |
| `Heuristic`, `Threshold`, `Perfect-CSI` | rule-based baselines and the perfect-CSI reference |

All parameters are in `config.py`.

## Output

Each run writes to its output directory: `RESULTS.md`, `tables/`, `figures/`, `eval/main.csv` (one row per scheme and training seed), `eval/sensitivity.csv`, and `agents/<scheme>_s<seed>/` (model checkpoints and per-episode training logs).

## Released results

The `results/` folder contains the outputs used in the paper:

- `default/main_per_seed.csv`: per-seed test metrics at the default condition (seeds 0-2; seeds 0-4 for `sac_only` and `sac_lstm`).
- `default/sensitivity_per_seed.csv`: per-seed metrics under the held-out conditions.
- `default/proposed_beta099_seeds0-2.csv`, `default/paired_vs_sac_only_seeds0-4.csv`: summary rows for the beta = 0.99 variant and the five-seed comparisons.
- `default/ua_threshold_default.csv`, `hover_vib/ua_threshold_hover_vib.csv`: all 72 UA-threshold configurations on the 50 test episodes.
- `hover_vib/summary_seeds0-2.csv`: rotor-vibration family (Table 13).

Means in Tables 8-11 are computed over seeds 0-2.
