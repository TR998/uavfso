from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

from scipy.special import erf


@dataclass
class Config:
    mode: str = "quick"

    altitude_m: float = 100.0
    horizontal_m: float = 1000.0

    alpha_att_per_km: float = 0.43
    gg_alpha: float = 4.0
    gg_beta: float = 1.9
    tau_c_ms: float = 10.0
    aperture_radius_m: float = 0.01
    beam_waist_m: float = 0.025
    jitter_urad: float = 5.0
    tau_pe_ms: float = 20.0
    sigma_aoa_rad: float = 1e-3
    fov_rad: float = 3.5e-3
    tau_aoa_ms: float = 20.0

    hover_vib_enable: bool = False
    hover_vib_amp_urad: float = 16.0
    hover_vib_freq_hz: float = 150.0
    hover_vib_freq_dev: float = 0.2
    hover_vib_tau_f_ms: float = 200.0

    responsivity: float = 0.5
    snr_ref_db: float = 33.2

    mod_orders: Tuple[int, ...] = (2, 4, 8, 16)
    ber_max: float = 1e-9
    n_symbols_per_slot: int = 333

    slot_s: float = 1e-3
    csi_delay: int = 2

    p_min_w: float = 0.1
    p_max_w: float = 1.0
    p_circuit_w: float = 5.0
    p_hover_w: float = 150.0

    battery_j: float = 1.8e6
    battery_init_frac: Tuple[float, float] = (0.6, 1.0)

    arrival_bits_per_slot: float = 500.0
    q_max_bits: float = 5000.0
    q_th_bits: float = 4000.0
    q0_bits: float = 250.0

    w_ee: float = 1.0
    w_out: float = 2.0
    w_ber: float = 1.0
    w_queue: float = 0.8
    eps_reward: float = 1e-9
    ber_term_clip: float = 10.0

    reward_ber_delivered_only: bool = True

    snr_norm_min_db: float = 0.0
    snr_norm_max_db: float = 50.0

    episode_slots: int = 1000
    lstm_window: int = 8
    transient_slots: int = 100
    eval_seed_base: int = 10000

    trace_slots: int = 100_000
    sweep_episodes: int = 10
    sanity_slots: int = 400_000

    lstm_hidden: Tuple[int, int] = (64, 32)
    lstm_dropout: float = 0.20
    lstm_lr: float = 1e-3
    lstm_batch: int = 64
    lstm_epochs: int = 150
    lstm_patience: int = 10

    it2_mf_snr: Tuple[Tuple[float, float, float], ...] = ((0.20, 0.12, 0.04), (0.50, 0.10, 0.03), (0.80, 0.12, 0.04))
    it2_mf_unc: Tuple[Tuple[float, float, float], ...] = ((0.15, 0.10, 0.04), (0.45, 0.10, 0.04), (0.75, 0.12, 0.05))
    it2_out_centers: Tuple[float, float, float] = (0.25, 0.55, 0.85)

    it2_rules: Tuple[Tuple[int, int, int], ...] = ((1, 2, 2), (0, 1, 1), (0, 0, 0))
    it2_e_percentile: float = 99.0
    state_append_conf: bool = False

    sac_actor_hidden: Tuple[int, int] = (128, 64)
    sac_critic_hidden: Tuple[int, int] = (128, 128)
    sac_lr_actor: float = 3e-4
    sac_lr_critic: float = 3e-4
    sac_lr_alpha: float = 3e-4
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_batch: int = 256
    sac_buffer: int = 1_000_000
    sac_warmup_steps: int = 5000
    sac_updates_per_step: int = 1
    sac_log_std_bounds: Tuple[float, float] = (-5.0, 2.0)

    sac_target_entropy_cont: float = -2.0
    sac_target_entropy_disc_frac: float = 0.2
    sac_alpha_max: float = 0.2

    sac_discrete_mode: str = "expected"
    sac_gumbel_tau: Tuple[float, float] = (0.5, 0.1)
    sac_gumbel_anneal_frac: float = 0.5
    sac_init_alpha: float = 0.2
    sac_mod_masking: bool = False
    sac_mask_margin_db: float = 0.0
    train_episodes: int = 100

    n_seeds: int = 1
    eval_episodes: int = 10
    sens_episodes: int = 5
    ckpt_every: int = 10
    train_seed_base: int = 100_000
    convergence_window: int = 50
    convergence_frac: float = 0.99
    sens_turbulence: Tuple[Tuple[str, float, float], ...] = (("Weak", 11.6, 10.1), ("Moderate", 4.0, 1.9), ("Strong", 4.2, 1.4))
    sens_jitter_urad: Tuple[float, ...] = (3.0, 5.0, 10.0)
    sens_delay: Tuple[int, ...] = (0, 2, 4)

    derived: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        d = {}
        L = math.hypot(self.altitude_m, self.horizontal_m)
        d["link_m"] = L
        d["h_l"] = math.exp(-self.alpha_att_per_km * L / 1000.0)
        v = math.sqrt(math.pi) * self.aperture_radius_m / (math.sqrt(2.0) * self.beam_waist_m)
        d["v"] = v
        d["A0"] = erf(v) ** 2
        d["wzeq2"] = (self.beam_waist_m ** 2 * math.sqrt(math.pi) * erf(v)
                      / (2.0 * v * math.exp(-v ** 2)))
        d["sigma_s_m"] = self.jitter_urad * 1e-6 * L
        d["xi2"] = d["wzeq2"] / (4.0 * d["sigma_s_m"] ** 2)
        d["h_ref"] = d["h_l"] * d["A0"]
        d["snr_ref_lin"] = 10 ** (self.snr_ref_db / 10.0)
        d["noise_var"] = (self.responsivity * self.p_max_w * d["h_ref"]) ** 2 / d["snr_ref_lin"]
        d["bandwidth_hz"] = self.n_symbols_per_slot / self.slot_s
        d["N0"] = d["noise_var"] / d["bandwidth_hz"]
        d["rho_c"] = math.exp(-self.slot_s * 1e3 / self.tau_c_ms)
        d["rho_pe"] = math.exp(-self.slot_s * 1e3 / self.tau_pe_ms)
        d["rho_aoa"] = math.exp(-self.slot_s * 1e3 / self.tau_aoa_ms)
        d["e_min_slot_j"] = (self.p_min_w + self.p_circuit_w + self.p_hover_w) * self.slot_s
        d["eta_ref"] = self.arrival_bits_per_slot / d["e_min_slot_j"]
        d["target_entropy_disc"] = self.sac_target_entropy_disc_frac * math.log(len(self.mod_orders))
        self.derived = d

    def __getitem__(self, k):
        return self.derived[k]


_MODE_SIZES = {
    "quick":  dict(trace_slots=100_000, sweep_episodes=10, sanity_slots=400_000,
                   lstm_epochs=30, train_episodes=40, n_seeds=1, eval_episodes=10, sens_episodes=5,
                   ckpt_every=10, convergence_window=10),
    "medium": dict(trace_slots=300_000, sweep_episodes=30, sanity_slots=1_000_000,
                   lstm_epochs=150, train_episodes=500, n_seeds=3, eval_episodes=50, sens_episodes=20,
                   ckpt_every=25),
    "full":   dict(trace_slots=500_000, sweep_episodes=100, sanity_slots=2_000_000,
                   lstm_epochs=150, train_episodes=800, n_seeds=5, eval_episodes=100, sens_episodes=50,
                   ckpt_every=25),
}


def get_config(mode: str = "quick", **override) -> Config:
    if mode not in _MODE_SIZES:
        raise ValueError(f"mode must be one of {list(_MODE_SIZES)}")
    kw = dict(_MODE_SIZES[mode])
    kw.update(override)
    return Config(mode=mode, **kw)


def with_overrides(cfg: Config, **override) -> Config:
    base = {k: v for k, v in asdict(cfg).items() if k != "derived"}
    base.update(override)
    return Config(**base)


if __name__ == "__main__":
    c = get_config("quick")
    for k, v in c.derived.items():
        print(f"{k:14s} = {v:.6g}")
