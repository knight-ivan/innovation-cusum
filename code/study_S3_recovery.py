#!/usr/bin/env python3
# =============================================================================
#  study_S3_recovery.py
#  Study S3: Recovery of I_t under non-stationarity
#
#  Compares four I_t estimators on non-stationary sequences:
#    (0) Oracle      — true I_t(u) from the data-generating process (upper bound)
#    (1) Naive       — constant |phi_center|  (lower bound: no adaptation)
#    (2) Forget-gate — LSTM forget gate (trained on stationary data)
#    (3) Bootstrap   — block-bootstrap surrogate (Chang 2026, JRSS-B)
#    (4) KLIEP+corr  — Pathway B: forget-gate + KLIEP density-ratio correction
#
#  Evaluation metric:
#    NMSE_t = MSE(hat_I_t, I_t^true) / Var(I_t^true)
#    averaged over time steps (excluding the KLIEP burn-in window).
#
#  Two non-stationary data-generating processes:
#    (A) Locally stationary AR  — smooth drift in phi(u)
#    (B) Change-point AR        — abrupt shift at T/2
#
#  Replications: N_RECOVERY (500) sequences for each DGP.
#  Results: results/S3_recovery.csv
#  Figures: figures/S3_*.png
#
#  Note: This study is reported in Supplementary Material Section S5.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os, time
from tqdm import tqdm

from config import (MASTER_SEED, N_RECOVERY, N_TRAIN_SEQ, T_TRAIN, T_TEST,
                    LS_PHI_CENTER, LS_PHI_AMP, CP_PHI_BEFORE, CP_PHI_AFTER,
                    CP_FRACTION, HIDDEN_DIM, KLIEP_WINDOW,
                    BB_BLOCK_LEN, BB_N_BOOT, RESULTS_DIR, FIGURES_DIR)
from data_generators import (generate_ar1, generate_locally_stationary_ar,
                              generate_changepoint_ar, phi_smooth)
from analytical_It import ls_ar_It, ar1_stationary_It
from models import InstrumentedLSTM
from training import train_model, extract_forget_gates, extract_cell_states
from kliep import RollingKLIEP, corrected_delta
from bootstrap_It import bootstrap_It, rolling_window_It


plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


# ─────────────────────────────────────────────────────────────────────────────
#  Core: compute all five I_t estimates for one sequence
# ─────────────────────────────────────────────────────────────────────────────

def estimate_It_all(x: np.ndarray,
                    lstm: InstrumentedLSTM,
                    kliep_model: RollingKLIEP,
                    phi_vals: np.ndarray,
                    phi_center: float,
                    alpha: float = 0.5) -> dict:
    """
    Compute all five I_t estimates for a single scalar sequence x of length T.

    Returns dict with keys: oracle, naive, fg, bootstrap, kliep_corr.
    All arrays have length T (nan where unavailable).
    """
    T = len(x)

    # 0. Oracle
    It_oracle = ls_ar_It(phi_vals)                    # (T,)

    # 1. Naive constant
    It_naive  = np.full(T, ar1_stationary_It(phi_center))

    # 2. Forget-gate proxy (mean over hidden dimension)
    fgs    = extract_forget_gates(lstm, x[None, :])   # (1, T, d)
    fg_seq = fgs[0].mean(axis=-1)                     # (T,)
    # Scale: multiply by stationary std to match I_t units
    scale  = ar1_stationary_It(phi_center) / max(fg_seq.mean(), 1e-10)
    It_fg  = fg_seq * scale

    # 3. Block-bootstrap surrogate (Chang 2026, JRSS-B)
    It_boot = np.full(T, np.nan)
    boot_vals = bootstrap_It(x, predictor_fit_fn=None,
                             block_len=BB_BLOCK_LEN, B=BB_N_BOOT,
                             seed=None)                # (T-1,)
    It_boot[1:] = boot_vals

    # 4. KLIEP-corrected (Pathway B)
    kliep_out = kliep_model.fit_transform(x)
    Et        = kliep_out['bias']                      # (T,) information-tracking bias
    # Replace nan in Et with 0 (no correction during burn-in)
    Et_filled = np.where(np.isfinite(Et), Et, 0.0)
    # Scale Et to I_t units
    Et_scale  = np.nanstd(It_oracle) / max(np.std(Et_filled[np.isfinite(Et)]), 1e-10) \
                if np.any(np.isfinite(Et)) else 1.0
    It_corr   = corrected_delta(It_fg, Et_filled * Et_scale, alpha=alpha)

    return {
        'oracle':     It_oracle,
        'naive':      It_naive,
        'fg':         It_fg,
        'bootstrap':  It_boot,
        'kliep_corr': It_corr,
    }


def compute_nmse(It_hat: np.ndarray, It_true: np.ndarray,
                 start: int = 0) -> float:
    """
    Normalised MSE = MSE(hat, true) / Var(true) on time steps [start:].
    """
    h = It_hat[start:]
    t = It_true[start:]
    mask = np.isfinite(h) & np.isfinite(t)
    if mask.sum() < 2:
        return np.nan
    mse  = float(np.mean((h[mask] - t[mask]) ** 2))
    var  = float(np.var(t[mask]))
    return mse / max(var, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
#  DGP A: locally stationary AR
# ─────────────────────────────────────────────────────────────────────────────

def run_dgp_A(lstm: InstrumentedLSTM,
              n_rep: int = N_RECOVERY) -> dict:
    """
    Run n_rep replications on locally stationary AR sequences.
    Returns dict of NMSE arrays, one per estimator.
    """
    kliep = RollingKLIEP(window=KLIEP_WINDOW)
    phi_c = LS_PHI_CENTER
    burn  = KLIEP_WINDOW                      # skip KLIEP burn-in

    nmse = {k: [] for k in ['naive', 'fg', 'bootstrap', 'kliep_corr']}
    example_results = None

    for rep in tqdm(range(n_rep), desc="  DGP A", leave=False):
        x_ls, phi_vals = generate_locally_stationary_ar(
            1, T_TEST, seed=MASTER_SEED + 300 + rep)
        x = x_ls[0]

        estimates = estimate_It_all(x, lstm, kliep, phi_vals, phi_c)
        It_oracle = estimates['oracle']

        for key in nmse:
            nmse[key].append(compute_nmse(estimates[key], It_oracle, start=burn))

        if rep == 0:
            example_results = {k: estimates[k].copy() for k in estimates}
            example_oracle  = It_oracle.copy()

    return {k: np.array(nmse[k]) for k in nmse}, example_results, example_oracle


# ─────────────────────────────────────────────────────────────────────────────
#  DGP B: abrupt change-point AR
# ─────────────────────────────────────────────────────────────────────────────

def run_dgp_B(lstm: InstrumentedLSTM,
              n_rep: int = N_RECOVERY) -> dict:
    """
    Run n_rep replications on change-point AR sequences.

    For DGP B, the 'oracle' uses piecewise-constant I_t:
      I_t = ar1_stationary_It(phi_b)  for t < tau
      I_t = ar1_stationary_It(phi_a)  for t >= tau
    """
    kliep  = RollingKLIEP(window=KLIEP_WINDOW)
    phi_b  = CP_PHI_BEFORE
    phi_a  = CP_PHI_AFTER
    T      = T_TEST
    tau    = int(T * CP_FRACTION)
    burn   = KLIEP_WINDOW

    It_oracle_template = np.concatenate([
        np.full(tau,     ar1_stationary_It(phi_b)),
        np.full(T - tau, ar1_stationary_It(phi_a)),
    ])

    # phi_vals for ls_ar_It: piecewise constant
    phi_vals_cp = np.concatenate([
        np.full(tau,     phi_b),
        np.full(T - tau, phi_a),
    ])

    nmse = {k: [] for k in ['naive', 'fg', 'bootstrap', 'kliep_corr']}
    example_results = None

    for rep in tqdm(range(n_rep), desc="  DGP B", leave=False):
        x_cp, _ = generate_changepoint_ar(1, T,
                                           phi_before=phi_b, phi_after=phi_a,
                                           seed=MASTER_SEED + 400 + rep)
        x = x_cp[0]

        estimates = estimate_It_all(x, lstm, kliep, phi_vals_cp, phi_b)

        for key in nmse:
            nmse[key].append(compute_nmse(estimates[key],
                                          It_oracle_template, start=burn))

        if rep == 0:
            example_results = {k: estimates[k].copy() for k in estimates}

    return ({k: np.array(nmse[k]) for k in nmse},
            example_results, It_oracle_template)


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'oracle':     ('k',          'Oracle (true $\\mathcal{I}_t$)'),
    'naive':      ('gray',       'Naive constant'),
    'fg':         ('steelblue',  'Forget gate (LSTM)'),
    'bootstrap':  ('darkorange', 'Block bootstrap'),
    'kliep_corr': ('green',      'KLIEP-corrected (Pathway B)'),
}


def plot_trajectories(example_results: dict, It_oracle: np.ndarray,
                      title: str, fname: str):
    """Plot I_t trajectories for one example sequence."""
    T   = len(It_oracle)
    t   = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, It_oracle, **{'color': 'k', 'lw': 2, 'label': 'Oracle (true $\\mathcal{I}_t$)', 'zorder': 5})
    for key, (col, lab) in COLORS.items():
        if key == 'oracle':
            continue
        v = example_results[key]
        if v is None:
            continue
        ax.plot(t, v, color=col, lw=1.4, alpha=0.85, label=lab)
    ax.set_xlabel('Time step $t$')
    ax.set_ylabel('$\\hat{\\mathcal{I}}_t$')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    plt.close(fig)


def plot_nmse_boxplot(nmse_dict_A: dict, nmse_dict_B: dict, fname: str):
    """Side-by-side boxplot of NMSE for both DGPs."""
    keys   = ['naive', 'fg', 'bootstrap', 'kliep_corr']
    labels = [COLORS[k][1].replace(' (LSTM)', '').replace(' (Pathway B)', '') for k in keys]
    colors = [COLORS[k][0] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, nmse_dict, title in zip(axes,
                                     [nmse_dict_A, nmse_dict_B],
                                     ['DGP A: Locally stationary AR',
                                      'DGP B: Abrupt change-point AR']):
        data = [nmse_dict[k] for k in keys]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops={'color': 'black', 'lw': 2})
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('NMSE($\\hat{\\mathcal{I}}_t$, $\\mathcal{I}_t^{\\mathrm{true}}$)')
        ax.set_title(title)
        ax.axhline(1.0, color='gray', ls=':', lw=1, label='NMSE=1 (trivial)')
        ax.legend(fontsize=7)

    plt.suptitle('Study S3 — Recovery of $\\mathcal{I}_t$ under non-stationarity\n'
                 f'({N_RECOVERY} replications per DGP)', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(nmse_A: dict, nmse_B: dict) -> pd.DataFrame:
    keys = ['naive', 'fg', 'bootstrap', 'kliep_corr']
    rows = []
    for key in keys:
        label = COLORS[key][1]
        for dgp, d in [('DGP A (LS-AR)', nmse_A), ('DGP B (CP-AR)', nmse_B)]:
            arr = np.array(d[key])
            arr = arr[np.isfinite(arr)]
            rows.append({
                'Estimator':  label,
                'DGP':        dgp,
                'NMSE_mean':  round(float(np.mean(arr)),   4),
                'NMSE_median':round(float(np.median(arr)), 4),
                'NMSE_std':   round(float(np.std(arr)),    4),
                'NMSE_q25':   round(float(np.percentile(arr, 25)), 4),
                'NMSE_q75':   round(float(np.percentile(arr, 75)), 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"Study S3 — Recovery of I_t under non-stationarity ({N_RECOVERY} reps)")
    print("=" * 70)
    t0 = time.time()

    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)

    # Train one LSTM on stationary AR(1) at phi_center
    print(f"\nTraining LSTM on stationary AR(1) phi={LS_PHI_CENTER}...")
    X_train = generate_ar1(N_TRAIN_SEQ, T_TRAIN, LS_PHI_CENTER,
                            seed=MASTER_SEED + 500)
    lstm = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    train_model(lstm, X_train, verbose=True)

    # ── DGP A: locally stationary ─────────────────────────────────────────
    print("\n── DGP A: Locally stationary AR ────────────────────────────────")
    nmse_A, ex_A, oracle_A = run_dgp_A(lstm, n_rep=N_RECOVERY)
    for k, v in nmse_A.items():
        print(f"  {k:15s}  NMSE mean={np.nanmean(v):.4f}  median={np.nanmedian(v):.4f}")

    # ── DGP B: change-point ───────────────────────────────────────────────
    print("\n── DGP B: Abrupt change-point AR ───────────────────────────────")
    nmse_B, ex_B, oracle_B = run_dgp_B(lstm, n_rep=N_RECOVERY)
    for k, v in nmse_B.items():
        print(f"  {k:15s}  NMSE mean={np.nanmean(v):.4f}  median={np.nanmedian(v):.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    df = make_summary_table(nmse_A, nmse_B)
    out_csv = os.path.join(RESULTS_DIR, "S3_recovery.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(df.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────
    # Trajectory plots for one example sequence
    ex_A_with_oracle = dict(ex_A)
    ex_A_with_oracle['oracle'] = oracle_A
    plot_trajectories(ex_A_with_oracle, oracle_A,
                      title='DGP A — Locally stationary AR (one example sequence)',
                      fname='S3A_It_trajectories.png')

    ex_B_with_oracle = dict(ex_B)
    ex_B_with_oracle['oracle'] = oracle_B
    plot_trajectories(ex_B_with_oracle, oracle_B,
                      title='DGP B — Change-point AR (one example sequence)',
                      fname='S3B_It_trajectories.png')

    # NMSE boxplot comparison
    plot_nmse_boxplot(nmse_A, nmse_B, fname='S3_nmse_comparison.png')

    # ── Additional figure: sensitivity to alpha (KLIEP mixing weight) ────
    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    kliep  = RollingKLIEP(window=KLIEP_WINDOW)
    phi_c  = LS_PHI_CENTER
    burn   = KLIEP_WINDOW

    nmse_alpha = {a: [] for a in alphas}
    for rep in tqdm(range(min(200, N_RECOVERY)), desc="  Alpha sensitivity"):
        x_ls, phi_vals = generate_locally_stationary_ar(
            1, T_TEST, seed=MASTER_SEED + 600 + rep)
        x = x_ls[0]
        estimates_base = estimate_It_all(x, lstm, kliep, phi_vals, phi_c, alpha=0.5)
        It_oracle = ls_ar_It(phi_vals)

        for a in alphas:
            kliep_out  = kliep.fit_transform(x)
            Et         = kliep_out['bias']
            Et_filled  = np.where(np.isfinite(Et), Et, 0.0)
            It_corr    = corrected_delta(estimates_base['fg'], Et_filled, alpha=a)
            nmse_alpha[a].append(compute_nmse(It_corr, It_oracle, start=burn))

    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    means_a = [np.nanmean(nmse_alpha[a]) for a in alphas]
    stds_a  = [np.nanstd(nmse_alpha[a])  for a in alphas]
    ax_a.errorbar(alphas, means_a, yerr=stds_a, marker='o',
                  color='steelblue', capsize=4, lw=1.5)
    ax_a.set_xlabel('Mixing weight $\\alpha$\n(0 = full KLIEP, 1 = forget gate only)')
    ax_a.set_ylabel('NMSE($\\hat{\\mathcal{I}}_t$, true)')
    ax_a.set_title('Study S3 — Sensitivity to KLIEP mixing weight $\\alpha$\n(DGP A, 200 reps)')
    plt.tight_layout()
    fig_a.savefig(os.path.join(FIGURES_DIR, "S3_alpha_sensitivity.png"))
    plt.close(fig_a)

    elapsed = time.time() - t0
    print(f"\nStudy S3 complete in {elapsed:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
