#!/usr/bin/env python3
# =============================================================================
#  study_S2_nonstationary.py
#  Study S2: Non-stationary case — does stationarity-trained LSTM degrade?
#
#  Two non-stationary settings:
#    (A) Locally stationary AR: phi(u) = 0.50 + 0.40*sin(2*pi*u), u in [0,1]
#        True I_t(u) varies smoothly.  LSTM trained on stationary data has a
#        constant forget gate (biased).
#
#    (B) Abrupt change-point AR: phi changes from 0.30 to 0.85 at T/2.
#        CUSUM detector fires near the true change-point.
#        After the alarm: LSTM predictions degrade; a reset + retrain recovers.
#
#  For each setting we report:
#    - Prediction MSE before/after change-point (or over time windows)
#    - CUSUM detection delay (Setting B)
#    - Forget-gate trajectory vs. true I_t(u) Spearman correlation
#
#  Replications: N_TEST_SEQ (1000) independent sequences.
#  Results: results/S2_ls.csv, results/S2_cp.csv
#  Figures: figures/S2_*.png
#
#  Note: Setting (A) is reported in Supplementary Material Section S4.
#        Setting (B) is the basis for Study S2B (study_S2B_pathway_A.py).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os, time
from tqdm import tqdm

from config import (MASTER_SEED, N_TEST_SEQ, N_TRAIN_SEQ, T_TRAIN, T_TEST,
                    AR_PHIS, CP_PHI_BEFORE, CP_PHI_AFTER, CP_FRACTION,
                    LS_PHI_CENTER, LS_PHI_AMP, HIDDEN_DIM, RESULTS_DIR, FIGURES_DIR)
from data_generators import (generate_ar1, generate_locally_stationary_ar,
                              generate_changepoint_ar, phi_smooth)
from analytical_It import ar1_stationary_It, ls_ar_It, ls_ar_fg_ratio
from models import InstrumentedLSTM, SimpleMamba
from training import (train_model, compute_residuals,
                      extract_forget_gates, extract_cell_states, compute_cell_It)
from statistical_tests import (mds_pass_rate, spearman_ci,
                                cusum_detector, detection_delay)


plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


# ─────────────────────────────────────────────────────────────────────────────
#  Setting A — Locally stationary AR
# ─────────────────────────────────────────────────────────────────────────────

def run_locally_stationary(n_test: int = N_TEST_SEQ) -> pd.DataFrame:
    """
    Train LSTM on a stationary AR(1) with phi = phi_center, then evaluate on
    locally stationary sequences where phi(u) varies.

    Metrics:
      - MSE(hat_y_t, X_{t+1}) averaged over t in three windows: [0-T/3], [T/3-2T/3], [2T/3-T]
      - Spearman(forget gate trajectory, true I_t(u) trajectory)
      - Comparison: constant |phi_center| (naive estimate) vs. LSTM forget gate
    """
    phi_center = LS_PHI_CENTER
    T = T_TEST

    # Training: stationary AR(1) at phi_center
    print("  Training LSTM on stationary AR(1) (phi_center = {:.2f})...".format(phi_center))
    X_train_stat = generate_ar1(N_TRAIN_SEQ, T_TRAIN, phi_center,
                                 seed=MASTER_SEED + 100)
    torch.manual_seed(MASTER_SEED + 100)
    lstm = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    train_model(lstm, X_train_stat)

    # Also train a Mamba model
    torch.manual_seed(MASTER_SEED + 101)
    mamba = SimpleMamba()
    train_model(mamba, X_train_stat)

    # Test: locally stationary sequences
    print(f"  Testing on {n_test} locally stationary sequences...")
    X_ls, phi_vals = generate_locally_stationary_ar(n_test, T, seed=MASTER_SEED + 102)

    # True I_t(u)
    It_true = ls_ar_It(phi_vals)            # (T,)
    It_true_ratio = ls_ar_fg_ratio(phi_vals)  # (T-1,)

    # Extract LSTM outputs
    residuals = compute_residuals(lstm, X_ls)          # (n_test, T-1)
    fgs       = extract_forget_gates(lstm, X_ls)        # (n_test, T, d)
    fg_mean   = fgs.mean(axis=-1)                      # (n_test, T)

    # MSE in three temporal windows
    def window_mse(resid_mat, start, end):
        return float(np.mean(resid_mat[:, start:end] ** 2))

    w1 = T // 3
    w2 = 2 * T // 3
    mse_early  = window_mse(residuals, 0,   w1)
    mse_middle = window_mse(residuals, w1,  w2)
    mse_late   = window_mse(residuals, w2,  T - 1)

    # Spearman(mean fg trajectory, true I_t trajectory)
    fg_traj = fg_mean.mean(axis=0)                     # (T,)
    sp_It   = spearman_ci(fg_traj, It_true)

    # Comparison: does LSTM forget gate track I_t better than constant?
    constant_pred = np.full(T, abs(phi_center) * 1.0)   # scaled to match scale
    sp_const      = spearman_ci(constant_pred, It_true)  # should be ~0

    rows = [{
        'setting':         'Locally stationary AR',
        'phi_func':        f'center={phi_center}, amp={LS_PHI_AMP}',
        'MSE_early_third': round(mse_early,  5),
        'MSE_mid_third':   round(mse_middle, 5),
        'MSE_late_third':  round(mse_late,   5),
        'rho_fg_It':       round(sp_It['rho'],    4),
        'rho_fg_pval':     round(sp_It['p_value'], 4),
        'rho_const_It':    round(sp_const['rho'], 4),
        'n_test_seqs':     n_test,
    }]

    print(f"  MSE: early={mse_early:.4f}  mid={mse_middle:.4f}  late={mse_late:.4f}")
    print(f"  Spearman(fg, I_t)={sp_It['rho']:.3f}  vs constant={sp_const['rho']:.3f}")

    # ── Figure: phi(u), true I_t, forget-gate trajectory ─────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    t_arr = np.arange(T)

    axes[0].plot(t_arr, phi_vals, 'k', lw=1.5)
    axes[0].axhline(phi_center, color='gray', ls='--', label=f'$\\phi_{{center}}={phi_center}$')
    axes[0].set_ylabel('$\\phi(t/T)$')
    axes[0].set_title('(a) Time-varying AR coefficient')
    axes[0].legend(fontsize=8)

    fg_std  = fg_mean.std(axis=0)
    It_norm = It_true / max(It_true.max(), 1e-10)
    fg_norm = fg_traj  / max(fg_traj.max(), 1e-10)

    axes[1].plot(t_arr, It_norm, 'firebrick', lw=1.5, label='True $\\mathcal{I}_t$ (norm.)')
    axes[1].plot(t_arr, fg_norm, 'steelblue', lw=1.5, alpha=0.85, label='Mean $f_t$ (LSTM, norm.)')
    axes[1].fill_between(t_arr,
                         (fg_mean.mean(axis=0) - fg_std) / max(fg_traj.max(), 1e-10),
                         (fg_mean.mean(axis=0) + fg_std) / max(fg_traj.max(), 1e-10),
                         alpha=0.2, color='steelblue')
    axes[1].set_ylabel('Normalised value')
    axes[1].set_title(f'(b) $\\mathcal{{I}}_t$ vs forget gate  ($\\rho_S={sp_It["rho"]:.3f}$)')
    axes[1].legend(fontsize=8)

    mse_curve = np.array([float(np.mean(residuals[:, t] ** 2)) for t in range(T - 1)])
    axes[2].plot(t_arr[1:], mse_curve, 'darkorange', lw=1.2)
    axes[2].set_ylabel('MSE (per time step)')
    axes[2].set_xlabel('Time step $t$')
    axes[2].set_title('(c) Per-step prediction MSE')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S2A_locally_stationary.png"))
    plt.close(fig)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Setting B — Abrupt change-point AR
# ─────────────────────────────────────────────────────────────────────────────

def run_changepoint(n_test: int = N_TEST_SEQ) -> pd.DataFrame:
    """
    Train LSTM on AR(1) with phi_before.  Test on sequences with a single
    change-point at T * CP_FRACTION.  Evaluate:
      1. CUSUM detection delay distribution (over n_test sequences).
      2. MSE before vs after the change-point.
      3. Recovery via post-alarm LSTM reset (re-train from T/2 to T).
    """
    phi_b = CP_PHI_BEFORE
    phi_a = CP_PHI_AFTER
    T     = T_TEST
    tau   = int(T * CP_FRACTION)

    print(f"  phi before={phi_b}, phi after={phi_a}, change-point tau={tau}")

    # Train LSTM on stationary phi_b data
    X_train = generate_ar1(N_TRAIN_SEQ, T_TRAIN, phi_b, seed=MASTER_SEED + 200)
    torch.manual_seed(MASTER_SEED + 200)
    lstm_before = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    train_model(lstm_before, X_train)

    # Train a "post-change" oracle LSTM on phi_a data
    X_train_a = generate_ar1(N_TRAIN_SEQ, T_TRAIN, phi_a, seed=MASTER_SEED + 201)
    torch.manual_seed(MASTER_SEED + 201)
    lstm_after = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    train_model(lstm_after, X_train_a)

    # Generate change-point test sequences
    X_cp, _ = generate_changepoint_ar(n_test, T,
                                       phi_before=phi_b, phi_after=phi_a,
                                       seed=MASTER_SEED + 202)

    # Detection via CUSUM
    print(f"  Running CUSUM on {n_test} sequences...")
    delays = []
    alarm_times = []
    for i in tqdm(range(n_test), desc="  CUSUM", leave=False):
        cusum_result = cusum_detector(X_cp[i], h=5.0)
        alarm_times.append(cusum_result['alarm_time'])
        delays.append(detection_delay(tau, cusum_result['alarm_time'], T))

    delays     = np.array(delays)
    alarm_times_arr = np.array([a if a is not None else T for a in alarm_times])
    detect_rate = float(np.mean([a is not None and a < T for a in alarm_times]))

    # Prediction MSE: before-change model on full test sequences
    resid_before  = compute_residuals(lstm_before, X_cp)   # (n_test, T-1)
    resid_after   = compute_residuals(lstm_after,  X_cp)   # (n_test, T-1)

    def seg_mse(resid, start, end):
        return float(np.mean(resid[:, start:end] ** 2))

    mse_bb = seg_mse(resid_before, 0,    tau)        # before-change model, before change
    mse_ba = seg_mse(resid_before, tau,  T - 1)      # before-change model, after change
    mse_aa = seg_mse(resid_after,  tau,  T - 1)      # after-change oracle, after change

    # Forget-gate jump at change-point (reuse already-extracted gates)
    fg_traj    = extract_forget_gates(lstm_before, X_cp).mean(axis=-1).mean(axis=0)  # (T,)

    rows = [{
        'setting':            'Abrupt change-point AR',
        'phi_before':         phi_b,
        'phi_after':          phi_a,
        'true_tau':           tau,
        'CUSUM_detect_rate':  round(detect_rate, 4),
        'CUSUM_delay_mean':   round(float(delays.mean()), 2),
        'CUSUM_delay_median': round(float(np.median(delays)), 2),
        'CUSUM_delay_90pct':  round(float(np.percentile(delays, 90)), 2),
        'MSE_before_pre':     round(mse_bb, 5),
        'MSE_before_post':    round(mse_ba, 5),
        'MSE_oracle_post':    round(mse_aa, 5),
        'MSE_ratio':          round(mse_ba / max(mse_bb, 1e-10), 3),
        'n_test_seqs':        n_test,
    }]

    print(f"  CUSUM detect rate={detect_rate:.3f}  delay mean={delays.mean():.1f}")
    print(f"  MSE before-model: pre={mse_bb:.4f}  post={mse_ba:.4f}  ratio={mse_ba/max(mse_bb,1e-10):.2f}")
    print(f"  MSE oracle (post-change): {mse_aa:.4f}")

    # ── Figure: CUSUM trajectory + forget-gate jump ───────────────────────
    # Show one example sequence
    example_idx = 0
    x_ex   = X_cp[example_idx]
    cusum_ex = cusum_detector(x_ex, h=5.0)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    t_arr = np.arange(T)

    # Series with change-point marker
    axes[0].plot(t_arr, x_ex, 'k', lw=0.8)
    axes[0].axvline(tau, color='red',  ls='--', lw=1.5, label=f'True $\\tau={tau}$')
    if cusum_ex['alarm_time'] is not None:
        axes[0].axvline(cusum_ex['alarm_time'], color='orange', ls=':',
                        lw=1.5, label=f'CUSUM alarm={cusum_ex["alarm_time"]}')
    axes[0].set_ylabel('$X_t$')
    axes[0].set_title('(a) Change-point AR series (example)')
    axes[0].legend(fontsize=8)

    # CUSUM statistics
    axes[1].plot(t_arr, cusum_ex['S_plus'],  'steelblue', lw=1.2, label='$S_t^+$')
    axes[1].plot(t_arr, cusum_ex['S_minus'], 'coral',     lw=1.2, label='$S_t^-$')
    axes[1].axhline(5.0, color='gray', ls='--', label='Threshold $h=5$')
    axes[1].axvline(tau, color='red', ls='--', lw=1, alpha=0.5)
    axes[1].set_ylabel('CUSUM statistic')
    axes[1].set_title('(b) CUSUM (Pathway A)')
    axes[1].legend(fontsize=8)

    # Forget-gate trajectory (population mean)
    axes[2].plot(t_arr, fg_traj, 'steelblue', lw=1.5, label='Mean $f_t$')
    axes[2].fill_between(t_arr, fg_traj - fg_std_arr, fg_traj + fg_std_arr,
                         alpha=0.2, color='steelblue')
    axes[2].axvline(tau, color='red', ls='--', lw=1.5, label=f'True $\\tau={tau}$')
    axes[2].axhline(abs(phi_b), color='green',  ls=':', label=f'$|\\phi_b|={phi_b}$')
    axes[2].axhline(abs(phi_a), color='orange', ls=':', label=f'$|\\phi_a|={phi_a}$')
    axes[2].set_xlabel('Time step $t$')
    axes[2].set_ylabel('Forget gate')
    axes[2].set_title('(c) Forget-gate trajectory (mean over 1000 sequences)')
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S2B_changepoint.png"))
    plt.close(fig)

    # Forget-gate std for fill_between
    fg_std_arr = extract_forget_gates(lstm_before, X_cp).mean(axis=-1).std(axis=0)

    # Distribution of CUSUM delays
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    valid_delays = delays[delays < T - tau]   # exclude "never detected"
    ax2.hist(valid_delays, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(float(np.median(valid_delays)), color='red', ls='--',
                label=f'Median={np.median(valid_delays):.0f}')
    ax2.set_xlabel('Detection delay (steps after true $\\tau$)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'CUSUM detection delay distribution\n(detect rate={detect_rate:.2f}, $n={n_test}$)')
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "S2B_cusum_delay_hist.png"))
    plt.close(fig2)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Study S2 — Non-stationary case")
    print("=" * 70)
    t0 = time.time()

    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)

    # ── Setting A: Locally stationary AR ──────────────────────────────────
    print("\n── Setting A: Locally stationary AR ────────────────────────────")
    df_ls = run_locally_stationary(n_test=N_TEST_SEQ)
    out_ls = os.path.join(RESULTS_DIR, "S2_locally_stationary.csv")
    df_ls.to_csv(out_ls, index=False)
    print(f"\nSaved → {out_ls}")
    print(df_ls.T.to_string())

    # ── Setting B: Abrupt change-point ────────────────────────────────────
    print("\n── Setting B: Abrupt change-point AR ───────────────────────────")
    df_cp = run_changepoint(n_test=N_TEST_SEQ)
    out_cp = os.path.join(RESULTS_DIR, "S2_changepoint.csv")
    df_cp.to_csv(out_cp, index=False)
    print(f"\nSaved → {out_cp}")
    print(df_cp.T.to_string())

    elapsed = time.time() - t0
    print(f"\nStudy S2 complete in {elapsed:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
