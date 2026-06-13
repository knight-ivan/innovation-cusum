#!/usr/bin/env python3
# =============================================================================
#  study_S1_stationary.py
#  Study S1: Empirical validation of the MDS decomposition (stationary case)
#
#  Tests:
#    (A) MDS property of LSTM hidden-state innovations (Proposition 2):
#        Ljung-Box test on residuals r_t = X_{t+1} - hat{y}_t.
#        Expected: high pass rate (close to 1 - MDS_LEVEL).
#
#    (B) Forget-gate ratio matches I_t / I_{t-1} (Conjecture 1):
#        For AR(1), optimal forget gate ≈ |phi|  (constant ratio).
#        Metric: Spearman(f_t, phi_target) and mean absolute error.
#
#    (C) SimpleMamba Delta_t correlates with I_t (Conjecture 1):
#        Spearman(Delta_t, I_t^cell).
#
#  Processes: AR(1) phi in {0.30, 0.70, 0.95}, two-state HMM, RS-AR.
#  Replications: N_TEST_SEQ independent test sequences per configuration.
#  Results saved to results/S1_mds.csv, results/S1_forget.csv,
#                   results/S1_mamba.csv
#  Figures saved to figures/S1_*.png
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
                    AR_PHIS, HMM_TRANS, HMM_MEANS, HMM_STD,
                    RSAR_PHIS, RSAR_TRANS, HIDDEN_DIM, MDS_LEVEL,
                    RESULTS_DIR, FIGURES_DIR)
from data_generators import (generate_ar1, generate_ar1_transient,
                              generate_hmm, generate_rsar)
from analytical_It import (ar1_stationary_It, ar1_transient_It,
                            ar1_forget_gate_target, hmm_It_numerical)
from models import InstrumentedLSTM, SimpleMamba
from training import (train_model, compute_residuals,
                      extract_forget_gates, extract_deltas,
                      extract_cell_states, compute_cell_It)
from statistical_tests import mds_ljung_box, mds_pass_rate, spearman_ci


plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: train one LSTM + Mamba on a given dataset
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(X_train: np.ndarray, seed: int, verbose: bool = False):
    torch.manual_seed(seed)
    lstm  = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    mamba = SimpleMamba()
    train_model(lstm,  X_train, verbose=verbose)
    train_model(mamba, X_train, verbose=verbose)
    return lstm, mamba


# ─────────────────────────────────────────────────────────────────────────────
#  Part A: MDS test
# ─────────────────────────────────────────────────────────────────────────────

def run_mds_study(configs: list) -> pd.DataFrame:
    """
    For each (process_name, X_train, X_test) configuration:
      1. Train LSTM.
      2. Compute residuals on N_TEST_SEQ test sequences.
      3. Apply Ljung-Box MDS test (pass rate).
      4. Baseline: AR(1) OLS residuals for comparison.

    configs : list of dicts with keys 'name', 'X_train', 'X_test'
    """
    rows = []
    for cfg in tqdm(configs, desc="MDS study"):
        name    = cfg['name']
        X_train = cfg['X_train']
        X_test  = cfg['X_test']

        torch.manual_seed(MASTER_SEED)
        lstm = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
        train_model(lstm, X_train)

        # LSTM residuals
        lstm_resid = compute_residuals(lstm, X_test)    # (N_TEST, T-1)
        lstm_rate  = mds_pass_rate(lstm_resid)

        # Baseline: raw AR(1) OLS residuals (no learning)
        def ols_residuals(X_seqs):
            N, T = X_seqs.shape
            resid = np.zeros((N, T - 1))
            for i in range(N):
                x = X_seqs[i]
                # Simple AR(1) OLS on full series
                phi_hat = np.dot(x[:-1], x[1:]) / max(np.dot(x[:-1], x[:-1]), 1e-12)
                resid[i] = x[1:] - phi_hat * x[:-1]
            return resid

        ols_resid = ols_residuals(X_test)
        ols_rate  = mds_pass_rate(ols_resid)

        rows.append({
            'process':           name,
            'LSTM_pass_rate':    round(lstm_rate, 4),
            'OLS_pass_rate':     round(ols_rate,  4),
            'expected_pass':     round(1 - MDS_LEVEL, 2),
            'n_test_seqs':       X_test.shape[0],
            'T_test':            X_test.shape[1],
        })
        print(f"  {name:30s}  LSTM={lstm_rate:.3f}  OLS={ols_rate:.3f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Part B: Forget-gate ratio test (Conjecture 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_forget_gate_study(ar_configs: list) -> pd.DataFrame:
    """
    For each AR(1) configuration:
      1. Train LSTM on stationary data (random init, no transient).
      2. Test on N_TEST_SEQ TRANSIENT sequences (h_0 = 0).
      3. Compute:
           - Mean forget-gate (averaged over t and sequences)
           - Target: |phi|
           - Spearman rho between mean f_t trajectory and I_t/I_{t-1} trajectory
    """
    rows = []
    # Figure for forget-gate trajectories
    n_phi = len(ar_configs)
    fig, axes = plt.subplots(1, n_phi, figsize=(4 * n_phi, 4), sharey=False)
    if n_phi == 1:
        axes = [axes]

    for col_idx, cfg in enumerate(tqdm(ar_configs, desc="Forget gate study")):
        phi     = cfg['phi']
        X_train = cfg['X_train_stat']
        X_test  = cfg['X_test_trans']    # transient sequences (h_0 = 0)
        name    = f"AR(1) phi={phi}"

        torch.manual_seed(MASTER_SEED + int(phi * 100))
        lstm = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
        train_model(lstm, X_train)

        # Extract forget gates on transient sequences
        fgs = extract_forget_gates(lstm, X_test)       # (N, T, d)
        # Mean across hidden dimension (Conjecture 1 holds coordinatewise)
        fg_mean = fgs.mean(axis=-1)                    # (N, T)

        # Theoretical target: |phi| (constant for AR(1))
        fg_target = ar1_forget_gate_target(phi)

        # Mean forget gate averaged over all (seq, t)
        fg_all    = fg_mean.ravel()
        mean_fg   = float(fg_all.mean())
        mae_fg    = float(np.abs(fg_all - fg_target).mean())

        # Trajectory comparison: mean forget gate over sequences at each t
        fg_traj   = fg_mean.mean(axis=0)              # (T,)
        It_traj   = ar1_transient_It(phi, T_TEST)     # (T,)
        # Forget-gate ratio trajectory (should ≈ It[1:]/It[:-1] = |phi|)
        ratio_traj = It_traj[1:] / np.maximum(It_traj[:-1], 1e-12)  # (T-1,)
        fg_ratio   = (fg_traj[1:] + fg_traj[:-1]) / 2               # smooth mean

        sp = spearman_ci(fg_ratio, ratio_traj)

        rows.append({
            'process':        name,
            'phi':            phi,
            'target_fg':      round(fg_target, 4),
            'mean_fg':        round(mean_fg, 4),
            'MAE_fg':         round(mae_fg, 4),
            'spearman_rho':   round(sp['rho'], 4),
            'spearman_pval':  round(sp['p_value'], 4),
            'CI_lo':          round(sp['ci_lo'], 4),
            'CI_hi':          round(sp['ci_hi'], 4),
        })
        print(f"  {name:20s}  target={fg_target:.2f}  mean_fg={mean_fg:.4f}"
              f"  MAE={mae_fg:.4f}  rho={sp['rho']:.3f}")

        # Plot: mean forget-gate trajectory vs |phi| reference
        ax = axes[col_idx]
        t_arr = np.arange(T_TEST)
        # Shade ±1 std across sequences
        fg_std = fg_mean.std(axis=0)
        ax.plot(t_arr, fg_traj, color='steelblue', lw=1.5, label='Mean $f_t$')
        ax.fill_between(t_arr, fg_traj - fg_std, fg_traj + fg_std,
                        alpha=0.25, color='steelblue')
        ax.axhline(fg_target, color='firebrick', ls='--', lw=1.5,
                   label=f'$|\\phi|={fg_target:.2f}$')
        ax.set_title(name)
        ax.set_xlabel('Time step $t$')
        ax.set_ylabel('Forget gate $(f_t)_j$')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S1_forget_gate_trajectories.png"))
    plt.close(fig)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Part C: SimpleMamba Delta_t vs cell-state I_t (Conjecture 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_mamba_study(ar_configs: list) -> pd.DataFrame:
    """
    For each AR(1) configuration:
      1. Train SimpleMamba.
      2. Extract Delta_t on transient test sequences.
      3. Compute I_t^cell from LSTM cell states as ground truth.
      4. Report Spearman(Delta_t, I_t^cell).
    """
    rows = []
    n_phi = len(ar_configs)
    fig, axes = plt.subplots(1, n_phi, figsize=(4 * n_phi, 4))
    if n_phi == 1:
        axes = [axes]

    for col_idx, cfg in enumerate(tqdm(ar_configs, desc="Mamba study")):
        phi     = cfg['phi']
        X_train = cfg['X_train_stat']
        X_test  = cfg['X_test_trans']
        name    = f"AR(1) phi={phi}"

        # Train both models
        torch.manual_seed(MASTER_SEED + int(phi * 100))
        lstm  = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
        mamba = SimpleMamba()
        train_model(lstm,  X_train)
        train_model(mamba, X_train)

        # I_t proxy from cell states (ground truth for comparison)
        cs     = extract_cell_states(lstm, X_test)     # (N, T, d)
        It_cell = compute_cell_It(cs)                  # (N, T)
        # Analytical reference
        It_true = ar1_transient_It(phi, T_TEST)        # (T,)

        # Delta_t from Mamba
        deltas = extract_deltas(mamba, X_test)         # (N, T)

        # Compare trajectories (mean over sequences)
        It_mean    = It_cell.mean(axis=0)              # (T,)
        delta_mean = deltas.mean(axis=0)               # (T,)

        # Normalise both to [0,1] for comparison
        def norm01(v):
            r = v - v.min()
            return r / max(r.max(), 1e-10)

        sp = spearman_ci(delta_mean, It_mean)

        rows.append({
            'process':       name,
            'phi':           phi,
            'rho_delta_It':  round(sp['rho'], 4),
            'pval':          round(sp['p_value'], 4),
            'CI_lo':         round(sp['ci_lo'], 4),
            'CI_hi':         round(sp['ci_hi'], 4),
        })
        print(f"  {name:20s}  Spearman(Delta_t, I_t^cell) = {sp['rho']:.3f}"
              f"  [p={sp['p_value']:.3f}]")

        # Scatter plot: Delta_t vs I_t^cell (one dot per time step, 10 seqs)
        ax = axes[col_idx]
        n_show = min(10, X_test.shape[0])
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_show))
        for i in range(n_show):
            ax.scatter(norm01(It_cell[i]), norm01(deltas[i]),
                       s=6, alpha=0.4, color=colors[i])
        ax.plot([0, 1], [0, 1], 'r--', lw=1, label='Identity')
        ax.set_xlabel('$\\mathcal{I}_t^{(\\mathrm{cell})}$ (norm.)')
        ax.set_ylabel('$\\Delta_t$ (norm.)')
        ax.set_title(f'{name}\n$\\rho_S={sp["rho"]:.3f}$')
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S1_mamba_delta_vs_It.png"))
    plt.close(fig)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Study S1 — Stationary case (Proposition 2 and Conjecture 1 validation)")
    print("=" * 70)
    t0 = time.time()

    np.random.seed(MASTER_SEED)
    torch.manual_seed(MASTER_SEED)

    # ── Build configurations ──────────────────────────────────────────────

    # AR(1) stationary training + transient test
    ar_configs = []
    for phi in AR_PHIS:
        X_train_stat  = generate_ar1(N_TRAIN_SEQ, T_TRAIN, phi,
                                      seed=MASTER_SEED + int(phi * 1000))
        X_test_stat   = generate_ar1(N_TEST_SEQ,  T_TEST,  phi,
                                      seed=MASTER_SEED + int(phi * 1000) + 1)
        X_test_trans  = generate_ar1_transient(N_TEST_SEQ, T_TEST, phi,
                                                seed=MASTER_SEED + int(phi * 1000) + 2)
        ar_configs.append({
            'phi':            phi,
            'name':           f'AR(1) phi={phi}',
            'X_train_stat':   X_train_stat,
            'X_test_stat':    X_test_stat,
            'X_test_trans':   X_test_trans,
        })

    # HMM
    X_hmm_train, _  = generate_hmm(N_TRAIN_SEQ, T_TRAIN, seed=MASTER_SEED + 10)
    X_hmm_test,  _  = generate_hmm(N_TEST_SEQ,  T_TEST,  seed=MASTER_SEED + 11)

    # RS-AR
    X_rsar_train, _ = generate_rsar(N_TRAIN_SEQ, T_TRAIN, seed=MASTER_SEED + 20)
    X_rsar_test,  _ = generate_rsar(N_TEST_SEQ,  T_TEST,  seed=MASTER_SEED + 21)

    # Unified config list for MDS study (uses stationary test sequences)
    mds_configs = (
        [{'name': c['name'], 'X_train': c['X_train_stat'],
          'X_test': c['X_test_stat']} for c in ar_configs]
        + [{'name': 'HMM',   'X_train': X_hmm_train,  'X_test': X_hmm_test},
           {'name': 'RS-AR', 'X_train': X_rsar_train, 'X_test': X_rsar_test}]
    )

    # ── Part A: MDS test ─────────────────────────────────────────────────
    print("\n── Part A: MDS (Ljung-Box) pass rates ───────────────────────────")
    df_mds = run_mds_study(mds_configs)
    out_mds = os.path.join(RESULTS_DIR, "S1_mds.csv")
    df_mds.to_csv(out_mds, index=False)
    print(f"\nSaved → {out_mds}")
    print(df_mds.to_string(index=False))

    # Figure: bar chart of pass rates
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df_mds))
    w = 0.35
    ax.bar(x - w/2, df_mds['LSTM_pass_rate'], w, label='LSTM', color='steelblue')
    ax.bar(x + w/2, df_mds['OLS_pass_rate'],  w, label='AR(1) OLS', color='coral')
    ax.axhline(1 - MDS_LEVEL, color='black', ls='--', lw=1,
               label=f'Expected ({1 - MDS_LEVEL:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(df_mds['process'], rotation=15, ha='right')
    ax.set_ylabel('MDS pass rate (Ljung-Box)')
    ax.set_title('Study S1A — MDS test pass rates\n(H$_0$: no autocorrelation in residuals)')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S1A_mds_pass_rates.png"))
    plt.close(fig)

    # ── Part B: Forget-gate study ────────────────────────────────────────
    print("\n── Part B: Forget-gate ratio test (Conjecture 1) ───────────────")
    df_fg = run_forget_gate_study(ar_configs)
    out_fg = os.path.join(RESULTS_DIR, "S1_forget_gate.csv")
    df_fg.to_csv(out_fg, index=False)
    print(f"\nSaved → {out_fg}")
    print(df_fg.to_string(index=False))

    # ── Part C: Mamba Delta_t study ──────────────────────────────────────
    print("\n── Part C: SimpleMamba Delta_t vs I_t^cell (Conjecture 1) ──────")
    df_mamba = run_mamba_study(ar_configs)
    out_mamba = os.path.join(RESULTS_DIR, "S1_mamba.csv")
    df_mamba.to_csv(out_mamba, index=False)
    print(f"\nSaved → {out_mamba}")
    print(df_mamba.to_string(index=False))

    # ── Summary figure: forget-gate mean vs |phi| across phi values ──────
    fig, ax = plt.subplots(figsize=(5, 4))
    phis_arr  = df_fg['phi'].values
    mean_fgs  = df_fg['mean_fg'].values
    target_fgs = df_fg['target_fg'].values
    ax.scatter(phis_arr, mean_fgs,   s=80, color='steelblue', zorder=3,
               label='Mean $\\bar{f}$ (LSTM)')
    ax.scatter(phis_arr, target_fgs, s=80, marker='x', color='firebrick', lw=2,
               zorder=3, label='Target $|\\phi|$')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    for p, mf, tf in zip(phis_arr, mean_fgs, target_fgs):
        ax.annotate(f'$\\phi={p}$', (p, mf), textcoords='offset points',
                    xytext=(6, 3), fontsize=8)
    ax.set_xlabel('True AR coefficient $|\\phi|$')
    ax.set_ylabel('Mean forget gate $\\bar{f}$')
    ax.set_title('Study S1B — Forget gate convergence\n(Conjecture 1, linear-Gaussian regime)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "S1B_forget_gate_summary.png"))
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nStudy S1 complete in {elapsed:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
