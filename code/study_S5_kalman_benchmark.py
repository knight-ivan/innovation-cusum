# =============================================================================
#  study_S5_kalman_benchmark.py — Kalman Filter Benchmark for I_t estimation
#
#  Research question:
#    How closely does the LSTM hidden state approximate the true marginal
#    information gain I_t, compared to the Kalman filter optimal estimator?
#
#  The Kalman filter is the optimal linear estimator for linear Gaussian
#  state-space models (SSMs).  For such models I_t has an exact closed form,
#  providing a ground-truth benchmark that was unavailable in S3.
#
#  Two DGPs:
#    Part A — AR(1) without observation noise
#             z_t = phi * z_{t-1} + eta_t,  x_t = z_t   (eta ~ N(0,1))
#             True I_t = |phi|^t * sigma / sqrt(1-phi^2)   [analytical]
#             Kalman estimator: plug-in formula with OLS-estimated phi, sigma
#
#    Part B — AR(1) with observation noise  (SNR = 1)
#             z_t = phi * z_{t-1} + eta_t,  eta ~ N(0, sigma_eta^2)
#             x_t = z_t + eps_t,            eps ~ N(0, sigma_eps^2)
#             True I_t: Kalman filter with *known* parameters (oracle)
#             Kalman estimator: Kalman filter with EM-estimated parameters
#
#  Four estimators compared (both parts):
#    1. Oracle/Analytical  — exact I_t (ground truth)
#    2. Kalman (estimated) — Kalman formula with estimated parameters
#    3. LSTM h-norm        — ||h_t - h_inf||_2 (hidden state distance)
#    4. Forget gate        — cumulative product of mean forget gates
#
#  Metrics: NMSE, Spearman rho (trajectory shape), bias, example trajectories
#
#  Outputs:
#    results/S5_kalman_partA.csv
#    results/S5_kalman_partB.csv
#    figures/S5_kalman_trajectories.png
#    figures/S5_kalman_nmse.png
# =============================================================================

import os, sys, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

from models   import InstrumentedLSTM
from training import train_model, compute_cell_It, extract_forget_gates
from config   import HIDDEN_DIM, N_EPOCHS, LR, BATCH_SIZE

# ── Configuration ─────────────────────────────────────────────────────────────
PHIS      = [0.30, 0.70, 0.95]
T         = 200
N_TRAIN   = 1000
N_TEST    = 1000
SNR       = 1.0          # sigma_eta^2 / sigma_eps^2 for Part B
SIGMA_ETA = 1.0
SEED      = 2025

rng = np.random.default_rng(SEED)


# ── Data generators ───────────────────────────────────────────────────────────

def gen_ar1_clean(N, T, phi, rng):
    """AR(1) without observation noise, X_0 = 0 (initialisation transient)."""
    X = np.zeros((N, T), dtype=np.float32)
    eps = rng.normal(0, SIGMA_ETA, (N, T-1)).astype(np.float32)
    for t in range(1, T):
        X[:, t] = phi * X[:, t-1] + eps[:, t-1]
    return X


def gen_ar1_noisy(N, T, phi, sigma_eta, sigma_eps, rng):
    """AR(1) with observation noise, z_0 = 0."""
    Z = np.zeros((N, T), dtype=np.float32)
    eta = rng.normal(0, sigma_eta, (N, T-1)).astype(np.float32)
    for t in range(1, T):
        Z[:, t] = phi * Z[:, t-1] + eta[:, t-1]
    eps = rng.normal(0, sigma_eps, (N, T)).astype(np.float32)
    X   = Z + eps
    return X, Z     # observations and true hidden states


# ── True I_t computations ──────────────────────────────────────────────────────

def It_analytical(phi, T, sigma=SIGMA_ETA):
    """
    True I_t for AR(1) without obs noise, initialised at X_0 = 0.
    I_t = |phi|^t * sigma / sqrt(1 - phi^2)
    Returns array of length T (index 0 = t=0, but we use t=1..T).
    """
    t_arr = np.arange(T, dtype=np.float64)
    return (abs(phi) ** t_arr * sigma / np.sqrt(1 - phi**2)).astype(np.float32)


def kalman_filter_known(X_seq, phi, sigma_eta, sigma_eps):
    """
    Kalman filter with *known* parameters for the noisy AR(1):
      z_t = phi * z_{t-1} + eta_t,  eta ~ N(0, sigma_eta^2)
      x_t = z_t + eps_t,            eps ~ N(0, sigma_eps^2)
    Initialised at z_0 = 0, P_0 = 0.

    Returns:
      z_filt : (T,) filtered state means
      P_filt : (T,) filtered state variances
      It_kalman : (T,) = |z_filt|  (proxy for information gain)
    """
    T = len(X_seq)
    Q = sigma_eta**2
    R = sigma_eps**2

    z  = 0.0; P = 0.0
    z_filt = np.zeros(T); P_filt = np.zeros(T)

    for t in range(T):
        # Predict
        z_pred = phi * z
        P_pred = phi**2 * P + Q
        # Update
        K = P_pred / (P_pred + R)
        z = z_pred + K * (X_seq[t] - z_pred)
        P = (1 - K) * P_pred
        z_filt[t] = z
        P_filt[t] = P

    # I_t proxy: magnitude of filtered state (deviation from prior mean 0)
    It = np.abs(z_filt)
    return z_filt, P_filt, It.astype(np.float32)


def kalman_filter_estimated(X_seq, phi_hat, sigma_eta_hat, sigma_eps_hat):
    """Same as above but with estimated parameters."""
    return kalman_filter_known(X_seq, phi_hat, sigma_eta_hat, sigma_eps_hat)


def estimate_ar1_params(X_seqs):
    """
    OLS AR(1) parameter estimation.
    X_seqs: (N, T)
    Returns phi_hat, sigma_hat (innovation std).
    """
    Y = X_seqs[:, 1:].ravel()
    Z = X_seqs[:, :-1].ravel()
    phi_hat = np.dot(Y, Z) / (np.dot(Z, Z) + 1e-12)
    resid = Y - phi_hat * Z
    sigma_hat = resid.std()
    return float(phi_hat), float(sigma_hat)


def em_ssm(X_seqs, phi_init=0.5, n_iter=30):
    """
    Simple EM for noisy AR(1): estimate phi, sigma_eta, sigma_eps.
    Uses single-sequence Kalman smoother (Rauch-Tung-Striebel).
    Averages sufficient statistics across all sequences.
    """
    N, T = X_seqs.shape
    phi = phi_init
    sigma_eta = 1.0
    sigma_eps = 1.0

    for _ in range(n_iter):
        # E-step: Kalman filter + smoother for each sequence
        sum_zz  = 0.0; sum_zpz = 0.0
        sum_xz  = 0.0; sum_xx  = 0.0
        sum_zz2 = 0.0; n_obs   = N * (T - 1)

        for i in range(min(N, 100)):   # use 100 seqs for speed
            x = X_seqs[i]
            Q = sigma_eta**2; R = sigma_eps**2

            # Forward pass
            z_p = np.zeros(T); P_p = np.zeros(T)
            z_f = np.zeros(T); P_f = np.zeros(T)
            z = 0.0; P = 0.0
            for t in range(T):
                zp = phi * z; Pp = phi**2 * P + Q
                K = Pp / (Pp + R + 1e-12)
                z = zp + K * (x[t] - zp); P = (1-K) * Pp
                z_p[t]=zp; P_p[t]=Pp; z_f[t]=z; P_f[t]=P

            # Backward smoother (RTS)
            z_s = z_f.copy(); P_s = P_f.copy()
            G   = np.zeros(T)
            for t in range(T-2, -1, -1):
                g = P_f[t] * phi / (P_p[t+1] + 1e-12)
                z_s[t] = z_f[t] + g * (z_s[t+1] - z_p[t+1])
                P_s[t] = P_f[t] + g**2 * (P_s[t+1] - P_p[t+1])
                G[t]   = g

            # Sufficient stats
            Ezz  = P_s + z_s**2
            for t in range(1, T):
                cross = G[t-1] * P_s[t] + z_s[t] * z_s[t-1]
                sum_zpz += cross
                sum_zz  += Ezz[t-1]
                sum_zz2 += Ezz[t]
            sum_xz  += np.sum(x * z_s)
            sum_xx  += np.sum(x**2)

        n = min(N, 100); nt = n * (T - 1)
        # M-step
        phi       = sum_zpz / (sum_zz + 1e-12)
        sigma_eta = np.sqrt(max((sum_zz2 - phi * sum_zpz) / nt, 1e-6))
        sigma_eps = np.sqrt(max((sum_xx / (n * T) - 2 * sum_xz / (n * T)
                                  + (sum_zz2 + sum_zz) / (n * T)), 1e-6))

    return float(phi), float(sigma_eta), float(sigma_eps)


# ── LSTM I_t estimators ───────────────────────────────────────────────────────

def train_lstm(X_train, seed=SEED):
    model = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    torch.manual_seed(seed)
    train_model(model, X_train, n_epochs=N_EPOCHS, lr=LR,
                batch_size=BATCH_SIZE, verbose=False)
    return model


def lstm_hnorm_It(model, X_seqs):
    """
    LSTM hidden state norm proxy:
      I_t^lstm = || h_t - h_inf ||_2  per dimension
    h_inf estimated as mean hidden state over last 20% of steps.

    Returns (N, T) array.
    """
    model.eval()
    x = torch.FloatTensor(X_seqs[:, :, None])   # (N, T, 1)
    with torch.no_grad():
        _, _, hs, _ = model(x)                   # hs: (N, T, hidden)
    hs_np = hs.numpy()
    tail  = max(1, int(0.2 * T))
    h_inf = hs_np[:, -tail:, :].mean(axis=1, keepdims=True)  # (N,1,hidden)
    return np.linalg.norm(hs_np - h_inf, axis=-1)            # (N, T)


def lstm_fg_It(model, X_seqs):
    """
    Forget-gate cumulative product proxy (normalised to start at 1):
      I_t^fg = prod_{s=1}^t mean_j(f_{s,j})
    Returns (N, T) array.
    """
    fgs = extract_forget_gates(model, X_seqs)     # (N, T, hidden)
    fg_mean = fgs.mean(axis=-1)                   # (N, T)
    It_fg   = np.cumprod(fg_mean, axis=1)
    # Normalise so t=0 is 1
    It_fg   = It_fg / (It_fg[:, 0:1] + 1e-12)
    return It_fg


# ── NMSE and Spearman ─────────────────────────────────────────────────────────

def nmse(pred, truth):
    """NMSE = MSE(pred, truth) / Var(truth), averaged over sequences."""
    diff = pred - truth
    v    = truth.var(axis=1, keepdims=True) + 1e-12
    return float(np.mean((diff**2) / v))


def mean_spearman(pred, truth):
    """Mean Spearman rho across N sequences."""
    rhos = []
    for i in range(len(pred)):
        r, _ = spearmanr(pred[i], truth[i])
        if np.isfinite(r): rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan


# ── Part A: AR(1) without observation noise ───────────────────────────────────

def run_part_A():
    print("\n" + "="*60)
    print("Part A: AR(1) without observation noise")
    print("="*60)

    rows = []

    for phi in PHIS:
        print(f"\n  phi = {phi}")
        rng2 = np.random.default_rng(SEED + int(phi*100))

        X_train = gen_ar1_clean(N_TRAIN, T, phi, rng2)
        X_test  = gen_ar1_clean(N_TEST,  T, phi, rng2)

        # ── True I_t (analytical) ──
        It_true_1d = It_analytical(phi, T)        # (T,)
        It_true    = np.tile(It_true_1d, (N_TEST, 1))  # (N, T)

        # ── Kalman estimator (plug-in) ──
        phi_hat, sigma_hat = estimate_ar1_params(X_test)
        It_kalman_1d = It_analytical(phi_hat, T, sigma=sigma_hat)
        It_kalman    = np.tile(It_kalman_1d, (N_TEST, 1))

        # ── LSTM: train once, extract h-norm and fg ──
        model    = train_lstm(X_train, seed=SEED + int(phi*100))
        It_hnorm = lstm_hnorm_It(model, X_test)             # (N, T)
        It_fg    = lstm_fg_It(model, X_test)

        # Normalise hnorm and fg to same scale as analytical I_t
        # (they are proxies up to a multiplicative constant; rescale by ratio)
        scale_h  = (It_true.mean() / (It_hnorm.mean() + 1e-12))
        scale_fg = (It_true.mean() / (It_fg.mean()    + 1e-12))
        It_hnorm_sc = It_hnorm * scale_h
        It_fg_sc    = It_fg    * scale_fg

        # ── Metrics ──
        res = {}
        for name, pred in [("kalman",  It_kalman),
                            ("hnorm",   It_hnorm_sc),
                            ("fg",      It_fg_sc)]:
            res[name] = dict(
                nmse    = round(nmse(pred, It_true), 4),
                spearman= round(mean_spearman(pred, It_true), 4),
                phi_hat = round(phi_hat, 4) if name == "kalman" else None,
            )
            print(f"    {name:8s}  NMSE={res[name]['nmse']:.4f}  "
                  f"rho_S={res[name]['spearman']:.4f}")

        rows.append(dict(
            phi=phi,
            phi_hat=round(phi_hat,4), sigma_hat=round(sigma_hat,4),
            kalman_NMSE  =res["kalman"]["nmse"],
            kalman_rhoS  =res["kalman"]["spearman"],
            hnorm_NMSE   =res["hnorm"]["nmse"],
            hnorm_rhoS   =res["hnorm"]["spearman"],
            fg_NMSE      =res["fg"]["nmse"],
            fg_rhoS      =res["fg"]["spearman"],
            It_true_t1   =round(float(It_true_1d[1]),4),
            It_true_t50  =round(float(It_true_1d[50]),4),
        ))

    # Save
    path = os.path.join(results_dir, "S5_kalman_partA.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: S5_kalman_partA.csv")
    return rows


# ── Part B: AR(1) with observation noise ─────────────────────────────────────

def run_part_B():
    print("\n" + "="*60)
    print("Part B: AR(1) with observation noise  (SNR=1)")
    print("="*60)

    sigma_eps = SIGMA_ETA / np.sqrt(SNR)
    rows = []

    for phi in PHIS:
        print(f"\n  phi = {phi}")
        rng2 = np.random.default_rng(SEED + 200 + int(phi*100))

        X_train, _ = gen_ar1_noisy(N_TRAIN, T, phi, SIGMA_ETA, sigma_eps, rng2)
        X_test,  Z_test = gen_ar1_noisy(N_TEST, T, phi, SIGMA_ETA, sigma_eps, rng2)

        # ── Oracle I_t: Kalman with *known* params ──
        It_oracle = np.array([
            kalman_filter_known(X_test[i], phi, SIGMA_ETA, sigma_eps)[2]
            for i in range(N_TEST)
        ])   # (N, T)

        # ── Kalman with EM-estimated params ──
        print("    EM estimation ...", end=" ", flush=True)
        phi_em, seta_em, seps_em = em_ssm(X_test, phi_init=0.5, n_iter=20)
        print(f"phi={phi_em:.3f}  s_eta={seta_em:.3f}  s_eps={seps_em:.3f}")
        It_kalman_em = np.array([
            kalman_filter_estimated(X_test[i], phi_em, seta_em, seps_em)[2]
            for i in range(N_TEST)
        ])

        # ── LSTM ──
        model    = train_lstm(X_train, seed=SEED + 200 + int(phi*100))
        It_hnorm = lstm_hnorm_It(model, X_test)
        It_fg    = lstm_fg_It(model, X_test)

        # Rescale to oracle scale
        scale_h  = It_oracle.mean() / (It_hnorm.mean() + 1e-12)
        scale_fg = It_oracle.mean() / (It_fg.mean()    + 1e-12)
        It_hnorm_sc = It_hnorm * scale_h
        It_fg_sc    = It_fg    * scale_fg

        for name, pred in [("kalman_em", It_kalman_em),
                            ("hnorm",    It_hnorm_sc),
                            ("fg",       It_fg_sc)]:
            n = nmse(pred, It_oracle)
            r = mean_spearman(pred, It_oracle)
            print(f"    {name:10s}  NMSE={n:.4f}  rho_S={r:.4f}")

        rows.append(dict(
            phi=phi,
            phi_em=round(phi_em,4), seta_em=round(seta_em,4), seps_em=round(seps_em,4),
            kalman_em_NMSE =round(nmse(It_kalman_em, It_oracle),4),
            kalman_em_rhoS =round(mean_spearman(It_kalman_em, It_oracle),4),
            hnorm_NMSE     =round(nmse(It_hnorm_sc, It_oracle),4),
            hnorm_rhoS     =round(mean_spearman(It_hnorm_sc, It_oracle),4),
            fg_NMSE        =round(nmse(It_fg_sc, It_oracle),4),
            fg_rhoS        =round(mean_spearman(It_fg_sc, It_oracle),4),
        ))

    path = os.path.join(results_dir, "S5_kalman_partB.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: S5_kalman_partB.csv")
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(rows_A, rows_B):
    # ── Fig 1: Example trajectories for phi=0.70 ─────────────────────────────
    phi = 0.70
    rng2 = np.random.default_rng(SEED + int(phi*100) + 999)
    sigma_eps = SIGMA_ETA / np.sqrt(SNR)

    # Part A
    X_train_a = gen_ar1_clean(N_TRAIN, T, phi, rng2)
    X_test_a  = gen_ar1_clean(20,      T, phi, rng2)
    model_a   = train_lstm(X_train_a, seed=SEED+99)
    phi_hat_a, sigma_hat_a = estimate_ar1_params(X_test_a)
    It_true_a  = np.tile(It_analytical(phi, T), (20, 1))
    It_kal_a   = np.tile(It_analytical(phi_hat_a, T, sigma_hat_a), (20, 1))
    It_h_a     = lstm_hnorm_It(model_a, X_test_a)
    It_h_a    *= It_true_a.mean() / (It_h_a.mean() + 1e-12)
    It_fg_a    = lstm_fg_It(model_a, X_test_a)
    It_fg_a   *= It_true_a.mean() / (It_fg_a.mean() + 1e-12)

    # Part B
    X_train_b, _ = gen_ar1_noisy(N_TRAIN, T, phi, SIGMA_ETA, sigma_eps, rng2)
    X_test_b, _  = gen_ar1_noisy(20,      T, phi, SIGMA_ETA, sigma_eps, rng2)
    model_b      = train_lstm(X_train_b, seed=SEED+299)
    phi_em, seta_em, seps_em = em_ssm(X_test_b, phi_init=0.5, n_iter=20)
    It_oracle_b  = np.array([
        kalman_filter_known(X_test_b[i], phi, SIGMA_ETA, sigma_eps)[2]
        for i in range(20)])
    It_kal_b = np.array([
        kalman_filter_estimated(X_test_b[i], phi_em, seta_em, seps_em)[2]
        for i in range(20)])
    It_h_b   = lstm_hnorm_It(model_b, X_test_b)
    It_h_b  *= It_oracle_b.mean() / (It_h_b.mean() + 1e-12)
    It_fg_b  = lstm_fg_It(model_b, X_test_b)
    It_fg_b *= It_oracle_b.mean() / (It_fg_b.mean() + 1e-12)

    t_arr = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, truth, kal, hn, fg, title in [
        (axes[0], It_true_a, It_kal_a, It_h_a, It_fg_a,
         "Part A: AR(1) no obs noise"),
        (axes[1], It_oracle_b, It_kal_b, It_h_b, It_fg_b,
         "Part B: AR(1) with obs noise  (SNR=1)"),
    ]:
        ax.plot(t_arr, truth.mean(0),    "k-",  lw=2,   label="True $\\mathcal{I}_t$")
        ax.plot(t_arr, kal.mean(0),   "--", color="steelblue",  lw=1.5, label="Kalman (est.)")
        ax.plot(t_arr, hn.mean(0),    "-.", color="darkorange",  lw=1.5, label="LSTM $\\|h_t-h_\\infty\\|$")
        ax.plot(t_arr, fg.mean(0),    ":",  color="firebrick",  lw=1.5, label="Forget gate $\\prod \\bar{f}_s$")
        ax.fill_between(t_arr,
            truth.mean(0) - truth.std(0), truth.mean(0) + truth.std(0),
            alpha=0.12, color="gray")
        ax.set_xlabel("$t$"); ax.set_ylabel("$\\hat{\\mathcal{I}}_t$")
        ax.set_title(f"{title}  ($\\phi={phi}$)")
        ax.legend(fontsize=8)

    plt.suptitle("Study S5: $\\mathcal{I}_t$ estimation — mean over 20 test sequences",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S5_kalman_trajectories.png"), dpi=120)
    plt.close()
    print("Saved: S5_kalman_trajectories.png")

    # ── Fig 2: NMSE comparison bar chart across phi ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colours = {"kalman": "steelblue", "hnorm": "darkorange", "fg": "firebrick",
               "kalman_em": "steelblue"}

    for ax, rows, part, keys, labels in [
        (axes[0], rows_A, "Part A",
         ["kalman_NMSE", "hnorm_NMSE", "fg_NMSE"],
         ["Kalman (plug-in)", "LSTM $\\|h_t-h_\\infty\\|$", "Forget gate"]),
        (axes[1], rows_B, "Part B",
         ["kalman_em_NMSE", "hnorm_NMSE", "fg_NMSE"],
         ["Kalman (EM)", "LSTM $\\|h_t-h_\\infty\\|$", "Forget gate"]),
    ]:
        phis_plot = [r["phi"] for r in rows]
        x = np.arange(len(phis_plot))
        width = 0.25
        cols   = ["steelblue", "darkorange", "firebrick"]
        for i, (key, label, col) in enumerate(zip(keys, labels, cols)):
            vals = [r[key] for r in rows]
            ax.bar(x + (i-1)*width, vals, width, label=label,
                   color=col, alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="gray", lw=1, ls="--", label="Naive (NMSE=1)")
        ax.set_xticks(x); ax.set_xticklabels([f"$\\phi={p}$" for p in phis_plot])
        ax.set_ylabel("NMSE"); ax.set_title(f"S5 {part}: NMSE by estimator")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S5_kalman_nmse.png"), dpi=120)
    plt.close()
    print("Saved: S5_kalman_nmse.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Study S5: Kalman Filter Benchmark for I_t estimation")
    print(f"  phis={PHIS}  T={T}  N_train={N_TRAIN}  N_test={N_TEST}  SNR={SNR}")

    rows_A = run_part_A()
    rows_B = run_part_B()
    make_figures(rows_A, rows_B)

    print(f"\nDone in {time.time()-t0:.1f}s")

    print("\n── Summary Table ──")
    print(f"{'':22s} {'Kalman':>10} {'LSTM h-norm':>12} {'Forget gate':>12}")
    print("Part A (no obs noise):")
    for r in rows_A:
        print(f"  phi={r['phi']:.2f}              "
              f"{r['kalman_NMSE']:>10.4f} {r['hnorm_NMSE']:>12.4f} {r['fg_NMSE']:>12.4f}")
    print("Part B (with obs noise, SNR=1):")
    for r in rows_B:
        print(f"  phi={r['phi']:.2f}              "
              f"{r['kalman_em_NMSE']:>10.4f} {r['hnorm_NMSE']:>12.4f} {r['fg_NMSE']:>12.4f}")
