# =============================================================================
#  study_S4_multivariate.py — Study S4: Multivariate / High-Dimensional Setting
#
#  Research question:
#    Does the martingale decomposition and information-gain framework extend
#    to d-dimensional VAR(1) processes?  Specifically:
#      (a) Do LSTM innovations remain MDS in d > 1?          [S4A]
#      (b) Does the true I_t scale as ||Phi^t||_op, and does
#          the LSTM forget gate track this norm?             [S4B]
#      (c) How does performance degrade as dimension d grows? [S4C]
#
#  DGP: VAR(1)   X_t = Phi * X_{t-1} + eps_t,  eps_t ~ N(0, I_d)
#  Phi constructed as  Phi = U * diag(lambda) * U',  U random orthogonal,
#  so that ||Phi||_op = max|lambda_i|.
#  Three operator-norm levels tested: rho_op in {0.30, 0.70, 0.95}
#  Three dimensions:                  d      in {2, 5, 10}
#
#  Outputs (saved to results/):
#    S4_mds.csv      — multivariate Portmanteau (Hosking) MDS pass rates
#    S4_It_norm.csv  — NMSE of ||Phi^t||_op vs various estimators
#    S4_dim.csv      — dimension-scaling of MDS pass rate and NMSE
#    S4_log.txt      — run log
#
#  Figures (saved to figures/):
#    S4_mds_heatmap.png
#    S4_It_norm_trajectories.png
#    S4_dim_scaling.png
# =============================================================================

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir  = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

log_path = os.path.join(results_dir, "S4_log.txt")
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ── Configuration ─────────────────────────────────────────────────────────────

RHO_OPS   = [0.30, 0.70, 0.95]   # operator norms of Phi
DIMS      = [2, 5, 10]            # d
T         = 200
N_TRAIN   = 500
N_TEST    = 300
HIDDEN    = 32
EPOCHS    = 60
LR        = 3e-3
BATCH     = 64
SEED_BASE = 2025
LB_LAG    = 10                    # lag for Hosking portmanteau test

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_var_matrix(d, rho_op, rng):
    """
    Construct a d×d VAR companion matrix with ||Phi||_op == rho_op.
    Phi = U diag(lambdas) U',  lambdas sampled uniform in [-rho_op, rho_op]
    then rescaled so max|lambda| == rho_op exactly.
    """
    # Random orthogonal U via QR decomposition
    A = rng.standard_normal((d, d))
    U, _ = np.linalg.qr(A)
    # Random eigenvalues in (-rho_op, rho_op)
    lam = rng.uniform(-rho_op, rho_op, size=d)
    # Ensure the operator norm equals rho_op exactly
    lam = lam / (np.max(np.abs(lam)) + 1e-12) * rho_op
    Phi = U @ np.diag(lam) @ U.T
    assert np.max(np.abs(np.linalg.eigvals(Phi))) <= rho_op + 1e-8
    return Phi.astype(np.float32)


def generate_var1(N, T, Phi, rng):
    """
    Generate N independent VAR(1) sequences of length T.
    X_t = Phi X_{t-1} + eps_t,  eps_t ~ N(0, I_d)
    Initialised from stationary distribution (vec(Gamma_0) = (I-Phi⊗Phi)^{-1} vec(I)).

    Returns X: (N, T, d)
    """
    d = Phi.shape[0]
    # Stationary covariance via Lyapunov: Gamma = Phi Gamma Phi' + I
    # Solved iteratively
    G = np.eye(d, dtype=np.float64)
    Phi64 = Phi.astype(np.float64)
    for _ in range(500):
        G_new = Phi64 @ G @ Phi64.T + np.eye(d)
        if np.max(np.abs(G_new - G)) < 1e-10:
            break
        G = G_new
    L = np.linalg.cholesky(G + 1e-8 * np.eye(d))

    X = np.zeros((N, T, d), dtype=np.float32)
    X[:, 0, :] = (L @ rng.standard_normal((d, N))).T.astype(np.float32)
    eps = rng.standard_normal((N, T - 1, d)).astype(np.float32)
    for t in range(1, T):
        X[:, t, :] = (X[:, t-1, :] @ Phi.T) + eps[:, t-1, :]
    return X


def true_It_norm(Phi, T):
    """
    Theoretical I_t ∝ ||Phi^t||_op for a VAR(1).
    Returns array of length T (index 0 = t=1).
    """
    d = Phi.shape[0]
    vals = np.zeros(T)
    Pt = np.eye(d, dtype=np.float64)
    Phi64 = Phi.astype(np.float64)
    for t in range(T):
        Pt = Pt @ Phi64
        vals[t] = np.linalg.norm(Pt, ord=2)  # operator norm = largest singular value
    return vals.astype(np.float32)


# ── Multivariate LSTM ─────────────────────────────────────────────────────────

class MultivariateLSTM(nn.Module):
    """
    LSTM for d-dimensional one-step-ahead prediction.
    Input:  (batch, T, d)
    Output: predictions (batch, T, d),
            forget gates (batch, T, hidden),
            hidden states (batch, T, hidden)
    """
    def __init__(self, d, hidden_size):
        super().__init__()
        self.d = d
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=d, hidden_size=hidden_size,
                            batch_first=True)
        # Spectral norm on recurrent weight (weight_hh_l0)
        self.lstm = nn.utils.spectral_norm(self.lstm, name='weight_hh_l0')
        self.head = nn.Linear(hidden_size, d)

    def forward(self, x):
        """x: (batch, T, d)"""
        out, _ = self.lstm(x)           # out: (batch, T, hidden)
        pred = self.head(out)           # (batch, T, d)
        return pred, out                # return preds and hidden states

    def get_forget_gates(self, x):
        """
        Extract forget gate f_t for each time step by manual unrolling.
        Returns f: (batch, T, hidden)
        """
        batch, T, d = x.shape
        h = torch.zeros(1, batch, self.hidden_size, device=x.device)
        c = torch.zeros(1, batch, self.hidden_size, device=x.device)
        fs = []
        # Access underlying LSTM weights
        lstm_raw = self.lstm  # spectral_norm wraps the module
        for t in range(T):
            xt = x[:, t, :]             # (batch, d)
            _, (h, c) = lstm_raw(xt.unsqueeze(1), (h, c))
            # Re-compute gates to extract f_t
            # weight_ih_l0: (4*hidden, d), weight_hh_l0: (4*hidden, hidden)
            try:
                W_ih = lstm_raw.weight_ih_l0
                W_hh = lstm_raw.weight_hh_l0
                b_ih = lstm_raw.bias_ih_l0
                b_hh = lstm_raw.bias_hh_l0
            except AttributeError:
                # After spectral_norm wrapping the attr name may differ
                W_ih = getattr(lstm_raw, 'weight_ih_l0', None)
                W_hh = getattr(lstm_raw, 'weight_hh_l0_orig',
                               getattr(lstm_raw, 'weight_hh_l0', None))
                b_ih = lstm_raw.bias_ih_l0
                b_hh = lstm_raw.bias_hh_l0

            gates = (xt @ W_ih.T + h.squeeze(0) @ W_hh.T
                     + b_ih.unsqueeze(0) + b_hh.unsqueeze(0))
            _, f_raw, _, _ = gates.chunk(4, dim=-1)
            fs.append(torch.sigmoid(f_raw).unsqueeze(1))
        return torch.cat(fs, dim=1)     # (batch, T, hidden)


def train_model(model, X_train, epochs=EPOCHS, lr=LR, batch=BATCH, seed=0):
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    N, T, d = X_train.shape
    losses = []
    for ep in range(epochs):
        idx = np.random.permutation(N)
        ep_loss = 0.0; n_batches = 0
        for start in range(0, N, batch):
            xb = torch.tensor(X_train[idx[start:start+batch]], dtype=torch.float32)
            optimizer.zero_grad()
            pred, _ = model(xb[:, :-1, :])        # predict t+1 from t
            target   = xb[:, 1:, :]
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1
        losses.append(ep_loss / n_batches)
    return losses


def extract_innovations(model, X):
    """
    Compute innovations M_t = h_t - E[h_t | h_{t-1}]  (approximated as
    h_t - pred_t for scalar case; for multivariate we use prediction residuals).

    Returns residuals: (N, T-1, d)
    """
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        pred, _ = model(xt[:, :-1, :])
        resid = xt[:, 1:, :] - pred
    return resid.numpy()


# ── Hosking multivariate portmanteau test ─────────────────────────────────────

def hosking_test(E, lag=LB_LAG):
    """
    Hosking (1980) multivariate portmanteau test for white noise.
    E: (N, T, d) array of residual sequences.
    Tests each sequence independently and returns average p-value and pass rate.

    For each sequence e_t (length T, dim d):
      Q = T * sum_{k=1}^{lag} tr( C_k' C_0^{-1} C_k C_0^{-1} )
      Q ~ chi^2(d^2 * lag) under H0 (white noise)
    """
    from scipy.stats import chi2
    N, T, d = E.shape
    df = d * d * lag
    pvals = []
    for i in range(N):
        e = E[i]                     # (T, d)
        C0 = (e.T @ e) / T
        try:
            C0inv = np.linalg.inv(C0 + 1e-8 * np.eye(d))
        except np.linalg.LinAlgError:
            pvals.append(np.nan); continue
        Q = 0.0
        for k in range(1, lag + 1):
            Ck = (e[k:].T @ e[:-k]) / T    # (d, d)
            Q += np.trace(Ck.T @ C0inv @ Ck @ C0inv)
        Q *= T
        pvals.append(1.0 - chi2.cdf(Q, df=df))
    pvals = np.array(pvals)
    pass_rate = np.mean(pvals[np.isfinite(pvals)] > 0.05)
    mean_pval = np.nanmean(pvals)
    return pass_rate, mean_pval


# ── VAR OLS baseline ──────────────────────────────────────────────────────────

def var_ols_residuals(X):
    """
    Fit VAR(1) by OLS: X_t = A X_{t-1} + eps_t.
    Returns residuals (N, T-1, d).
    """
    N, T, d = X.shape
    # Stack all sequences
    Y = X[:, 1:, :].reshape(-1, d)      # ((N*(T-1)), d)
    Z = X[:, :-1, :].reshape(-1, d)     # ((N*(T-1)), d)
    A_hat = np.linalg.lstsq(Z, Y, rcond=None)[0]  # (d, d)
    resid = (Y - Z @ A_hat).reshape(N, T-1, d)
    return resid.astype(np.float32), A_hat


# ── Study S4A: MDS test across dimensions and rho_op ─────────────────────────

def run_S4A():
    log("\n" + "="*60)
    log("Study S4A: Multivariate MDS (Hosking) Test")
    log("="*60)

    rows = []
    for d in DIMS:
        for rho_op in RHO_OPS:
            rng = np.random.default_rng(SEED_BASE + d * 100 + int(rho_op * 100))
            Phi = make_var_matrix(d, rho_op, rng)

            # Generate data
            X_train = generate_var1(N_TRAIN, T, Phi, rng)
            X_test  = generate_var1(N_TEST,  T, Phi, rng)

            # Train LSTM
            model = MultivariateLSTM(d=d, hidden_size=HIDDEN)
            train_model(model, X_train, seed=SEED_BASE)

            # LSTM residuals
            E_lstm = extract_innovations(model, X_test)  # (N, T-1, d)
            lstm_pass, lstm_pval = hosking_test(E_lstm, lag=LB_LAG)

            # OLS residuals
            E_ols, A_hat = var_ols_residuals(X_test)
            ols_pass, ols_pval = hosking_test(E_ols, lag=LB_LAG)

            # Op-norm of fitted A
            A_opnorm = np.linalg.norm(A_hat, ord=2)

            log(f"  d={d:2d}  rho_op={rho_op:.2f} | "
                f"LSTM pass={lstm_pass:.3f} (p={lstm_pval:.4f})  "
                f"OLS  pass={ols_pass:.3f} (p={ols_pval:.4f})  "
                f"||A_hat||_op={A_opnorm:.3f}")

            rows.append(dict(
                d=d, rho_op=rho_op,
                LSTM_pass_rate=round(lstm_pass, 4),
                LSTM_mean_pval=round(lstm_pval, 4),
                OLS_pass_rate=round(ols_pass, 4),
                OLS_mean_pval=round(ols_pval, 4),
                A_hat_opnorm=round(A_opnorm, 4),
            ))

    # Save
    import csv
    path = os.path.join(results_dir, "S4_mds.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    log(f"Saved: S4_mds.csv")

    # Heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, method, key in zip(axes, ["LSTM", "OLS"],
                                ["LSTM_pass_rate", "OLS_pass_rate"]):
        mat = np.array([[r[key] for r in rows if r["d"] == d] for d in DIMS])
        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto",
                       cmap="RdYlGn", origin="upper")
        ax.set_xticks(range(len(RHO_OPS))); ax.set_xticklabels(RHO_OPS)
        ax.set_yticks(range(len(DIMS)));    ax.set_yticklabels(DIMS)
        ax.set_xlabel(r"$\|\Phi\|_{\mathrm{op}}$"); ax.set_ylabel("Dimension $d$")
        ax.set_title(f"{method} — MDS pass rate (Hosking, lag={LB_LAG})")
        for i, d in enumerate(DIMS):
            for j, rho in enumerate(RHO_OPS):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S4_mds_heatmap.png"), dpi=120)
    plt.close()
    log("Saved: S4_mds_heatmap.png")
    return rows


# ── Study S4B: I_t norm tracking ─────────────────────────────────────────────

def run_S4B():
    log("\n" + "="*60)
    log("Study S4B: True ||Phi^t||_op vs LSTM forget gate")
    log("="*60)

    rows = []
    fig, axes = plt.subplots(len(DIMS), len(RHO_OPS),
                             figsize=(12, 3 * len(DIMS)), sharex=True)

    for i, d in enumerate(DIMS):
        for j, rho_op in enumerate(RHO_OPS):
            rng = np.random.default_rng(SEED_BASE + d * 100 + int(rho_op * 100))
            Phi = make_var_matrix(d, rho_op, rng)

            X_train = generate_var1(N_TRAIN, T, Phi, rng)
            X_test  = generate_var1(N_TEST,  T, Phi, rng)

            model = MultivariateLSTM(d=d, hidden_size=HIDDEN)
            train_model(model, X_train, seed=SEED_BASE)

            # True I_t proxy: ||Phi^t||_op
            It_true = true_It_norm(Phi, T)   # length T

            # LSTM forget gate mean over hidden units and test sequences
            model.eval()
            try:
                xt = torch.tensor(X_test, dtype=torch.float32)
                fg = model.get_forget_gates(xt)   # (N, T, hidden)
                fg_mean = fg.detach().numpy().mean(axis=(0, 2))  # (T,)
                fg_std  = fg.detach().numpy().mean(axis=2).std(axis=0)
            except Exception as e:
                log(f"  WARNING: forget gate extraction failed d={d} rho={rho_op}: {e}")
                fg_mean = np.full(T, np.nan)
                fg_std  = np.full(T, np.nan)

            # Spearman correlation between fg_mean and It_true
            from scipy.stats import spearmanr
            rho_s, p_s = spearmanr(fg_mean[np.isfinite(fg_mean)],
                                    It_true[np.isfinite(fg_mean)])

            log(f"  d={d:2d}  rho_op={rho_op:.2f} | "
                f"Spearman(fg, ||Phi^t||_op) = {rho_s:.3f}  p={p_s:.4f}")

            rows.append(dict(
                d=d, rho_op=rho_op,
                spearman_rho=round(float(rho_s), 4),
                spearman_pval=round(float(p_s), 4),
                fg_mean_final=round(float(np.nanmean(fg_mean[-10:])), 4),
                It_final=round(float(It_true[-1]), 6),
            ))

            # Plot
            ax = axes[i][j] if len(DIMS) > 1 else axes[j]
            t_arr = np.arange(1, T + 1)
            It_norm = It_true / (It_true[0] + 1e-12)   # normalise for overlay
            ax.plot(t_arr, It_norm, "k-", lw=1.5, label=r"$\|\Phi^t\|_{op}$ (norm.)")
            ax.plot(t_arr, fg_mean, color="steelblue", lw=1.2, label=r"$\bar{f}_t$")
            ax.fill_between(t_arr, fg_mean - fg_std, fg_mean + fg_std,
                            alpha=0.25, color="steelblue")
            ax.set_title(f"d={d}, ρ_op={rho_op}  ρ_S={rho_s:.2f}", fontsize=9)
            if i == len(DIMS) - 1: ax.set_xlabel("t")
            if j == 0: ax.set_ylabel(r"$\bar{f}_t$ / $\|\Phi^t\|$")
            ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(r"Study S4B: Mean forget gate vs $\|\Phi^t\|_{\mathrm{op}}$",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S4_It_norm_trajectories.png"), dpi=120)
    plt.close()
    log("Saved: S4_It_norm_trajectories.png")

    import csv
    path = os.path.join(results_dir, "S4_It_norm.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    log("Saved: S4_It_norm.csv")
    return rows


# ── Study S4C: Dimension scaling ─────────────────────────────────────────────

def run_S4C():
    log("\n" + "="*60)
    log("Study S4C: Dimension scaling (fixed rho_op=0.70)")
    log("="*60)

    rho_op = 0.70
    dims_ext = [1, 2, 5, 10, 20]
    rows = []

    for d in dims_ext:
        rng = np.random.default_rng(SEED_BASE + d * 100 + int(rho_op * 100))
        Phi = make_var_matrix(d, rho_op, rng) if d > 1 else np.array([[rho_op]], dtype=np.float32)

        X_train = generate_var1(N_TRAIN, T, Phi, rng)
        X_test  = generate_var1(N_TEST,  T, Phi, rng)

        model = MultivariateLSTM(d=d, hidden_size=HIDDEN)
        train_model(model, X_train, seed=SEED_BASE)

        # MDS test
        E_lstm = extract_innovations(model, X_test)
        lstm_pass, lstm_pval = hosking_test(E_lstm, lag=LB_LAG)

        # OLS
        E_ols, A_hat = var_ols_residuals(X_test)
        ols_pass, ols_pval = hosking_test(E_ols, lag=LB_LAG)

        # Prediction MSE
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_test, dtype=torch.float32)
            pred, _ = model(xt[:, :-1, :])
            mse = ((pred - xt[:, 1:, :]) ** 2).mean().item()

        log(f"  d={d:2d} | LSTM pass={lstm_pass:.3f}  OLS pass={ols_pass:.3f}  "
            f"MSE={mse:.4f}")

        rows.append(dict(
            d=d, rho_op=rho_op,
            LSTM_pass_rate=round(lstm_pass, 4),
            OLS_pass_rate=round(ols_pass, 4),
            LSTM_MSE=round(mse, 5),
        ))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ds = [r["d"] for r in rows]
    axes[0].plot(ds, [r["LSTM_pass_rate"] for r in rows], "o-",
                 color="steelblue", label="LSTM")
    axes[0].plot(ds, [r["OLS_pass_rate"] for r in rows], "s--",
                 color="coral", label="OLS")
    axes[0].axhline(0.95, color="gray", lw=1, ls=":")
    axes[0].set_xlabel("Dimension $d$"); axes[0].set_ylabel("MDS pass rate")
    axes[0].set_title(r"MDS pass rate vs $d$  ($\|\Phi\|_{\mathrm{op}}=0.70$)")
    axes[0].legend(); axes[0].set_xticks(ds)

    axes[1].plot(ds, [r["LSTM_MSE"] for r in rows], "o-", color="steelblue")
    axes[1].set_xlabel("Dimension $d$"); axes[1].set_ylabel("Prediction MSE")
    axes[1].set_title("LSTM prediction MSE vs $d$")
    axes[1].set_xticks(ds)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S4_dim_scaling.png"), dpi=120)
    plt.close()
    log("Saved: S4_dim_scaling.png")

    import csv
    path = os.path.join(results_dir, "S4_dim.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    log("Saved: S4_dim.csv")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    log(f"Study S4: Multivariate/High-Dimensional VAR(1)")
    log(f"Dims={DIMS}  rho_ops={RHO_OPS}  T={T}  N_train={N_TRAIN}  N_test={N_TEST}")

    rows_S4A = run_S4A()
    rows_S4B = run_S4B()
    rows_S4C = run_S4C()

    elapsed = time.time() - t0
    log(f"\nStudy S4 complete in {elapsed:.1f}s")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    # ── Summary ──
    log("\n── S4A Summary: LSTM MDS pass rates ──")
    for r in rows_S4A:
        log(f"  d={r['d']:2d}  rho_op={r['rho_op']:.2f}  "
            f"LSTM={r['LSTM_pass_rate']:.3f}  OLS={r['OLS_pass_rate']:.3f}")

    log("\n── S4B Summary: Forget gate vs ||Phi^t||_op ──")
    for r in rows_S4B:
        log(f"  d={r['d']:2d}  rho_op={r['rho_op']:.2f}  "
            f"rho_S={r['spearman_rho']:.3f}  p={r['spearman_pval']:.4f}")
