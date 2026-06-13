# =============================================================================
#  study_S6_pathway_A_multivariate.py — Pathway A CUSUM in d > 1
#
#  Research question:
#    Does the Pathway A advantage (lower false-alarm rate, better ARL_0)
#    demonstrated in S2B for scalar AR(1) carry over to d-dimensional
#    VAR(1) data?
#
#  DGP:  VAR(1) with single abrupt change-point
#    X_t = Phi_before * X_{t-1} + eps_t,   t <= tau
#    X_t = Phi_after  * X_{t-1} + eps_t,   t >  tau
#    eps_t ~ N(0, I_d),  tau = 100,  T = 200
#    ||Phi_before||_op = 0.30,  ||Phi_after||_op = 0.85
#
#  Three methods compared at each d in {1, 2, 5, 10}:
#    M1  Raw CUSUM       — CUSUM on mean-sq standardised score
#    M2  VAR-whitened    — CUSUM on OLS VAR(1) residuals
#    M3  Pathway A       — CUSUM on LSTM prediction residuals
#
#  Multivariate aggregation (same threshold h=5 throughout):
#    Scalar input to CUSUM: Z_t = (1/d) * ||L^{-1}(X_t - mu0)||^2
#    where LL' = Sigma_0 (Cholesky), then centre by subtracting in-control
#    mean E[Z_t] ~ 1 and standardise by in-control std.
#    For LSTM: Z_t^{lstm} = (1/d) * ||X_t - X_hat_t||^2 / sigma_lstm^2
#
#  Metrics:
#    - Detection rate, false-alarm rate, mean/median/90th-pct delay (N=1000)
#    - ARL_0 under stationarity                                      (N=2000)
#
#  Outputs:
#    results/S6_pathway_A_multivariate.csv
#    results/S6_pathway_A_arl.csv
#    figures/S6_pathway_A_heatmap.png
#    figures/S6_pathway_A_delay_profile.png
# =============================================================================

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

script_dir  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
DIMS        = [1, 2, 5, 10]
RHO_BEFORE  = 0.30
RHO_AFTER   = 0.85
T           = 200
TAU         = 100
BURN        = 40            # first 20% for in-control estimation
K_REF       = 0.5
H           = 5.0
N_TRAIN     = 600
N_TEST      = 1000
N_ARL       = 2000
N_CAL       = 300           # calibration sequences for LSTM in-control stats
HIDDEN      = 32
EPOCHS      = 60
LR          = 3e-3
BATCH       = 64
SEED        = 2025


# ── VAR matrix constructor ─────────────────────────────────────────────────────

def make_var_matrix(d, rho_op, rng):
    if d == 1:
        return np.array([[rho_op]], dtype=np.float32)
    A = rng.standard_normal((d, d))
    U, _ = np.linalg.qr(A)
    lam = rng.uniform(-rho_op, rho_op, size=d)
    lam = lam / (np.max(np.abs(lam)) + 1e-12) * rho_op
    return (U @ np.diag(lam) @ U.T).astype(np.float32)


# ── Data generators ───────────────────────────────────────────────────────────

def stationary_cov(Phi):
    """Solve discrete Lyapunov: Gamma = Phi Gamma Phi' + I"""
    d = Phi.shape[0]
    G = np.eye(d, dtype=np.float64)
    Phi64 = Phi.astype(np.float64)
    for _ in range(500):
        G_new = Phi64 @ G @ Phi64.T + np.eye(d)
        if np.max(np.abs(G_new - G)) < 1e-10:
            break
        G = G_new
    return G


def gen_var1(N, T, Phi, rng):
    """VAR(1) from stationary init. Returns (N, T, d)."""
    d = Phi.shape[0]
    G = stationary_cov(Phi)
    L = np.linalg.cholesky(G + 1e-8 * np.eye(d))
    X = np.zeros((N, T, d), dtype=np.float32)
    X[:, 0, :] = (L @ rng.standard_normal((d, N))).T.astype(np.float32)
    eps = rng.standard_normal((N, T-1, d)).astype(np.float32)
    for t in range(1, T):
        X[:, t, :] = (X[:, t-1, :] @ Phi.T) + eps[:, t-1, :]
    return X


def gen_var1_cp(N, T, Phi_b, Phi_a, tau, rng):
    """VAR(1) with change-point at tau. Returns (N, T, d)."""
    d = Phi_b.shape[0]
    G = stationary_cov(Phi_b)
    L = np.linalg.cholesky(G + 1e-8 * np.eye(d))
    X = np.zeros((N, T, d), dtype=np.float32)
    X[:, 0, :] = (L @ rng.standard_normal((d, N))).T.astype(np.float32)
    eps = rng.standard_normal((N, T-1, d)).astype(np.float32)
    for t in range(1, T):
        Phi = Phi_b if t <= tau else Phi_a
        X[:, t, :] = (X[:, t-1, :] @ Phi.T) + eps[:, t-1, :]
    return X


# ── Multivariate LSTM ─────────────────────────────────────────────────────────

class MultivariateLSTM(nn.Module):
    def __init__(self, d, hidden_size=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d, hidden_size=hidden_size,
                            batch_first=True)
        self.lstm = nn.utils.spectral_norm(self.lstm, name='weight_hh_l0')
        self.head = nn.Linear(hidden_size, d)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out), out


def train_mv_lstm(d, X_train, seed=SEED):
    torch.manual_seed(seed)
    model = MultivariateLSTM(d=d, hidden_size=HIDDEN)
    opt   = optim.Adam(model.parameters(), lr=LR)
    N     = len(X_train)
    for ep in range(EPOCHS):
        idx = np.random.permutation(N)
        for s in range(0, N, BATCH):
            xb = torch.tensor(X_train[idx[s:s+BATCH]], dtype=torch.float32)
            opt.zero_grad()
            pred, _ = model(xb[:, :-1, :])
            loss = ((pred - xb[:, 1:, :]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


# ── Signal constructors ───────────────────────────────────────────────────────

def signal_raw_mv(x, burn=BURN):
    """
    M1: aggregate multivariate raw score.
    Estimate Sigma_0 from burn-in, compute Z_t = (1/d)||Sigma_0^{-1/2}(x_t-mu0)||^2.
    Centre and standardise using burn-in stats.
    Returns z: (T,)
    """
    d = x.shape[1]
    mu0    = x[:burn].mean(axis=0)                          # (d,)
    cov0   = np.cov(x[:burn].T) if d > 1 else np.array([[x[:burn].var()]])
    cov0  += 1e-6 * np.eye(d)
    L_inv  = np.linalg.inv(np.linalg.cholesky(cov0))       # (d, d)
    scores = np.array([
        (L_inv @ (x[t] - mu0)) @ (L_inv @ (x[t] - mu0)) / d
        for t in range(len(x))
    ])
    # Standardise: under H0, E[Z_t] ~ 1
    z_mu  = scores[:burn].mean()
    z_std = max(scores[:burn].std(), 1e-8)
    return (scores - z_mu) / z_std


def signal_whitened_mv(x, burn=BURN):
    """
    M2: VAR(1) OLS whitening.
    Fit A_hat on burn-in window, compute residuals e_t = x_t - A_hat x_{t-1}.
    Aggregate: Z_t = (1/d)||Sigma_e^{-1/2} e_t||^2, centre and standardise.
    """
    d = x.shape[1]
    Y = x[1:burn];  Z = x[:burn-1]
    # OLS: A_hat = (Z'Z)^{-1} Z'Y  => (d, d)
    A_hat = np.linalg.lstsq(Z, Y, rcond=None)[0]           # (d, d)
    resid = x[1:] - x[:-1] @ A_hat                         # (T-1, d)
    cov_e = np.cov(resid[:burn].T) if d > 1 else np.array([[resid[:burn].var()]])
    cov_e += 1e-6 * np.eye(d)
    L_inv = np.linalg.inv(np.linalg.cholesky(cov_e))
    scores = np.zeros(len(x))
    for t in range(1, len(x)):
        e = x[t] - x[t-1] @ A_hat
        scores[t] = (L_inv @ e) @ (L_inv @ e) / d
    z_mu  = scores[1:burn].mean()
    z_std = max(scores[1:burn].std(), 1e-8)
    return (scores - z_mu) / z_std


def build_lstm_signal_fn(model, mu_r, sig_r, d):
    """Returns a function x -> z using the trained LSTM."""
    def signal_pathway_a(x):
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[None, :, :], dtype=torch.float32)  # (1,T,d)
            pred, _ = model(xt[:, :-1, :])                          # (1,T-1,d)
            pred_np  = pred.squeeze(0).numpy()                      # (T-1, d)
        scores = np.zeros(len(x))
        for t in range(1, len(x)):
            e = x[t] - pred_np[t-1]
            scores[t] = (e @ e) / d                                  # mean sq residual
        z_mu  = mu_r
        z_std = max(sig_r, 1e-8)
        return (scores - z_mu) / z_std
    return signal_pathway_a


def calibrate_lstm(model, X_cal, d):
    """Estimate in-control (mean, std) of (1/d)||e_t||^2 on stationary data."""
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i in range(len(X_cal)):
            x = X_cal[i]
            xt = torch.tensor(x[None, :, :], dtype=torch.float32)
            pred, _ = model(xt[:, :-1, :])
            pred_np  = pred.squeeze(0).numpy()
            for t in range(1, len(x)):
                e = x[t] - pred_np[t-1]
                all_scores.append((e @ e) / d)
    arr = np.array(all_scores)
    return float(arr.mean()), float(arr.std())


# ── Page CUSUM ────────────────────────────────────────────────────────────────

def page_cusum(z, k=K_REF, h=H, start=BURN):
    Sp = Sm = 0.0
    for t in range(start, len(z)):
        zt = float(z[t])
        Sp = max(0.0, Sp + zt - k)
        Sm = max(0.0, Sm - zt - k)
        if Sp > h or Sm > h:
            return t
    return None


# ── Run one (d, method) configuration ────────────────────────────────────────

def evaluate(d, Phi_b, Phi_a, rng, model=None, mu_r=None, sig_r=None):
    """
    Run detection analysis for one (d, Phi_b, Phi_a) setting.
    Returns dict with metrics for M1, M2, M3.
    """
    X_cp = gen_var1_cp(N_TEST, T, Phi_b, Phi_a, TAU, rng)

    sig_fns = {
        "raw":      lambda x: signal_raw_mv(x),
        "whitened": lambda x: signal_whitened_mv(x),
        "pathway_a": build_lstm_signal_fn(model, mu_r, sig_r, d),
    }

    out = {}
    for name, fn in sig_fns.items():
        delays = []; fa = 0
        for i in range(N_TEST):
            z     = fn(X_cp[i])
            alarm = page_cusum(z)
            if alarm is None:
                delays.append(T - TAU)
            else:
                delays.append(alarm - TAU)
                if alarm < TAU:
                    fa += 1
        d_arr   = np.array(delays)
        det     = d_arr < (T - TAU)
        d_det   = d_arr[det] if det.any() else np.array([np.nan])
        out[name] = dict(
            detect_rate      = round(float(det.mean()), 4),
            false_alarm_rate = round(fa / N_TEST, 4),
            delay_mean       = round(float(d_det.mean()), 2),
            delay_median     = round(float(np.median(d_det)), 2),
            delay_90pct      = round(float(np.percentile(d_det, 90)), 2),
        )
    return out


def evaluate_arl(d, Phi_b, rng, model=None, mu_r=None, sig_r=None):
    """ARL_0 on pure stationary data (no change-point)."""
    T_long = T * 5
    X_pure = gen_var1(N_ARL, T_long, Phi_b, rng)

    sig_fns = {
        "raw":      lambda x: signal_raw_mv(x),
        "whitened": lambda x: signal_whitened_mv(x),
        "pathway_a": build_lstm_signal_fn(model, mu_r, sig_r, d),
    }

    arls = {}
    for name, fn in sig_fns.items():
        runs = []
        for i in range(N_ARL):
            z     = fn(X_pure[i])
            alarm = page_cusum(z, start=BURN)
            runs.append(alarm if alarm is not None else T_long)
        arls[name] = round(float(np.mean(runs)), 1)
    return arls


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Study S6: Pathway A CUSUM — multivariate extension")
    print(f"  dims={DIMS}  rho_before={RHO_BEFORE}  rho_after={RHO_AFTER}")
    print(f"  tau={TAU}  T={T}  h={H}  N_test={N_TEST}  N_arl={N_ARL}")

    det_rows = []
    arl_rows = []

    for d in DIMS:
        print(f"\n{'='*55}")
        print(f"  d = {d}")
        rng = np.random.default_rng(SEED + d * 37)

        Phi_b = make_var_matrix(d, RHO_BEFORE, rng)
        Phi_a = make_var_matrix(d, RHO_AFTER,  rng)

        # Train LSTM on pre-change stationary data
        print(f"  Training LSTM ({N_TRAIN} seqs) ...", end=" ", flush=True)
        X_train = gen_var1(N_TRAIN, T, Phi_b, rng)
        model   = train_mv_lstm(d, X_train, seed=SEED + d)
        print("done.")

        # Calibrate LSTM in-control statistics
        X_cal  = gen_var1(N_CAL, T, Phi_b, rng)
        mu_r, sig_r = calibrate_lstm(model, X_cal, d)
        print(f"  LSTM in-control: mean={mu_r:.4f}  std={sig_r:.4f}")

        # Detection analysis
        print(f"  Detection analysis (N={N_TEST}) ...")
        det = evaluate(d, Phi_b, Phi_a, rng, model, mu_r, sig_r)
        for name in ["raw", "whitened", "pathway_a"]:
            m = det[name]
            print(f"    {name:12s} detect={m['detect_rate']:.3f}  "
                  f"FA={m['false_alarm_rate']:.3f}  "
                  f"delay={m['delay_mean']:+.1f}/{m['delay_median']:+.1f}/"
                  f"{m['delay_90pct']:+.1f}")
            det_rows.append(dict(d=d, method=name, **m))

        # ARL_0
        print(f"  ARL_0 (N={N_ARL}) ...")
        arls = evaluate_arl(d, Phi_b, rng, model, mu_r, sig_r)
        for name, arl in arls.items():
            print(f"    {name:12s} ARL_0={arl:.0f}")
            arl_rows.append(dict(d=d, method=name, ARL0=arl))

    # ── Save ──────────────────────────────────────────────────────────────────
    path = os.path.join(results_dir, "S6_pathway_A_multivariate.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=det_rows[0].keys())
        w.writeheader(); w.writerows(det_rows)
    print(f"\nSaved: S6_pathway_A_multivariate.csv")

    path_arl = os.path.join(results_dir, "S6_pathway_A_arl.csv")
    with open(path_arl, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=arl_rows[0].keys())
        w.writeheader(); w.writerows(arl_rows)
    print(f"Saved: S6_pathway_A_arl.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    methods = ["raw", "whitened", "pathway_a"]
    labels  = ["M1: Raw", "M2: Whitened", "M3: Pathway A"]
    colours = ["firebrick", "darkorange", "steelblue"]

    # Fig 1: FA rate and ARL_0 vs dimension
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ylabel, title, ref in [
        (axes[0], "false_alarm_rate", "False-alarm rate",
         "FA rate vs dimension", 0.05),
        (axes[1], "delay_mean", "Mean detection delay",
         "Mean delay vs dimension", 0),
        (axes[2], "detect_rate", "Detection rate",
         "Detection rate vs dimension", 1.0),
    ]:
        for method, label, col in zip(methods, labels, colours):
            vals = [r[metric] for r in det_rows if r["method"] == method]
            axes_d = [r["d"] for r in det_rows if r["method"] == method]
            ax.plot(axes_d, vals, "o-", color=col, label=label, lw=1.5)
        ax.axhline(ref, color="gray", lw=1, ls="--")
        ax.set_xlabel("Dimension $d$"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.set_xticks(DIMS)
        ax.legend(fontsize=8)

    plt.suptitle(f"S6: Three-way CUSUM — multivariate VAR(1)  "
                 f"($\\rho_{{\\mathrm{{op}}}}: {RHO_BEFORE}\\to{RHO_AFTER}$, "
                 f"$\\tau={TAU}$, $h={H}$)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S6_pathway_A_heatmap.png"), dpi=120)
    plt.close()
    print("Saved: S6_pathway_A_heatmap.png")

    # Fig 2: ARL_0 vs dimension
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label, col in zip(methods, labels, colours):
        vals = [r["ARL0"] for r in arl_rows if r["method"] == method]
        ds   = [r["d"]    for r in arl_rows if r["method"] == method]
        ax.plot(ds, vals, "o-", color=col, label=label, lw=1.5)
    ax.axhline(285, color="gray", lw=1, ls="--", label="Theoretical i.i.d. (h=5)")
    ax.set_xlabel("Dimension $d$"); ax.set_ylabel("$\\mathrm{ARL}_0$")
    ax.set_title("In-control $\\mathrm{ARL}_0$ vs dimension")
    ax.set_xticks(DIMS); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S6_pathway_A_delay_profile.png"), dpi=120)
    plt.close()
    print("Saved: S6_pathway_A_delay_profile.png")

    print(f"\nDone in {time.time()-t0:.1f}s")

    # Summary
    print(f"\n── Summary (h={H}) ──")
    print(f"{'d':>3} {'method':>12} {'detect':>7} {'FA':>7} "
          f"{'mean delay':>11} {'ARL_0':>7}")
    for d in DIMS:
        for name in methods:
            dr = next(r for r in det_rows if r["d"]==d and r["method"]==name)
            ar = next(r for r in arl_rows  if r["d"]==d and r["method"]==name)
            print(f"{d:>3} {name:>12} {dr['detect_rate']:>7.3f} "
                  f"{dr['false_alarm_rate']:>7.3f} "
                  f"{dr['delay_mean']:>+11.1f} {ar['ARL0']:>7.0f}")
