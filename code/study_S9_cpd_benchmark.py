# =============================================================================
#  study_S9_cpd_benchmark.py — CPD benchmark: PELT vs Pathway A (M3)
#
#  Research question:
#    How does PELT (Killick et al. 2012) compare with Pathway A across
#    dimensions d in {1, 2, 5, 10} on the same VAR(1) change-point DGP
#    used in Study S6?
#
#  Design:
#    - Identical DGP to S6 (VAR(1), tau=100, T=200, rho_before=0.30,
#      rho_after=0.85, N_test=1000, N_arl=500)
#    - PELT uses ruptures (l2 cost, penalty=7.0 calibrated at d=1 to
#      match Pathway A's ARL_0; held fixed across d to test stability)
#    - Pathway A re-trained with same architecture as S6 (LSTM, 32 units)
#    - Metrics: detection rate, false-alarm rate, mean delay, ARL_0
#
#  Outputs:
#    results/S9_cpd_benchmark.csv
#    figures/S9_pelt_vs_pathwayA.png
#
#  Runtime: ~15–25 min (LSTM training dominates; PELT itself is fast)
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

try:
    import ruptures as rpt
except ImportError:
    raise ImportError("ruptures not found. Run: pip install ruptures")

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

script_dir  = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ── Parameters (identical to S6) ─────────────────────────────────────────────
DIMS        = [1, 2, 5, 10]
RHO_BEFORE  = 0.30
RHO_AFTER   = 0.85
T           = 200
TAU         = 100
BURN        = 40
K_REF       = 0.5
H           = 5.0
N_TRAIN     = 600
N_TEST      = 1000
N_ARL       = 500     # reduced for speed (1000 gives stable estimates)
N_CAL       = 300
HIDDEN      = 32
EPOCHS      = 60
LR          = 3e-3
BATCH       = 64
SEED        = 2025

# PELT parameters
# model="l2" is O(n log n) expected; much faster than rbf for long sequences.
# L2 cost detects changes in mean+variance, appropriate for VAR(1) persistence change.
# Penalty pre-calibrated to ARL_0 ≈ 200 at d=1 on T_long=600 in-control sequences
# (grid search: pen=7 → ARL_0=202.3; pen=6 → ARL_0=150.3).
PELT_MODEL  = "l2"
PELT_PEN    = 7.0     # fixed, matches Pathway A's ARL_0 ≈ 187 at d=1


# ── VAR(1) helpers (same as S6) ───────────────────────────────────────────────

def make_var_matrix(d, rho_op, rng):
    if d == 1:
        return np.array([[rho_op]], dtype=np.float32)
    A = rng.standard_normal((d, d))
    U, _ = np.linalg.qr(A)
    lam = rng.uniform(-rho_op, rho_op, size=d)
    lam = lam / (np.max(np.abs(lam)) + 1e-12) * rho_op
    return (U @ np.diag(lam) @ U.T).astype(np.float32)


def stationary_cov(Phi):
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


# ── LSTM (same as S6) ─────────────────────────────────────────────────────────

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


def calibrate_lstm(model, X_cal, d):
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


# ── PELT detection ────────────────────────────────────────────────────────────

def pelt_detect(x, pen, model=PELT_MODEL, start=BURN):
    """
    Run PELT on (T, d) array x.  Returns first detected change-point index
    (0-indexed) in the window [start, T], or None if no change detected.
    bkps from ruptures are 1-indexed and include T as the final element.
    """
    signal = x.astype(np.float64)  # (T, d)
    algo = rpt.Pelt(model=model, min_size=5, jump=5).fit(signal)
    try:
        bkps = algo.predict(pen=pen)  # list of ints, last = T
    except Exception:
        return None
    # Remove the last sentinel element (T)
    cps = [b - 1 for b in bkps[:-1] if start <= b - 1 < len(x)]
    return cps[0] if cps else None


# ── Pathway A signal ─────────────────────────────────────────────────────────

def build_lstm_signal_fn(model, mu_r, sig_r, d):
    def signal_pathway_a(x):
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[None, :, :], dtype=torch.float32)
            pred, _ = model(xt[:, :-1, :])
            pred_np  = pred.squeeze(0).numpy()
        scores = np.zeros(len(x))
        for t in range(1, len(x)):
            e = x[t] - pred_np[t-1]
            scores[t] = (e @ e) / d
        z_mu  = mu_r
        z_std = max(sig_r, 1e-8)
        return (scores - z_mu) / z_std
    return signal_pathway_a


# ── Calibrate PELT penalty ────────────────────────────────────────────────────

def calibrate_pelt_pen(Phi_b, rng_cal, target_arl=180.0, n_seqs=100,
                        t_long=T):
    """
    Binary search over PELT penalty at d=1 to achieve target ARL_0.
    Uses shorter sequences (length T=200) for speed; hold penalty fixed across d.
    """
    X_arl = gen_var1(n_seqs, t_long, Phi_b, rng_cal)

    def arl_at_pen(pen):
        runs = []
        for i in range(n_seqs):
            x = X_arl[i]
            alarm = pelt_detect(x, pen, start=BURN)
            runs.append(alarm if alarm is not None else t_long)
        return float(np.mean(runs))

    lo, hi = 0.5, 100.0
    for it in range(8):
        mid = (lo + hi) / 2.0
        arl = arl_at_pen(mid)
        if arl < target_arl:
            lo = mid
        else:
            hi = mid
        print(f"    iter {it+1}: pen={mid:.2f}  ARL_0={arl:.1f}  target={target_arl:.0f}")
    cal_pen = (lo + hi) / 2.0
    print(f"  Calibrated PELT penalty: {cal_pen:.3f}")
    return cal_pen


# ── Evaluate one dimension ────────────────────────────────────────────────────

def evaluate_dim(d, Phi_b, Phi_a, rng, model, mu_r, sig_r, pelt_pen):
    """Evaluate Pathway A + PELT on change-point sequences."""
    X_cp = gen_var1_cp(N_TEST, T, Phi_b, Phi_a, TAU, rng)
    lstm_fn = build_lstm_signal_fn(model, mu_r, sig_r, d)

    methods = {
        "pathway_a": lambda x: page_cusum(lstm_fn(x)),
        "pelt":      lambda x: pelt_detect(x, pelt_pen),
    }

    results = {}
    for name, detect_fn in methods.items():
        delays = []; fa = 0; detected = 0
        for i in range(N_TEST):
            alarm = detect_fn(X_cp[i])
            if alarm is None:
                delays.append(T - TAU)
            else:
                detected += 1
                delays.append(alarm - TAU)
                if alarm < TAU:
                    fa += 1
        d_arr = np.array(delays)
        det_mask = d_arr < (T - TAU)
        d_det = d_arr[det_mask] if det_mask.any() else np.array([np.nan])
        results[name] = dict(
            detect_rate      = round(float(det_mask.mean()), 4),
            false_alarm_rate = round(fa / N_TEST, 4),
            delay_mean       = round(float(np.nanmean(d_det)), 2),
            delay_median     = round(float(np.nanmedian(d_det)), 2),
        )
    return results


def evaluate_arl_dim(d, Phi_b, rng, model, mu_r, sig_r, pelt_pen):
    """ARL_0 on pure stationary data (no change-point)."""
    T_long = T * 3   # 600 steps — enough to distinguish ARL ranges
    X_pure = gen_var1(N_ARL, T_long, Phi_b, rng)
    lstm_fn = build_lstm_signal_fn(model, mu_r, sig_r, d)

    methods = {
        "pathway_a": lambda x: page_cusum(lstm_fn(x), start=BURN),
        "pelt":      lambda x: pelt_detect(x, pelt_pen, start=BURN),
    }

    arls = {}
    for name, detect_fn in methods.items():
        runs = []
        for i in range(N_ARL):
            alarm = detect_fn(X_pure[i])
            runs.append(alarm if alarm is not None else T_long)
        arls[name] = round(float(np.mean(runs)), 1)
    return arls


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Study S9: PELT vs Pathway A — CPD benchmark across dimensions")
    print(f"  dims={DIMS}  rho_before={RHO_BEFORE}  rho_after={RHO_AFTER}")
    print(f"  tau={TAU}  T={T}  h={H}  N_test={N_TEST}  N_arl={N_ARL}")

    # PELT penalty pre-calibrated to ARL_0 ≈ 200 at d=1 (grid search):
    # pen=7.0 → ARL_0=202; matches Pathway A's ARL_0≈187 at d=1 in Study S6.
    pelt_pen = PELT_PEN
    print(f"\n  PELT penalty (pre-calibrated): {pelt_pen:.1f}")

    rows = []

    for d in DIMS:
        print(f"\n{'='*55}")
        print(f"  d = {d}")
        rng = np.random.default_rng(SEED + d * 37)

        Phi_b = make_var_matrix(d, RHO_BEFORE, rng)
        Phi_a = make_var_matrix(d, RHO_AFTER,  rng)

        # Train LSTM
        print(f"  Training LSTM ({N_TRAIN} seqs) ...", end=" ", flush=True)
        X_train = gen_var1(N_TRAIN, T, Phi_b, rng)
        model   = train_mv_lstm(d, X_train, seed=SEED + d)
        print("done.")

        # Calibrate LSTM in-control stats
        X_cal = gen_var1(N_CAL, T, Phi_b, rng)
        mu_r, sig_r = calibrate_lstm(model, X_cal, d)

        # Evaluate on change-point sequences
        print(f"  Evaluating on {N_TEST} change-point sequences ...")
        det_res = evaluate_dim(d, Phi_b, Phi_a, rng, model, mu_r, sig_r,
                               pelt_pen)

        # Evaluate ARL_0
        print(f"  Evaluating ARL_0 ({N_ARL} stationary sequences) ...")
        arl_res = evaluate_arl_dim(d, Phi_b, rng, model, mu_r, sig_r,
                                   pelt_pen)

        for method in ["pathway_a", "pelt"]:
            rows.append(dict(
                d             = d,
                method        = method,
                pelt_pen      = round(pelt_pen, 3),
                detect_rate   = det_res[method]["detect_rate"],
                false_alarm   = det_res[method]["false_alarm_rate"],
                delay_mean    = det_res[method]["delay_mean"],
                delay_median  = det_res[method]["delay_median"],
                arl0          = arl_res[method],
            ))
            print(f"  {method:12s}  detect={det_res[method]['detect_rate']:.3f}"
                  f"  FA={det_res[method]['false_alarm_rate']:.3f}"
                  f"  delay={det_res[method]['delay_mean']:.1f}"
                  f"  ARL0={arl_res[method]:.1f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "S9_cpd_benchmark.csv")
    fieldnames = ["d", "method", "pelt_pen", "detect_rate", "false_alarm",
                  "delay_mean", "delay_median", "arl0"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    dims      = DIMS
    pa_arl    = [r["arl0"]        for r in rows if r["method"] == "pathway_a"]
    pelt_arl  = [r["arl0"]        for r in rows if r["method"] == "pelt"]
    pa_fa     = [r["false_alarm"]  for r in rows if r["method"] == "pathway_a"]
    pelt_fa   = [r["false_alarm"]  for r in rows if r["method"] == "pelt"]
    pa_delay  = [r["delay_mean"]   for r in rows if r["method"] == "pathway_a"]
    pelt_delay= [r["delay_mean"]   for r in rows if r["method"] == "pelt"]

    x = np.arange(len(dims))
    w = 0.35

    colors = {"pathway_a": "#2ca02c", "pelt": "#d62728"}

    # ARL_0
    axes[0].bar(x - w/2, pa_arl,   width=w, color=colors["pathway_a"],
                label="Pathway A (M3)")
    axes[0].bar(x + w/2, pelt_arl, width=w, color=colors["pelt"],
                label=f"PELT (pen={pelt_pen:.1f})")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"d={d}" for d in dims])
    axes[0].set_ylabel("ARL$_0$"); axes[0].set_title("ARL$_0$ (higher is better)")
    axes[0].legend(fontsize=8)

    # False-alarm rate
    axes[1].bar(x - w/2, pa_fa,   width=w, color=colors["pathway_a"])
    axes[1].bar(x + w/2, pelt_fa, width=w, color=colors["pelt"])
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"d={d}" for d in dims])
    axes[1].set_ylabel("False-alarm rate"); axes[1].set_title("False-alarm rate (lower is better)")
    axes[1].axhline(0.05, color="k", ls="--", lw=0.8, label="5%")

    # Mean delay
    axes[2].bar(x - w/2, pa_delay,   width=w, color=colors["pathway_a"])
    axes[2].bar(x + w/2, pelt_delay, width=w, color=colors["pelt"])
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"d={d}" for d in dims])
    axes[2].set_ylabel("Mean delay (steps)"); axes[2].set_title("Detection delay (positive = after $\\tau$)")
    axes[2].axhline(0, color="k", ls="--", lw=0.8)

    fig.suptitle(f"Study S9: PELT vs Pathway A  ($h={H}$, $\\tau={TAU}$, $T={T}$)",
                 fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "S9_pelt_vs_pathwayA.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {fig_path}")

    elapsed = time.time() - t0
    print(f"\nStudy S9 complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
