# =============================================================================
#  study_S2B_pathway_A.py — Three-way CUSUM comparison for S2B
#
#  The paper's Pathway A uses the trained LSTM's prediction residuals
#  as the CUSUM input signal.  Under stationarity these residuals approximate
#  the MDS innovations (Theorem 2, ARL lower bound) — near i.i.d. with known
#  variance — so the CUSUM is well-calibrated without any manual whitening.
#  The ARL_0 lower bound is (1/(2*sqrt(2))) * exp(kappa*h / (2*sigma^2)).
#  After a change-point the residuals spike, triggering detection.
#
#  Three methods compared (all using Page CUSUM, k=0.5):
#
#    M1  Raw CUSUM        input: (X_t - mu0) / sigma0
#    M2  Whitened CUSUM   input: AR(1) residuals / sigma_eps
#    M3  Pathway A        input: LSTM prediction residuals / sigma_lstm
#                         (paper's proposed method)
#
#  Evaluated on:
#    - Detection rate, false-alarm rate, mean/median/90th-pct delay  (2000 seqs)
#    - ARL_0 under pure AR(0.30) stationarity                        (3000 seqs)
#    - Example trajectory figure
#
#  Outputs:
#    results/S2B_pathway_A.csv
#    results/S2B_pathway_A_arl.csv
#    figures/S2B_pathway_A_comparison.png
#    figures/S2B_pathway_A_trajectory.png
# =============================================================================

import os, sys, warnings, time
import numpy as np
import torch
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

from models   import InstrumentedLSTM
from training import train_model
from config   import HIDDEN_DIM, N_EPOCHS as EPOCHS, LR, BATCH_SIZE

# ── Parameters ────────────────────────────────────────────────────────────────
PHI_BEFORE = 0.30
PHI_AFTER  = 0.85
T          = 200
TAU        = 100          # true change-point
BURN       = 40           # pre-change estimation window (first 20%)
K_REF      = 0.5
H          = 5.0          # single threshold — now fairly compared across methods
N_TRAIN    = 800          # sequences for LSTM training
N_TEST     = 2000         # sequences for delay analysis
N_ARL      = 3000         # sequences for ARL_0
SIGMA_EPS  = 1.0
SEED       = 2025

rng_global = np.random.default_rng(SEED)


# ── Data generation ───────────────────────────────────────────────────────────

def gen_ar1(N, T, phi, rng, init="stationary"):
    X = np.zeros((N, T), dtype=np.float32)
    sigma_s = SIGMA_EPS / np.sqrt(1 - phi**2)
    X[:, 0] = rng.normal(0, sigma_s, N) if init == "stationary" else 0.0
    eps = rng.normal(0, SIGMA_EPS, (N, T-1)).astype(np.float32)
    for t in range(1, T):
        X[:, t] = phi * X[:, t-1] + eps[:, t-1]
    return X


def gen_cp(N, T, phi_b, phi_a, tau, rng):
    X = np.zeros((N, T), dtype=np.float32)
    sigma_s = SIGMA_EPS / np.sqrt(1 - phi_b**2)
    X[:, 0] = rng.normal(0, sigma_s, N).astype(np.float32)
    eps = rng.normal(0, SIGMA_EPS, (N, T-1)).astype(np.float32)
    for t in range(1, T):
        phi = phi_b if t <= tau else phi_a
        X[:, t] = phi * X[:, t-1] + eps[:, t-1]
    return X


# ── CUSUM kernel ──────────────────────────────────────────────────────────────

def page_cusum(z_seq, k=K_REF, h=H, start=BURN):
    """Run two-sided Page CUSUM on standardised sequence z_seq from index start."""
    Sp = Sm = 0.0
    for t in range(start, len(z_seq)):
        z = float(z_seq[t])
        Sp = max(0.0, Sp + z - k)
        Sm = max(0.0, Sm - z - k)
        if Sp > h or Sm > h:
            return t
    return None


# ── M1: Raw CUSUM ─────────────────────────────────────────────────────────────

def signal_raw(x):
    mu0    = x[:BURN].mean()
    sigma0 = max(x[:BURN].std(), 1e-8)
    return (x - mu0) / sigma0


# ── M2: Whitened CUSUM ────────────────────────────────────────────────────────

def signal_whitened(x):
    burn = x[:BURN]
    phi_hat = np.dot(burn[1:], burn[:-1]) / (np.dot(burn[:-1], burn[:-1]) + 1e-12)
    resid   = burn[1:] - phi_hat * burn[:-1]
    sig_eps = max(resid.std(), 1e-8)

    z = np.zeros_like(x)
    for t in range(1, len(x)):
        e = x[t] - phi_hat * x[t-1]
        z[t] = e / sig_eps
    return z


# ── M3: Pathway A — LSTM prediction residuals ────────────────────────────────

def train_lstm_on_stationary(rng, seed=SEED):
    """Train LSTM on pure AR(phi_before) data."""
    X_train = gen_ar1(N_TRAIN, T, PHI_BEFORE, rng)
    model   = InstrumentedLSTM(hidden_size=HIDDEN_DIM)
    train_model(model, X_train, n_epochs=EPOCHS, lr=LR,
                batch_size=BATCH_SIZE, verbose=False)
    return model


def estimate_lstm_residual_stats(model, rng):
    """
    Estimate in-control mean and std of |LSTM residuals| on pure stationary data.
    Used to standardise the CUSUM input for M3.
    """
    X_cal = gen_ar1(500, T, PHI_BEFORE, rng)
    model.eval()
    all_resid = []
    with torch.no_grad():
        for i in range(len(X_cal)):
            x = X_cal[i]
            preds, _, _, _ = model.predict_sequence(x)
            # prediction residuals: e_t = x_t - x_hat_{t-1}
            resid = x[1:] - preds[:-1]
            all_resid.append(resid)
    all_resid = np.concatenate(all_resid)
    mu_r  = float(all_resid.mean())
    sig_r = float(max(all_resid.std(), 1e-8))
    return mu_r, sig_r


def signal_pathway_a(x, model, mu_r, sig_r):
    """
    M3 signal: standardised LSTM one-step-ahead prediction residuals.
    z_t = (x_t - x_hat_{t-1} - mu_r) / sig_r
    """
    model.eval()
    with torch.no_grad():
        preds, _, _, _ = model.predict_sequence(x)
    z = np.zeros(len(x))
    for t in range(1, len(x)):
        e    = x[t] - preds[t-1]
        z[t] = (e - mu_r) / sig_r
    return z


# ── Detection analysis ────────────────────────────────────────────────────────

def run_detection(model, mu_r, sig_r, rng):
    print(f"  Generating {N_TEST} change-point sequences ...")
    X_cp = gen_cp(N_TEST, T, PHI_BEFORE, PHI_AFTER, TAU, rng)

    results = {m: {"delays": [], "fa": 0, "miss": 0}
               for m in ["raw", "whitened", "pathway_a"]}

    for i in range(N_TEST):
        x = X_cp[i]
        signals = {
            "raw":       signal_raw(x),
            "whitened":  signal_whitened(x),
            "pathway_a": signal_pathway_a(x, model, mu_r, sig_r),
        }
        for name, z in signals.items():
            alarm = page_cusum(z)
            if alarm is None:
                results[name]["miss"] += 1
                results[name]["delays"].append(T - TAU)   # max-penalty miss
            else:
                delay = alarm - TAU
                results[name]["delays"].append(delay)
                if alarm < TAU:
                    results[name]["fa"] += 1

    rows = []
    for name in ["raw", "whitened", "pathway_a"]:
        d  = np.array(results[name]["delays"])
        detected = d < (T - TAU)
        dr   = detected.mean()
        fa_r = results[name]["fa"] / N_TEST
        d_det = d[detected] if detected.any() else np.array([np.nan])
        row = dict(
            method         = name,
            h              = H,
            detect_rate    = round(float(dr), 4),
            false_alarm_rate = round(float(fa_r), 4),
            delay_mean     = round(float(d_det.mean()), 2),
            delay_median   = round(float(np.median(d_det)), 2),
            delay_90pct    = round(float(np.percentile(d_det, 90)), 2),
        )
        rows.append(row)
        print(f"  {name:12s} | detect={dr:.3f}  FA={fa_r:.3f}  "
              f"delay mean={row['delay_mean']:+.1f}  "
              f"median={row['delay_median']:+.1f}  "
              f"90pct={row['delay_90pct']:+.1f}")
    return rows, X_cp


def run_arl(model, mu_r, sig_r, rng):
    """ARL_0 on pure stationary AR(phi_before) data."""
    print(f"\n  ARL_0 simulation ({N_ARL} sequences, T={T*5}) ...")
    # Use longer sequences so we see the actual run length
    T_long = T * 5
    X_pure = gen_ar1(N_ARL, T_long, PHI_BEFORE, rng)

    runs = {"raw": [], "whitened": [], "pathway_a": []}
    for i in range(N_ARL):
        x = X_pure[i]
        signals = {
            "raw":       signal_raw(x),
            "whitened":  signal_whitened(x),
            "pathway_a": signal_pathway_a(x, model, mu_r, sig_r),
        }
        for name, z in signals.items():
            alarm = page_cusum(z, start=BURN)
            runs[name].append(alarm if alarm is not None else T_long)

    rows = []
    for name in ["raw", "whitened", "pathway_a"]:
        arl = float(np.mean(runs[name]))
        print(f"  {name:12s} | ARL_0 = {arl:.0f}")
        rows.append(dict(method=name, h=H, ARL0=round(arl, 1)))
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(det_rows, arl_rows, X_cp, model, mu_r, sig_r):
    methods     = ["raw", "whitened", "pathway_a"]
    labels      = ["M1: Raw CUSUM", "M2: Whitened CUSUM", "M3: Pathway A (LSTM)"]
    colours     = ["firebrick", "darkorange", "steelblue"]

    # ── Fig 1: Bar chart comparison ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["false_alarm_rate", "delay_mean", "detect_rate"]
    titles  = ["False-alarm rate\n(lower is better)",
                "Mean detection delay\n(lower / closer to 0 is better)",
                "Detection rate\n(higher is better)"]
    refs    = [0.05, 0, 1.0]

    for ax, metric, title, ref in zip(axes, metrics, titles, refs):
        vals = [next(r[metric] for r in det_rows if r["method"] == m)
                for m in methods]
        bars = ax.bar(labels, vals, color=colours, alpha=0.8, edgecolor="white")
        ax.axhline(ref, color="gray", lw=1.2, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["M1\nRaw", "M2\nWhitened", "M3\nPathway A"],
                            fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"S2B: Three-way CUSUM comparison  "
                 f"(h={H}, k={K_REF}, φ: {PHI_BEFORE}→{PHI_AFTER}, τ={TAU})",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S2B_pathway_A_comparison.png"), dpi=120)
    plt.close()
    print("Saved: S2B_pathway_A_comparison.png")

    # ── Fig 2: Example trajectory for one test sequence ───────────────────────
    rng_ex = np.random.default_rng(SEED + 99)
    x_ex   = gen_cp(1, T, PHI_BEFORE, PHI_AFTER, TAU, rng_ex)[0]

    z_raw  = signal_raw(x_ex)
    z_wh   = signal_whitened(x_ex)
    z_pa   = signal_pathway_a(x_ex, model, mu_r, sig_r)

    alarms = {
        "raw":       page_cusum(z_raw),
        "whitened":  page_cusum(z_wh),
        "pathway_a": page_cusum(z_pa),
    }
    cumS = {}
    for name, z in [("raw", z_raw), ("whitened", z_wh), ("pathway_a", z_pa)]:
        Sp = np.zeros(T); Sm = np.zeros(T)
        for t in range(1, T):
            Sp[t] = max(0, Sp[t-1] + z[t] - K_REF)
            Sm[t] = max(0, Sm[t-1] - z[t] - K_REF)
        cumS[name] = (Sp, Sm)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    t_arr = np.arange(T)

    # Panel 0: raw series
    axes[0].plot(t_arr, x_ex, color="gray", lw=1)
    axes[0].axvline(TAU, color="firebrick", ls="--", lw=1.5, label=f"True τ={TAU}")
    axes[0].set_ylabel("$X_t$"); axes[0].set_title("Observed series")
    axes[0].legend(fontsize=8)

    # Panels 1–3: CUSUM statistics for each method
    for ax, name, label, col in zip(axes[1:],
                                     ["raw", "whitened", "pathway_a"],
                                     labels, colours):
        Sp, Sm = cumS[name]
        ax.plot(t_arr, Sp, color=col, lw=1.4, label="$S_t^+$")
        ax.plot(t_arr, Sm, color=col, lw=1.4, ls="--", alpha=0.6, label="$S_t^-$")
        ax.axhline(H, color="gray", lw=1, ls=":", label=f"h={H}")
        ax.axvline(TAU, color="firebrick", ls="--", lw=1.5)
        alarm = alarms[name]
        if alarm is not None:
            ax.axvline(alarm, color="darkorange", ls=":", lw=2,
                       label=f"Alarm t={alarm} (delay={alarm-TAU:+d})")
        ax.set_ylabel("CUSUM stat"); ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time $t$")
    plt.suptitle("S2B: CUSUM trajectories on one example sequence", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S2B_pathway_A_trajectory.png"), dpi=120)
    plt.close()
    print("Saved: S2B_pathway_A_trajectory.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Study S2B Pathway A — three-way CUSUM comparison")
    print(f"  phi_before={PHI_BEFORE}  phi_after={PHI_AFTER}  tau={TAU}  T={T}  h={H}")

    # Step 1: train LSTM on pre-change (stationary) data
    print(f"\nTraining LSTM on AR({PHI_BEFORE}) data ({N_TRAIN} sequences) ...")
    model = train_lstm_on_stationary(rng_global, seed=SEED)
    print("  Training done.")

    # Step 2: estimate in-control residual distribution
    print("Calibrating LSTM residual distribution ...")
    mu_r, sig_r = estimate_lstm_residual_stats(model, rng_global)
    print(f"  In-control residuals: mu={mu_r:.4f}  sigma={sig_r:.4f}")

    # Step 3: detection analysis
    print(f"\nDetection analysis (N={N_TEST}) ...")
    det_rows, X_cp = run_detection(model, mu_r, sig_r, rng_global)

    # Step 4: ARL_0
    arl_rows = run_arl(model, mu_r, sig_r, rng_global)

    # Step 5: save
    path = os.path.join(results_dir, "S2B_pathway_A.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=det_rows[0].keys())
        w.writeheader(); w.writerows(det_rows)
    print(f"\nSaved: S2B_pathway_A.csv")

    path_arl = os.path.join(results_dir, "S2B_pathway_A_arl.csv")
    with open(path_arl, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=arl_rows[0].keys())
        w.writeheader(); w.writerows(arl_rows)
    print(f"Saved: S2B_pathway_A_arl.csv")

    # Step 6: figures
    make_figures(det_rows, arl_rows, X_cp, model, mu_r, sig_r)

    print(f"\nDone in {time.time()-t0:.1f}s")

    # ── Summary table ──
    print("\n── Summary (h=5.0) ──")
    print(f"{'Method':<14} {'Detect':>7} {'FA rate':>8} {'Mean delay':>11} "
          f"{'Median':>8} {'90pct':>7} {'ARL_0':>7}")
    for dr, ar in zip(det_rows, arl_rows):
        print(f"{dr['method']:<14} {dr['detect_rate']:>7.3f} "
              f"{dr['false_alarm_rate']:>8.3f} {dr['delay_mean']:>+11.1f} "
              f"{dr['delay_median']:>+8.1f} {dr['delay_90pct']:>+7.1f} "
              f"{ar['ARL0']:>7.0f}")
