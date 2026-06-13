# =============================================================================
#  study_S2B_cusum_revisit.py — Revisit S2B CUSUM delay problem
#
#  Problem identified: with AR(phi_before=0.30) data and threshold h=5,
#  the CUSUM fires on average 28.5 steps *before* the true change-point.
#  Root cause: the Page CUSUM assumes i.i.d. input, but (X_t - mu0)/sigma0
#  for an AR(phi) series is still autocorrelated with corr ~ phi.  This
#  inflates the CUSUM statistic and shortens the average run length (ARL_0)
#  far below the i.i.d.-calibrated value, causing early false alarms.
#
#  This script runs three analyses:
#
#  (A) Threshold sensitivity: h in {3,5,8,10,15,20}
#      For each h: detect rate, mean/median delay, false-alarm rate (fire
#      before tau on clean pre-change data).
#      Reports the detection-delay frontier.
#
#  (B) Pre-whitened CUSUM: fit AR(1) on pre-change window, apply CUSUM
#      to the AR(1) residuals (which are approx i.i.d.).  Compare with
#      raw CUSUM at the same h.
#
#  (C) ARL_0 simulation: estimate the in-control ARL for both raw and
#      whitened CUSUM at each h on pure AR(0.30) data (no change-point).
#      Alongside the theoretical i.i.d. ARL_0 for reference.
#
#  Outputs:
#    results/S2B_cusum_threshold.csv
#    results/S2B_cusum_whitened.csv
#    results/S2B_cusum_arl.csv
#    figures/S2B_cusum_frontier.png
#    figures/S2B_cusum_arl.png
# =============================================================================

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings("ignore")

script_dir  = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ── Parameters matching S2B ───────────────────────────────────────────────────
PHI_BEFORE  = 0.30
PHI_AFTER   = 0.85
TAU_FRAC    = 0.50     # change-point at t=100 out of T=200
T           = 200
TAU         = int(T * TAU_FRAC)   # = 100
N_SEQS      = 2000     # sequences for delay analysis
N_ARL       = 5000     # sequences for ARL_0 estimation (no change-point)
K_REF       = 0.5      # CUSUM reference value
H_GRID      = [3, 5, 8, 10, 15, 20]
SIGMA_EPS   = 1.0
BURN        = int(0.20 * T)       # pre-change estimation window (40 steps)
SEED        = 42

rng = np.random.default_rng(SEED)


# ── Data generation ───────────────────────────────────────────────────────────

def gen_cp_series(N, T, phi_b, phi_a, tau, rng):
    """VAR(1) with single change-point at tau."""
    X = np.zeros((N, T))
    sigma_b = SIGMA_EPS / np.sqrt(1 - phi_b**2)
    X[:, 0] = rng.normal(0, sigma_b, N)
    for t in range(1, T):
        phi = phi_b if t <= tau else phi_a
        X[:, t] = phi * X[:, t-1] + rng.normal(0, SIGMA_EPS, N)
    return X


def gen_pure_ar1(N, T, phi, rng):
    """Pure AR(1), no change-point."""
    X = np.zeros((N, T))
    sigma_s = SIGMA_EPS / np.sqrt(1 - phi**2)
    X[:, 0] = rng.normal(0, sigma_s, N)
    for t in range(1, T):
        X[:, t] = phi * X[:, t-1] + rng.normal(0, SIGMA_EPS, N)
    return X


# ── CUSUM implementations ─────────────────────────────────────────────────────

def cusum_raw(x, mu0, sigma0, k=K_REF, h=5.0):
    """Page CUSUM on the raw series."""
    T = len(x)
    Sp = Sm = 0.0
    for t in range(BURN, T):
        z = (x[t] - mu0) / sigma0
        Sp = max(0, Sp + z - k)
        Sm = max(0, Sm - z - k)
        if Sp > h or Sm > h:
            return t
    return None


def cusum_whitened(x, phi_hat, mu0, sigma_eps_hat, k=K_REF, h=5.0):
    """
    CUSUM on AR(1)-whitened residuals:
      e_t = x_t - phi_hat * x_{t-1}
    Residuals are approximately i.i.d. N(0, sigma_eps^2).
    """
    T = len(x)
    Sp = Sm = 0.0
    for t in range(BURN + 1, T):
        e = x[t] - phi_hat * x[t-1]
        z = e / sigma_eps_hat
        Sp = max(0, Sp + z - k)
        Sm = max(0, Sm - z - k)
        if Sp > h or Sm > h:
            return t
    return None


def estimate_ar1(x_burn):
    """OLS AR(1) on burn-in window."""
    y = x_burn[1:]
    x = x_burn[:-1]
    phi = np.dot(y, x) / np.dot(x, x)
    resid = y - phi * x
    sigma_eps = resid.std()
    mu0 = np.mean(x_burn)
    sigma0 = x_burn.std()
    return phi, max(sigma_eps, 1e-8), mu0, max(sigma0, 1e-8)


# ── Analysis A: threshold sensitivity ────────────────────────────────────────

def run_threshold_sensitivity():
    print("Analysis A: threshold sensitivity")
    X_cp = gen_cp_series(N_SEQS, T, PHI_BEFORE, PHI_AFTER, TAU, rng)

    rows = []
    for h in H_GRID:
        delays = []
        false_alarms = 0
        missed = 0
        for i in range(N_SEQS):
            x = X_cp[i]
            phi_hat, sig_eps, mu0, sig0 = estimate_ar1(x[:BURN])
            alarm = cusum_raw(x, mu0, sig0, h=h)
            if alarm is None:
                missed += 1
                delays.append(T - TAU)   # penalise as max delay
            else:
                delay = alarm - TAU
                delays.append(delay)
                if alarm < TAU:
                    false_alarms += 1

        delays = np.array(delays)
        detected = delays < (T - TAU)
        detect_rate = detected.mean()
        fa_rate     = false_alarms / N_SEQS
        d_mean      = delays[detected].mean() if detected.any() else np.nan
        d_med       = np.median(delays[detected]) if detected.any() else np.nan
        d_90        = np.percentile(delays[detected], 90) if detected.any() else np.nan

        print(f"  h={h:4.0f} | detect={detect_rate:.3f}  FA={fa_rate:.3f}  "
              f"delay mean={d_mean:+.1f}  median={d_med:+.1f}  90pct={d_90:+.1f}")
        rows.append(dict(h=h, detect_rate=round(detect_rate,4),
                         false_alarm_rate=round(fa_rate,4),
                         delay_mean=round(float(d_mean),2),
                         delay_median=round(float(d_med),2),
                         delay_90pct=round(float(d_90),2)))

    path = os.path.join(results_dir, "S2B_cusum_threshold.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: S2B_cusum_threshold.csv")
    return rows


# ── Analysis B: raw vs whitened CUSUM at each h ───────────────────────────────

def run_whitened_comparison():
    print("\nAnalysis B: raw vs pre-whitened CUSUM")
    X_cp = gen_cp_series(N_SEQS, T, PHI_BEFORE, PHI_AFTER, TAU, rng)

    rows = []
    for h in H_GRID:
        raw_delays = []; wh_delays = []
        raw_fa = 0; wh_fa = 0

        for i in range(N_SEQS):
            x = X_cp[i]
            phi_hat, sig_eps, mu0, sig0 = estimate_ar1(x[:BURN])

            a_raw = cusum_raw(x, mu0, sig0, h=h)
            a_wh  = cusum_whitened(x, phi_hat, mu0, sig_eps, h=h)

            for a, dl, fa in [(a_raw, raw_delays, 'raw'),
                               (a_wh,  wh_delays,  'wh')]:
                if a is None:
                    (raw_delays if fa == 'raw' else wh_delays).append(T - TAU)
                else:
                    d = a - TAU
                    (raw_delays if fa == 'raw' else wh_delays).append(d)
                    if a < TAU:
                        if fa == 'raw': raw_fa += 1
                        else:           wh_fa  += 1

        raw_delays = np.array(raw_delays); wh_delays = np.array(wh_delays)
        raw_det = raw_delays < (T - TAU);  wh_det = wh_delays < (T - TAU)

        def summ(delays, det):
            if not det.any(): return np.nan, np.nan, np.nan
            return (float(delays[det].mean()),
                    float(np.median(delays[det])),
                    float(np.percentile(delays[det], 90)))

        r_m, r_md, r_90 = summ(raw_delays, raw_det)
        w_m, w_md, w_90 = summ(wh_delays,  wh_det)

        print(f"  h={h:4.0f} | RAW  detect={raw_det.mean():.3f} FA={raw_fa/N_SEQS:.3f} "
              f"delay={r_m:+.1f}/{r_md:+.1f}/{r_90:+.1f}")
        print(f"        | WHIT detect={wh_det.mean():.3f}  FA={wh_fa/N_SEQS:.3f} "
              f"delay={w_m:+.1f}/{w_md:+.1f}/{w_90:+.1f}")

        rows.append(dict(h=h, method="raw",
                         detect_rate=round(float(raw_det.mean()),4),
                         false_alarm_rate=round(raw_fa/N_SEQS,4),
                         delay_mean=round(r_m,2),
                         delay_median=round(r_md,2), delay_90pct=round(r_90,2)))
        rows.append(dict(h=h, method="whitened",
                         detect_rate=round(float(wh_det.mean()),4),
                         false_alarm_rate=round(wh_fa/N_SEQS,4),
                         delay_mean=round(w_m,2),
                         delay_median=round(w_md,2), delay_90pct=round(w_90,2)))

    path = os.path.join(results_dir, "S2B_cusum_whitened.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: S2B_cusum_whitened.csv")
    return rows


# ── Analysis C: ARL_0 simulation ──────────────────────────────────────────────

def run_arl_simulation():
    print("\nAnalysis C: in-control ARL_0 simulation")
    X_pure = gen_pure_ar1(N_ARL, T * 5, PHI_BEFORE, rng)   # long sequences

    # Theoretical ARL_0 for i.i.d. CUSUM (Siegmund 1985 approx):
    # ARL_0 ≈ exp(2*delta*h) / (2*delta)  where delta = k  (for k=0.5)
    def theoretical_arl(h, k=K_REF):
        # Wald-type approx: ARL ≈ (exp(2*k*h) - 2*k*h - 1) / (2*k^2)
        return (np.exp(2*k*h) - 2*k*h - 1) / (2 * k**2)

    rows = []
    for h in H_GRID:
        raw_runs = []
        wh_runs  = []
        for i in range(N_ARL):
            x = X_pure[i]
            burn_x = x[:BURN]
            phi_hat, sig_eps, mu0, sig0 = estimate_ar1(burn_x)

            # Raw: scan full long sequence
            Sp = Sm = 0.0; fired = None
            for t in range(BURN, len(x)):
                z = (x[t] - mu0) / sig0
                Sp = max(0, Sp + z - K_REF)
                Sm = max(0, Sm - z - K_REF)
                if Sp > h or Sm > h:
                    fired = t; break
            raw_runs.append(fired if fired is not None else len(x))

            # Whitened
            Sp = Sm = 0.0; fired = None
            for t in range(BURN + 1, len(x)):
                e = x[t] - phi_hat * x[t-1]
                z = e / sig_eps
                Sp = max(0, Sp + z - K_REF)
                Sm = max(0, Sm - z - K_REF)
                if Sp > h or Sm > h:
                    fired = t; break
            wh_runs.append(fired if fired is not None else len(x))

        arl_raw = float(np.mean(raw_runs))
        arl_wh  = float(np.mean(wh_runs))
        arl_th  = theoretical_arl(h)
        print(f"  h={h:4.0f} | ARL_0 raw={arl_raw:.0f}  whitened={arl_wh:.0f}  "
              f"theoretical(iid)={arl_th:.0f}")
        rows.append(dict(h=h, ARL0_raw=round(arl_raw,1),
                         ARL0_whitened=round(arl_wh,1),
                         ARL0_theoretical_iid=round(arl_th,1)))

    path = os.path.join(results_dir, "S2B_cusum_arl.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved: S2B_cusum_arl.csv")
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(rows_A, rows_B, rows_C):
    # Fig 1: Detection-delay frontier (A and B combined)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: false-alarm rate vs h
    hs = [r["h"] for r in rows_A]
    ax = axes[0]
    ax.plot(hs, [r["false_alarm_rate"] for r in rows_A], "o-",
            color="firebrick", label="Raw CUSUM")
    raw_B  = [r for r in rows_B if r["method"] == "raw"]
    wh_B   = [r for r in rows_B if r["method"] == "whitened"]
    ax.plot([r["h"] for r in wh_B], [r["false_alarm_rate"] for r in wh_B],
            "s--", color="steelblue", label="Whitened CUSUM")
    ax.axhline(0.05, color="gray", lw=1, ls=":")
    ax.set_xlabel("Threshold $h$"); ax.set_ylabel("False-alarm rate")
    ax.set_title("False-alarm rate vs $h$")
    ax.legend(fontsize=8); ax.set_xticks(hs)

    # Panel 2: mean delay vs h
    ax = axes[1]
    ax.plot(hs, [r["delay_mean"] for r in rows_A], "o-",
            color="firebrick", label="Raw CUSUM")
    ax.plot([r["h"] for r in wh_B], [r["delay_mean"] for r in wh_B],
            "s--", color="steelblue", label="Whitened CUSUM")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Threshold $h$"); ax.set_ylabel("Mean detection delay (steps)")
    ax.set_title("Mean delay vs $h$\n(negative = pre-alarm)")
    ax.legend(fontsize=8); ax.set_xticks(hs)

    # Panel 3: ARL_0 vs h
    ax = axes[2]
    hs_c = [r["h"] for r in rows_C]
    ax.plot(hs_c, [r["ARL0_raw"] for r in rows_C], "o-",
            color="firebrick", label="Raw CUSUM (AR data)")
    ax.plot(hs_c, [r["ARL0_whitened"] for r in rows_C], "s--",
            color="steelblue", label="Whitened CUSUM")
    ax.plot(hs_c, [r["ARL0_theoretical_iid"] for r in rows_C], "k:",
            lw=1.5, label="Theoretical i.i.d.")
    ax.set_xlabel("Threshold $h$"); ax.set_ylabel("$\\mathrm{ARL}_0$ (steps)")
    ax.set_title("In-control $\\mathrm{ARL}_0$ vs $h$")
    ax.legend(fontsize=8); ax.set_xticks(hs_c)
    ax.set_yscale("log")

    plt.suptitle("S2B Revisit: CUSUM threshold sensitivity and pre-whitening",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S2B_cusum_frontier.png"), dpi=120)
    plt.close()
    print("Saved: S2B_cusum_frontier.png")

    # Fig 2: Delay distribution at h=5 (raw) vs h=10 (whitened) — the "sweet spots"
    rng2 = np.random.default_rng(SEED + 1)
    X_cp = gen_cp_series(2000, T, PHI_BEFORE, PHI_AFTER, TAU, rng2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, h, method, title in [
        (axes[0], 5,  "raw",      "Raw CUSUM  h=5  (original S2B)"),
        (axes[1], 10, "whitened", "Whitened CUSUM  h=10"),
    ]:
        delays = []
        for i in range(2000):
            x = X_cp[i]
            phi_hat, sig_eps, mu0, sig0 = estimate_ar1(x[:BURN])
            if method == "raw":
                a = cusum_raw(x, mu0, sig0, h=h)
            else:
                a = cusum_whitened(x, phi_hat, mu0, sig_eps, h=h)
            if a is not None:
                delays.append(a - TAU)
        delays = np.array(delays)
        ax.hist(delays, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="firebrick", lw=1.5, ls="--", label="True $\\tau$")
        ax.axvline(float(np.median(delays)), color="orange", lw=1.5, ls="-",
                   label=f"Median={np.median(delays):.0f}")
        ax.set_xlabel("Detection delay (steps after $\\tau$)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\ndetect={len(delays)/2000:.2f}  "
                     f"FA={np.mean(delays<0):.2f}  mean={delays.mean():+.1f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "S2B_cusum_arl.png"), dpi=120)
    plt.close()
    print("Saved: S2B_cusum_arl.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()
    print(f"S2B CUSUM revisit  phi_before={PHI_BEFORE}  phi_after={PHI_AFTER}  "
          f"tau={TAU}  T={T}")
    rows_A = run_threshold_sensitivity()
    rows_B = run_whitened_comparison()
    rows_C = run_arl_simulation()
    make_figures(rows_A, rows_B, rows_C)
    print(f"\nDone in {time.time()-t0:.1f}s")
