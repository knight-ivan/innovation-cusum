#!/usr/bin/env python3
# =============================================================================
#  run_all.py — Master script: run all nine empirical studies in sequence
#
#  Paper: "Martingale Innovations from Contractive Recurrent Networks
#          and Dimension-Robust Change-Point Detection"
#  Author: Yuan-chin Ivan Chang, Academia Sinica, 2026
#
#  Studies in main text:
#    S1  — MDS validation & forget-gate diagnostics      [Proposition 2, Conjecture 1]
#    S2B — CUSUM at a known change-point (scalar AR)     [Theorem 2]
#    S4  — Multivariate MDS on VAR(1)                   [Theorem 1]
#    S5  — Kalman filter benchmark for J_t estimation
#    S6  — Pathway A CUSUM dimension stability           [Corollary 3]
#    S7  — Nile River real-data application              [Theorem 2]
#    S8  — US equity ETF real multivariate application   [Corollary 3]
#    S9  — PELT vs Pathway A CPD benchmark               [Corollary 3]
#
#  Studies in Supplementary Material:
#    S2A — Locally stationary AR  (Supplement Section S4)  [Corollary 2]
#    S3  — I_t surrogate comparison (Supplement Section S5)
#
#  Usage:
#    cd P1_Theory/code
#    python run_all.py                # run all studies
#    python run_all.py --study S1     # run only Study S1
#    python run_all.py --study S2     # run only Study S2 (locally stationary)
#    python run_all.py --study S3     # run only Study S3 (I_t recovery)
#    python run_all.py --study S4     # run only Study S4 (multivariate MDS)
#    python run_all.py --study S5     # run only Study S5 (Kalman benchmark)
#    python run_all.py --study S6     # run only Study S6 (Pathway A multivariate)
#    python run_all.py --study S7     # run only Study S7 (real data)
#    python run_all.py --study S8     # run only Study S8 (equity ETF)
#    python run_all.py --study S9     # run only Study S9 (CPD benchmark)
#    python run_all.py --nile         # run Nile River R analysis only
#
#  Prerequisites:
#    pip install -r requirements.txt
#    (For Nile analysis: Rscript and packages in nile_analysis.R)
#
#  Outputs (results/):
#    S1_mds.csv, S1_forget_gate.csv, S1_mamba.csv
#    S2_locally_stationary.csv, S2_changepoint.csv
#    S3_recovery.csv
#    S4_multivariate_mds.csv
#    S5_kalman_benchmark.csv
#    S6_pathway_A_multivariate.csv
#    S7_real_data.csv
#    S8_equity_etf.csv
#    S9_cpd_benchmark.csv
#    nile_mds_tests.csv, nile_changepoint.csv  (from R)
#
#  Outputs (figures/):
#    S1A_mds_pass_rates.png, S1B_forget_gate_summary.png
#    S1_forget_gate_trajectories.png, S1_mamba_delta_vs_It.png
#    S2A_locally_stationary.png, S2B_changepoint.png, S2B_cusum_delay_hist.png
#    S3A_It_trajectories.png, S3B_It_trajectories.png
#    S3_nmse_comparison.png, S3_alpha_sensitivity.png
#    S4_multivariate_mds.png
#    S5_kalman_benchmark.png
#    S6_arl0_vs_dim.png
#    S7_real_data.png
#    S8_equity_etf.png
#    S9_pelt_vs_pathwayA.png
#    nile_series.png, nile_ar1_residuals.png, nile_cusum_test.png  (from R)
#
#  Expected total runtime on a modern laptop CPU:
#    Study S1:  ~10–15 min  (training + 1000 test sequences × 3 processes)
#    Study S2:  ~8–12 min   (1000 sequences, CUSUM)
#    Study S3:  ~20–30 min  (500 reps × KLIEP per sequence)
#    Study S4:  ~15 min     (multivariate VAR(1), d in {1,2,5,10})
#    Study S5:  ~10 min     (Kalman filter comparisons)
#    Study S6:  ~20 min     (Pathway A CUSUM, d in {1,2,5,10,20})
#    Study S7:  ~5 min      (Nile River real data)
#    Study S8:  ~10 min     (equity ETF, 5 assets)
#    Study S9:  ~4 min      (PELT vs Pathway A, 4 dims × 1000 sequences)
#    Nile R:    ~1 min
#    Total:     ~105–125 min (~2 hours on a single-core laptop)
# =============================================================================

import argparse
import os
import subprocess
import sys
import time

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run P1 empirical studies")
parser.add_argument("--study", choices=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "all"],
                    default="all", help="Which study to run (default: all)")
parser.add_argument("--nile",  action="store_true",
                    help="Run the Nile River R analysis")
parser.add_argument("--no-nile", action="store_true",
                    help="Skip the Nile River R analysis")
args = parser.parse_args()

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_python_study(script: str):
    """Run a Python study script and print elapsed time."""
    path = os.path.join(_HERE, script)
    print(f"\n{'='*70}")
    print(f"Running: {script}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[WARNING] {script} exited with code {result.returncode}")
    else:
        print(f"[OK] {script} finished in {elapsed:.1f} s")
    return result.returncode == 0


def run_nile_r():
    """Run the Nile River R analysis script."""
    r_script = os.path.join(_HERE, "nile_analysis.R")
    print(f"\n{'='*70}")
    print("Running: nile_analysis.R  (requires Rscript)")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = subprocess.run(["Rscript", r_script], check=False,
                                capture_output=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"[WARNING] nile_analysis.R exited with code {result.returncode}")
            print("  Make sure Rscript is installed and required R packages are available.")
        else:
            print(f"[OK] nile_analysis.R finished in {elapsed:.1f} s")
        return result.returncode == 0
    except FileNotFoundError:
        print("[SKIP] Rscript not found. Skipping Nile River analysis.")
        print("  Install R from https://cran.r-project.org/ and re-run.")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    overall_start = time.time()
    print("P1 Empirical Studies — Master Runner")
    print(f"Working directory: {_HERE}")

    successes = []

    # Nile River R analysis (optional, runs first to create data/nile.csv)
    if args.nile or (not args.no_nile and args.study == "all"):
        ok = run_nile_r()
        successes.append(("Nile (R)", ok))

    # Python studies
    if args.study in ("S1", "all"):
        ok = run_python_study("study_S1_stationary.py")
        successes.append(("Study S1", ok))

    if args.study in ("S2", "all"):
        ok = run_python_study("study_S2_nonstationary.py")
        successes.append(("Study S2", ok))

    if args.study in ("S3", "all"):
        ok = run_python_study("study_S3_recovery.py")
        successes.append(("Study S3", ok))

    if args.study in ("S4", "all"):
        ok = run_python_study("study_S4_multivariate.py")
        successes.append(("Study S4", ok))

    if args.study in ("S5", "all"):
        ok = run_python_study("study_S5_kalman_benchmark.py")
        successes.append(("Study S5", ok))

    if args.study in ("S6", "all"):
        ok = run_python_study("study_S6_pathway_A_multivariate.py")
        successes.append(("Study S6", ok))

    if args.study in ("S7", "all"):
        ok = run_python_study("study_S7_real_data.py")
        successes.append(("Study S7", ok))

    if args.study in ("S8", "all"):
        ok = run_python_study("study_S8_equity_etf.py")
        successes.append(("Study S8", ok))

    if args.study in ("S9", "all"):
        ok = run_python_study("study_S9_cpd_benchmark.py")
        successes.append(("Study S9", ok))

    # ── Final summary ─────────────────────────────────────────────────────
    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"All studies complete in {total:.1f} s ({total/60:.1f} min)")
    print(f"{'='*70}")
    for name, ok in successes:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6s}] {name}")

    print(f"\nResults in: {os.path.join(_HERE, 'results')}")
    print(f"Figures in: {os.path.join(_HERE, 'figures')}")

    # Return non-zero exit code if any study failed
    if not all(ok for _, ok in successes):
        sys.exit(1)


if __name__ == "__main__":
    main()
