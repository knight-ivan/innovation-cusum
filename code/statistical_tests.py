# =============================================================================
#  statistical_tests.py — Hypothesis tests for the empirical studies
#
#  Tests:
#    1. mds_ljung_box  — Ljung-Box test for zero autocorrelation in residuals
#                        (MDS property: E[r_t | F_{t-1}] = 0)
#    2. mds_f_test     — F-test for joint significance of OLS coefficients
#                        when regressing r_t on r_{t-1}, ..., r_{t-K}
#    3. spearman_ci    — Spearman correlation with bootstrap CI
#    4. cusum_detect   — CUSUM change-point detector (Pathway A)
# =============================================================================

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from config import MDS_LAGS, MDS_LEVEL, MASTER_SEED


# ── 1. Ljung–Box MDS test ─────────────────────────────────────────────────────

def mds_ljung_box(residuals: np.ndarray,
                  lags: list = MDS_LAGS,
                  level: float = MDS_LEVEL) -> dict:
    """
    Apply the Ljung–Box test to a 1-D residual series.

    H_0: residuals are uncorrelated at all specified lags.
    Reject H_0 ⟹ FAIL (evidence of autocorrelation ⟹ not an MDS).

    Parameters
    ----------
    residuals : (T,) array
    lags      : list of lags to test jointly (default [1..5])
    level     : significance level

    Returns
    -------
    dict with keys: lb_stat, p_value, pass_test (bool), max_lag
    """
    r = np.asarray(residuals).ravel()
    result = acorr_ljungbox(r, lags=[max(lags)], return_df=False)
    lb_stat = float(result['lb_stat'].iloc[-1])
    p_value = float(result['lb_pvalue'].iloc[-1])
    return {
        'lb_stat':   lb_stat,
        'p_value':   p_value,
        'pass_test': p_value > level,   # True = not rejected = MDS
        'max_lag':   max(lags),
    }


def mds_pass_rate(residuals_matrix: np.ndarray,
                  lags: list = MDS_LAGS,
                  level: float = MDS_LEVEL) -> float:
    """
    Fraction of test sequences whose residuals pass the Ljung–Box MDS test.

    residuals_matrix : (N, T) array
    Returns float in [0, 1].
    """
    N = residuals_matrix.shape[0]
    passes = 0
    for i in range(N):
        result = mds_ljung_box(residuals_matrix[i], lags=lags, level=level)
        if result['pass_test']:
            passes += 1
    return passes / N


# ── 2. OLS F-test for autocorrelation ────────────────────────────────────────

def mds_f_test(residuals: np.ndarray,
               lags: list = MDS_LAGS,
               level: float = MDS_LEVEL) -> dict:
    """
    Breusch-Godfrey-style OLS F-test: regress r_t on r_{t-1}, ..., r_{t-K}
    and test the joint null that all coefficients are zero.

    Returns
    -------
    dict with keys: F_stat, p_value, pass_test, df1, df2
    """
    r = np.asarray(residuals).ravel()
    K = max(lags)
    T = len(r)
    if T <= K + 1:
        return {'F_stat': np.nan, 'p_value': np.nan,
                'pass_test': True, 'df1': K, 'df2': T - K - 1}

    # Design matrix: intercept + lags
    y = r[K:]
    X = np.column_stack([r[K - k:T - k] for k in lags])
    X = np.column_stack([np.ones(len(y)), X])

    # OLS
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        res   = y - y_hat
        ss_res = res @ res
        ss_tot = (y - y.mean()) @ (y - y.mean())
        ss_reg = ss_tot - ss_res
        df1    = K
        df2    = len(y) - K - 1
        F_stat = (ss_reg / df1) / (ss_res / df2) if df2 > 0 else np.nan
        p_val  = 1 - stats.f.cdf(F_stat, df1, df2) if not np.isnan(F_stat) else np.nan
    except Exception:
        F_stat, p_val = np.nan, np.nan
        df1, df2 = K, 0

    return {
        'F_stat':    F_stat,
        'p_value':   p_val,
        'pass_test': (p_val > level) if not np.isnan(p_val) else True,
        'df1':       df1,
        'df2':       df2,
    }


# ── 3. Spearman correlation with bootstrap CI ─────────────────────────────────

def spearman_ci(x: np.ndarray, y: np.ndarray,
                n_boot: int = 1000,
                alpha: float = 0.05,
                seed: int = MASTER_SEED) -> dict:
    """
    Spearman rank correlation with (1 - alpha) bootstrap confidence interval.

    Parameters
    ----------
    x, y   : paired arrays of the same length
    n_boot : number of bootstrap replicates
    alpha  : significance level for CI

    Returns
    -------
    dict with keys: rho, p_value, ci_lo, ci_hi
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)

    rho, p_value = stats.spearmanr(x, y)

    # Guard: if either array is constant, Spearman is undefined
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return {'rho': np.nan, 'p_value': np.nan,
                'ci_lo': np.nan, 'ci_hi': np.nan}

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        xi, yi = x[idx], y[idx]
        if np.ptp(xi) < 1e-12 or np.ptp(yi) < 1e-12:
            boot_rhos[b] = np.nan
        else:
            boot_rhos[b], _ = stats.spearmanr(xi, yi)

    valid = boot_rhos[np.isfinite(boot_rhos)]
    if len(valid) == 0:
        return {'rho': rho, 'p_value': p_value, 'ci_lo': np.nan, 'ci_hi': np.nan}
    ci_lo = np.percentile(valid, 100 * alpha / 2)
    ci_hi = np.percentile(valid, 100 * (1 - alpha / 2))
    return {'rho': rho, 'p_value': p_value, 'ci_lo': ci_lo, 'ci_hi': ci_hi}


# ── 4. CUSUM change-point detector (Pathway A) ───────────────────────────────

def cusum_detector(X: np.ndarray,
                   mu0: float = None,
                   sigma0: float = None,
                   k: float = 0.5,
                   h: float = 5.0) -> dict:
    """
    Two-sided CUSUM for detecting a shift in the mean of a scalar series.

    Page's CUSUM:
      S_t^+ = max(0, S_{t-1}^+ + (X_t - mu0)/sigma0 - k)
      S_t^- = max(0, S_{t-1}^- - (X_t - mu0)/sigma0 - k)
    Alarm when S_t^+ > h or S_t^- > h.

    Parameters
    ----------
    X      : (T,) array
    mu0    : in-control mean (estimated from first 20% of series if None)
    sigma0 : in-control std  (estimated from first 20% of series if None)
    k      : reference value (0.5 * shift_size in sigma units)
    h      : decision threshold (in sigma units)

    Returns
    -------
    dict with keys:
      alarm_time  : first alarm index (or None)
      S_plus      : (T,) upper CUSUM
      S_minus     : (T,) lower CUSUM
      detected    : bool
    """
    X = np.asarray(X, dtype=float).ravel()
    T = len(X)

    burn = max(5, int(0.2 * T))
    if mu0 is None:
        mu0 = X[:burn].mean()
    if sigma0 is None:
        sigma0 = max(X[:burn].std(), 1e-8)

    S_p = np.zeros(T)
    S_m = np.zeros(T)
    alarm_time = None

    for t in range(1, T):
        z = (X[t] - mu0) / sigma0
        S_p[t] = max(0.0, S_p[t - 1] + z - k)
        S_m[t] = max(0.0, S_m[t - 1] - z - k)
        if alarm_time is None and (S_p[t] > h or S_m[t] > h):
            alarm_time = t

    return {
        'alarm_time': alarm_time,
        'S_plus':     S_p,
        'S_minus':    S_m,
        'detected':   alarm_time is not None,
    }


def detection_delay(true_cp: int, alarm_time, T: int) -> float:
    """
    Detection delay = alarm_time - true_cp.
    Returns T (full miss) if no alarm fired.
    """
    if alarm_time is None:
        return float(T - true_cp)
    return float(alarm_time - true_cp)
