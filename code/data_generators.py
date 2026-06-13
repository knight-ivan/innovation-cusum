# =============================================================================
#  data_generators.py — Synthetic process generators for P1 empirical studies
#
#  Processes implemented:
#    1. AR(1)                   — stationary baseline
#    2. Two-state HMM           — stationary, with discrete latent state
#    3. Regime-switching AR     — stationary, nonlinear
#    4. Locally stationary AR   — smooth time-varying phi(u)
#    5. Change-point AR         — abrupt single break at T*cp_frac
#    6. Nile River loader       — loads data/nile.csv (written by nile_analysis.R)
# =============================================================================

import numpy as np
from config import (SIGMA_EPS, HMM_TRANS, HMM_MEANS, HMM_STD,
                    RSAR_PHIS, RSAR_TRANS, LS_PHI_CENTER, LS_PHI_AMP,
                    CP_PHI_BEFORE, CP_PHI_AFTER, CP_FRACTION)
import os


# ── 1. AR(1) ─────────────────────────────────────────────────────────────────

def generate_ar1(N: int, T: int, phi: float,
                 sigma: float = SIGMA_EPS,
                 seed: int = None) -> np.ndarray:
    """
    Generate N independent AR(1) sequences of length T.

    X_t = phi * X_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)

    Initialised from the stationary distribution N(0, sigma^2/(1-phi^2)).

    Returns
    -------
    X : ndarray of shape (N, T)
    """
    rng = np.random.default_rng(seed)
    stat_std = sigma / np.sqrt(1.0 - phi**2) if abs(phi) < 1 else sigma
    X = np.zeros((N, T))
    X[:, 0] = rng.normal(0.0, stat_std, size=N)
    eps = rng.normal(0.0, sigma, size=(N, T - 1))
    for t in range(1, T):
        X[:, t] = phi * X[:, t - 1] + eps[:, t - 1]
    return X


def generate_ar1_transient(N: int, T: int, phi: float,
                            sigma: float = SIGMA_EPS,
                            seed: int = None) -> np.ndarray:
    """
    Like generate_ar1 but initialised from X_0 = 0 (far from stationary).
    Used to study the initialisation transient in Section 4 / Study S1.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N, T))
    # X[:, 0] = 0  already
    eps = rng.normal(0.0, sigma, size=(N, T - 1))
    for t in range(1, T):
        X[:, t] = phi * X[:, t - 1] + eps[:, t - 1]
    return X


# ── 2. Two-state HMM ──────────────────────────────────────────────────────────

def generate_hmm(N: int, T: int,
                 trans: np.ndarray = HMM_TRANS,
                 means: np.ndarray = HMM_MEANS,
                 std: float = HMM_STD,
                 seed: int = None):
    """
    Generate N independent sequences from a two-state Gaussian HMM.

    States s_t in {0, 1}.  X_t | s_t ~ N(means[s_t], std^2).
    Transition matrix P[i,j] = P(s_t=j | s_{t-1}=i).

    Returns
    -------
    X      : ndarray (N, T)
    states : ndarray (N, T)  — hidden state labels
    """
    rng = np.random.default_rng(seed)
    K = trans.shape[0]
    # Stationary distribution
    evals, evecs = np.linalg.eig(trans.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    pi = pi / pi.sum()

    states = np.zeros((N, T), dtype=int)
    X = np.zeros((N, T))

    states[:, 0] = rng.choice(K, size=N, p=pi)
    for t in range(1, T):
        for k in range(K):
            mask = states[:, t - 1] == k
            if mask.any():
                states[mask, t] = rng.choice(K, size=mask.sum(), p=trans[k])
    X = means[states] + rng.normal(0.0, std, size=(N, T))
    return X, states


# ── 3. Regime-switching AR ────────────────────────────────────────────────────

def generate_rsar(N: int, T: int,
                  phis: list = RSAR_PHIS,
                  trans: np.ndarray = RSAR_TRANS,
                  sigma: float = SIGMA_EPS,
                  seed: int = None):
    """
    Regime-switching AR(1): the AR coefficient phi_t switches between
    phis[0] and phis[1] according to a Markov chain with trans matrix `trans`.

    Returns
    -------
    X      : ndarray (N, T)
    states : ndarray (N, T)
    """
    rng = np.random.default_rng(seed)
    K = len(phis)
    evals, evecs = np.linalg.eig(trans.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    pi = pi / pi.sum()

    states = np.zeros((N, T), dtype=int)
    states[:, 0] = rng.choice(K, size=N, p=pi)
    for t in range(1, T):
        for k in range(K):
            mask = states[:, t - 1] == k
            if mask.any():
                states[mask, t] = rng.choice(K, size=mask.sum(), p=trans[k])

    phi_t = np.array(phis)[states]           # (N, T)
    X = np.zeros((N, T))
    sigma_sq = sigma**2
    X[:, 0] = rng.normal(0.0, sigma, size=N)
    eps = rng.normal(0.0, sigma, size=(N, T - 1))
    for t in range(1, T):
        X[:, t] = phi_t[:, t] * X[:, t - 1] + eps[:, t - 1]
    return X, states


# ── 4. Locally stationary AR ─────────────────────────────────────────────────

def phi_smooth(u: np.ndarray,
               center: float = LS_PHI_CENTER,
               amp: float = LS_PHI_AMP) -> np.ndarray:
    """phi(u) = center + amp * sin(2*pi*u),  u in [0,1].

    Ranges from center-amp to center+amp; kept in (-1, 1) by design.
    """
    return center + amp * np.sin(2.0 * np.pi * u)


def generate_locally_stationary_ar(N: int, T: int,
                                    phi_func=None,
                                    sigma: float = SIGMA_EPS,
                                    seed: int = None) -> np.ndarray:
    """
    Generate N sequences from a locally stationary AR(1) process:
      X_{t,T} = phi(t/T) * X_{t-1,T} + eps_t,  eps_t ~ N(0, sigma^2).

    phi_func : callable u -> phi, default phi_smooth.
    """
    if phi_func is None:
        phi_func = phi_smooth
    rng = np.random.default_rng(seed)
    u = np.arange(T) / T
    phi_vals = phi_func(u)                   # (T,)

    X = np.zeros((N, T))
    eps = rng.normal(0.0, sigma, size=(N, T - 1))
    # Start near 0 (no stationary init for non-stationary process)
    X[:, 0] = rng.normal(0.0, sigma, size=N)
    for t in range(1, T):
        X[:, t] = phi_vals[t] * X[:, t - 1] + eps[:, t - 1]
    return X, phi_vals


# ── 5. Change-point AR ────────────────────────────────────────────────────────

def generate_changepoint_ar(N: int, T: int,
                             phi_before: float = CP_PHI_BEFORE,
                             phi_after: float = CP_PHI_AFTER,
                             cp_frac: float = CP_FRACTION,
                             sigma: float = SIGMA_EPS,
                             seed: int = None):
    """
    AR(1) with a single abrupt change-point at t* = floor(T * cp_frac).

    Before t*: phi = phi_before.
    At and after t*: phi = phi_after.

    Returns
    -------
    X   : ndarray (N, T)
    tau : int — change-point index
    """
    rng = np.random.default_rng(seed)
    tau = int(np.floor(T * cp_frac))
    stat_std = sigma / np.sqrt(1.0 - phi_before**2)

    X = np.zeros((N, T))
    X[:, 0] = rng.normal(0.0, stat_std, size=N)
    eps = rng.normal(0.0, sigma, size=(N, T - 1))
    for t in range(1, T):
        phi_t = phi_before if t < tau else phi_after
        X[:, t] = phi_t * X[:, t - 1] + eps[:, t - 1]
    return X, tau


# ── 6. Nile River data loader ─────────────────────────────────────────────────

def load_nile(data_dir: str = None) -> np.ndarray:
    """
    Load the Nile River annual discharge series (1871–1970, T=100).

    The CSV file `nile.csv` is created by running `nile_analysis.R`.
    Falls back to a hand-typed copy if the file is not found.

    Returns
    -------
    nile : ndarray of shape (100,)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    csv_path = os.path.join(data_dir, "nile.csv")
    if os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df["flow"].values.astype(float)

    # Built-in fallback: exact Nile River values from R's Nile dataset
    nile = np.array([
        1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140,
        995,  935,  1110, 994,  1020, 960,  1180, 799,  958,  1140,
        1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774,  840,
        874,  694,  940,  833,  701,  916,  692,  1020, 1050, 969,
        831,  726,  456,  824,  702,  1120, 1100, 832,  764,  821,
        768,  845,  864,  862,  698,  845,  744,  796,  1040, 759,
        781,  865,  845,  944,  984,  897,  822,  1010, 771,  676,
        649,  846,  812,  742,  801,  1040, 860,  874,  848,  890,
        744,  749,  838,  1050, 918,  986,  797,  923,  975,  815,
        1020, 906,  901,  1170, 912,  746,  919,  718,  714,  740
    ], dtype=float)
    return nile
