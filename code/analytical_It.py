# =============================================================================
#  analytical_It.py — Analytical computation of the marginal information gain
#
#  Definition (P1 §3.3):
#    I_t = || E[Z | F_t] - E[Z | T_inf] ||_{L^2}
#
#  For numerical study, we take Z = X_{t+1} (one-step-ahead target).
#  The "initialisation-transient" interpretation (Proof of Prop 3.4):
#    I_t^(path) ≈ alpha * ||c_t - c_inf||_2 is path-specific and decays
#    geometrically during the transient from h_0 = 0 to the stationary regime.
#    The forget gate ratio I_t / I_{t-1} ≈ |phi| (constant for AR(1)).
#
#  For the non-stationary case, I_t(u) depends on the local coefficient phi(u).
# =============================================================================

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from config import SIGMA_EPS


# ── Stationary AR(1) ─────────────────────────────────────────────────────────

def ar1_stationary_It(phi: float, sigma: float = SIGMA_EPS) -> float:
    """
    Constant (unconditional) marginal information gain for stationary AR(1).

    I = || E[X_{t+1} | F_t] - E[X_{t+1} | T_inf] ||_{L^2}
      = || phi * X_t - 0 ||_{L^2}
      = |phi| * sigma / sqrt(1 - phi^2)

    This is CONSTANT over t for a stationary process; the "decay" manifests
    only during the initialisation transient (see ar1_transient_It).
    """
    return abs(phi) * sigma / np.sqrt(1.0 - phi**2)


def ar1_forget_gate_target(phi: float) -> float:
    """
    For stationary AR(1) the optimal forget gate is phi (Prop 3.4 applied
    to the linear-Gaussian / Kalman-filter regime).  Returns |phi|.
    """
    return abs(phi)


def ar1_transient_It(phi: float, T: int, sigma: float = SIGMA_EPS,
                     alpha: float = 1.0) -> np.ndarray:
    """
    I_t during the initialisation transient starting from h_0 = 0.

    From the proof of Prop 3.4 (linear-Gaussian regime):
      I_t ≈ alpha * |phi|^t * sigma_inf
    where sigma_inf = sigma / sqrt(1 - phi^2) is the stationary std.

    Parameters
    ----------
    phi   : AR coefficient
    T     : length of sequence
    sigma : innovation std
    alpha : proportionality constant (set to 1 for comparison)

    Returns
    -------
    It_values : ndarray of shape (T,)  — I_t at t = 0, 1, ..., T-1
    """
    sigma_inf = sigma / np.sqrt(1.0 - phi**2) if abs(phi) < 1 else sigma
    t = np.arange(T, dtype=float)
    return alpha * (abs(phi) ** t) * sigma_inf


def ar1_forget_gate_ratio(phi: float, T: int) -> np.ndarray:
    """
    I_t / I_{t-1} for the AR(1) transient.  Constant = |phi| for all t >= 1.
    """
    return np.full(T - 1, fill_value=abs(phi))


# ── HMM: numerical I_t via forward algorithm ─────────────────────────────────

def hmm_It_numerical(X: np.ndarray,
                     trans: np.ndarray,
                     means: np.ndarray,
                     std: float,
                     mu_global: float = None) -> np.ndarray:
    """
    Numerically compute I_t for a path X from a two-state Gaussian HMM.

    I_t^(path) = | E[X_{t+1} | X_{1:t}] - mu_global |

    where mu_global = pi_0 * means[0] + pi_1 * means[1] is the stationary mean.

    Uses the forward algorithm (log-space) to get P(s_t | X_{1:t}).

    Parameters
    ----------
    X       : ndarray (T,)  — observed sequence
    trans   : (2, 2) transition matrix
    means   : (2,) emission means
    std     : emission std
    mu_global : marginal mean; computed from stationary dist if None

    Returns
    -------
    It : ndarray (T-1,)  — I_t for t = 0, 1, ..., T-2
    """
    from scipy.stats import norm as stats_norm

    T = len(X)
    K = trans.shape[0]

    # Stationary distribution
    evals, evecs = np.linalg.eig(trans.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    pi = pi / pi.sum()

    if mu_global is None:
        mu_global = pi @ means

    # Forward pass — alpha[t, k] = P(s_t = k, X_{1:t}) (scaled)
    log_alpha = np.zeros((T, K))
    log_emit_0 = stats_norm.logpdf(X[0], means, std)
    log_alpha[0] = np.log(pi) + log_emit_0
    log_alpha[0] -= np.max(log_alpha[0])   # stabilise

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = (
                np.log(stats_norm.pdf(X[t], means[k], std))
                + np.log(np.sum(np.exp(log_alpha[t - 1]) * trans[:, k]))
            )
        log_alpha[t] -= np.max(log_alpha[t])

    # Posterior P(s_t | X_{1:t})
    alpha = np.exp(log_alpha)
    posterior = alpha / alpha.sum(axis=1, keepdims=True)

    # E[X_{t+1} | F_t] = sum_k P(s_{t+1}=k | F_t) * means[k]
    #                   = means @ (trans.T @ posterior[t])
    It = np.zeros(T - 1)
    for t in range(T - 1):
        p_next = trans.T @ posterior[t]        # P(s_{t+1} | F_t)
        pred = means @ p_next                  # E[X_{t+1} | F_t]
        It[t] = abs(pred - mu_global)
    return It


# ── Locally stationary AR: time-varying I_t ──────────────────────────────────

def ls_ar_It(phi_vals: np.ndarray, sigma: float = SIGMA_EPS) -> np.ndarray:
    """
    I_t for a locally stationary AR(1) with time-varying phi_t.

    I_t(u) = |phi(u)| * sigma / sqrt(1 - phi(u)^2)

    phi_vals : ndarray (T,) — phi values at each time step
    """
    return np.abs(phi_vals) * sigma / np.sqrt(1.0 - phi_vals**2)


def ls_ar_fg_ratio(phi_vals: np.ndarray, sigma: float = SIGMA_EPS) -> np.ndarray:
    """
    True I_t / I_{t-1} for locally stationary AR.
    Returns ndarray of length T-1.
    """
    It = ls_ar_It(phi_vals, sigma)
    return It[1:] / It[:-1]
