# =============================================================================
#  bootstrap_It.py — Block-bootstrap estimator of I_t  (Algorithm 1, P2)
#
#  Used here as a "gold-standard" surrogate against which the LSTM forget-gate
#  proxy and the KLIEP-corrected estimate are evaluated in Study S3.
# =============================================================================

import numpy as np
from config import BB_BLOCK_LEN, BB_N_BOOT, MASTER_SEED


def block_bootstrap_resample(X: np.ndarray,
                              block_len: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Circular block-bootstrap resample of a 1-D series X of length T.

    Draws T // block_len + 1 overlapping blocks of length block_len from
    the circular extension, then trims to length T.
    """
    T = len(X)
    X_circ = np.concatenate([X, X[:block_len]])    # circular padding
    n_blocks = T // block_len + 1
    starts = rng.integers(0, T, size=n_blocks)
    blocks = [X_circ[s: s + block_len] for s in starts]
    return np.concatenate(blocks)[:T]


def bootstrap_It(X: np.ndarray,
                 predictor_fit_fn,
                 block_len: int = BB_BLOCK_LEN,
                 B: int = BB_N_BOOT,
                 long_window: int = None,
                 seed: int = MASTER_SEED) -> np.ndarray:
    """
    Block-bootstrap estimator of I_t (Algorithm 1, Paper 2, §3.1).

    hat_I_t^boot = sqrt( B^{-1} sum_{b=1}^B (hat_mu_t^(b) - hat_mu_inf^(b))^2 )

    where
      hat_mu_t^(b)   = one-step-ahead prediction of x^(b)_{t+1} given x^(b)_{1:t}
      hat_mu_inf^(b) = T^{-1} sum_{s=t}^T x^(b)_s  (tail mean proxy)

    Parameters
    ----------
    X               : (T,) observed series
    predictor_fit_fn: callable(X_train, t) -> float
                      Given the bootstrap resample X and time index t,
                      returns the one-step-ahead prediction at t.
                      Default: AR(1) OLS predictor.
    block_len       : block length l
    B               : number of bootstrap replications
    long_window     : number of tail steps for hat_mu_inf (default T//5)
    seed            : random seed

    Returns
    -------
    It_boot : (T-1,)  — I_t estimate for t = 0, 1, ..., T-2
    """
    T = len(X)
    if long_window is None:
        long_window = max(10, T // 5)

    rng = np.random.default_rng(seed)

    # Bootstrap estimates: mu_t^(b) and mu_inf^(b) for each b and t
    mu_t   = np.zeros((B, T - 1))
    mu_inf = np.zeros((B, T - 1))

    for b in range(B):
        Xb = block_bootstrap_resample(X, block_len, rng)
        for t in range(T - 1):
            # One-step-ahead prediction using AR(1) OLS on Xb[:t+1]
            if predictor_fit_fn is None:
                mu_t[b, t] = _ar1_ols_predict(Xb, t)
            else:
                mu_t[b, t] = predictor_fit_fn(Xb, t)
            # Tail mean: proxy for E[X_{t+1} | T_inf]
            start = max(t + 1, T - long_window)
            mu_inf[b, t] = Xb[start:].mean()

    diff_sq = (mu_t - mu_inf) ** 2              # (B, T-1)
    It_boot = np.sqrt(diff_sq.mean(axis=0))    # (T-1,)
    return It_boot


def _ar1_ols_predict(X: np.ndarray, t: int) -> float:
    """
    AR(1) OLS predictor: fit phi on X[0:t+1] and predict X[t+1].
    Falls back to the sample mean for t < 2.
    """
    if t < 2:
        return X[:t + 1].mean()
    y = X[1:t + 1]
    x = X[:t]
    phi_hat = (x @ y) / max(x @ x, 1e-12)
    return float(phi_hat * X[t])


def rolling_window_It(X: np.ndarray,
                      k: int = 5,
                      K: int = 40) -> np.ndarray:
    """
    Surrogate A (P2 §3.3.1): rolling-window prediction gap.

    hat_I_{t,A} = | X_bar_{t,k} - X_bar_{t,K} |

    where X_bar_{t,w} = (1/w) sum_{s=max(0,t-w+1)}^t X_s.

    Returns ndarray of length T.
    """
    T = len(X)
    It_A = np.full(T, np.nan)
    for t in range(K - 1, T):
        mean_k = X[max(0, t - k + 1): t + 1].mean()
        mean_K = X[max(0, t - K + 1): t + 1].mean()
        It_A[t] = abs(mean_k - mean_K)
    return It_A
