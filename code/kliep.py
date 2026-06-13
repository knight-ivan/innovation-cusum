# =============================================================================
#  kliep.py — KLIEP density-ratio estimator (Sugiyama et al. 2012)
#
#  Estimates w(x) = p_target(x) / p_source(x) using Kullback-Leibler
#  Importance Estimation Procedure with Gaussian (RBF) kernel centres.
#
#  Used in Study S3 (Pathway B) to:
#    1. Estimate L_t = dP_t / dP_0 on a rolling window
#    2. Extract the drift component A_t = L_t - E[L_t | B_{t+1}]
#    3. Compute the bias correction E_t (information-tracking bias)
#    4. Form Delta_t^corr = alpha * I_t^baseline + (1-alpha) * E_t
# =============================================================================

import numpy as np
from config import KLIEP_SIGMA, KLIEP_N_ITER, KLIEP_LR, KLIEP_WINDOW


class KLIEP:
    """
    Kernelised KLIEP density ratio estimator.

    Minimises KL(p_target || p_source * w) subject to
      E_{p_source}[w(X)] = 1 (normalisation).

    Parameters
    ----------
    sigma    : RBF kernel bandwidth
    n_iter   : gradient ascent iterations
    lr       : learning rate
    n_centres: number of kernel centres (subset of target samples)
    """

    def __init__(self,
                 sigma: float = KLIEP_SIGMA,
                 n_iter: int = KLIEP_N_ITER,
                 lr: float = KLIEP_LR,
                 n_centres: int = 50):
        self.sigma     = sigma
        self.n_iter    = n_iter
        self.lr        = lr
        self.n_centres = n_centres
        self.centres_  = None
        self.alpha_    = None

    def _rbf(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix K[i,j] = exp(-||X_i - C_j||^2 / (2*sigma^2))."""
        X = np.atleast_2d(X)
        C = np.atleast_2d(C)
        diffs = X[:, :, None] - C.T[None, :, :]   # broadcast
        # Handle 1-D case
        if X.shape[1] == 1:
            sq = (X - C.T) ** 2                    # (n_X, n_C)
        else:
            sq = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)
        return np.exp(-sq / (2.0 * self.sigma ** 2))

    def fit(self, X_source: np.ndarray, X_target: np.ndarray):
        """
        Fit the density ratio w(x) = p_target / p_source.

        Parameters
        ----------
        X_source : (n_s,) or (n_s, d)
        X_target : (n_t,) or (n_t, d)
        """
        X_s = np.atleast_2d(X_source).reshape(-1, 1) if X_source.ndim == 1 \
              else np.atleast_2d(X_source)
        X_t = np.atleast_2d(X_target).reshape(-1, 1) if X_target.ndim == 1 \
              else np.atleast_2d(X_target)

        n_t = X_t.shape[0]
        n_s = X_s.shape[0]

        # Choose kernel centres from target
        n_c = min(self.n_centres, n_t)
        idx = np.random.choice(n_t, size=n_c, replace=False)
        self.centres_ = X_t[idx]

        # Kernel matrices
        Phi_t = self._rbf(X_t, self.centres_)   # (n_t, n_c)
        Phi_s = self._rbf(X_s, self.centres_)   # (n_s, n_c)

        # Initialise alpha
        alpha = np.ones(n_c) / n_c

        # Projected gradient ascent (maximise E_{target}[log w(X)])
        for _ in range(self.n_iter):
            w_s = Phi_s @ alpha                    # (n_s,)
            # Gradient of KL objective + normalisation via projection
            grad = Phi_t.mean(axis=0) - Phi_s.T @ w_s / max(w_s.mean(), 1e-10)
            alpha += self.lr * grad
            # Project: alpha >= 0 and normalisation
            alpha = np.maximum(alpha, 0.0)
            norm_val = Phi_s @ alpha
            norm_mean = norm_val.mean()
            if norm_mean > 1e-10:
                alpha /= norm_mean

        self.alpha_ = alpha
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate density ratio w(x) = p_target(x) / p_source(x).

        Parameters
        ----------
        X : (n,) or (n, d)

        Returns
        -------
        w : (n,) — estimated importance weights (clipped to [0.01, 100])
        """
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        X_ = np.atleast_2d(X).reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
        w = self._rbf(X_, self.centres_) @ self.alpha_
        return np.clip(w, 0.01, 100.0)


# ── Rolling-window KLIEP for online density-ratio tracking ───────────────────

class RollingKLIEP:
    """
    Computes KLIEP density ratio L_t = dP_t / dP_0 on a rolling window.

    At each time t, fits KLIEP using:
      - Source: X_{t-W:t-W/2}  (earlier half of window = reference)
      - Target: X_{t-W/2:t}    (recent half of window = current distribution)

    Returns L_t estimates and a drift indicator A_t.

    Parameters
    ----------
    window : rolling window width W
    """

    def __init__(self,
                 window: int = KLIEP_WINDOW,
                 sigma: float = KLIEP_SIGMA,
                 n_iter: int = KLIEP_N_ITER,
                 lr: float = KLIEP_LR):
        self.window = window
        self.kliep_kwargs = dict(sigma=sigma, n_iter=n_iter, lr=lr,
                                 n_centres=min(30, window // 2))

    def fit_transform(self, X: np.ndarray) -> dict:
        """
        Apply rolling KLIEP to the scalar series X.

        Parameters
        ----------
        X : (T,) array

        Returns
        -------
        dict with keys:
          L_t    : (T,) density ratio (np.nan for t < window)
          A_t    : (T,) drift component = L_t - rolling_mean(L_t)
          bias   : (T,) |A_t| — information-tracking bias proxy
        """
        T = X.shape[0]
        W = self.window
        half = W // 2

        L = np.full(T, np.nan)
        for t in range(W, T):
            source = X[t - W     : t - half]
            target = X[t - half  : t       ]
            if len(source) < 2 or len(target) < 2:
                continue
            try:
                kliep = KLIEP(**self.kliep_kwargs)
                kliep.fit(source, target)
                # Evaluate at the current point X[t]
                L[t] = float(kliep.predict(np.array([[X[t]]]))[0])
            except Exception:
                pass

        # Drift component A_t: deviation of L_t from its rolling mean
        # (proxy for the quasi-martingale drift in Pathway B)
        A = np.full(T, np.nan)
        for t in range(W, T):
            window_L = L[max(0, t - half): t]
            valid = window_L[np.isfinite(window_L)]
            if len(valid) > 0:
                A[t] = L[t] - valid.mean() if np.isfinite(L[t]) else np.nan

        bias = np.where(np.isfinite(A), np.abs(A), np.nan)
        return {'L_t': L, 'A_t': A, 'bias': bias}


# ── Corrected information-gain surrogate (supplement Eq. 5) ──────────────────

def corrected_delta(It_baseline: np.ndarray,
                    Et: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Delta_t^corr = alpha * I_t^baseline + (1 - alpha) * E_t

    Parameters
    ----------
    It_baseline : (T,) — uncorrected I_t estimate (from LSTM forget gate
                         or block-bootstrap)
    Et          : (T,) — information-tracking bias |A_t| from RollingKLIEP
    alpha       : mixing weight (0 = full correction, 1 = no correction)

    Returns
    -------
    delta_corr : (T,)
    """
    return alpha * np.asarray(It_baseline) + (1.0 - alpha) * np.asarray(Et)
