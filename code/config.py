# =============================================================================
#  config.py — Global configuration for P1 empirical studies
#  Paper 1: Martingale Structure of RNN and LSTM Hidden States
#  Yuan-chin Ivan Chang, Academia Sinica
# =============================================================================

import os

# ── Random seed ───────────────────────────────────────────────────────────────
MASTER_SEED = 2024

# ── Replication counts ────────────────────────────────────────────────────────
# "1000 replications" = 1000 independent test sequences from a fixed trained
# model.  For studies that require model retraining, 100 seeds × 10 sequences
# achieves the same statistical power more efficiently.
N_TEST_SEQ   = 1000   # independent test sequences for S1, S2
N_RECOVERY   = 500    # replications for the recovery comparison (S3)

# ── Sequence lengths ──────────────────────────────────────────────────────────
T_TRAIN      = 150    # training sequence length
T_TEST       = 200    # test / transient sequence length (long enough to see decay)
T_NILE       = 100    # Nile River dataset length

# ── Training hyperparameters ─────────────────────────────────────────────────
N_TRAIN_SEQ  = 1000   # number of training sequences per model
N_EPOCHS     = 40
BATCH_SIZE   = 64
LR           = 1e-3

# ── LSTM hyperparameters ──────────────────────────────────────────────────────
HIDDEN_DIM   = 32     # d — chosen small for CPU tractability
N_LAYERS     = 1

# ── Simplified Mamba / SSM hyperparameters ────────────────────────────────────
MAMBA_D_STATE = 16

# ── AR(1) process parameters ─────────────────────────────────────────────────
AR_PHIS      = [0.30, 0.70, 0.95]   # autocorrelation coefficients
SIGMA_EPS    = 1.0                   # innovation standard deviation

# ── HMM parameters ───────────────────────────────────────────────────────────
import numpy as np
HMM_TRANS    = np.array([[0.9, 0.1],
                          [0.1, 0.9]])
HMM_MEANS    = np.array([-1.5, 1.5])
HMM_STD      = 1.0

# ── Regime-switching AR parameters ───────────────────────────────────────────
RSAR_PHIS    = [0.30, 0.85]         # AR coefficients in each regime
RSAR_TRANS   = HMM_TRANS.copy()

# ── Locally stationary AR parameters ─────────────────────────────────────────
# phi(u) = phi_center + phi_amp * sin(2*pi*u), u = t/T in [0,1]
LS_PHI_CENTER = 0.50
LS_PHI_AMP    = 0.40   # range: 0.10 to 0.90

# ── Change-point AR parameters ───────────────────────────────────────────────
CP_PHI_BEFORE = 0.30
CP_PHI_AFTER  = 0.85
CP_FRACTION   = 0.50   # change-point at T * CP_FRACTION

# ── Statistical tests ─────────────────────────────────────────────────────────
MDS_LAGS     = [1, 2, 3, 4, 5]
MDS_LEVEL    = 0.05    # significance level for MDS test

# ── KLIEP hyperparameters ─────────────────────────────────────────────────────
KLIEP_SIGMA   = 1.0    # RBF kernel bandwidth
KLIEP_N_ITER  = 200
KLIEP_LR      = 0.01
KLIEP_WINDOW  = 50     # rolling window width for online KLIEP

# ── Block bootstrap hyperparameters ─────────────────────────────────────────
BB_BLOCK_LEN  = 10     # block length l
BB_N_BOOT     = 200    # B replications

# ── Output directories ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_HERE, "results")
FIGURES_DIR  = os.path.join(_HERE, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
