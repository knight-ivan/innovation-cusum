"""
Study S7: Real-data application — Nile River annual flows
============================================================
Dataset : Nile River annual discharge at Aswan, 1871–1970 (T=100)
          Source: Cobb (1978); statsmodels nile dataset.
Known break: t=28 (year 1898) — construction of the first Aswan Dam
             caused a permanent mean reduction (~1100 → ~850 × 10^8 m³/yr)
             and increased autocorrelation.

Three CUSUM methods compared (same as S2B and S6):
  M1  Raw (standardised deviation from in-control mean)
  M2  AR(1)-whitened residuals
  M3  Pathway A: LSTM innovation CUSUM

Training strategy: block bootstrap on the 27 pre-break observations
  to generate 200 synthetic in-control sequences of length 27.

Outputs (saved to results/):
  S7_nile_detection.csv     — alarm time and delay per method
  S7_nile_cusum_traj.png    — CUSUM trajectories
  S7_nile_series.png        — raw series with change-point marked
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import statsmodels.api as sm

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
FIGURES = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TAU_TRUE   = 28          # known change-point (1898)
H_CUSUM    = 4.0         # threshold (lower because T=100 is short)
HIDDEN_DIM = 16
N_EPOCHS   = 80
LR         = 5e-4
BATCH_SIZE = 16
N_BOOT     = 200         # bootstrap training sequences
BLOCK_LEN  = 5           # block length for block bootstrap
BURN       = 20          # in-control burn-in steps for calibration
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
nile_raw = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
T = len(nile_raw)   # 100
# Standardise for LSTM stability
mu_all  = nile_raw[:TAU_TRUE].mean()
sig_all = nile_raw[:TAU_TRUE].std()
nile    = (nile_raw - mu_all) / sig_all    # standardised; (100,)

X_ic    = nile[:TAU_TRUE].reshape(-1, 1)   # (27, 1) in-control window
X_full  = nile.reshape(-1, 1)              # (100, 1)

print("Study S7: Nile River — Pathway A CUSUM")
print(f"  T={T}, tau_true={TAU_TRUE} (1898 Aswan Dam), h={H_CUSUM}")
print(f"  In-control window: {TAU_TRUE} obs (1871–1897)")

# ── Block bootstrap to generate training sequences ────────────────────────────
def block_bootstrap(x, n_seqs, block_len, seq_len, rng):
    """Generate n_seqs synthetic sequences of length seq_len
    by randomly concatenating blocks from x."""
    n = len(x); seqs = []
    for _ in range(n_seqs):
        seq = []
        while len(seq) < seq_len:
            start = rng.integers(0, max(1, n - block_len))
            seq.extend(x[start:start+block_len].tolist())
        seqs.append(np.array(seq[:seq_len]).reshape(-1, 1))
    return seqs

rng = np.random.default_rng(SEED)
train_seqs = block_bootstrap(X_ic.ravel(), N_BOOT, BLOCK_LEN, TAU_TRUE, rng)

# ── LSTM model ────────────────────────────────────────────────────────────────
class UniLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def train_lstm(seqs, hidden=HIDDEN_DIM, epochs=N_EPOCHS, lr=LR, bs=BATCH_SIZE):
    model   = UniLSTM(hidden)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    T_seq   = tensors[0].shape[0]
    for ep in range(epochs):
        idx = np.random.permutation(len(tensors))
        for i in range(0, len(tensors), bs):
            batch = torch.stack([tensors[j] for j in idx[i:i+bs]])  # (B,T,1)
            inp   = batch[:, :-1, :]
            tgt   = batch[:, 1:, :]
            pred  = model(inp)
            loss  = loss_fn(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

print(f"  Training LSTM on {N_BOOT} bootstrap sequences ...", end='', flush=True)
model = train_lstm(train_seqs)
print(" done.")

# ── Compute signals ───────────────────────────────────────────────────────────
def lstm_innov(model, x):
    """One-step prediction errors for x:(T,1). Returns (T-1,1) errors."""
    with torch.no_grad():
        inp  = torch.tensor(x[:-1], dtype=torch.float32).unsqueeze(0)  # (1,T-1,1)
        pred = model(inp).squeeze(0).numpy()                             # (T-1,1)
    return x[1:] - pred   # (T-1,1)

# M1: Raw — z_t = (x_t - mu_ic) / sig_ic
mu_ic  = X_ic.mean()
sig_ic = X_ic.std() + 1e-8
score_raw = ((X_full - mu_ic) / sig_ic).ravel()   # (T,)

# M2: AR(1) whitened — OLS on in-control; residuals on full series
Y_ic  = X_ic[1:].ravel()
Z_ic  = X_ic[:-1].ravel()
phi_hat = (Z_ic @ Y_ic) / (Z_ic @ Z_ic)
resid_full = X_full[1:, 0] - phi_hat * X_full[:-1, 0]   # (T-1,)
mu_r   = resid_full[:TAU_TRUE-1].mean()
sig_r  = resid_full[:TAU_TRUE-1].std() + 1e-8
score_white = (resid_full - mu_r) / sig_r              # (T-1,)

# M3: Pathway A — LSTM innovations
innov_full = lstm_innov(model, X_full).ravel()          # (T-1,)
mu_i  = innov_full[:TAU_TRUE-1].mean()
sig_i = innov_full[:TAU_TRUE-1].std() + 1e-8
score_lstm = (innov_full - mu_i) / sig_i                # (T-1,)

# ── Page CUSUM ────────────────────────────────────────────────────────────────
def page_cusum(z, k=0.5, h=H_CUSUM):
    Sp = np.zeros(len(z)+1); Sm = np.zeros(len(z)+1)
    for t, zt in enumerate(z):
        Sp[t+1] = max(0.0, Sp[t] + zt - k)
        Sm[t+1] = max(0.0, Sm[t] - zt - k)
        if Sp[t+1] > h or Sm[t+1] > h:
            return t+1, Sp[1:], Sm[1:]
    return len(z)+1, Sp[1:], Sm[1:]

al_raw,   Sp_raw,   Sm_raw   = page_cusum(score_raw)
al_white, Sp_white, Sm_white = page_cusum(score_white)
al_lstm,  Sp_lstm,  Sm_lstm  = page_cusum(score_lstm)

def fmt_delay(a, tau=TAU_TRUE):
    if a > T: return "no alarm"
    return f"{a - tau:+d}"

print()
print(f"  {'Method':<12} {'Alarm':>8} {'Delay':>8}")
print(f"  {'raw':<12} {al_raw:>8d} {fmt_delay(al_raw):>8}")
print(f"  {'whitened':<12} {al_white:>8d} {fmt_delay(al_white):>8}")
print(f"  {'pathway_a':<12} {al_lstm:>8d} {fmt_delay(al_lstm):>8}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    'method':  ['raw', 'whitened', 'pathway_a'],
    'alarm_t': [al_raw, al_white, al_lstm],
    'delay':   [al_raw - TAU_TRUE, al_white - TAU_TRUE, al_lstm - TAU_TRUE],
})
csv_path = os.path.join(RESULTS, 'S7_nile_detection.csv')
df_out.to_csv(csv_path, index=False)
print(f"\n  Saved: {os.path.basename(csv_path)}")

# ── Figure 1: Raw series ──────────────────────────────────────────────────────
years = np.arange(1871, 1971)
fig1, ax = plt.subplots(figsize=(9, 3))
ax.plot(years, nile_raw, 'navy', lw=1.2)
ax.axvline(1871 + TAU_TRUE - 1, color='red', lw=1.5, ls='--', label=f'τ=1898 (t={TAU_TRUE})')
ax.set_xlabel('Year'); ax.set_ylabel('Annual discharge (10⁸ m³/yr)')
ax.set_title('Nile River annual flows, 1871–1970')
ax.legend(); plt.tight_layout()
fig1.savefig(os.path.join(RESULTS, 'S7_nile_series.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)

# ── Figure 2: CUSUM trajectories ─────────────────────────────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
# score_raw has length T=100, others T-1=99; plot on common 99-length axis
t_ax = np.arange(1, T)    # t = 1..99 (index into series after lag)

methods_plot = [
    ('Raw (M1)',       score_raw[1:],  Sp_raw[:T-1],   Sm_raw[:T-1],   al_raw,   'steelblue'),
    ('Whitened (M2)',  score_white,    Sp_white,        Sm_white,        al_white, 'darkorange'),
    ('Pathway A (M3)', score_lstm,     Sp_lstm,         Sm_lstm,         al_lstm,  'forestgreen'),
]

for ax, (label, score, Sp, Sm, alarm, col) in zip(axes, methods_plot):
    n = min(len(t_ax), len(Sp))
    ax.plot(t_ax[:n], Sp[:n],  color=col, lw=1.5, label='$S^+_t$')
    ax.plot(t_ax[:n], Sm[:n],  color=col, lw=1.5, ls='--', alpha=0.6, label='$S^-_t$')
    ax.axhline(H_CUSUM, color='red', lw=1.0, ls=':', label=f'$h={H_CUSUM}$')
    ax.axvline(TAU_TRUE, color='black', lw=1.2, ls='--', alpha=0.7, label=f'$\\tau={TAU_TRUE}$')
    if alarm <= T:
        ax.axvline(alarm, color=col, lw=1.5, ls='-', alpha=0.9,
                   label=f'alarm $t={alarm}$ (delay {alarm-TAU_TRUE:+d})')
    else:
        ax.text(0.98, 0.9, 'no alarm', transform=ax.transAxes,
                ha='right', color=col, fontsize=9)
    ax.set_ylabel('CUSUM', fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, ncol=6, loc='upper left')
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel('$t$ (year offset from 1871)', fontsize=9)
fig2.suptitle('Study S7: CUSUM trajectories — Nile River (known break $\\tau=28$, year 1898)',
              fontsize=11)
plt.tight_layout()
fig2.savefig(os.path.join(FIGURES, 'S7_nile_cusum_traj.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: S7_nile_cusum_traj.png")
print(f"  Saved: S7_nile_series.png")
print(f"\nDone in {time.time():.0f}s (wall)")
