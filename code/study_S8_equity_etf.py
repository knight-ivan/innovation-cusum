"""
Study S8: Real multivariate application — US equity sector ETF returns
=======================================================================
Dataset : Daily log-returns of US equity sector SPDR ETFs, 2017-01-01
          to 2022-06-30, sourced from Yahoo Finance via yfinance.
Tickers : XLF (Financial), XLK (Technology), XLE (Energy)
Known break : 2020-02-19 (S&P 500 all-time high immediately before
              the COVID-19 pandemic crash; widely cited as the most
              abrupt cross-sector distributional shift in recent history)

Two configurations run in PARALLEL (ProcessPoolExecutor, 2 workers):
  Part A  d=1  XLF only         (univariate, bridges Study S7)
  Part B  d=3  XLF + XLK + XLE (genuinely multivariate)

Note: companion paper (JRSS-B) uses a two-sector panel (d=2); d=1 and d=3
      are chosen to complement rather than overlap.

Hardware: iMac Pro, 8-core Intel Xeon W, 128 GB RAM.
  Each worker uses 4 PyTorch intra-op threads (8 threads total).

Three CUSUM variants (same as Studies S2B, S6, S7):
  M1  Raw Mahalanobis CUSUM
  M2  VAR(1)-whitened Mahalanobis CUSUM
  M3  Pathway A: LSTM innovation CUSUM

ARL0 calibration: binary search over h using Monte Carlo simulation
  with N_MC=300 rolling-window in-control sequences drawn from
  the in-control training set.

Outputs (saved to results/):
  S8_etf_detection.csv      — alarm time, delay, ARL0, threshold per method x config
  S8_etf_cusum_partA.png    — CUSUM trajectories Part A (d=1)
  S8_etf_cusum_partB.png    — CUSUM trajectories Part B (d=3)
  S8_etf_returns.png        — raw log-return series with change-point marked

Usage:
  cd P1_Theory/code
  python study_S8_equity_etf.py
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE   = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')
FIGURES = os.path.join(_HERE, 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── Global config ─────────────────────────────────────────────────────────────
CFG = dict(
    tickers      = ['XLF', 'XLK', 'XLE'],
    ic_start     = '2017-01-01',
    ic_end       = '2019-12-31',
    test_start   = '2020-01-01',
    test_end     = '2022-06-30',
    break_date   = '2020-02-19',   # S&P 500 all-time high; COVID crash onset
    hidden_dim   = 32,
    n_epochs     = 60,
    lr           = 1e-3,
    batch_size   = 32,
    win_len      = 100,            # rolling window for LSTM training
    h_cusum      = 5.0,            # fixed threshold, consistent with Studies S2B and S6
    n_mc         = 500,            # Monte Carlo sequences for ARL0 estimation
    mc_seq_len   = 500,            # length of each MC sequence (block-bootstrapped)
    block_len    = 10,             # block length for block bootstrap
    k_cusum      = 0.5,
    seed         = 42,
    n_threads    = 4,
)

# ── Data ──────────────────────────────────────────────────────────────────────
CACHE = os.path.join(_HERE, 'data', 'S8_etf_prices_covid.csv')

def fetch_prices(tickers):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")
    df = yf.download(tickers, start='2016-12-01', end='2022-07-31',
                     auto_adjust=True, progress=False)['Close']
    return df[tickers].dropna()

def load_data(cfg):
    tickers = cfg['tickers']
    if os.path.exists(CACHE):
        prices = pd.read_csv(CACHE, index_col=0, parse_dates=True)[tickers]
    else:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        print("  Downloading ETF prices via yfinance ...", end='', flush=True)
        prices = fetch_prices(tickers)
        prices.to_csv(CACHE)
        print(" done.")

    log_ret = np.log(prices / prices.shift(1)).dropna()
    log_ret.index = pd.to_datetime(log_ret.index)

    ic_ret   = log_ret[cfg['ic_start']:cfg['ic_end']].values
    test_ret = log_ret[cfg['test_start']:cfg['test_end']].values
    test_idx = log_ret[cfg['test_start']:cfg['test_end']].index

    mu_ic  = ic_ret.mean(axis=0)
    std_ic = ic_ret.std(axis=0) + 1e-8
    ic_norm   = (ic_ret  - mu_ic) / std_ic
    test_norm = (test_ret - mu_ic) / std_ic

    tau_star = int(np.searchsorted(test_idx, pd.Timestamp(cfg['break_date'])))
    return ic_norm, test_norm, test_idx, tau_star, log_ret, mu_ic, std_ic

# ── Statistics helpers ────────────────────────────────────────────────────────
def fit_var1(X):
    Y, Z = X[1:], X[:-1]
    Phi  = np.linalg.lstsq(Z, Y, rcond=None)[0].T
    return Phi, Y - Z @ Phi.T

def safe_chol(M, d):
    M = M.reshape(d, d)
    return np.linalg.cholesky(M + 1e-6 * np.eye(d))

def mahal_score(X, L):
    d = X.shape[1]
    z = np.linalg.solve(L, X.T).T
    return (z ** 2).sum(axis=1) / d - 1.0

def whitened_score(X, Phi, L_e):
    resid = X[1:] - (Phi @ X[:-1].T).T
    return mahal_score(resid, L_e)

def page_cusum(z, k, h=None):
    S = np.zeros(len(z) + 1)
    for t, zt in enumerate(z):
        S[t+1] = max(0.0, S[t] + zt - k)
        if h is not None and S[t+1] >= h:
            return t + 1, S[1:]
    return len(z) + 1, S[1:]

def block_bootstrap_scores(z_ic, n_mc, seq_len, block_len, rng):
    """Generate n_mc synthetic in-control score sequences of length seq_len
    by block-bootstrapping from z_ic."""
    n = len(z_ic)
    seqs = []
    for _ in range(n_mc):
        seq = []
        while len(seq) < seq_len:
            start = int(rng.integers(0, max(1, n - block_len)))
            seq.extend(z_ic[start:start + block_len].tolist())
        seqs.append(np.array(seq[:seq_len]))
    return seqs

def estimate_arl0(z_ic, k, h, n_mc, seq_len, block_len, rng):
    """Estimate ARL0 using block-bootstrapped in-control sequences."""
    seqs   = block_bootstrap_scores(z_ic, n_mc, seq_len, block_len, rng)
    alarms = [page_cusum(s, k=k, h=h)[0] for s in seqs]
    return float(np.mean(alarms))

# ── LSTM ──────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MultiLSTM(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.lstm = nn.LSTM(d, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, d)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def train_lstm(X, d, cfg):
    torch.set_num_threads(cfg['n_threads'])
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    win = cfg['win_len']
    seqs = [X[s:s+win] for s in range(0, len(X) - win, 1)]
    tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
    model   = MultiLSTM(d, cfg['hidden_dim'])
    opt     = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.MSELoss()
    bs      = cfg['batch_size']
    for _ in range(cfg['n_epochs']):
        idx = rng.permutation(len(tensors))
        for i in range(0, len(tensors), bs):
            batch = torch.stack([tensors[j] for j in idx[i:i+bs]])
            inp, tgt = batch[:, :-1, :], batch[:, 1:, :]
            loss = loss_fn(model(inp), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

def lstm_innovations(model, X):
    with torch.no_grad():
        inp  = torch.tensor(X[:-1], dtype=torch.float32).unsqueeze(0)
        pred = model(inp).squeeze(0).numpy()
    return X[1:] - pred

# ── Worker ────────────────────────────────────────────────────────────────────
def run_part(part_label, d_idx, ic_norm, test_norm, test_idx, tau_star, cfg):
    torch.set_num_threads(cfg['n_threads'])
    np.random.seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])

    d    = len(d_idx)
    ic   = ic_norm[:, d_idx]
    test = test_norm[:, d_idx]
    T1   = len(test)
    k    = cfg['k_cusum']
    win  = cfg['win_len']
    n_mc      = cfg['n_mc']
    seq_len   = cfg['mc_seq_len']
    block_len = cfg['block_len']
    h         = cfg['h_cusum']

    # Covariance structures
    Sig_ic = (np.cov(ic.T) if d > 1 else np.array([[ic.var()]])).reshape(d, d)
    L_ic   = safe_chol(Sig_ic, d)
    Phi, ic_resid = fit_var1(ic)
    Sig_e  = (np.cov(ic_resid.T) if d > 1 else np.array([[ic_resid.var()]])).reshape(d, d)
    L_e    = safe_chol(Sig_e, d)

    # Train LSTM
    print(f"  [Part {part_label}, d={d}] Training LSTM ...", flush=True)
    t0    = time.time()
    model = train_lstm(ic, d, cfg)
    print(f"  [Part {part_label}, d={d}] Training done in {time.time()-t0:.1f}s", flush=True)

    # In-control innovation statistics
    innov_ic = lstm_innovations(model, ic)
    mu_i     = innov_ic.mean(axis=0)
    sig_i    = innov_ic.std(axis=0) + 1e-8

    # In-control scores
    z_ic_raw   = mahal_score(ic, L_ic)
    z_ic_white = whitened_score(np.vstack([ic[:1], ic]), Phi, L_e)
    innov_nrm  = (innov_ic - mu_i) / sig_i
    z_ic_lstm  = mahal_score(innov_nrm, np.eye(d))

    # Estimate ARL0 at fixed h via block bootstrap
    print(f"  [Part {part_label}, d={d}] Estimating ARL0 (MC, n_mc={n_mc}) ...",
          end='', flush=True)
    arl_raw   = estimate_arl0(z_ic_raw,   k, h, n_mc, seq_len, block_len, rng)
    arl_white = estimate_arl0(z_ic_white, k, h, n_mc, seq_len, block_len, rng)
    arl_lstm  = estimate_arl0(z_ic_lstm,  k, h, n_mc, seq_len, block_len, rng)
    print(f" done.", flush=True)
    print(f"  [Part {part_label}, d={d}] h={h}  "
          f"ARL0_raw={arl_raw:.0f}  "
          f"ARL0_white={arl_white:.0f}  "
          f"ARL0_lstm={arl_lstm:.0f}", flush=True)

    # Test scores
    z_raw   = mahal_score(test, L_ic)
    z_white = whitened_score(np.vstack([ic[-1:], test]), Phi, L_e)
    innov_t = lstm_innovations(model, np.vstack([ic[-1:], test]))
    z_lstm  = mahal_score((innov_t - mu_i) / sig_i, np.eye(d))

    rows      = []
    plot_data = {}
    for method, z, arl_cal in [
            ('Raw (M1)',        z_raw,   arl_raw),
            ('Whitened (M2)',   z_white, arl_white),
            ('Pathway A (M3)', z_lstm,  arl_lstm)]:
        alarm, S = page_cusum(z, k=k, h=h)
        delay    = (alarm - tau_star) if alarm <= T1 else None
        delay_str = f'{delay:+d}' if delay is not None else 'NO ALARM'
        print(f"  [Part {part_label}, d={d}] {method:<18}  "
              f"alarm={alarm:>5}  tau*={tau_star}  delay={delay_str}", flush=True)
        rows.append(dict(part=part_label, d=d, method=method,
                         tau_star=tau_star, alarm=alarm,
                         delay=delay, detected=int(alarm <= T1),
                         h=h, arl0=round(arl_cal, 1)))
        plot_data[method] = dict(z=z, S=S, alarm=alarm, h=h)

    return rows, plot_data, part_label, d, tau_star, T1, test_idx

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_cusum(plot_data, part_label, d, tau_star, T1, test_idx, k):
    ticker_str = 'XLF' if part_label == 'A' else 'XLF / XLK / XLE'
    colours    = {'Raw (M1)': 'steelblue',
                  'Whitened (M2)': 'darkorange',
                  'Pathway A (M3)': 'forestgreen'}
    fig, axes  = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t_ax       = np.arange(1, T1 + 1)

    for ax, method in zip(axes, ['Raw (M1)', 'Whitened (M2)', 'Pathway A (M3)']):
        pd_  = plot_data[method]
        S, h, alarm = pd_['S'], pd_['h'], pd_['alarm']
        col  = colours[method]
        n    = min(len(t_ax), len(S))
        ax.plot(t_ax[:n], S[:n], color=col, lw=1.4, label='$S_t$')
        ax.axhline(h, color='red', lw=1.0, ls=':', label=f'$h={h:.2f}$')
        ax.axvline(tau_star, color='black', lw=1.2, ls='--', alpha=0.7,
                   label=f'$\\tau^*={tau_star}$ (COVID peak)')
        if alarm <= T1:
            ax.axvline(alarm, color=col, lw=1.5, ls='-', alpha=0.9,
                       label=f'alarm $t={alarm}$ (delay {alarm-tau_star:+d})')
        else:
            ax.text(0.98, 0.85, 'no alarm before end',
                    transform=ax.transAxes, ha='right', color=col, fontsize=8)
        ax.set_ylabel('CUSUM $S_t$', fontsize=9)
        ax.set_title(method, fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc='upper left')
        ax.set_ylim(bottom=0)

    xticks  = np.arange(0, T1, 125)
    xlabels = [str(test_idx[min(i, T1-1)].year) for i in xticks]
    axes[-1].set_xticks(xticks); axes[-1].set_xticklabels(xlabels)
    axes[-1].set_xlabel('Date (approximate year)', fontsize=9)
    fig.suptitle(
        f'Study S8 Part {part_label} ($d={d}$, {ticker_str}): '
        f'CUSUM trajectories  (known break $\\tau^*={tau_star}$, 2020-02-19)',
        fontsize=10)
    plt.tight_layout()
    fname = os.path.join(FIGURES, f'S8_etf_cusum_part{part_label}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fname)}")

def plot_returns(log_ret, cfg):
    tickers = cfg['tickers']
    colours = ['steelblue', 'darkorange', 'forestgreen']
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, ticker, col in zip(axes, tickers, colours):
        ret_pct = log_ret[ticker].values * 100
        ax.plot(log_ret.index, ret_pct, color=col, lw=0.5, alpha=0.8)
        ax.axvline(pd.Timestamp(cfg['break_date']), color='red', lw=1.5,
                   ls='--', label='COVID peak (2020-02-19)')
        ax.axvspan(pd.Timestamp(cfg['ic_start']), pd.Timestamp(cfg['ic_end']),
                   alpha=0.08, color='gray', label='In-control window')
        ax.set_ylabel(f'{ticker} (%)', fontsize=8)
        ax.legend(fontsize=7, loc='lower left')
    axes[-1].set_xlabel('Date')
    fig.suptitle(
        'Study S8: Daily log-returns of XLF, XLK, XLE  (2017–2022)\n'
        'Grey shading = in-control period (2017–2019);  '
        'red dashed = COVID break (2020-02-19)',
        fontsize=10)
    plt.tight_layout()
    fname = os.path.join(RESULTS, 'S8_etf_returns.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fname)}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wall_start = time.time()
    print("Study S8: US Equity Sector ETF Returns — Pathway A CUSUM")
    print(f"  Hardware: 8-core Xeon W, 2 parallel workers × {CFG['n_threads']} threads\n")

    ic_norm, test_norm, test_idx, tau_star, log_ret, mu_ic, std_ic = load_data(CFG)
    T0, T1 = len(ic_norm), len(test_norm)
    print(f"  In-control: {CFG['ic_start']} to {CFG['ic_end']}  (T0={T0})")
    print(f"  Test:       {CFG['test_start']} to {CFG['test_end']}  (T1={T1})")
    print(f"  tau* = {tau_star}  ({CFG['break_date']}, COVID market peak)")

    plot_returns(log_ret, CFG)

    configs = [('A', [0]), ('B', [0, 1, 2])]
    all_rows  = []
    all_plots = {}

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(run_part, label, d_idx,
                        ic_norm, test_norm, test_idx, tau_star, CFG): label
            for label, d_idx in configs
        }
        for future in as_completed(futures):
            rows, plot_data, part_label, d, tau_s, T1_r, tidx = future.result()
            all_rows.extend(rows)
            all_plots[part_label] = (plot_data, d, tau_s, T1_r, tidx)

    df_out   = pd.DataFrame(all_rows).sort_values(['part', 'method'])
    csv_path = os.path.join(RESULTS, 'S8_etf_detection.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Saved: {os.path.basename(csv_path)}")
    print(df_out.to_string(index=False))

    for part_label, (plot_data, d, tau_s, T1_r, tidx) in sorted(all_plots.items()):
        plot_cusum(plot_data, part_label, d, tau_s, T1_r, tidx, CFG['k_cusum'])

    elapsed = time.time() - wall_start
    print(f"\nStudy S8 complete in {elapsed:.1f}s ({elapsed/60:.1f} min).")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
