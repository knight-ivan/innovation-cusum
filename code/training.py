# =============================================================================
#  training.py — Training and evaluation utilities
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from config import N_EPOCHS, BATCH_SIZE, LR


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(model: nn.Module,
                X_seqs: np.ndarray,
                n_epochs: int = N_EPOCHS,
                batch_size: int = BATCH_SIZE,
                lr: float = LR,
                verbose: bool = False) -> nn.Module:
    """
    Train a sequence model (InstrumentedLSTM or SimpleMamba) to minimise
    one-step-ahead MSE.

    Parameters
    ----------
    model   : nn.Module whose forward(x) returns (preds, *extras)
              where preds has shape (batch, T, 1)
    X_seqs  : (N, T) array of scalar sequences
    n_epochs, batch_size, lr : training hyperparameters

    Returns
    -------
    model : trained in-place, also returned for chaining
    """
    N, T = X_seqs.shape
    # Input: X_{1:T-1},  Target: X_{2:T}
    X_in  = torch.FloatTensor(X_seqs[:, :-1, None])   # (N, T-1, 1)
    X_out = torch.FloatTensor(X_seqs[:, 1:,  None])   # (N, T-1, 1)

    dataset = TensorDataset(X_in, X_out)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.1)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            out = model(x_batch)
            preds = out[0]                         # first element is predictions
            loss  = F.mse_loss(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(x_batch)
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            avg = epoch_loss / N
            print(f"  epoch {epoch+1:3d}/{n_epochs}  train MSE = {avg:.5f}")

    model.eval()
    return model


# ── Evaluation utilities ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_residuals(model: nn.Module,
                      X_seqs: np.ndarray) -> np.ndarray:
    """
    Compute one-step-ahead prediction residuals r_t = X_{t+1} - ŷ_t.

    Parameters
    ----------
    X_seqs : (N, T)

    Returns
    -------
    residuals : (N, T-1)
    """
    N, T = X_seqs.shape
    X_in  = torch.FloatTensor(X_seqs[:, :-1, None])
    X_out = X_seqs[:, 1:]                             # (N, T-1), numpy

    model.eval()
    out   = model(X_in)
    preds = out[0].squeeze(-1).numpy()                # (N, T-1)
    return X_out - preds


@torch.no_grad()
def extract_forget_gates(model, X_seqs: np.ndarray) -> np.ndarray:
    """
    Extract per-step forget gates from InstrumentedLSTM.

    Parameters
    ----------
    X_seqs : (N, T)

    Returns
    -------
    fgs : (N, T, hidden_size)
    """
    x = torch.FloatTensor(X_seqs[:, :, None])   # (N, T, 1)
    model.eval()
    _, fgs, _, _ = model(x)
    return fgs.numpy()


@torch.no_grad()
def extract_deltas(model, X_seqs: np.ndarray) -> np.ndarray:
    """
    Extract per-step Delta_t from SimpleMamba.

    Returns
    -------
    deltas : (N, T) — scalar Delta_t per step
    """
    x = torch.FloatTensor(X_seqs[:, :, None])   # (N, T, 1)
    model.eval()
    _, deltas, _ = model(x)
    return deltas.squeeze(-1).numpy()            # (N, T)


@torch.no_grad()
def extract_cell_states(model, X_seqs: np.ndarray) -> np.ndarray:
    """
    Extract LSTM cell states c_t.

    Returns
    -------
    cs : (N, T, hidden_size)
    """
    x = torch.FloatTensor(X_seqs[:, :, None])
    model.eval()
    _, _, _, cs = model(x)
    return cs.numpy()


def compute_cell_It(cs: np.ndarray,
                    c_inf: np.ndarray = None) -> np.ndarray:
    """
    Compute the cell-state proxy for I_t:
      I_t^(cell) = || c_t - c_inf || / || c_0 - c_inf ||

    normalised so that I_0^(cell) = 1 for each sequence.

    Parameters
    ----------
    cs    : (N, T, d)  — cell states
    c_inf : (d,) or (N, d)  — stationary cell state; if None, use
            the mean of the last 20% of time steps as proxy

    Returns
    -------
    It_cell : (N, T)
    """
    if c_inf is None:
        T = cs.shape[1]
        tail = max(1, int(0.2 * T))
        c_inf = cs[:, -tail:, :].mean(axis=1, keepdims=True)  # (N, 1, d)
    else:
        c_inf = np.array(c_inf)
        if c_inf.ndim == 1:
            c_inf = c_inf[None, None, :]   # (1, 1, d)

    diff  = cs - c_inf                                          # (N, T, d)
    norms = np.linalg.norm(diff, axis=-1)                       # (N, T)
    # Normalise by I_0^(cell) to get relative decay
    norm0 = norms[:, 0:1]
    norm0 = np.where(norm0 < 1e-10, 1.0, norm0)
    return norms / norm0
