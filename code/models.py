# =============================================================================
#  models.py — LSTM with gate extraction and simplified selective SSM
#
#  InstrumentedLSTM:
#    - One-layer LSTM for scalar sequences
#    - Spectral normalisation on W_hh (enforces Assumption 2.1: ||W_h||_op < 1)
#    - step() returns hidden state, cell state, AND forget gate
#
#  SimpleMamba:
#    - Minimal selective SSM (Mamba-like) with learnable selection parameter Delta_t
#    - Implemented in plain PyTorch (CPU-friendly, no specialised CUDA kernels)
#    - Returns predictions AND Delta_t sequence for Study S1
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import HIDDEN_DIM, MAMBA_D_STATE


# ── Instrumented LSTM ─────────────────────────────────────────────────────────

class InstrumentedLSTMCell(nn.Module):
    """
    Manual LSTM cell that returns the forget gate f_t explicitly.

    Spectral normalisation is applied to W_hh so that ||W_hh||_op <= 1,
    enforcing Assumption 2.1 (operator-norm contractivity).
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Combined input → gates (4 * hidden_size for i, f, g, o)
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)

        # Recurrent weight matrix with spectral normalisation
        W_hh_raw = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.W_hh = nn.utils.spectral_norm(W_hh_raw, name='weight')

    def forward(self, x: torch.Tensor, hidden):
        """
        Parameters
        ----------
        x      : (batch, input_size)
        hidden : tuple (h, c) each of shape (batch, hidden_size)

        Returns
        -------
        h_new : (batch, hidden_size)
        c_new : (batch, hidden_size)
        f     : (batch, hidden_size)  — forget gate
        """
        h, c = hidden
        gates = self.W_ih(x) + self.W_hh(h)
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_raw)
        f = torch.sigmoid(f_raw)
        g = torch.tanh(g_raw)
        o = torch.sigmoid(o_raw)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, f


class InstrumentedLSTM(nn.Module):
    """
    Full sequence LSTM for scalar (univariate) one-step-ahead prediction.

    Input : (batch, T, 1)
    Output: predictions (batch, T, 1),
            forget gates (batch, T, hidden_size),
            hidden states (batch, T, hidden_size),
            cell states   (batch, T, hidden_size)
    """

    def __init__(self, hidden_size: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = InstrumentedLSTMCell(1, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, T, 1)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)

        preds, fgs, hs, cs = [], [], [], []
        for t in range(T):
            h, c, f = self.cell(x[:, t, :], (h, c))
            preds.append(self.readout(h))
            fgs.append(f)
            hs.append(h)
            cs.append(c)

        preds = torch.stack(preds, dim=1)  # (B, T, 1)
        fgs   = torch.stack(fgs,   dim=1)  # (B, T, hidden_size)
        hs    = torch.stack(hs,    dim=1)  # (B, T, hidden_size)
        cs    = torch.stack(cs,    dim=1)  # (B, T, hidden_size)
        return preds, fgs, hs, cs

    @torch.no_grad()
    def predict_sequence(self, X: np.ndarray):
        """
        Convenience wrapper: numpy in, numpy out.

        X : (T,) univariate sequence

        Returns
        -------
        preds : (T,) — one-step-ahead predictions
        fgs   : (T, hidden_size) — forget gates
        hs    : (T, hidden_size) — hidden states
        cs    : (T, hidden_size) — cell states
        """
        x = torch.FloatTensor(X).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        preds, fgs, hs, cs = self.forward(x)
        return (preds.squeeze().numpy(),
                fgs.squeeze().numpy(),
                hs.squeeze().numpy(),
                cs.squeeze().numpy())


# ── Simplified Selective SSM (Mamba proxy) ───────────────────────────────────

class SimpleMamba(nn.Module):
    """
    Minimal selective state-space model with learnable Delta_t.

    Architecture:
      - State update:  h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
      - Discretisation: A_bar_t = exp(-exp(Delta_t) * dt)
      - Selection:      Delta_t = softplus(W_delta * x_t + b_delta)
      - Output:         y_t = C * h_t

    Delta_t is an input-dependent (selective) timescale analogous to the
    Mamba selection parameter.  Under the theory (Conjecture 1 / Prop 3.4),
    Delta_t should correlate with the marginal information gain I_t.

    This CPU-compatible implementation avoids the specialised CUDA kernels
    of the original Mamba library while preserving the key selectivity idea.
    """

    def __init__(self, d_state: int = MAMBA_D_STATE):
        super().__init__()
        self.d_state = d_state

        # Fixed log-A (learnable bias for each state dimension)
        self.log_A = nn.Parameter(torch.ones(d_state) * np.log(0.5))

        # Input projections
        self.W_B     = nn.Linear(1, d_state)
        self.W_delta = nn.Linear(1, 1)          # scalar Delta_t for simplicity
        self.W_C     = nn.Linear(d_state, 1)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, T, 1)

        Returns
        -------
        preds  : (batch, T, 1)   — one-step-ahead predictions
        deltas : (batch, T, 1)   — selection parameter Delta_t
        hs     : (batch, T, d_state) — hidden states
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.d_state, device=x.device)

        preds, deltas, hs = [], [], []
        for t in range(T):
            x_t = x[:, t, :]                                   # (B, 1)

            # Selection parameter
            delta = F.softplus(self.W_delta(x_t))              # (B, 1)
            deltas.append(delta)

            # Discretise: A_bar = exp(-exp(log_A) * delta)
            A_bar = torch.exp(
                -torch.exp(self.log_A).unsqueeze(0) * delta    # (B, d_state)
            )
            B_bar = self.W_B(x_t)                              # (B, d_state)

            # State update
            h = A_bar * h + B_bar * x_t                        # (B, d_state)
            hs.append(h)

            # Output
            y = self.W_C(h)                                    # (B, 1)
            preds.append(y)

        preds  = torch.stack(preds,  dim=1)   # (B, T, 1)
        deltas = torch.stack(deltas, dim=1)   # (B, T, 1)
        hs     = torch.stack(hs,     dim=1)   # (B, T, d_state)
        return preds, deltas, hs

    @torch.no_grad()
    def predict_sequence(self, X: np.ndarray):
        """Numpy convenience wrapper."""
        x = torch.FloatTensor(X).unsqueeze(0).unsqueeze(-1)
        preds, deltas, hs = self.forward(x)
        return (preds.squeeze().numpy(),
                deltas.squeeze().numpy(),
                hs.squeeze().numpy())
