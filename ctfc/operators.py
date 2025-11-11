
"""
operators.py
------------
Chrono-operator definitions for the Chrono-Tensorial Flux Calculus (CTFC) framework.
Implements chrono-derivative, chrono-trace, flux contraction, and Laplacian curvature operators.
"""

import numpy as np

def chrono_derivative(C_t, C_prev, psi):
    """Chrono-derivative operator Φ_cd = ||C_t - C_prev||_F * (1 + ψ)."""
    dC = C_t - C_prev
    return np.linalg.norm(dC, 'fro') * (1 + psi)

def chrono_trace(C_t, psi, r=1):
    """Flux-dressed trace operator Tr(C_t) * (1 + (r/(r+2))ψ²)."""
    trC = np.trace(C_t)
    return trC * (1 + (r/(r+2)) * psi**2)

def chrono_contraction(C_t, psi, r=1):
    """Flux-dressed contraction operator Tr(C_t) * (1 + (r/(r+1))ψ²)."""
    trC = np.trace(C_t)
    return trC * (1 + (r/(r+1)) * psi**2)

def laplacian_curvature(C_t, corr, I_t, r=1):
    """Laplacian curvature R_t = Tr(C_t L_t) * (1 + (r/(r+1))I_t)."""
    n = C_t.shape[0]
    np.fill_diagonal(corr, 1.0)
    W = np.exp(-2 * (1 - corr))
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return float(np.trace(C_t @ L) * (1 + (r/(r+1)) * I_t))
