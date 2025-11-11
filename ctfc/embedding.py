"""
embedding.py
-------------
Implements the discrete-time CTFC≈ embedding algorithm using chrono-operators.
"""

import numpy as np
from .operators import chrono_derivative, chrono_trace, chrono_contraction, laplacian_curvature

def ctfc_embedding(R_arr, window=20, r=1, kappa=0.9, k_eig=1):
    T, n = R_arr.shape
    feats = []
    I_prev = 0.0
    C_prev = np.cov(R_arr[:window].T)

    for t in range(window, T):
        W = R_arr[t-window:t, :]
        C_t = np.cov(W.T)
        x_t = W[-1, :]
        mean_W = W.mean(axis=0)
        psi_vec = x_t - mean_W
        psi = float(np.linalg.norm(psi_vec))

        I_t = kappa * I_prev + psi
        phi_cd = chrono_derivative(C_t, C_prev, psi)
        trC = chrono_trace(C_t, psi, r)
        contract = chrono_contraction(C_t, psi, r)

        corr = np.corrcoef(W.T)
        R_t = laplacian_curvature(C_t, corr, I_t, r)

        eigs = np.sort(np.real(np.linalg.eigvalsh(C_t)))[::-1]
        lam = eigs[:k_eig] * (1 + psi)

        feats.append([I_t, psi, trC, contract, phi_cd, R_t] + lam.tolist())
        I_prev, C_prev = I_t, C_t

    return np.array(feats, dtype=np.float64)

