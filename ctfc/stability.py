"""
stability.py
-------------
Stability and boundedness diagnostics for CTFC embeddings.
"""
import numpy as np

def spectral_radius(matrix):
    """Compute the spectral radius ρ(A) = max(|λ_i|)."""
    eigs = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigs))

def is_stable(matrix, threshold=1.0):
    """Check if system is stable: spectral radius < threshold."""
    return spectral_radius(matrix) < threshold

