
"""
Chrono-Tensorial Flux Calculus (CTFC)
=====================================

This package implements the Chrono-Tensorial Flux Calculus framework introduced in:
    "Chrono-Tensorial Flux Calculus for Dimension Reduction of Spatiotemporal Data"
    (Dhruvil, 2025)

CTFC provides a tensorial formulation for representing spatiotemporal flux,
curvature, and memory dynamics in high-dimensional systems. It generalizes
classical covariance-based models by embedding chrono-differential operators
into tensor contractions.

Modules
-------
operators.py
    Defines chrono-operators: chrono-derivative, chrono-trace, flux contraction,
    and Laplacian curvature.
embedding.py
    Implements the discrete-time realization (CTFC≈) for time-series embedding.
discrete_map.py
    Continuous-to-discrete operator approximations and symbolic mappings.
stability.py
    Stability and boundedness diagnostics for chrono-flux systems.
metrics.py
    Embedding-quality metrics including temporal stability, redundancy, Fisher score,
    and compactness.
experiment.py
    End-to-end experimental pipeline comparing CTFC≈, PCA, and GARCH embeddings.

Usage
-----
>>> from ctfc.embedding import ctfc_embedding
>>> from ctfc.experiment import run_experiment
>>> results, quality = run_experiment()

Author
------
Dhruvil (2025)
"""

__version__ = "1.0.0"
__author__ = "Dhruvil"

# Public API
__all__ = [
    "operators",
    "embedding",
    "discrete_map",
    "stability",
    "metrics",
    "experiment",
]
