"""
metrics.py
-----------
Embedding-quality metrics: temporal stability, redundancy, condition number, Fisher information, etc.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

def embedding_quality(X, y):
    acs = [np.corrcoef(X[:-1,j], X[1:,j])[0,1] for j in range(X.shape[1])]
    temporal_stability = np.nanmean(acs)
    cov = np.cov(X.T)
    cond_num = np.linalg.cond(cov)
    off_diag_energy = np.sum(cov**2) - np.sum(np.diag(cov)**2)
    redundancy_ratio = off_diag_energy / np.sum(cov**2)
    f_vals, p_vals = f_classif(X, y)
    fisher_mean, mean_p = np.mean(f_vals), np.mean(p_vals)
    pca = PCA(n_components=min(2, X.shape[1]))
    pca.fit(StandardScaler().fit_transform(X))
    compactness = np.sum(pca.explained_variance_ratio_)
    return temporal_stability, redundancy_ratio, cond_num, fisher_mean, mean_p, compactness
