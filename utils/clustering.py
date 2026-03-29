"""Clustering utilities."""

import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(X: torch.Tensor, tau: float) -> torch.Tensor:
    """Partition patches into clusters using cosine-distance agglomerative clustering.

    Args:
        X: (N, C) L2-normalized patch features.
        tau: similarity threshold (clusters are split below 1 - tau distance).

    Returns:
        (N,) integer cluster labels.
    """
    S = (X @ X.T).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    ac = AgglomerativeClustering(
        n_clusters=None, metric='precomputed',
        linkage='average', distance_threshold=float(1.0 - tau),
    )
    labels = ac.fit_predict(D)
    return torch.from_numpy(labels).long().to(X.device)


def compute_cluster_prototypes(X: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Compute an L2-normalized prototype for each cluster.

    Args:
        X: (N, C) patch features.
        labels: (N,) integer cluster assignments.
        K: number of clusters.

    Returns:
        (K, C) L2-normalized prototypes.
    """
    protos = []
    for k in range(K):
        idx = (labels == k)
        mu = X[idx].mean(dim=0) if idx.any() else torch.randn_like(X[0])
        protos.append(F.normalize(mu, p=2, dim=0).unsqueeze(0))
    return torch.cat(protos, dim=0)
