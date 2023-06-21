import numpy as np

from typing import Tuple
from collections import Counter, defaultdict
from sklearn.neighbors import KernelDensity


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    eucl_dist_top = defaultdict(list)
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            eucl_dist_top[i].append(dist)
            eucl_dist_top[j].append(dist)
    uniq_estimates = np.zeros(len(embeddings))
    for emb, dists in eucl_dist_top.items():
        d = np.sort(dists)[:num_neighbors]
        print(d)
        uniq_estimates[emb] = np.sum(d) / num_neighbors
    return uniq_estimates


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity(kernel='gaussian').fit(embeddings)
    uniqueness = np.exp(-kde.score_samples(embeddings))
    return uniqueness
