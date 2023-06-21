import numpy as np 

from typing import Tuple
from uniqueness import kde_uniqueness

def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    group_score = kde_uniqueness(embeddings).mean()

    return (group_score < threshold, group_score)