import numpy as np

from typing import Dict, Tuple


@staticmethod
def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
    """Calculate pairwise similarities between each item
    in embedding.

    Args:
        embeddings (Dict[int, np.ndarray]): Items embeddings.

    Returns:
        Tuple[List[str], Dict[Tuple[int, int], float]]:
        List of all items + Pairwise similarities dict
        Keys are in form of (i, j) - combinations pairs of item_ids
        with i < j.
        Round each value to 8 decimal places.
    """
    return pair_sims