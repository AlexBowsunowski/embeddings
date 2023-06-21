import numpy as np

from typing import Dict, Tuple, List
from scipy import spatial
from collections import defaultdict


class SimilarItems:

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
        pair_sims = {}
        items = list(embeddings.items())
        for i, (k, emb1) in enumerate(items):
            for j, emb2 in items[i + 1:]:
                pair_sims[(k, j)] = round(1 - spatial.distance.cosine(emb1, emb2), 8)
        return pair_sims
    
    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = defaultdict(list)
        sim_sorted = sorted(sim.items(), key=lambda kv: kv[1], reverse=True)
        for pair, score in sim_sorted:
            if len(knn_dict[pair[0]]) < top:
                knn_dict[pair[0]].append((pair[1], score))
            if len(knn_dict[pair[1]]) < top:
                knn_dict[pair[1]].append((pair[0], score))
        return knn_dict
    
    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for i, top_n in knn_dict.items():
            weights = np.array([cos_dist + 1 for _, cos_dist in top_n])
            weights_norm = weights / np.sum(weights)
            price_top_n = np.array([prices[k]  for k, _ in top_n])
            price = np.sum(price_top_n * weights_norm)
            knn_price_dict[i] = round(price, 2)

        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        similiraty_scores = SimilarItems.similarity(embeddings)
        top_n_neigbours = SimilarItems.knn(similiraty_scores, top)
        knn_price_dict = SimilarItems.knn_price(top_n_neigbours, prices)

        return knn_price_dict