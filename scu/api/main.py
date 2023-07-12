import os
from typing import Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity

DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


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
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity(kernel="gaussian").fit(embeddings)
    uniqueness = np.exp(-kde.score_samples(embeddings))
    return uniqueness


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


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    item_embeddings = kde_uniqueness(
        np.array([embeddings[item_id] for item_id in item_ids])
    )
    for i, item_id in enumerate(item_ids):
        item_uniqueness[item_id] = item_embeddings[i]
    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    answer = {"diversity": 0.0, "reject": True}

    group_emb = np.array([embeddings[item_id] for item_id in item_ids])
    diversity, reject = group_diversity(group_emb, DIVERSITY_THRESHOLD)
    answer["diversity"] = float(diversity)
    answer["reject"] = bool(reject)
    return answer


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000)
