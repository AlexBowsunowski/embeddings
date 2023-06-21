from typing import Tuple

import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity

DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {
    1: np.array([-26.57, -76.61, 81.61, -9.11, 74.8, 54.23, 32.56, -22.62, -72.44, -82.78]),
    2: np.array([-55.98, 82.87, 86.07, 18.71, -18.66, -46.74, -68.18, 60.29, 98.92, -78.95]),
    3: np.array([-27.97, 25.39, -96.85, 3.51, 95.57, -27.48, -80.27, 8.39, 89.96, -36.68]),
    4: np.array([-37.0, -49.39, 43.3, 73.36, 29.98, -56.44, -15.91, -56.46, 24.54, 12.43]),
    5: np.array([-22.71, 4.47, -65.42, 10.11, 98.34, 17.96, -10.77, 2.5, -26.55, 69.16]),
}


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
    kde = KernelDensity(kernel='gaussian').fit(embeddings)
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
        np.array(
            [embeddings[item_id] for item_id in item_ids]
        )
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
    print(diversity, reject)
    answer["diversity"] = float(diversity)
    answer["reject"] = bool(reject)
    print(answer)
    return answer


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000)
