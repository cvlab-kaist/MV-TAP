"""Camera-view sampling utilities for the multi-view DexYCB dataset.

The raw DexYCB-multiview capture provides 8 synchronized camera views per
sequence.  For the MV-TAP evaluation protocol we sub-sample ``K`` of those
views with one of three strategies described in the paper:

1. ``fps``     -- Farthest Point Sampling on the camera centers.  Picks views
                  that are as far apart as possible (maximally spread baseline).
2. ``nearest`` -- Picks the ``K`` views that form the most compact cluster
                  (smallest pairwise camera distances).
3. ``random``  -- Uniformly random subset (reproducible via ``seed``).

Camera centers are derived from the world->camera ``extrinsics`` matrix
``E = [R | t]`` as ``C = -R^T t`` (the standard pinhole convention used by the
stored ``intrinsics_extrinsics.npz`` files).

The functions here only decide *which* views to keep; the actual on-disk
dataset materialization lives in ``scripts/sample_dexycb_views.py``.
"""

import numpy as np


def camera_center_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Return the camera center in world coordinates.

    Args:
        extrinsics: ``(4, 4)`` or ``(3, 4)`` world->camera matrix ``[R | t]``.

    Returns:
        ``(3,)`` camera center ``C = -R^T t``.
    """
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return -R.T @ t


def camera_centers(extrinsics_list) -> np.ndarray:
    """Stack camera centers for a list/array of extrinsics into ``(V, 3)``."""
    return np.stack([camera_center_from_extrinsics(e) for e in extrinsics_list], axis=0)


def sample_fps(centers: np.ndarray, k: int) -> list:
    """Farthest Point Sampling on camera centers.

    The first seed is chosen deterministically as the camera farthest from the
    centroid, so the result depends only on the geometry (no RNG).

    Returns the selected view indices in selection order.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    k = min(k, n)
    if k <= 0:
        return []

    centroid = centers.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(centers - centroid, axis=1)))

    selected = [start]
    # min distance from every point to the currently selected set
    min_dist = np.linalg.norm(centers - centers[start], axis=1)
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(centers - centers[nxt], axis=1))
    return selected


def sample_nearest(centers: np.ndarray, k: int) -> list:
    """Select the ``K`` views forming the most compact cluster.

    For every candidate anchor we take it together with its ``K-1`` nearest
    neighbors and score the cluster by the sum of within-cluster pairwise
    distances; the anchor yielding the smallest score wins.  Deterministic.

    Returns the selected view indices sorted ascending.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = len(centers)
    k = min(k, n)
    if k <= 0:
        return []

    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)  # (V, V)

    best_set, best_score = None, np.inf
    for anchor in range(n):
        # anchor + its (k-1) closest neighbors
        order = np.argsort(dist[anchor])  # includes anchor itself at distance 0
        cluster = order[:k]
        sub = dist[np.ix_(cluster, cluster)]
        score = sub.sum()  # total intra-cluster spread (symmetric, double-counted -> fine for argmin)
        if score < best_score:
            best_score, best_set = score, cluster
    return sorted(int(i) for i in best_set)


def sample_random(centers: np.ndarray, k: int, seed: int = 0) -> list:
    """Uniformly random subset of ``K`` view indices (reproducible)."""
    n = len(centers)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def sample_views(method: str, centers: np.ndarray, k: int, seed: int = 0) -> list:
    """Dispatch to a sampling strategy and return selected view indices.

    The returned indices are always sorted ascending so that the on-disk view
    ordering (``view_00``, ``view_01``, ...) stays aligned with the order in
    which the view-axis arrays (``tracks_2d`` etc.) are sliced.
    """
    method = method.lower()
    if method == "fps":
        return sorted(sample_fps(centers, k))
    if method == "nearest":
        return sample_nearest(centers, k)
    if method == "random":
        return sample_random(centers, k, seed=seed)
    raise ValueError(f"unknown sampling method: {method!r} (expected fps/nearest/random)")
