"""Module 3: drone hub placement (idea 3).

Candidate hub sites are the hospitals themselves. For every candidate we know the
drone delivery time to every hospital (`time_matrix`). Two ways to choose k hubs:

* `best_hubs_exhaustive`  - tries every combination (p-median, optimal; fine for
  ~12-25 sites, the search space is C(n, k)).
* `kmeans_hubs`           - k-means on coordinates, snapped to the nearest hospital.
  A cheap heuristic used as a baseline to compare against the optimal answer.
"""
import itertools
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config
from src.geo import drone_path_km, to_xy_km
from src.simulator import drone_time_min


def time_matrix(hospitals):
    """DataFrame [hub candidate x hospital] of drone delivery minutes.

    Diagonal is 0 (hub serves its own hospital). Pairs beyond drone range or
    inside a no-fly zone are math.inf.
    """
    names = hospitals["name"].tolist()
    mat = pd.DataFrame(math.inf, index=names, columns=names, dtype=float)
    for a in hospitals.itertuples():
        for b in hospitals.itertuples():
            if a.name == b.name:
                mat.loc[a.name, b.name] = 0.0
                continue
            flight = drone_path_km(a.lat, a.lon, b.lat, b.lon,
                                   config.NO_FLY_ZONES, config.REF_LAT, config.REF_LON)
            if flight <= config.DRONE_MAX_RANGE_KM:
                mat.loc[a.name, b.name] = drone_time_min(flight)
    return mat


def evaluate(hub_names, matrix, threshold=None):
    """Score a set of hubs. Each hospital is served by its fastest hub."""
    threshold = config.COVERAGE_TARGET_MIN if threshold is None else threshold
    hub_names = list(hub_names)
    if not hub_names:
        raise ValueError("need at least one hub")
    unknown = [h for h in hub_names if h not in matrix.index]
    if unknown:
        raise ValueError(f"unknown hubs: {unknown}")
    sub = matrix.loc[hub_names]
    best_time = sub.min(axis=0)
    assignment = sub.idxmin(axis=0).where(np.isfinite(best_time))
    reachable = np.isfinite(best_time)
    cost_series = best_time.where(reachable, config.UNCOVERED_PENALTY_MIN)
    return {
        "hubs": hub_names,
        "avg_min": float(cost_series.mean()),
        "max_min": float(best_time[reachable].max()) if reachable.any() else math.inf,
        "pct_within_target": float((best_time <= threshold).mean() * 100.0),
        "uncovered": best_time.index[~reachable].tolist(),
        "assignment": assignment.to_dict(),
        "best_time": best_time.to_dict(),
    }


def best_hubs_exhaustive(matrix, k, threshold=None):
    """Optimal k hubs minimising average delivery time (uncovered are penalised)."""
    n = len(matrix.index)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    best = None
    for combo in itertools.combinations(sorted(matrix.index), k):
        res = evaluate(combo, matrix, threshold)
        key = (res["avg_min"], list(combo))
        if best is None or key < best[0]:
            best = (key, res)
    return best[1]


def kmeans_hubs(hospitals, matrix, k, seed=0, threshold=None):
    """k-means on projected coordinates, each centroid snapped to a distinct hospital."""
    n = len(hospitals)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    pts = np.array([to_xy_km(r.lat, r.lon, config.REF_LAT, config.REF_LON)
                    for r in hospitals.itertuples()])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pts)
    names = hospitals["name"].tolist()
    chosen = []
    for c in km.cluster_centers_:
        order = np.argsort(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]))
        for idx in order:
            if names[idx] not in chosen:
                chosen.append(names[idx])
                break
    return evaluate(chosen, matrix, threshold)


def coverage_curve(hospitals, matrix, k_max=None, threshold=None):
    """Optimal and k-means results for k = 1..k_max, as a comparison table."""
    k_max = min(k_max or 6, len(hospitals))
    rows = []
    for k in range(1, k_max + 1):
        opt = best_hubs_exhaustive(matrix, k, threshold)
        km = kmeans_hubs(hospitals, matrix, k, threshold=threshold)
        rows.append({
            "k": k,
            "optimal_avg_min": round(opt["avg_min"], 2),
            "optimal_pct_within_target": round(opt["pct_within_target"], 1),
            "kmeans_avg_min": round(km["avg_min"], 2),
            "kmeans_pct_within_target": round(km["pct_within_target"], 1),
            "optimal_hubs": ", ".join(opt["hubs"]),
        })
    return pd.DataFrame(rows)
