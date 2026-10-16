import math

import pandas as pd
import pytest

import config
from src.hubs import (best_hubs_exhaustive, coverage_curve, evaluate,
                      kmeans_hubs, time_matrix)
from src.simulator import load_hospitals


@pytest.fixture(scope="module")
def hospitals():
    return load_hospitals()


@pytest.fixture(scope="module")
def matrix(hospitals):
    return time_matrix(hospitals)


def synthetic():
    # 4 sites on a line A-B ... C-D; B-C far apart (inf)
    inf = math.inf
    names = list("ABCD")
    data = [[0, 5, inf, inf],
            [5, 0, inf, inf],
            [inf, inf, 0, 6],
            [inf, inf, 6, 0]]
    return pd.DataFrame(data, index=names, columns=names, dtype=float)


def test_matrix_diagonal_zero_and_symmetric(matrix):
    assert (matrix.values.diagonal() == 0).all()
    finite = matrix.replace(math.inf, float("nan"))
    assert finite.equals(finite.T)


def test_all_hubs_gives_zero_cost(matrix):
    res = evaluate(matrix.index, matrix)
    assert res["avg_min"] == 0
    assert res["pct_within_target"] == 100
    assert res["uncovered"] == []


def test_synthetic_best_pair_is_one_per_cluster():
    res = best_hubs_exhaustive(synthetic(), 2)
    assert len(set(res["hubs"]) & {"A", "B"}) == 1
    assert len(set(res["hubs"]) & {"C", "D"}) == 1
    assert res["uncovered"] == []


def test_synthetic_one_hub_leaves_other_cluster_uncovered():
    res = evaluate(["A"], synthetic())
    assert set(res["uncovered"]) == {"C", "D"}
    # penalty is charged in the average
    assert res["avg_min"] > config.UNCOVERED_PENALTY_MIN / 3


def test_optimal_cost_never_increases_with_k(matrix):
    costs = [best_hubs_exhaustive(matrix, k)["avg_min"] for k in range(1, 5)]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))


def test_optimal_beats_or_ties_kmeans(hospitals, matrix):
    for k in (1, 2, 3, 4):
        opt = best_hubs_exhaustive(matrix, k)["avg_min"]
        km = kmeans_hubs(hospitals, matrix, k)["avg_min"]
        assert opt <= km + 1e-9


def test_kmeans_returns_k_distinct_hubs(hospitals, matrix):
    res = kmeans_hubs(hospitals, matrix, 4)
    assert len(res["hubs"]) == 4 == len(set(res["hubs"]))


def test_invalid_k_and_unknown_hub(hospitals, matrix):
    with pytest.raises(ValueError):
        best_hubs_exhaustive(matrix, 0)
    with pytest.raises(ValueError):
        best_hubs_exhaustive(matrix, len(matrix) + 1)
    with pytest.raises(ValueError):
        kmeans_hubs(hospitals, matrix, 0)
    with pytest.raises(ValueError):
        evaluate(["Nowhere"], matrix)
    with pytest.raises(ValueError):
        evaluate([], matrix)


def test_coverage_curve_shape(hospitals, matrix):
    df = coverage_curve(hospitals, matrix, k_max=3)
    assert list(df["k"]) == [1, 2, 3]
    assert (df["optimal_avg_min"] <= df["kmeans_avg_min"] + 1e-9).all()
