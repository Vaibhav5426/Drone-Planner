import math

import pytest

import config
from src.simulator import (compare_from_depot, compare_pair, drone_time_min,
                           hourly_profile, load_hospitals, road_time_min)


@pytest.fixture(scope="module")
def hospitals():
    return load_hospitals()


def test_dataset_shape_and_bounds(hospitals):
    assert len(hospitals) == 12
    assert hospitals["name"].is_unique
    assert hospitals["lat"].between(28.3, 28.9).all()
    assert hospitals["lon"].between(76.9, 77.5).all()


def test_road_slower_at_peak_than_night():
    assert road_time_min(20, 18) > road_time_min(20, 3)


def test_road_time_rejects_bad_hour():
    with pytest.raises(ValueError):
        road_time_min(10, 24)


def test_drone_time_hand_calc():
    expected = config.DRONE_OVERHEAD_MIN + 30 / config.DRONE_CRUISE_KMH * 60
    assert abs(drone_time_min(30) - expected) < 1e-9
    assert math.isinf(drone_time_min(math.inf))


def test_drone_beats_road_at_peak_for_all_pairs(hospitals):
    for depot in hospitals["name"]:
        df = compare_from_depot(depot, hospitals, hour=18)
        ok = df[df["in_range"]]
        assert (ok["drone_min"] < ok["road_min"]).all(), depot


def test_compare_from_depot_excludes_self_and_counts(hospitals):
    df = compare_from_depot("AIIMS New Delhi", hospitals, 9)
    assert len(df) == 11
    assert "AIIMS New Delhi" not in set(df["destination"])


def test_unknown_depot_raises(hospitals):
    with pytest.raises(ValueError):
        compare_from_depot("Nowhere Clinic", hospitals, 9)


def test_drone_time_hour_independent(hospitals):
    a, b = hospitals.iloc[0], hospitals.iloc[5]
    assert compare_pair(a, b, 3)["drone_min"] == compare_pair(a, b, 18)["drone_min"]


def test_hourly_profile_has_24_rows(hospitals):
    df = hourly_profile(hospitals.iloc[0], hospitals.iloc[10])
    assert list(df["hour"]) == list(range(24))
