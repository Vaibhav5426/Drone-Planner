import math

import pytest

import config
from src.battery import (can_deliver, cruise_power_w, ground_speeds, make_training_data,
                         max_range_km, predict_range, sortie_energy_wh, train_range_model,
                         usable_energy_wh)


def test_no_wind_ground_speed_equals_airspeed():
    out, back = ground_speeds(0, 0)
    assert out == back == config.DRONE_CRUISE_KMH


def test_headwind_and_tailwind():
    out, back = ground_speeds(20, 0)       # pure headwind outbound
    assert out == pytest.approx(config.DRONE_CRUISE_KMH - 20)
    assert back == pytest.approx(config.DRONE_CRUISE_KMH + 20)
    out2, back2 = ground_speeds(20, 180)   # pure tailwind outbound
    assert out2 == pytest.approx(config.DRONE_CRUISE_KMH + 20)
    assert back2 == pytest.approx(config.DRONE_CRUISE_KMH - 20)


def test_crosswind_slows_track_speed():
    out, back = ground_speeds(30, 90)
    expected = math.sqrt(config.DRONE_CRUISE_KMH ** 2 - 30 ** 2)
    assert out == pytest.approx(expected) == pytest.approx(back)
    assert out < config.DRONE_CRUISE_KMH


def test_impossible_crosswind_returns_zero():
    assert ground_speeds(config.DRONE_CRUISE_KMH + 5, 90) == (0.0, 0.0)


def test_calibrated_to_synopsis_about_40km():
    r = max_range_km(config.DRONE_MAX_PAYLOAD_KG)
    assert 38 <= r <= 43


def test_range_monotonic_in_payload_and_wind():
    ranges = [max_range_km(p) for p in (0, 1, 2, 3, 3.5)]
    assert all(a > b for a, b in zip(ranges, ranges[1:]))
    assert max_range_km(2, wind_kmh=30, wind_angle_deg=0) < max_range_km(2)


def test_limits_cancel_flight():
    assert max_range_km(config.DRONE_MAX_PAYLOAD_KG + 0.1) == 0
    assert max_range_km(1, wind_kmh=config.MAX_OPERATING_WIND_KMH + 1) == 0


def test_cold_old_and_low_battery_reduce_range():
    base = max_range_km(2)
    assert max_range_km(2, temp_c=-5) < base
    assert max_range_km(2, soh=0.7) < base
    assert max_range_km(2, soc=0.5) < base
    assert max_range_km(2, soc=0.2) == 0


def test_usable_energy_full_battery():
    assert usable_energy_wh() == pytest.approx(config.BATTERY_CAPACITY_WH * 0.8)
    with pytest.raises(ValueError):
        usable_energy_wh(soc=1.2)


def test_sortie_energy_consistent_with_max_range():
    r = max_range_km(2.0)
    assert sortie_energy_wh(r, 2.0) == pytest.approx(usable_energy_wh())
    assert can_deliver(r - 0.01, 2.0) and not can_deliver(r + 0.5, 2.0)


def test_validation_errors():
    with pytest.raises(ValueError):
        cruise_power_w(-1)
    with pytest.raises(ValueError):
        ground_speeds(-1, 0)
    with pytest.raises(ValueError):
        sortie_energy_wh(-1, 1)


def test_training_data_deterministic_and_valid():
    a, b = make_training_data(200, seed=3), make_training_data(200, seed=3)
    assert a.equals(b)
    assert (a["range_km"] >= 0).all()


@pytest.fixture(scope="module")
def trained():
    return train_range_model(n=3000, seed=1)


def test_model_accuracy_on_held_out_data(trained):
    _model, m = trained
    assert m["r2"] > 0.9
    assert m["mae_km"] < 3.0
    assert m["n_test"] == 750


def test_model_prediction_sane(trained):
    model, _ = trained
    truth = max_range_km(1.0, 10, 45, 0.9, 0.95, 18)
    pred = predict_range(model, 1.0, 10, 45, 0.9, 0.95, 18)
    assert abs(pred - truth) < 4


def test_model_accurate_at_full_fresh_battery_corner(trained):
    """Regression: the first model underpredicted by ~3.5 km at soc=1, soh=1."""
    import numpy as np
    model, _ = trained
    errs = []
    for payload in np.linspace(0, config.DRONE_MAX_PAYLOAD_KG, 8):
        for wind in (0, 10, 20):
            truth = max_range_km(payload, wind, 45, 1.0, 1.0, 25)
            errs.append(abs(predict_range(model, payload, wind, 45, 1.0, 1.0, 25) - truth))
    assert np.mean(errs) < 1.5
