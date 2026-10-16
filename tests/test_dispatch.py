import math

import pandas as pd
import pytest

import config
from src.dispatch import (add_road_baseline, generate_orders, simulate, summarize)
from src.simulator import drone_time_min, load_hospitals


@pytest.fixture(scope="module")
def hospitals():
    return load_hospitals()


def mini():
    """Four sites north of the airport zone: hubs A,B (20 km apart) and nearby destinations."""
    return pd.DataFrame({
        "name": ["A", "B", "Da", "Db", "Far"],
        "lat": [28.68, 28.68, 28.69, 28.69, 28.95],
        "lon": [77.10, 77.30, 77.11, 77.29, 77.95],
        "type": ["x"] * 5,
    })


def orders_df(rows):
    return pd.DataFrame(rows, columns=["order_id", "t_min", "dest", "priority", "payload_kg"])


def test_generate_orders_deterministic_and_valid(hospitals):
    hubs = ["AIIMS New Delhi", "Medanta Gurugram"]
    a = generate_orders(hospitals, hubs, 50, 240, seed=7)
    b = generate_orders(hospitals, hubs, 50, 240, seed=7)
    assert a.equals(b)
    assert a["t_min"].is_monotonic_increasing
    assert set(a["priority"]) <= {1, 2, 3}
    assert not set(a["dest"]) & set(hubs)
    assert a["t_min"].between(0, 240).all()


def test_generate_orders_validation(hospitals):
    with pytest.raises(ValueError):
        generate_orders(hospitals, list(hospitals["name"]), 5, 60)
    with pytest.raises(ValueError):
        generate_orders(hospitals, ["AIIMS New Delhi"], 5, 0)


def test_single_order_timing_matches_simulator_formula():
    h = mini()
    res, log = simulate(h, ["A"], 1, orders_df([(0, 5.0, "Da", 1, 1.0)]))
    r = res.iloc[0]
    assert r["status"] == "served"
    assert r["wait_min"] == pytest.approx(0.0)
    assert r["delivery_min"] == pytest.approx(drone_time_min(r["flight_km"]))
    assert log.iloc[0]["free_at_min"] > log.iloc[0]["delivered_min"]


def test_hungarian_assigns_nearest_hub_even_when_listed_in_reverse():
    h = mini()
    o = orders_df([(0, 0.0, "Db", 1, 1.0), (1, 0.0, "Da", 1, 1.0)])
    res, _ = simulate(h, ["A", "B"], 1, o)
    got = dict(zip(res["dest"], res["hub"]))
    assert got == {"Da": "A", "Db": "B"}


def test_priority_policy_serves_emergency_before_queued_routine():
    h = mini()
    o = orders_df([(0, 0.0, "Da", 3, 1.0), (1, 1.0, "Da", 3, 1.0), (2, 2.0, "Da", 1, 1.0)])
    pri, _ = simulate(h, ["A"], 1, o, policy="priority")
    fifo, _ = simulate(h, ["A"], 1, o, policy="fifo")
    p1 = pri.loc[pri["order_id"] == 2, "delivery_min"].iloc[0]
    f1 = fifo.loc[fifo["order_id"] == 2, "delivery_min"].iloc[0]
    assert p1 < f1


def test_out_of_range_order_is_flagged_not_dropped_silently():
    h = mini()
    o = orders_df([(0, 0.0, "Da", 2, 1.0), (1, 0.0, "Far", 2, 1.0)])
    res, _ = simulate(h, ["A"], 1, o)
    assert len(res) == 2
    assert res.loc[res["dest"] == "Far", "status"].iloc[0].startswith("unserved")
    assert res.loc[res["dest"] == "Da", "status"].iloc[0] == "served"


def test_overweight_payload_is_unserved():
    h = mini()
    o = orders_df([(0, 0.0, "Da", 1, config.DRONE_MAX_PAYLOAD_KG + 1.0)])
    res, _ = simulate(h, ["A"], 1, o)
    assert res.iloc[0]["status"].startswith("unserved")


def test_invariants_on_realistic_run(hospitals):
    hubs = ["AIIMS New Delhi", "LNJP Hospital", "Medanta Gurugram"]
    orders = generate_orders(hospitals, hubs, 120, 240, seed=3)
    res, log = simulate(hospitals, hubs, 1, orders, policy="priority")
    served = res[res["status"] == "served"]
    assert len(served) == len(res) == 120
    assert served["order_id"].is_unique
    assert (served["dispatch_min"] >= served["t_min"] - 1e-9).all()
    assert (served["delivered_min"] > served["dispatch_min"]).all()
    # no drone is double-booked
    for _, g in log.sort_values("start_min").groupby("drone_id"):
        starts, ends = g["start_min"].to_numpy(), g["free_at_min"].to_numpy()
        assert (starts[1:] >= ends[:-1] - 1e-9).all()


def test_more_drones_do_not_increase_mean_delivery_time(hospitals):
    hubs = ["AIIMS New Delhi", "LNJP Hospital", "Medanta Gurugram"]
    orders = generate_orders(hospitals, hubs, 120, 120, seed=5)
    r1, _ = simulate(hospitals, hubs, 1, orders)
    r3, _ = simulate(hospitals, hubs, 3, orders)
    assert r3["delivery_min"].mean() <= r1["delivery_min"].mean() + 1e-9


def test_headwind_slows_delivery():
    h = mini()
    o = orders_df([(0, 0.0, "Da", 1, 1.0)])
    calm, _ = simulate(h, ["A"], 1, o)
    windy, _ = simulate(h, ["A"], 1, o, wind_kmh=30, wind_angle_deg=0)
    assert windy.iloc[0]["delivery_min"] > calm.iloc[0]["delivery_min"]


def test_summary_and_road_baseline(hospitals):
    hubs = ["AIIMS New Delhi", "LNJP Hospital", "Medanta Gurugram"]
    orders = generate_orders(hospitals, hubs, 15, 180, seed=2)   # lightly loaded fleet
    res, log = simulate(hospitals, hubs, 2, orders)
    table, util = summarize(res, log, 6)
    assert list(table["priority"]) == ["Emergency", "Urgent", "Routine"]
    assert table["orders"].sum() == 15
    assert 0 < util <= 100
    withroad = add_road_baseline(res, hospitals, hubs)
    assert (withroad["road_min"] > 0).all()
    assert withroad["road_min"].mean() > res["delivery_min"].mean()


def test_overloaded_fleet_loses_advantage_and_priority_protects_emergencies(hospitals):
    hubs = ["AIIMS New Delhi", "LNJP Hospital", "Medanta Gurugram"]
    orders = generate_orders(hospitals, hubs, 60, 180, seed=2)   # far more than 3 drones can serve
    pri, _ = simulate(hospitals, hubs, 1, orders, policy="priority")
    fifo, _ = simulate(hospitals, hubs, 1, orders, policy="fifo")
    e_pri = pri.loc[pri["priority"] == 1, "delivery_min"].mean()
    e_fifo = fifo.loc[fifo["priority"] == 1, "delivery_min"].mean()
    assert e_pri < e_fifo / 2
    assert fifo["delivery_min"].mean() > 60     # queueing dominates when under-provisioned


def test_validation_errors(hospitals):
    o = orders_df([(0, 0.0, "AIIMS New Delhi", 1, 1.0)])
    with pytest.raises(ValueError):
        simulate(hospitals, ["Nowhere"], 1, o)
    with pytest.raises(ValueError):
        simulate(hospitals, ["Safdarjung Hospital"], 1, o, policy="random")
    with pytest.raises(ValueError):
        simulate(hospitals, ["Safdarjung Hospital"], 0, o)
    bad = orders_df([(0, 0.0, "Ghost", 1, 1.0)])
    with pytest.raises(ValueError):
        simulate(hospitals, ["Safdarjung Hospital"], 1, bad)
