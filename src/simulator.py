"""Phase 2: drone vs road delivery-time simulator (idea 1)."""
import math
from pathlib import Path

import pandas as pd

import config
from src.geo import drone_path_km, haversine_km


DEFAULT_HOSPITALS = Path(__file__).resolve().parent.parent / "data" / "hospitals.csv"


def load_hospitals(path=DEFAULT_HOSPITALS):
    df = pd.read_csv(path)
    required = {"name", "lat", "lon", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"hospitals file missing columns: {sorted(missing)}")
    return df


def road_distance_km(straight_km):
    return straight_km * config.ROAD_CIRCUITY


def road_time_min(straight_km, hour):
    """Road delivery time (minutes) for a given straight-line distance and hour."""
    if hour not in config.ROAD_SPEED_KMH:
        raise ValueError(f"hour must be an int in 0..23, got {hour!r}")
    speed = config.ROAD_SPEED_KMH[hour]
    return config.ROAD_DISPATCH_MIN + road_distance_km(straight_km) / speed * 60.0


def drone_time_min(flight_km):
    """Drone delivery time (minutes) for a given flight distance. Hour-independent."""
    if math.isinf(flight_km):
        return math.inf
    return config.DRONE_OVERHEAD_MIN + flight_km / config.DRONE_CRUISE_KMH * 60.0


def compare_pair(origin, dest, hour):
    """Compare delivery from `origin` to `dest` (each a row/dict with lat, lon, name)."""
    straight = haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    flight = drone_path_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"],
                           config.NO_FLY_ZONES, config.REF_LAT, config.REF_LON)
    in_range = flight <= config.DRONE_MAX_RANGE_KM
    road = road_time_min(straight, hour)
    drone = drone_time_min(flight) if in_range else math.inf
    return {
        "destination": dest["name"],
        "straight_km": round(straight, 2),
        "drone_flight_km": round(flight, 2) if not math.isinf(flight) else math.inf,
        "in_range": bool(in_range),
        "road_min": round(road, 1),
        "drone_min": round(drone, 1) if not math.isinf(drone) else math.inf,
        "minutes_saved": round(road - drone, 1) if not math.isinf(drone) else float("nan"),
    }


def compare_from_depot(depot_name, hospitals, hour):
    """Compare delivery from one depot to every other hospital at a given hour."""
    match = hospitals[hospitals["name"] == depot_name]
    if match.empty:
        raise ValueError(f"unknown depot: {depot_name!r}")
    origin = match.iloc[0]
    rows = [compare_pair(origin, d, hour)
            for _, d in hospitals.iterrows() if d["name"] != depot_name]
    return pd.DataFrame(rows).sort_values("straight_km").reset_index(drop=True)


def hourly_profile(origin, dest):
    """Road vs drone time for every hour 0..23 between two hospitals."""
    rows = []
    for h in range(24):
        r = compare_pair(origin, dest, h)
        rows.append({"hour": h, "road_min": r["road_min"], "drone_min": r["drone_min"]})
    return pd.DataFrame(rows)
