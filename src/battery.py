"""Module 4: battery, range and payload model (idea 4).

Physics-style energy model for a fixed-wing delivery drone that flies out loaded
and returns empty. Includes headwind / crosswind effects on ground speed, cold
and ageing effects on usable battery, plus an ML surrogate (gradient boosting) that
predicts the achievable one-way radius from flight conditions.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config

FEATURES = ["payload_kg", "wind_kmh", "wind_angle_deg", "soc", "soh", "temp_c"]


def usable_energy_wh(soc=1.0, soh=1.0, temp_c=20.0):
    """Energy available for the sortie after the reserve, ageing and cold derating."""
    if not 0 <= soc <= 1 or not 0 < soh <= 1:
        raise ValueError("soc must be in [0,1] and soh in (0,1]")
    temp_factor = max(0.5, 1.0 - config.COLD_DERATE_PER_C * max(0.0, config.COLD_REF_C - temp_c))
    return config.BATTERY_CAPACITY_WH * soh * temp_factor * max(0.0, soc - config.BATTERY_RESERVE_FRACTION)


def cruise_power_w(payload_kg):
    if payload_kg < 0:
        raise ValueError("payload must be >= 0")
    return config.POWER_EMPTY_W + config.POWER_PER_KG_W * payload_kg


def ground_speeds(wind_kmh, wind_angle_deg, airspeed=None):
    """(outbound, return) ground speeds in km/h.

    wind_angle_deg is the angle between the outbound track and the direction the
    wind blows FROM: 0 = pure headwind outbound, 180 = pure tailwind outbound,
    90 = pure crosswind. The drone crabs into a crosswind, which costs speed.
    Returns (0, 0) if the aircraft cannot hold the track.
    """
    v = config.DRONE_CRUISE_KMH if airspeed is None else airspeed
    if wind_kmh < 0:
        raise ValueError("wind must be >= 0")
    theta = math.radians(wind_angle_deg)
    head = wind_kmh * math.cos(theta)
    cross = abs(wind_kmh * math.sin(theta))
    if cross >= v:
        return 0.0, 0.0
    along = math.sqrt(v * v - cross * cross)
    out, back = along - head, along + head
    if out <= 0 or back <= 0:
        return 0.0, 0.0
    return out, back


def energy_per_km_round_trip(payload_kg, wind_kmh=0.0, wind_angle_deg=0.0):
    """Wh consumed per km of one-way radius (loaded out + empty back)."""
    out, back = ground_speeds(wind_kmh, wind_angle_deg)
    if out == 0:
        return math.inf
    return cruise_power_w(payload_kg) / out + cruise_power_w(0.0) / back


def max_range_km(payload_kg, wind_kmh=0.0, wind_angle_deg=0.0, soc=1.0, soh=1.0, temp_c=20.0):
    """Maximum one-way radius (km) that still lets the drone return with reserve."""
    if payload_kg > config.DRONE_MAX_PAYLOAD_KG or wind_kmh > config.MAX_OPERATING_WIND_KMH:
        return 0.0
    per_km = energy_per_km_round_trip(payload_kg, wind_kmh, wind_angle_deg)
    if math.isinf(per_km):
        return 0.0
    budget = usable_energy_wh(soc, soh, temp_c) - config.OVERHEAD_ENERGY_WH
    return max(0.0, budget / per_km)


def sortie_energy_wh(distance_km, payload_kg, wind_kmh=0.0, wind_angle_deg=0.0):
    """Total Wh for a round trip to a destination `distance_km` away."""
    if distance_km < 0:
        raise ValueError("distance must be >= 0")
    per_km = energy_per_km_round_trip(payload_kg, wind_kmh, wind_angle_deg)
    return math.inf if math.isinf(per_km) else config.OVERHEAD_ENERGY_WH + per_km * distance_km


def can_deliver(distance_km, payload_kg, wind_kmh=0.0, wind_angle_deg=0.0,
                soc=1.0, soh=1.0, temp_c=20.0):
    return distance_km <= max_range_km(payload_kg, wind_kmh, wind_angle_deg, soc, soh, temp_c)


# ----------------------------- ML surrogate ---------------------------------
def make_training_data(n=4000, seed=0):
    """Simulated flights: physics range plus measurement noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "payload_kg": rng.uniform(0.0, config.DRONE_MAX_PAYLOAD_KG, n),
        "wind_kmh": rng.uniform(0.0, config.MAX_OPERATING_WIND_KMH, n),
        "wind_angle_deg": rng.uniform(0.0, 180.0, n),
        "soc": rng.uniform(0.3, 1.0, n),
        "soh": rng.uniform(0.7, 1.0, n),
        "temp_c": rng.uniform(-5.0, 45.0, n),
    })
    # Oversample the edges of the operating envelope (full / brand-new battery) so the
    # model is not forced to extrapolate exactly where drones usually fly.
    df.loc[rng.random(n) < 0.25, "soc"] = 1.0
    df.loc[rng.random(n) < 0.25, "soh"] = 1.0
    clean = np.array([max_range_km(*row) for row in df[FEATURES].itertuples(index=False, name=None)])
    df["range_km"] = np.clip(clean + rng.normal(0.0, config.ML_NOISE_KM, n), 0.0, None)
    df["physics_range_km"] = clean
    return df


def train_range_model(n=4000, seed=0):
    """Train/test a gradient-boosting model. Returns (model, metrics dict). Test set is held out."""
    data = make_training_data(n, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(data[FEATURES], data["range_km"],
                                              test_size=0.25, random_state=seed)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                          random_state=seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    truth_clean = data.loc[X_te.index, "physics_range_km"]
    metrics = {
        "mae_km": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "mae_vs_noise_free_km": float(mean_absolute_error(truth_clean, pred)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    return model, metrics


def predict_range(model, payload_kg, wind_kmh, wind_angle_deg, soc, soh, temp_c):
    X = pd.DataFrame([[payload_kg, wind_kmh, wind_angle_deg, soc, soh, temp_c]], columns=FEATURES)
    return float(max(0.0, model.predict(X)[0]))
