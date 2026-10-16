"""All tunable assumptions live here, so they are easy to cite in the report.

EVERY number below is an ASSUMPTION for simulation, not a measured value.
Cross-check the drone figures against the synopsis (12-20 min for a 25 km
radius, 1.5-3.5 kg payload, ~40 km range) and state them in your report.
"""

# Projection centre for the flat-earth (equirectangular) approximation, NCR.
REF_LAT = 28.60
REF_LON = 77.20

# ---- Road model -----------------------------------------------------------
# Real roads are longer than the straight line. Typical urban factor 1.2-1.5.
ROAD_CIRCUITY = 1.35
# Time before the vehicle actually leaves (dispatch + loading), minutes.
ROAD_DISPATCH_MIN = 5.0
# Assumed average road speed (km/h) for each hour of day 0..23.
# Peaks 8-10 and 17-20 reflect Delhi-style congestion.
ROAD_SPEED_KMH = {
    0: 40, 1: 40, 2: 40, 3: 40, 4: 40, 5: 38,
    6: 32, 7: 25, 8: 17, 9: 16, 10: 18,
    11: 24, 12: 24, 13: 24, 14: 24, 15: 23, 16: 21,
    17: 15, 18: 14, 19: 15, 20: 18,
    21: 30, 22: 34, 23: 38,
}

# ---- Drone model ----------------------------------------------------------
DRONE_CRUISE_KMH = 90.0
# Launch + climb + precision drop overhead, minutes.
DRONE_OVERHEAD_MIN = 3.0
DRONE_MAX_RANGE_KM = 40.0      # one-way planning limit (synopsis: ~40 km/sortie)
DRONE_MAX_PAYLOAD_KG = 3.5     # synopsis: 1.5-3.5 kg

# ---- No-fly zones: (name, lat, lon, radius_km) ----------------------------
# Demo assumption: 5 km circle around IGI Airport. Not the real regulation.
NO_FLY_ZONES = [
    ("IGI Airport (assumed 5 km)", 28.5562, 77.1000, 5.0),
]

# ---- Module 3: hub placement ----------------------------------------------
COVERAGE_TARGET_MIN = 20.0        # synopsis: 12-20 min delivery goal
UNCOVERED_PENALTY_MIN = 999.0     # cost charged for a hospital no hub can reach

# ---- Module 4: battery / range / payload ----------------------------------
# Calibrated so a full battery, max payload, no wind gives ~40 km one-way
# radius (synopsis: ~40 km per sortie). Illustrative, not a real aircraft spec.
BATTERY_CAPACITY_WH = 350.0
BATTERY_RESERVE_FRACTION = 0.20     # never plan to dip below 20% state of charge
OVERHEAD_ENERGY_WH = 15.0           # take-off + climb + precision drop
POWER_EMPTY_W = 220.0               # cruise power with no payload
POWER_PER_KG_W = 40.0               # extra cruise power per kg of payload
MAX_OPERATING_WIND_KMH = 40.0       # above this, flights are cancelled
COLD_DERATE_PER_C = 0.006           # capacity lost per deg C below 20 C
COLD_REF_C = 20.0
ML_NOISE_KM = 1.0                   # simulated measurement noise for training data

# ---- Module 5: dispatch & priority routing --------------------------------
PRIORITY_NAMES = {1: "Emergency", 2: "Urgent", 3: "Routine"}
PRIORITY_MIX = (0.2, 0.3, 0.5)                   # share of orders per priority 1,2,3
PRIORITY_WEIGHT = {1: 100.0, 2: 10.0, 3: 1.0}    # cost weight in the assignment step
SLA_MIN = {1: 20.0, 2: 45.0, 3: 90.0}            # delivery-time targets (assumed)
PAYLOAD_CHOICES_KG = (1.0, 2.0, 3.0)
TURNAROUND_MIN = 6.0                             # battery swap + reload at the hub
SIM_START_HOUR = 8                               # clock hour at t=0 (for road baseline)
