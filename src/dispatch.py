"""Module 5: multi-drone dispatch with priority routing (idea 5).

Event-driven simulation. Whenever an order arrives or a drone becomes free:
  1. choose which waiting orders to serve now (policy decides the order),
  2. assign them to free drones with the Hungarian algorithm, minimising the
     priority-weighted completion time (exact optimum for an assignment problem).

Policies:  "priority" - emergency orders first, then urgent, then routine
           "fifo"     - first come, first served (baseline for comparison)

A drone flies out loaded, drops, returns empty, then needs a turnaround (battery
swap) before the next sortie. Range feasibility comes from the battery model.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config
from src import battery
from src.geo import drone_path_km, haversine_km
from src.simulator import road_time_min

BIG = 1e9
POLICIES = ("priority", "fifo")


def generate_orders(hospitals, hub_names, n_orders, horizon_min, seed=0):
    """Random orders (Poisson-like arrivals) to hospitals that do not host a hub."""
    if n_orders < 0 or horizon_min <= 0:
        raise ValueError("n_orders must be >= 0 and horizon_min > 0")
    dests = hospitals.loc[~hospitals["name"].isin(hub_names), "name"].to_numpy()
    if len(dests) == 0:
        raise ValueError("every hospital is a hub; no destinations to order for")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_id": np.arange(n_orders),
        "t_min": np.sort(rng.uniform(0.0, horizon_min, n_orders)),
        "dest": rng.choice(dests, n_orders),
        "priority": rng.choice([1, 2, 3], n_orders, p=list(config.PRIORITY_MIX)),
        "payload_kg": rng.choice(list(config.PAYLOAD_CHOICES_KG), n_orders),
    })


def _flight_table(hospitals):
    pts = {r.name: (r.lat, r.lon) for r in hospitals.itertuples()}
    table = {}
    for a, (la, lo) in pts.items():
        for b, (lb, lob) in pts.items():
            table[(a, b)] = drone_path_km(la, lo, lb, lob, config.NO_FLY_ZONES,
                                          config.REF_LAT, config.REF_LON)
    return table


def simulate(hospitals, hub_names, drones_per_hub, orders, policy="priority",
             wind_kmh=0.0, wind_angle_deg=0.0):
    """Run the dispatch simulation. Returns (results DataFrame, drone_log DataFrame)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    hub_names = list(hub_names)
    known = set(hospitals["name"])
    if not hub_names or any(h not in known for h in hub_names):
        raise ValueError("hub_names must be a non-empty list of known hospitals")
    per_hub = (dict(drones_per_hub) if isinstance(drones_per_hub, dict)
               else {h: int(drones_per_hub) for h in hub_names})
    if any(per_hub.get(h, 0) < 0 for h in hub_names):
        raise ValueError("drones per hub must be >= 0")
    if sum(per_hub.get(h, 0) for h in hub_names) < 1:
        raise ValueError("need at least one drone")
    missing = set(orders["dest"]) - known
    if missing:
        raise ValueError(f"orders reference unknown hospitals: {sorted(missing)}")

    ft = _flight_table(hospitals)
    out_gs, back_gs = battery.ground_speeds(wind_kmh, wind_angle_deg)

    drones = []
    for h in hub_names:
        for _ in range(per_hub.get(h, 0)):
            drones.append({"id": len(drones), "hub": h, "free_at": 0.0})
    active_hubs = {d["hub"] for d in drones}

    def feasible(hub, dest, payload):
        flight = ft[(hub, dest)]
        if math.isinf(flight) or out_gs == 0:
            return False
        r = battery.max_range_km(payload, wind_kmh, wind_angle_deg)
        return r > 0 and flight <= r

    od = orders.sort_values(["t_min", "order_id"]).reset_index(drop=True)
    rows, servable = {}, []
    for o in od.itertuples():
        base = {"order_id": o.order_id, "priority": o.priority, "dest": o.dest,
                "t_min": o.t_min, "payload_kg": o.payload_kg}
        if any(feasible(h, o.dest, o.payload_kg) for h in active_hubs):
            servable.append(o)
        else:
            rows[o.order_id] = {**base, "status": "unserved (out of range)"}

    log, pending, nxt, t, served = [], [], 0, 0.0, 0
    total = len(servable)
    sv_ids = {o.order_id for o in servable}
    by_id = {o.order_id: o for o in servable}

    while served < total:
        while nxt < len(od) and od.loc[nxt, "t_min"] <= t:
            oid = od.loc[nxt, "order_id"]
            if oid in sv_ids:
                pending.append(oid)
            nxt += 1
        free = [d for d in drones if d["free_at"] <= t]
        progressed = False
        if pending and free:
            if policy == "priority":
                pending.sort(key=lambda i: (by_id[i].priority, by_id[i].t_min, i))
            else:
                pending.sort(key=lambda i: (by_id[i].t_min, i))
            cand = []
            for oid in pending:
                o = by_id[oid]
                if any(feasible(d["hub"], o.dest, o.payload_kg) for d in free):
                    cand.append(oid)
                if len(cand) == len(free):
                    break
            if cand:
                cost = np.full((len(cand), len(free)), BIG)
                for ri, oid in enumerate(cand):
                    o = by_id[oid]
                    w = config.PRIORITY_WEIGHT[o.priority] if policy == "priority" else 1.0
                    for ci, d in enumerate(free):
                        if feasible(d["hub"], o.dest, o.payload_kg):
                            fl = ft[(d["hub"], o.dest)]
                            done = config.DRONE_OVERHEAD_MIN + fl / out_gs * 60.0
                            cost[ri, ci] = w * ((t - o.t_min) + done)
                for ri, ci in zip(*linear_sum_assignment(cost)):
                    if cost[ri, ci] >= BIG:
                        continue
                    o, d = by_id[cand[ri]], free[ci]
                    fl = ft[(d["hub"], o.dest)]
                    delivered = t + config.DRONE_OVERHEAD_MIN + fl / out_gs * 60.0
                    free_at = delivered + fl / back_gs * 60.0 + config.TURNAROUND_MIN
                    d["free_at"] = free_at
                    log.append({"drone_id": d["id"], "hub": d["hub"], "order_id": o.order_id,
                                "start_min": t, "delivered_min": delivered, "free_at_min": free_at})
                    rows[o.order_id] = {
                        "order_id": o.order_id, "priority": o.priority, "dest": o.dest,
                        "t_min": o.t_min, "payload_kg": o.payload_kg, "status": "served",
                        "hub": d["hub"], "drone_id": d["id"], "flight_km": fl,
                        "dispatch_min": t, "delivered_min": delivered,
                        "wait_min": t - o.t_min, "delivery_min": delivered - o.t_min,
                    }
                    pending.remove(o.order_id)
                    served += 1
                    progressed = True
        if progressed:
            continue
        events = []
        if nxt < len(od):
            events.append(float(od.loc[nxt, "t_min"]))
        events += [d["free_at"] for d in drones if d["free_at"] > t]
        if not events:
            break
        t = min(events)

    res = pd.DataFrame(sorted(rows.values(), key=lambda r: r["order_id"]))
    if "delivery_min" in res:
        res["sla_met"] = [
            (bool(dm <= config.SLA_MIN[p]) if st == "served" else False)
            for dm, p, st in zip(res["delivery_min"], res["priority"], res["status"])
        ]
    return res, pd.DataFrame(log)


def add_road_baseline(results, hospitals, hub_names, sim_start_hour=None):
    """Road time from the nearest hub at the order's clock hour (no queueing: optimistic for road)."""
    start = config.SIM_START_HOUR if sim_start_hour is None else sim_start_hour
    pts = {r.name: (r.lat, r.lon) for r in hospitals.itertuples()}
    out = results.copy()
    road = []
    for r in out.itertuples():
        hour = int((start + r.t_min / 60.0) % 24)
        d = min(haversine_km(*pts[h], *pts[r.dest]) for h in hub_names)
        road.append(road_time_min(d, hour))
    out["road_min"] = road
    return out


def summarize(results, drone_log, n_drones):
    """Per-priority delivery statistics plus fleet utilisation."""
    rows = []
    for p in (1, 2, 3):
        sub = results[results["priority"] == p]
        served = sub[sub["status"] == "served"]
        rows.append({
            "priority": config.PRIORITY_NAMES[p],
            "orders": len(sub),
            "served": len(served),
            "mean_min": round(float(served["delivery_min"].mean()), 1) if len(served) else float("nan"),
            "p90_min": round(float(served["delivery_min"].quantile(0.9)), 1) if len(served) else float("nan"),
            f"sla_met_pct": round(float(served["sla_met"].mean() * 100), 1) if len(served) else float("nan"),
        })
    table = pd.DataFrame(rows)
    if len(drone_log):
        busy = float((drone_log["free_at_min"] - drone_log["start_min"]).sum())
        span = float(drone_log["free_at_min"].max())
        util = busy / (n_drones * span) * 100.0 if span > 0 else 0.0
    else:
        util = 0.0
    return table, round(util, 1)
