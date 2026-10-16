"""Medical Drone Network Planner - Streamlit app.

Run:  streamlit run app.py
Tabs: Module 1 drone vs road | Module 3 hub placement | Module 4 battery & range |
      Module 5 dispatch & priority routing.
"""
import math

import pandas as pd
import streamlit as st

import config
from src import battery, dispatch, hubs
from src.simulator import compare_from_depot, hourly_profile, load_hospitals

st.set_page_config(page_title="Medical Drone Network Planner", layout="wide")


@st.cache_data
def get_hospitals():
    return load_hospitals()


@st.cache_data
def get_matrix():
    return hubs.time_matrix(get_hospitals())


@st.cache_data
def get_curve(k_max, threshold):
    return hubs.coverage_curve(get_hospitals(), get_matrix(), k_max=k_max, threshold=threshold)


@st.cache_resource
def get_range_model():
    return battery.train_range_model(n=4000, seed=0)


hospitals = get_hospitals()
names = hospitals["name"].tolist()

st.title("Medical Drone Network Planner")
st.caption("Delhi NCR | simulated, assumption-based model (all assumptions in config.py)")

tab1, tab3, tab4, tab5 = st.tabs([
    "Module 1: Drone vs Road", "Module 3: Hub Placement",
    "Module 4: Battery & Range", "Module 5: Dispatch",
])

# ---------------------------------------------------------------- Module 1
with tab1:
    c1, c2 = st.columns(2)
    depot = c1.selectbox("Depot (delivery origin)", names, index=0, key="sim_depot")
    hour = c2.slider("Hour of day", 0, 23, 18, key="sim_hour")

    df = compare_from_depot(depot, hospitals, hour)
    shown = df.replace({math.inf: float("nan")})
    reachable = df[df["in_range"]]
    m1, m2, m3 = st.columns(3)
    if len(reachable):
        m1.metric("Avg road time", f"{reachable['road_min'].mean():.0f} min")
        m2.metric("Avg drone time", f"{reachable['drone_min'].mean():.0f} min")
        m3.metric("Avg time saved", f"{reachable['minutes_saved'].mean():.0f} min")
    else:
        st.warning("No destination is within drone range from this depot.")

    st.subheader("Delivery time by destination")
    st.bar_chart(shown.set_index("destination")[["road_min", "drone_min"]])
    st.dataframe(shown, hide_index=True)

    st.subheader("Time of day effect")
    dest = st.selectbox("Destination", df["destination"].tolist(), index=0, key="sim_dest")
    origin_row = hospitals[hospitals["name"] == depot].iloc[0]
    dest_row = hospitals[hospitals["name"] == dest].iloc[0]
    prof = hourly_profile(origin_row, dest_row).replace({math.inf: float("nan")})
    st.line_chart(prof.set_index("hour")[["road_min", "drone_min"]])

    with st.expander("Assumptions used"):
        st.write({
            "Road circuity factor": config.ROAD_CIRCUITY,
            "Drone cruise speed (km/h)": config.DRONE_CRUISE_KMH,
            "Drone max one-way range (km)": config.DRONE_MAX_RANGE_KM,
            "No-fly zones": [z[0] for z in config.NO_FLY_ZONES],
        })

# ---------------------------------------------------------------- Module 3
with tab3:
    st.write("Choose where to base drones. Each hospital is served by its fastest hub.")
    a, b, c = st.columns(3)
    k = a.slider("Number of hubs (k)", 1, 6, 3, key="hub_k")
    thr = b.slider("Delivery-time target (min)", 4, 20, 10, key="hub_thr")
    method = c.radio("Method", ["Optimal (exhaustive)", "k-means (heuristic)"], key="hub_method")

    matrix = get_matrix()
    if method.startswith("Optimal"):
        res = hubs.best_hubs_exhaustive(matrix, k, thr)
    else:
        res = hubs.kmeans_hubs(hospitals, matrix, k, threshold=thr)

    m1, m2, m3 = st.columns(3)
    m1.metric("Average delivery time", f"{res['avg_min']:.1f} min")
    m2.metric(f"Hospitals within {thr} min", f"{res['pct_within_target']:.0f}%")
    m3.metric("Slowest hospital", f"{res['max_min']:.1f} min")
    st.write("**Hubs:** " + ", ".join(res["hubs"]))
    if res["uncovered"]:
        st.warning("Not reachable by any hub: " + ", ".join(res["uncovered"]))

    pts = hospitals.copy()
    pts["color"] = ["#E63946" if n in res["hubs"] else "#137C8B" for n in pts["name"]]
    pts["size"] = [400 if n in res["hubs"] else 150 for n in pts["name"]]
    st.map(pts, latitude="lat", longitude="lon", color="color", size="size")
    st.caption("Red = drone hub, teal = other hospital")

    assign = pd.DataFrame({
        "hospital": list(res["best_time"].keys()),
        "served_by_hub": [res["assignment"][h] for h in res["best_time"]],
        "delivery_min": [round(v, 1) if math.isfinite(v) else float("nan")
                         for v in res["best_time"].values()],
    })
    st.dataframe(assign, hide_index=True)

    st.subheader("How many hubs are enough?")
    st.dataframe(get_curve(6, thr), hide_index=True)
    st.caption("With a 40 km range the whole city is reachable from one hub, so the "
               "trade-off shows up in average time and in a strict target, not in reachability.")

# ---------------------------------------------------------------- Module 4
with tab4:
    st.write("How far can the drone fly and still return with reserve?")
    a, b, c = st.columns(3)
    payload = a.slider("Payload (kg)", 0.0, float(config.DRONE_MAX_PAYLOAD_KG), 2.0, 0.1, key="bat_payload")
    wind = b.slider("Wind speed (km/h)", 0, int(config.MAX_OPERATING_WIND_KMH), 10, key="bat_wind")
    angle = c.slider("Wind angle (0 = headwind out, 180 = tailwind out)", 0, 180, 0, key="bat_angle")
    d, e, f = st.columns(3)
    soc = d.slider("Battery charge", 0.3, 1.0, 1.0, 0.05, key="bat_soc")
    soh = e.slider("Battery health", 0.7, 1.0, 1.0, 0.05, key="bat_soh")
    temp = f.slider("Temperature (C)", -5, 45, 25, key="bat_temp")
    dist = st.slider("Delivery distance (km)", 1, 50, 20, key="bat_dist")

    model, metrics = get_range_model()
    phys = battery.max_range_km(payload, wind, angle, soc, soh, temp)
    ml = battery.predict_range(model, payload, wind, angle, soc, soh, temp)
    ok = battery.can_deliver(dist, payload, wind, angle, soc, soh, temp)
    energy = battery.sortie_energy_wh(dist, payload, wind, angle)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max range (physics)", f"{phys:.1f} km")
    m2.metric("Max range (ML model)", f"{ml:.1f} km")
    m3.metric("Sortie energy", "n/a" if math.isinf(energy) else f"{energy:.0f} Wh")
    m4.metric(f"Can deliver {dist} km?", "Yes" if ok else "No")
    if payload > config.DRONE_MAX_PAYLOAD_KG or wind > config.MAX_OPERATING_WIND_KMH:
        st.warning("Outside operating limits: flight cancelled.")

    pay_curve = pd.DataFrame({
        "payload_kg": [round(x * 0.25, 2) for x in range(0, int(config.DRONE_MAX_PAYLOAD_KG * 4) + 1)],
    })
    pay_curve["range_km"] = [battery.max_range_km(p, wind, angle, soc, soh, temp) for p in pay_curve["payload_kg"]]
    wind_curve = pd.DataFrame({"wind_kmh": list(range(0, int(config.MAX_OPERATING_WIND_KMH) + 1, 2))})
    wind_curve["range_km"] = [battery.max_range_km(payload, w, angle, soc, soh, temp) for w in wind_curve["wind_kmh"]]
    g1, g2 = st.columns(2)
    g1.write("Range vs payload")
    g1.line_chart(pay_curve.set_index("payload_kg"))
    g2.write("Range vs wind speed")
    g2.line_chart(wind_curve.set_index("wind_kmh"))

    with st.expander("ML model quality (held-out test flights)"):
        st.write({
            "Mean absolute error (km)": round(metrics["mae_km"], 2),
            "R-squared": round(metrics["r2"], 3),
            "Training / test flights": f"{metrics['n_train']} / {metrics['n_test']}",
        })
        st.caption("The model is trained on simulated flights (physics + noise). With real "
                   "flight logs you would train on those instead.")

# ---------------------------------------------------------------- Module 5
with tab5:
    st.write("Simulate a fleet serving random orders. Compare priority routing with first-come-first-served.")
    default_hubs = [h for h in ("AIIMS New Delhi", "LNJP Hospital", "Medanta Gurugram") if h in names]
    chosen = st.multiselect("Hubs", names, default=default_hubs, key="dsp_hubs")
    a, b, c, d = st.columns(4)
    per_hub = a.slider("Drones per hub", 1, 5, 2, key="dsp_drones")
    n_orders = b.slider("Orders", 5, 120, 40, key="dsp_orders")
    horizon = c.slider("Arrival window (min)", 60, 480, 180, 30, key="dsp_horizon")
    seed = d.number_input("Random seed", 0, 9999, 1, key="dsp_seed")
    w = st.slider("Headwind on outbound leg (km/h)", 0, int(config.MAX_OPERATING_WIND_KMH), 0, key="dsp_wind")

    if not chosen:
        st.info("Pick at least one hub.")
    elif len(chosen) == len(names):
        st.error("Every hospital is a hub, so there is nowhere to deliver. Remove a hub.")
    else:
        orders = dispatch.generate_orders(hospitals, chosen, int(n_orders), float(horizon), int(seed))
        res_p, log_p = dispatch.simulate(hospitals, chosen, per_hub, orders, "priority", wind_kmh=float(w))
        res_f, log_f = dispatch.simulate(hospitals, chosen, per_hub, orders, "fifo", wind_kmh=float(w))
        tab_p, util_p = dispatch.summarize(res_p, log_p, per_hub * len(chosen))
        tab_f, util_f = dispatch.summarize(res_f, log_f, per_hub * len(chosen))

        unserved = int((res_p["status"] != "served").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Fleet size", per_hub * len(chosen))
        m2.metric("Fleet utilisation (priority)", f"{util_p:.0f}%")
        m3.metric("Orders not served", unserved)
        if util_p > 85:
            st.warning("Fleet is close to saturation: queues build up. Add drones or hubs.")

        l, r = st.columns(2)
        l.subheader("Priority routing")
        l.dataframe(tab_p, hide_index=True)
        r.subheader("First come, first served")
        r.dataframe(tab_f, hide_index=True)

        served_p = dispatch.add_road_baseline(res_p[res_p["status"] == "served"], hospitals, chosen)
        if len(served_p):
            comp = pd.DataFrame({
                "Drone (priority)": served_p.groupby("priority")["delivery_min"].mean(),
                "Drone (FIFO)": res_f[res_f["status"] == "served"].groupby("priority")["delivery_min"].mean(),
                "Road (no queue)": served_p.groupby("priority")["road_min"].mean(),
            })
            comp.index = [config.PRIORITY_NAMES[i] for i in comp.index]
            st.subheader("Mean delivery time by priority (min)")
            st.bar_chart(comp)
        with st.expander("Order-level results (priority policy)"):
            st.dataframe(res_p, hide_index=True)
        st.caption("Road baseline assumes an ambulance is always available (no queueing), "
                   "which favours road transport. Priority routing only matters once demand "
                   "exceeds fleet capacity.")
