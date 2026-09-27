# SPDX-License-Identifier: CC0-1.0
# Cold-Chain Shipment Spoilage dataset generator, and the synthetic data it produces.
# Author: bubs1-dev. Released into the public domain under CC0 1.0 Universal:
# https://creativecommons.org/publicdomain/zero/1.0/
# Commercial use, modification and redistribution are permitted without restriction.

"""
Reproducible generator for the Cold-Chain Shipment Spoilage dataset.

Running this script twice produces byte-identical files in ./raw/:
    shipments.csv        one row per shipment, with the spoilage label
    sensor_readings.csv  raw temperature-logger readings (long format)
    products.csv         product master data with storage specifications

Simulation outline
------------------
1. A product catalogue (40 SKUs, 10 categories) defines the allowed temperature
   window, freeze sensitivity and a stability budget (minutes above the upper
   limit the product can tolerate).
2. Each shipment picks an origin facility (with its own climate and hemisphere),
   a destination, a carrier, a transport mode, a packaging system and a logger.
3. The *true* product temperature is simulated with a first-order thermal model:
   the payload relaxes towards the environment temperature; active reefers pull
   it back towards the set-point unless the unit fails; passive packaging holds
   the set-point until its cooling capacity is consumed by heat load. Hub
   dwells (tarmac exposure), customs holds and door openings add heat.
4. The spoilage label is drawn from a logistic model of the true exposure:
   time above the product's upper limit relative to its stability budget, peak
   exceedance, freezing of freeze-sensitive products, plus unobserved lot
   quality noise.
5. The logger *measures* the true temperature with calibration drift, noise,
   dropouts, error sentinels (-999), battery failure, duplicated uploads and,
   for the TL-200US model, in degrees Fahrenheit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250927
N_SHIPMENTS = 26000
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-09-30 18:00")
OUT = Path(__file__).resolve().parent / "raw"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
# category: storage_class, min_c, max_c, freeze_sensitive, budget range (min), fragility, value range
CATEGORIES = {
    "vaccine":        ("refrigerated",    2.0,   8.0, True,  (120, 480),   1.0, (40, 300)),
    "biologic":       ("refrigerated",    2.0,   8.0, True,  (60, 240),    1.3, (200, 2500)),
    "insulin":        ("refrigerated",    2.0,   8.0, True,  (600, 1440),  0.6, (20, 90)),
    "blood_plasma":   ("frozen",        -30.0, -18.0, False, (30, 90),     1.4, (150, 600)),
    "frozen_seafood": ("frozen",        -25.0, -15.0, False, (240, 720),   0.8, (8, 60)),
    "ice_cream":      ("frozen",        -25.0, -18.0, False, (60, 180),    1.1, (3, 12)),
    "fresh_berries":  ("chilled",         0.0,   4.0, True,  (180, 600),   0.9, (2, 15)),
    "dairy":          ("chilled",         1.0,   5.0, True,  (240, 720),   0.7, (1, 8)),
    "cut_flowers":    ("chilled",         1.0,   6.0, True,  (300, 900),   0.8, (1, 10)),
    "oral_solid_crt": ("controlled_room", 15.0, 25.0, False, (1440, 4320), 0.4, (5, 120)),
}

products = []
fragility = {}
sku_no = 1001
for cat, (sclass, lo, hi, fsens, (b0, b1), frag, (v0, v1)) in CATEGORIES.items():
    for _ in range(4):
        sku = f"SKU-{sku_no}"
        sku_no += 1
        budget = int(rng.integers(b0, b1 + 1) // 10 * 10)
        fragility[sku] = frag * rng.uniform(0.75, 1.25)
        products.append(dict(
            sku=sku, product_category=cat, storage_class=sclass,
            min_temp_c=lo, max_temp_c=hi, freeze_sensitive=fsens,
            stability_budget_minutes=budget,
            unit_value_usd=round(float(rng.uniform(v0, v1)), 2),
        ))
products = pd.DataFrame(products)
prod_idx = products.set_index("sku")

# ---------------------------------------------------------------------------
# Facilities, hubs, carriers, loggers
# ---------------------------------------------------------------------------
# code: (annual mean C, seasonal amplitude C, hemisphere (+1 north, -1 south), region)
FACILITIES = {
    "FAC-MUM": (27.5, 3.0, 1, "south_asia"),
    "FAC-DEL": (25.0, 9.0, 1, "south_asia"),
    "FAC-SIN": (27.8, 0.8, 1, "southeast_asia"),
    "FAC-ROT": (10.5, 7.0, 1, "western_europe"),
    "FAC-CHI": (10.0, 14.0, 1, "north_america"),
    "FAC-PHX": (24.0, 10.5, 1, "north_america"),
    "FAC-SAO": (20.5, 3.5, -1, "south_america"),
    "FAC-JNB": (16.0, 5.0, -1, "southern_africa"),
    "FAC-OSL": (6.0, 10.0, 1, "northern_europe"),
    "FAC-MEX": (17.0, 3.0, 1, "north_america"),
}
FAC_WEIGHTS = np.array([12, 9, 8, 11, 10, 7, 8, 6, 5, 7], dtype=float)
DEST_REGIONS = ["south_asia", "southeast_asia", "western_europe", "northern_europe",
                "north_america", "south_america", "southern_africa", "middle_east"]
DEST_CLIMATE = {  # annual mean, amplitude, hemisphere
    "south_asia": (27, 5, 1), "southeast_asia": (28, 1, 1),
    "western_europe": (11, 7, 1), "northern_europe": (7, 9, 1),
    "north_america": (14, 11, 1), "south_america": (21, 4, -1),
    "southern_africa": (18, 5, -1), "middle_east": (29, 9, 1),
}
# hub: (annual mean, amplitude, hemisphere, tarmac solar gain C)
HUBS = {
    "DXB": (28, 8, 1, 9), "DOH": (28, 8, 1, 8), "SIN": (28, 1, 1, 5),
    "FRA": (10, 8, 1, 3), "AMS": (10, 6, 1, 2), "ORD": (10, 14, 1, 3),
    "MEM": (17, 10, 1, 5), "LHR": (11, 6, 1, 2), "HKG": (23, 6, 1, 5),
    "JFK": (13, 11, 1, 3), "GRU": (20, 3, -1, 5), "NBO": (18, 2, -1, 5),
    "BOM": (27, 3, 1, 7), "MIA": (25, 4, 1, 6),
}
HUB_CODES = list(HUBS)
# carrier: (reefer failure prob per shipment, handling quality 0..1, door-event rate per 10h, reports door events)
CARRIERS = {
    "Arctis Logistics":  (0.025, 0.90, 0.6, True),
    "BlueLine Cold":     (0.045, 0.80, 0.9, True),
    "Cryonet":           (0.030, 0.85, 0.5, False),
    "FrostWay":          (0.070, 0.65, 1.4, True),
    "Meridian Freight":  (0.055, 0.70, 1.1, True),
    "NorthStar Reefer":  (0.020, 0.92, 0.4, True),
    "Tundra Express":    (0.060, 0.60, 1.6, False),
    "Polar Freight":     (0.110, 0.55, 1.8, True),   # launched Feb 2025
}
POLAR_LAUNCH = pd.Timestamp("2025-02-01")
# logger: (interval minutes, noise sd, sentinel error rate)
LOGGERS = {
    "TL-100":   (30, 0.30, 0.004),
    "TL-200":   (60, 0.20, 0.001),
    "TL-200US": (60, 0.20, 0.001),   # records in Fahrenheit
    "XT-9":     (20, 0.10, 0.000),
}

PACKAGING_BY_CLASS = {
    "refrigerated":    (["active_reefer", "passive_vip_box", "passive_eps_box", "gel_pack_cooler"], [0.40, 0.30, 0.20, 0.10]),
    "chilled":         (["active_reefer", "passive_eps_box", "gel_pack_cooler"], [0.55, 0.25, 0.20]),
    "frozen":          (["active_reefer", "dry_ice_shipper"], [0.60, 0.40]),
    "controlled_room": (["active_reefer", "passive_eps_box", "insulated_blanket"], [0.35, 0.35, 0.30]),
}
QUALIFIED_HOURS = {
    "passive_vip_box": (96, 120), "passive_eps_box": (48, 72), "gel_pack_cooler": (20, 36),
    "dry_ice_shipper": (48, 96), "insulated_blanket": (60, 84),
}


def seasonal_temp(mean, amp, hemi, ts):
    """Monthly-mean ambient temperature; peaks mid-July in the north, mid-January in the south."""
    doy = ts.dayofyear + ts.hour / 24.0
    phase = 2 * np.pi * (doy - 196) / 365.25
    return np.asarray(mean + hemi * amp * np.cos(phase), dtype=float)


# ---------------------------------------------------------------------------
# Shipment-level attributes
# ---------------------------------------------------------------------------
span_h = (END - START) / pd.Timedelta(hours=1)
# volume grows ~30% over the period
u = rng.uniform(size=N_SHIPMENTS)
frac = (np.sqrt(1 + u * (1.3 ** 2 - 1)) - 1) / 0.3
dispatch = START + pd.to_timedelta(np.sort(frac) * span_h, unit="h")
dispatch = dispatch.round("min")

fac_codes = list(FACILITIES)
origin = rng.choice(fac_codes, size=N_SHIPMENTS, p=FAC_WEIGHTS / FAC_WEIGHTS.sum())
skus = rng.choice(products.sku.values, size=N_SHIPMENTS)

rows = []
readings = []
risk_parts = []

id_pool = rng.choice(16 ** 7, size=N_SHIPMENTS, replace=False)

for i in range(N_SHIPMENTS):
    sid = f"SH{id_pool[i]:07X}"
    t0 = dispatch[i]
    sku = skus[i]
    p = prod_idx.loc[sku]
    sclass = p.storage_class
    lo, hi = p.min_temp_c, p.max_temp_c
    setpoint = (lo + hi) / 2 if sclass != "frozen" else hi - 5.0
    fac = origin[i]
    f_mean, f_amp, f_hemi, f_region = FACILITIES[fac]

    dest = rng.choice([d for d in DEST_REGIONS if d != f_region] + [f_region] * 2)
    domestic = dest == f_region

    # carrier (Polar Freight only after launch, and it grows fast)
    carriers = list(CARRIERS)
    cw = np.array([14, 12, 10, 11, 12, 9, 10, 0], dtype=float)
    if t0 >= POLAR_LAUNCH:
        cw[-1] = 14 + 4 * (t0 - POLAR_LAUNCH).days / 30
    carrier = rng.choice(carriers, p=cw / cw.sum())
    fail_p, handling, door_rate, reports_doors = CARRIERS[carrier]

    # packaging & mode
    pk_opts, pk_w = PACKAGING_BY_CLASS[sclass]
    packaging = rng.choice(pk_opts, p=pk_w)
    if packaging == "active_reefer":
        mode = rng.choice(["road_reefer", "rail_reefer", "air"], p=[0.6, 0.15, 0.25] if not domestic else [0.85, 0.15, 0.0])
    elif packaging in ("gel_pack_cooler",):
        mode = "courier_van" if domestic else "air"
    else:
        mode = rng.choice(["air", "courier_van"], p=[0.2, 0.8] if domestic else [0.9, 0.1])

    qual_h = 0.0  # not applicable for active reefers
    if packaging in QUALIFIED_HOURS:
        a, b = QUALIFIED_HOURS[packaging]
        qual_h = float(rng.integers(a // 12, b // 12 + 1) * 12)

    # route & timing
    if mode == "air":
        n_hubs = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
        planned = rng.uniform(10, 28) + 6 * n_hubs
    elif mode == "road_reefer":
        n_hubs = int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        planned = rng.uniform(8, 60 if not domestic else 30)
    elif mode == "rail_reefer":
        n_hubs = int(rng.choice([0, 1], p=[0.5, 0.5]))
        planned = rng.uniform(40, 110)
    else:
        n_hubs = 0
        planned = rng.uniform(3, 14)
    hubs = list(rng.choice(HUB_CODES, size=n_hubs, replace=False)) if n_hubs else []
    planned = round(float(planned), 1)
    customs = bool((not domestic) and rng.uniform() < 0.12)
    delay = float(rng.lognormal(0.0, 0.12 + 0.25 * (1 - handling)))
    actual = planned * max(delay, 0.9) + (rng.uniform(6, 40) if customs else 0.0)
    actual = round(float(actual), 1)

    # logger
    months = (t0 - START).days / 30.4
    us_share = 0.10 + 0.30 * months / 21
    lw = np.array([0.30, 0.35, us_share, 0.20])
    logger = rng.choice(list(LOGGERS), p=lw / lw.sum())
    interval, noise_sd, sentinel_rate = LOGGERS[logger]
    calib_age = int(rng.integers(0, 720))

    precooled_true = rng.uniform() < (0.85 if handling > 0.75 else 0.65)

    # ------------------------------------------------------------------
    # Thermal simulation at logger resolution
    # ------------------------------------------------------------------
    dt_h = interval / 60.0
    n = int(np.ceil(actual / dt_h)) + 1
    t_h = np.arange(n) * dt_h
    ts = t0 + pd.to_timedelta(t_h, unit="h")

    o_amb = seasonal_temp(f_mean, f_amp, f_hemi, ts)
    d_mean, d_amp, d_hemi = DEST_CLIMATE[dest]
    d_amb = seasonal_temp(d_mean, d_amp, d_hemi, ts)
    w = np.clip(t_h / max(actual, 1e-6), 0, 1)
    ambient = (1 - w) * o_amb + w * d_amb
    ambient = ambient + 5.0 * np.sin(2 * np.pi * (np.asarray(ts.hour + ts.minute / 60) - 9) / 24)
    ambient = ambient + rng.normal(0, 2.0)  # weather anomaly for this trip

    env = ambient.copy()
    if mode == "air":
        env = np.where((w > 0.1) & (w < 0.9), 12.0 + 0.3 * (ambient - 12.0), ambient)
    elif mode == "courier_van":
        env = ambient + 3.0

    # hub dwell windows (tarmac exposure)
    exposed = np.zeros(n, dtype=bool)
    for k, h in enumerate(hubs):
        centre = (k + 1) / (n_hubs + 1) * actual
        dwell = rng.uniform(1, 5) * (1.6 - handling)
        if customs and k == n_hubs - 1:
            dwell += actual - planned * max(delay, 0.9)
        m = (t_h >= centre - dwell / 2) & (t_h <= centre + dwell / 2)
        hm, ha, hh, solar = HUBS[h]
        hub_amb = seasonal_temp(hm, ha, hh, ts[m]) + rng.normal(0, 1.5)
        env[m] = hub_amb + solar * (1.2 - handling)
        exposed |= m
    if customs and n_hubs == 0:
        m = t_h >= planned * max(delay, 0.9)
        env[m] = ambient[m] + 4.0
        exposed |= m

    T = np.empty(n)
    T[0] = setpoint + (0.0 if precooled_true else rng.uniform(4, 10))
    active = packaging == "active_reefer"
    fail_start, fail_end = np.inf, np.inf
    if active and rng.uniform() < fail_p * (1.0 + 0.6 * (ambient.mean() > 25)):
        fail_start = rng.uniform(0.05, 0.9) * actual
        fail_end = fail_start + rng.exponential(8.0) + 1.0
    # passive cooling capacity (in C*hours of heat load at nominal 30C ambient)
    capacity = (qual_h * (30.0 - setpoint) if not active else 0.0) * rng.uniform(0.85, 1.1)
    if packaging == "dry_ice_shipper":
        setpoint = -45.0
        T[0] = setpoint
    door_times = []
    if mode in ("road_reefer", "courier_van"):
        n_doors = rng.poisson(door_rate * actual / 10.0)
        door_times = list(rng.uniform(0, actual, size=n_doors))
    door_idx = set(np.searchsorted(t_h, door_times))

    k_leak = 0.06 if active else 0.25
    for j in range(1, n):
        tj = t_h[j]
        e = env[j]
        prev = T[j - 1]
        if active:
            unplugged = exposed[j] and mode == "air" and rng.uniform() < 0.3
            working = not (fail_start <= tj <= fail_end) and not unplugged
            drift = k_leak * (e - prev) * dt_h
            pull = 0.9 * (setpoint - prev) * dt_h if working else 0.0
            # winter failure mode: reefer heater fault drives temperature down
            if not working and e < setpoint:
                drift *= 3.0
            T[j] = prev + drift + pull
        else:
            load = max(e - setpoint, 0.0) * dt_h * (1.8 if exposed[j] else 1.0)
            capacity -= load
            if capacity > 0:
                T[j] = prev + 0.5 * (setpoint - prev) * dt_h + 0.01 * (e - prev) * dt_h
            else:
                T[j] = prev + k_leak * (e - prev) * dt_h
            # passive packs also cool below range in freezing weather
            if e < 0 and capacity <= 0:
                T[j] = prev + 0.3 * (e - prev) * dt_h
        if j in door_idx:
            T[j] += rng.uniform(2, 7) * (1 if e > T[j] else -0.5)
        T[j] += rng.normal(0, 0.15)

    # ------------------------------------------------------------------
    # True exposure -> spoilage risk
    # ------------------------------------------------------------------
    dt_min = interval
    above = np.clip(T - hi, 0, None)
    minutes_above = float((above > 0).sum() * dt_min)
    peak_above = float(above.max())
    freeze_minutes = float(((T < -0.5).sum() * dt_min) if (p.freeze_sensitive and sclass != "frozen") else 0.0)
    ratio = minutes_above / p.stability_budget_minutes
    lot_quality = rng.normal(0, 0.7)
    risk = (2.4 * np.clip(ratio, 0, 3.0) + 0.35 * min(peak_above, 12) * fragility[sku]
            + 3.2 * (freeze_minutes > 30) + 1.2 * (freeze_minutes > 180)
            + 0.5 * customs + 0.9 * (1 - handling) + lot_quality)
    risk_parts.append(risk)

    # ------------------------------------------------------------------
    # Logger measurement
    # ------------------------------------------------------------------
    meas = T + rng.normal(0, 0.12) + 0.0011 * calib_age + rng.normal(0, noise_sd, size=n)
    meas = np.round(meas, 1)
    valid = np.ones(n, dtype=bool)
    valid &= rng.uniform(size=n) > 0.004                                 # random dropouts
    valid &= ~((T < -20) & (rng.uniform(size=n) < 0.15))                 # cold battery sag
    if rng.uniform() < 0.025:                                           # battery death
        cut = int(n * rng.uniform(0.35, 0.9))
        valid[cut:] = False
    if logger == "TL-200US":
        meas = np.round(meas * 9 / 5 + 32, 1)
    sentinel = rng.uniform(size=n) < sentinel_rate
    meas = np.where(sentinel, -999.0, meas)
    rts = ts[valid]
    rv = meas[valid]
    if rng.uniform() < 0.015 and len(rv) > 4:                           # duplicated upload
        s = int(rng.integers(0, len(rv) // 2))
        rts = rts.append(rts[s:s + len(rv) // 3])
        rv = np.concatenate([rv, rv[s:s + len(rv) // 3]])
    readings.append(pd.DataFrame({"shipment_id": sid, "reading_ts": rts, "temp_value": rv}))

    # ------------------------------------------------------------------
    # Messy shipment metadata
    # ------------------------------------------------------------------
    if rng.uniform() < 0.08:
        precooled_str = np.nan
    else:
        precooled_str = rng.choice(["Y", "yes", "TRUE", "1"] if precooled_true else ["N", "no", "FALSE", "0"])
    doors_reported = len(door_times) if reports_doors else np.nan
    forecast_max = round(float(o_amb[0] + 5.0 + rng.normal(0, 1.8)), 1)
    qty = int(rng.integers(10, 800))
    rows.append(dict(
        shipment_id=sid,
        dispatch_time=t0.strftime("%Y-%m-%d %H:%M:%S"),
        sku=sku,
        quantity_units=qty,
        origin_facility=fac,
        destination_region=dest,
        route_hubs="|".join(hubs) if hubs else np.nan,
        carrier=carrier,
        transport_mode=mode,
        packaging_type=packaging,
        packaging_qualified_hours=qual_h,
        planned_transit_hours=planned,
        actual_transit_hours=actual,
        customs_hold=int(customs),
        logger_model=logger,
        logger_interval_min=interval,
        logger_calibration_age_days=calib_age,
        precooled=precooled_str,
        origin_forecast_max_c=forecast_max,
        door_open_events=doors_reported,
        declared_value_usd=round(qty * float(p.unit_value_usd), 2),
    ))

ship = pd.DataFrame(rows)

# Calibrate the intercept so that ~7% of shipments spoil, then draw labels.
risk = np.array(risk_parts)
lo_b, hi_b = -20.0, 5.0
for _ in range(60):
    mid = (lo_b + hi_b) / 2
    if (1 / (1 + np.exp(-(risk + mid)))).mean() > 0.07:
        hi_b = mid
    else:
        lo_b = mid
prob = 1 / (1 + np.exp(-(risk + lo_b)))
spoiled = (rng.uniform(size=N_SHIPMENTS) < prob).astype(int)
ship["spoiled"] = spoiled

# Post-arrival field (target leakage): recorded by the receiving dock *after* QA.
disp = np.where(
    spoiled == 1,
    rng.choice(["QUARANTINE", "REJECTED", "ACCEPTED_WITH_NOTE", "ACCEPTED"], size=N_SHIPMENTS, p=[0.55, 0.25, 0.12, 0.08]),
    rng.choice(["QUARANTINE", "REJECTED", "ACCEPTED_WITH_NOTE", "ACCEPTED"], size=N_SHIPMENTS, p=[0.03, 0.005, 0.10, 0.865]),
)
ship["dock_disposition"] = disp

readings = pd.concat(readings, ignore_index=True)
readings["reading_ts"] = readings["reading_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

OUT.mkdir(parents=True, exist_ok=True)
products.to_csv(OUT / "products.csv", index=False)
ship.to_csv(OUT / "shipments.csv", index=False)
readings.to_csv(OUT / "sensor_readings.csv", index=False)
print(f"shipments={len(ship)} spoiled_rate={spoiled.mean():.4f} readings={len(readings)}")
