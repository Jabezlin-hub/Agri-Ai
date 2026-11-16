"""Decision-support features built on top of the trained crop model.

soil_fix      - counterfactual search: the smallest soil changes (N, P, K, pH) that make a
                crop the farmer wants to grow become the model's top recommendation.
stress_test   - re-scores the shortlist under warming and rainfall-change scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURES, TARGET

SOIL = ["N", "P", "K", "ph"]
MIN_TOP_PROB = 0.25
CLIMATE = ["temperature", "humidity", "rainfall"]


def _score(model, X: pd.DataFrame, crop: str) -> tuple[np.ndarray, np.ndarray]:
    """Probability of `crop` and its rank (1 = top) for every row of X."""
    proba = model.predict_proba(X[FEATURES])
    idx = list(model.classes_).index(crop)
    p = proba[:, idx]
    rank = (proba > p[:, None]).sum(axis=1) + 1
    return p, rank


def soil_fix(model, inputs: dict, target: str, df: pd.DataFrame) -> dict:
    """Find the smallest change to the field's soil that makes `target` the top crop.

    Climate stays fixed (a farmer can fertilise or lime a field, not change its weather).
    Candidates are drawn by moving the field's soil part-way or all the way toward each
    real soil profile of the target crop in the dataset; the model scores every candidate
    and the cheapest one (in standard-deviation units) that reaches rank 1 is chosen.
    If none reaches rank 1, the best reachable rank is reported, and the climate is
    checked against the crop's usual range to explain why.
    """
    current = pd.Series(inputs)[FEATURES].astype(float)
    p0, r0 = _score(model, pd.DataFrame([current]), target)
    result = {"target": target, "current_prob": float(p0[0]), "current_rank": int(r0[0])}
    if r0[0] == 1:
        return {**result, "status": "already"}

    profiles = df.loc[df[TARGET] == target, SOIL].drop_duplicates().to_numpy(float)
    steps = np.array([0.25, 0.5, 0.75, 1.0])[:, None, None]
    soil_now = current[SOIL].to_numpy(float)
    cand = (soil_now + steps * (profiles[None] - soil_now)).reshape(-1, len(SOIL))
    cand[:, :3] = np.round(cand[:, :3])            # N, P, K as whole kg/ha
    cand[:, 3] = np.round(cand[:, 3], 1)           # pH to 0.1
    cand = np.unique(cand, axis=0)

    X = pd.DataFrame(np.repeat(current.to_numpy()[None], len(cand), axis=0), columns=FEATURES)
    X[SOIL] = cand
    p, rank = _score(model, X, target)
    # Cost = total change in standard-deviation units. Lowering a nutrient is slow in
    # practice (you can only stop adding it), so decreases count double.
    scale = df[SOIL].std().to_numpy()
    delta = cand - soil_now
    cost = (np.where(delta < 0, 2.0, 1.0) * np.abs(delta) / scale).sum(axis=1)

    # "#1" only counts with real confidence; #1 at 5% just means nothing fits well.
    for status, ok in (("top", (rank == 1) & (p >= MIN_TOP_PROB)), ("top3", rank <= 3)):
        if ok.any():
            i = np.flatnonzero(ok)[np.argmin(cost[ok])]
            break
    else:
        i, status = int(np.argmax(p)), "climate"

    new_soil = dict(zip(SOIL, cand[i]))
    crop_climate = df.loc[df[TARGET] == target, CLIMATE]
    climate_issues = [
        {"feature": f, "yours": float(current[f]),
         "low": float(crop_climate[f].min()), "high": float(crop_climate[f].max())}
        for f in CLIMATE
        if not crop_climate[f].min() <= current[f] <= crop_climate[f].max()
    ]
    return {**result, "status": status, "new_soil": new_soil,
            "new_prob": float(p[i]), "new_rank": int(rank[i]),
            "changes": {f: float(new_soil[f] - current[f]) for f in SOIL},
            "climate_issues": climate_issues}


def fertiliser_advice(feature: str, delta: float) -> str | None:
    """Plain-language action for one soil change. Amounts are rough guides only."""
    if feature == "ph":
        if abs(delta) < 0.15:
            return None
        if delta > 0:
            return (f"Raise soil pH by about {delta:.1f}: apply agricultural lime "
                    "(the amount depends on soil texture; ask a soil lab).")
        return (f"Lower soil pH by about {-delta:.1f}: apply elemental sulphur or use an "
                "acidifying fertiliser such as ammonium sulphate.")
    if abs(delta) < 3:           # below what a field application can target
        return None
    name = {"N": "nitrogen", "P": "phosphorus", "K": "potassium"}[feature]
    if delta < 0:
        return (f"Reduce {name} by about {-delta:.0f} kg/ha: stop applying {name} "
                "fertiliser; levels fall over one or more seasons as crops take it up.")
    if feature == "N":
        return (f"Add about {delta:.0f} kg/ha of nitrogen, e.g. about {delta / 0.46:.0f} kg/ha "
                "of urea (46% N), split across the season.")
    if feature == "P":
        return (f"Add about {delta:.0f} kg/ha of phosphorus, e.g. as single super phosphate "
                "(SSP) or DAP.")
    return f"Add about {delta:.0f} kg/ha of potassium, e.g. as muriate of potash (MOP)."


SCENARIOS = [(dt, rf) for dt in (0, 1, 2, 3) for rf in (0.8, 1.0, 1.2)]


def scenario_label(dt: float, rf: float) -> str:
    rain = "rain same" if rf == 1.0 else f"rain {rf - 1:+.0%}"
    return f"+{dt} °C, {rain}" if dt else f"today, {rain}"


def stress_test(model, inputs: dict, crops: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score `crops` under warming (+0..+3 °C) x rainfall (-20%, 0, +20%) scenarios.

    Returns (probabilities, ranks): rows = crops, columns = scenario labels.
    """
    base = pd.Series(inputs)[FEATURES].astype(float)
    rows = []
    for dt, rf in SCENARIOS:
        s = base.copy()
        s["temperature"] += dt
        s["rainfall"] *= rf
        rows.append(s)
    X = pd.DataFrame(rows)
    proba = model.predict_proba(X[FEATURES])
    classes = list(model.classes_)
    labels = [scenario_label(dt, rf) for dt, rf in SCENARIOS]
    probs = pd.DataFrame({c: proba[:, classes.index(c)] for c in crops}, index=labels).T
    ranks = pd.DataFrame(
        {c: (proba > proba[:, [classes.index(c)]]).sum(axis=1) + 1 for c in crops},
        index=labels).T
    return probs, ranks
