"""AgriAI - Crop Recommender (Streamlit app).

Run:  streamlit run app.py
"""
import os

# One thread for every numeric library, set before numpy / scikit-learn load. On a cloud
# container these libraries see the host's many cores while the app gets a fraction of
# one, and dozens of competing threads made each run take minutes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import warnings
from pathlib import Path

from threadpoolctl import threadpool_limits

threadpool_limits(limits=1)  # also caps pools that were started before the env vars

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.advisor import SOIL, fertiliser_advice, soil_fix, stress_test
from src.data import FEATURE_INFO, FEATURES, TARGET, load_data
from src.explain import shapley_values
from src.weather import find_places, season_climate

FAMILIES_PATH = Path(__file__).parent / "models" / "crop_families.joblib"

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘", "pigeonpeas": "🫛",
    "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘", "lentil": "🫘", "pomegranate": "🍎",
    "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "apple": "🍏", "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "☁️",
    "jute": "🧵", "coffee": "☕", "wheat": "🌾", "barley": "🌾", "oats": "🌾", "rye": "🌾",
    "buckwheat": "🌾", "quinoa": "🌾", "potato": "🥔", "carrot": "🥕", "onion": "🧅",
    "garlic": "🧄", "peas": "🫛", "faba_bean": "🫛", "broccoli": "🥦", "cauliflower": "🥦",
    "cabbage": "🥬", "kale": "🥬", "lettuce": "🥬", "spinach": "🥬", "brussels_sprouts": "🥬",
    "strawberry": "🍓", "blueberry": "🫐", "cranberry": "🫐", "raspberry": "🍓",
    "cherry": "🍒", "pear": "🍐", "plum": "🍑", "hazelnut": "🌰", "walnut": "🌰",
}

st.set_page_config(page_title="AgriAI - Crop Recommender", page_icon="🌱", layout="wide")


def crop_name(c: str) -> str:
    return c.replace("_", " ").title()


def crop_label(c: str) -> str:
    return f"{CROP_EMOJI.get(c, '🌱')} {crop_name(c)}"


# Each loader takes the metrics file's timestamp, so re-running train.py
# refreshes the cache without restarting the app.
# Harmless on its own, but printed once per tree per prediction on cloud hosts.
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


def _single_core(est):
    """Make every fitted part of the model predict on one core. The model was trained with
    n_jobs=-1 (all cores); on a cloud container Python sees the host's many cores but the
    app gets a fraction of one, so parallel prediction spawns dozens of competing threads.
    A single core is also faster for the small batches the app predicts."""
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1
    for part in (getattr(est, "estimators_", None) or []):
        _single_core(part)
    for _, step in (getattr(est, "steps", None) or []):
        _single_core(step)
    return est


@st.cache_resource
def get_model(version: float):
    return _single_core(joblib.load(MODEL_PATH))


@st.cache_data
def get_data(version: float):
    return load_data()


@st.cache_data
def get_metrics(version: float):
    return json.loads(METRICS_PATH.read_text())


@st.cache_resource
def get_families(version: float):
    return joblib.load(FAMILIES_PATH) if FAMILIES_PATH.exists() else None


@st.cache_data(show_spinner=False)
def explain(version: float, field: tuple):
    """Exact Shapley values for this field, against the 64 crops' median profiles."""
    background = df.groupby(TARGET)[FEATURES].median()
    return shapley_values(model, dict(zip(FEATURES, field)), background)


# Streamlit reruns the whole script on every click and renders every tab, so the
# model-heavy steps are cached per field: repeat visits and other tabs cost nothing.
@st.cache_data(show_spinner=False)
def predict(version: float, field: tuple):
    return model.predict_proba(pd.DataFrame([field], columns=FEATURES))[0]


@st.cache_data(show_spinner=False)
def cached_soil_fix(version: float, field: tuple, crop: str):
    return soil_fix(model, dict(zip(FEATURES, field)), crop, df)


@st.cache_data(show_spinner=False)
def cached_stress_test(version: float, field: tuple, crops: tuple):
    return stress_test(model, dict(zip(FEATURES, field)), list(crops))


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

version = METRICS_PATH.stat().st_mtime
model = get_model(version)
df, source = get_data(version)
metrics = get_metrics(version)
families = get_families(version)
best_res = metrics.get("final") or next(
    r for r in metrics["results"] if r["model"] == metrics["best_model"])

st.title("🌱 AgriAI — Crop Recommender")
st.caption("Recommends crops for your soil nutrients (N-P-K, pH) and local climate "
           f"· Model: **{metrics['best_model']}** · Data: {metrics['n_rows']:,} records, "
           f"{metrics['n_crops']} crops · Right crop in the top 5: "
           f"**{best_res.get('test_top5', best_res['test_accuracy']):.0%}** of test cases")
if "synthetic" in source:
    st.info("Running on the synthetic stand-in dataset. Put a real dataset CSV in `data/` "
            "and re-run `python train.py`.", icon="ℹ️")

tab_rec, tab_fix, tab_climate, tab_fam, tab_data, tab_model, tab_about = st.tabs(
    ["🔮 Recommend", "🛠️ Soil fix advisor", "🌡️ Climate stress test", "🧬 Crop families",
     "📊 Data explorer", "🧪 Model performance", "ℹ️ About"])


def family_name(crop: str) -> str | None:
    if not families:
        return None
    return families["names"][families["family_of"][crop]]


def family_traits(crop: str) -> str:
    return families.get("descriptions", {}).get(families["family_of"][crop], "")


def family_members(crop: str) -> list[str]:
    f = families["family_of"][crop]
    return [c for c, g in families["family_of"].items() if g == f]

defaults = df[df[TARGET] == "rice"][FEATURES].median()
limits = df[FEATURES].agg(["min", "max"])
STEPS = {"N": 1, "P": 1, "K": 1, "temperature": 0.1, "humidity": 0.1, "ph": 0.1, "rainfall": 1.0}
BOUNDS = {}
for _f in FEATURES:
    _lo, _hi = float(limits.loc["min", _f]), float(limits.loc["max", _f])
    if STEPS[_f] == 1:
        BOUNDS[_f] = (int(_lo), int(_hi))
    else:  # widen to whole 0.1 steps so the limits are valid widget values
        BOUNDS[_f] = (round(_lo // 0.1 * 0.1, 1), round(-(-_hi // 0.1) * 0.1, 1))

# Field values live in session state so the weather lookup can fill them in.
for _f in FEATURES:
    if f"in_{_f}" not in st.session_state:
        v = defaults[_f]
        st.session_state[f"in_{_f}"] = int(round(v)) if STEPS[_f] == 1 else round(float(v), 1)


def field_input(feat: str):
    """A number box for one feature, limited to the range seen in the dataset."""
    label, unit = FEATURE_INFO[feat]
    text = f"{label} ({unit})" if unit else label
    lo, hi = BOUNDS[feat]
    if STEPS[feat] == 1:
        return st.number_input(text, lo, hi, step=1, key=f"in_{feat}")
    return st.number_input(text, lo, hi, step=float(STEPS[feat]), format="%.1f",
                           key=f"in_{feat}")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_places(name: str):
    return find_places(name)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_season(lat: float, lon: float, start_month: int, months: int):
    return season_climate(lat, lon, start_month, months)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


# ---------------- Recommendation ----------------
with tab_rec:
    st.subheader("1. Enter your field conditions")

    with st.expander("📍 Fill the climate from your location (optional)"):
        st.caption("Looks up last season's real weather for your town from the free "
                   "Open-Meteo service: average temperature and humidity, and total rainfall "
                   "over the growing season you choose. Needs an internet connection.")
        w1, w2, w3 = st.columns([2, 1.2, 1])
        place_name = w1.text_input("Town or village", placeholder="e.g. Coimbatore")
        start_month = w2.selectbox("Season starts in", range(1, 13), index=5,
                                   format_func=lambda m: MONTHS[m - 1])
        n_months = w3.number_input("Season length (months)", 2, 8, 4)
        places, failed = [], False
        if place_name.strip():
            try:
                places = cached_places(place_name.strip())
            except Exception as e:  # network or service error
                failed = True
                st.error(f"Couldn't reach the weather service: {e}")
            if not places and not failed:
                st.warning("No place found with that name. Try a nearby town.")
        if places:
            pick = st.selectbox("Matching places", range(len(places)),
                                format_func=lambda i: places[i]["label"])
            if st.button("☁️ Use this place's weather", type="secondary"):
                p = places[pick]
                try:
                    with st.spinner("Fetching last season's weather…"):
                        c = cached_season(p["lat"], p["lon"], start_month, int(n_months))
                    clipped = []
                    for f in ("temperature", "humidity", "rainfall"):
                        lo, hi = BOUNDS[f]
                        v = min(max(c[f], lo), hi)
                        if v != c[f]:
                            clipped.append(f"{FEATURE_INFO[f][0].lower()} {c[f]} → {v}")
                        st.session_state[f"in_{f}"] = round(float(v), 1)
                    st.session_state["_weather_note"] = (
                        f"Filled from {p['label']}, {c['start']:%d %b %Y} – {c['end']:%d %b %Y} "
                        f"({c['days']} days): {c['temperature']} °C, {c['humidity']}% humidity, "
                        f"{c['rainfall']} mm rain."
                        + (f" Clipped to the dataset's range: {', '.join(clipped)}." if clipped else ""))
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't fetch the weather: {e}")
        st.caption("⚠️ The dataset doesn't say over what period its rainfall was measured, so "
                   "season totals are the closest match, not an exact one.")

    if st.session_state.get("_weather_note"):
        st.success(st.session_state["_weather_note"], icon="📍")
    st.caption("Type your own soil-test and weather values (pre-filled with a typical rice "
               "field), then press **Recommend crop**.")
    inputs = {}
    with st.form("field_conditions"):
        soil, climate = st.columns(2)
        with soil:
            st.markdown("**🧪 Soil (from a soil test)**")
            for feat in ("N", "P", "K", "ph"):
                inputs[feat] = field_input(feat)
        with climate:
            st.markdown("**🌦️ Climate**")
            for feat in ("temperature", "humidity", "rainfall"):
                inputs[feat] = field_input(feat)
        st.form_submit_button("🌱 Recommend crop", type="primary", width="stretch")

    st.subheader("2. Recommendation")
    proba = predict(version, tuple(inputs[f] for f in FEATURES))
    ranking = (pd.DataFrame({"crop": model.classes_, "confidence": proba})
               .sort_values("confidence", ascending=False).reset_index(drop=True))
    best = ranking.iloc[0]

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown(f"<div style='font-size:64px;line-height:1'>{CROP_EMOJI.get(best.crop, '🌱')}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"## {crop_name(best.crop)}")
        st.metric("Model confidence", f"{best.confidence:.1%}")
        if best.confidence < 0.4:
            st.warning("Several crops suit these conditions about equally well. Treat the top "
                       "few as a shortlist rather than one answer.")
    with c2:
        st.subheader("Top 5 suitable crops")
        top = ranking.head(5).copy()
        top["crop"] = top.crop.map(crop_label)
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"crop": "Crop", "confidence": st.column_config.ProgressColumn(
                         "Confidence", format="percent", min_value=0, max_value=1)})

    if families:
        fam_crops = family_members(best.crop)
        fam_share = ranking[ranking.crop.isin(fam_crops)].confidence.sum()
        st.info(f"**Crop family: {family_name(best.crop)}** ({family_traits(best.crop)}, "
                f"compared with the average of all {metrics['n_crops']} crops). Members: "
                f"{', '.join(crop_name(c) for c in fam_crops)}. The model gives this family "
                f"{fam_share:.0%} in total, so even if the exact crop is uncertain, the family "
                "is a stronger signal (see the 🧬 Crop families tab).", icon="🧬")

    st.subheader("🧠 Why this crop? (explainable AI)")
    # Exact Shapley values need ~8,000 model predictions, so they run on request (then
    # stay cached for this field) instead of slowing every click.
    if st.toggle("Explain the recommendation", key="show_why",
                 help="Computes exact Shapley values: a few seconds on the free server."):
        explain_crop = st.selectbox("Explain the model's score for",
                                    list(ranking.crop.head(5)),
                                    format_func=crop_label, key="explain_crop")
        with st.spinner("Computing Shapley values…"):
            contrib, base = explain(version, tuple(inputs[f] for f in FEATURES))
        col = list(model.classes_).index(explain_crop)
        phi = contrib[explain_crop]
        pred_p = ranking.set_index("crop").confidence[explain_crop]
        shap_df = pd.DataFrame({
            "input": [f"{FEATURE_INFO[f][0]} = {inputs[f]:g} {FEATURE_INFO[f][1]}".strip()
                      for f in FEATURES],
            "contribution": phi.values * 100,
        }).sort_values("contribution", key=abs, ascending=False)
        shap_df["direction"] = shap_df.contribution.map(
            lambda v: "raises" if v >= 0 else "lowers")
        chart = (alt.Chart(shap_df)
                 .mark_bar(cornerRadiusEnd=3)
                 .encode(x=alt.X("contribution:Q",
                                 title="Effect on this crop's score (percentage points)"),
                         y=alt.Y("input:N", sort=None, title=None,
                                 axis=alt.Axis(labelLimit=260, labelFontSize=12)),
                         color=alt.Color("direction:N", title=None,
                                         scale=alt.Scale(domain=["raises", "lowers"],
                                                         range=["#2e7d32", "#c62828"])),
                         tooltip=["input", alt.Tooltip("contribution:Q", format="+.1f")])
                 .properties(height=260))
        st.altair_chart(chart, width="stretch")
        top_up = shap_df[shap_df.contribution > 0].head(2).input.tolist()
        top_down = shap_df[shap_df.contribution < 0].head(1).input.tolist()
        st.caption(
            f"An average crop profile scores {crop_name(explain_crop)} at {base[col]:.1%}. Your "
            f"field's values move it to {pred_p:.1%}"
            + (f"; the biggest boosts come from {' and '.join(top_up)}" if top_up else "")
            + (f", while {top_down[0]} holds it back" if top_down else "")
            + ". These are exact Shapley values: each bar is the input's average effect across "
            "all 128 combinations of the 7 inputs, and the bars add up to the difference.")
    else:
        st.caption("Switch on to see which of your inputs pushed the model toward each crop.")

    st.subheader(f"Your field vs. typical {crop_name(best.crop).lower()} conditions")
    crop_rows = df[df[TARGET] == best.crop][FEATURES]
    compare = pd.DataFrame({
        # Fixed FEATURES order: the form collects inputs in a different (soil, climate) order
        "Your value": pd.Series(inputs)[FEATURES],
        "Crop average": crop_rows.mean().round(2),
        "Crop min": crop_rows.min().round(2),
        "Crop max": crop_rows.max().round(2),
    })
    compare["In range"] = [
        "✅" if compare.loc[f, "Crop min"] <= inputs[f] <= compare.loc[f, "Crop max"] else "⚠️"
        for f in FEATURES]
    compare.index = [FEATURE_INFO[f][0] for f in FEATURES]
    st.dataframe(compare, width="stretch")

FIELD_LINE = ("Using your field from the Recommend tab: "
              + " · ".join(f"{FEATURE_INFO[f][0]} {inputs[f]:g}" for f in FEATURES))

# ---------------- Soil fix advisor ----------------
with tab_fix:
    st.subheader("Want to grow a specific crop? See what your soil needs")
    st.caption("Pick the crop you'd like to grow. The app searches hundreds of possible soil "
               "adjustments, scores each with the model, and shows the smallest change to N, P, "
               "K and pH that makes your chosen crop the top recommendation. Your climate stays "
               "the same, since fertiliser and lime can change soil but not weather.")
    st.caption(FIELD_LINE)
    all_crops = sorted(model.classes_, key=crop_name)
    # Default to the field's #2 crop (one search instead of trying many to pick a demo).
    field_key = tuple(inputs[f] for f in FEATURES)
    suggested = ranking.crop.iloc[1]
    target = st.selectbox("Crop you want to grow", all_crops, index=all_crops.index(suggested),
                          format_func=crop_label)
    fix = cached_soil_fix(version, field_key, target)
    name = crop_name(target)

    if fix["status"] == "already":
        st.success(f"{name} is already the top recommendation for your field "
                   f"({fix['current_prob']:.0%} confidence). No soil changes needed.", icon="✅")
    else:
        changed = any(fertiliser_advice(f, d) for f, d in fix["changes"].items())
        if fix["status"] == "top":
            st.success(f"With the changes below, **{name}** moves from #{fix['current_rank']} to "
                       f"**#1** ({fix['current_prob']:.0%} → {fix['new_prob']:.0%} confidence).",
                       icon="🛠️")
        elif fix["status"] == "top3" and changed and fix["new_rank"] == 1:
            st.warning(f"Soil changes can lift {name} to #1, but only to {fix['new_prob']:.0%} "
                       "confidence: the model doesn't consider any crop a good fit for this "
                       "field after the change. Treat this as a weak suggestion.", icon="🛠️")
        elif fix["status"] == "top3" and changed:
            st.warning(f"Soil changes alone can't make {name} a confident top pick, but they move "
                       f"it from #{fix['current_rank']} to #{fix['new_rank']} "
                       f"({fix['current_prob']:.0%} → {fix['new_prob']:.0%}).", icon="🛠️")
        elif fix["status"] == "top3":
            st.info(f"{name} is already #{fix['current_rank']} for your field, and no soil change "
                    "moves it higher. The difference comes from the climate.", icon="ℹ️")
        else:
            st.error(f"No realistic soil change makes {name} suitable for this field. The best "
                     f"reachable is #{fix['new_rank']} ({fix['new_prob']:.0%}).", icon="⛔")

        if changed:
            table = pd.DataFrame({
                "Now": [inputs[f] for f in SOIL],
                "Target": [fix["new_soil"][f] for f in SOIL],
                "Change": [fix["changes"][f] for f in SOIL],
            }, index=[f"{FEATURE_INFO[f][0]}" + (f" ({FEATURE_INFO[f][1]})" if FEATURE_INFO[f][1] else "")
                      for f in SOIL])
            st.dataframe(table.style.format("{:+.1f}", subset=["Change"])
                         .format("{:g}", subset=["Now", "Target"]), width="stretch")
            st.markdown("**What to do**")
            for f, d in fix["changes"].items():
                advice = fertiliser_advice(f, d)
                if advice:
                    st.markdown(f"- {advice}")

        if fix.get("climate_issues"):
            lines = [f"- **{FEATURE_INFO[c['feature']][0]}**: yours is {c['yours']:g}, but "
                     f"{name.lower()} in the dataset grows with {c['low']:.1f}–{c['high']:.1f} "
                     f"{FEATURE_INFO[c['feature']][1]}" for c in fix["climate_issues"]]
            st.warning(f"**Climate check:** your climate is outside {name.lower()}'s usual range. "
                       "Soil changes can't fix this, so treat the result with caution "
                       "(irrigation or drainage may help with rainfall).\n\n" + "\n".join(lines),
                       icon="🌦️")
    st.caption("Amounts are rough guides from the model, not an agronomic prescription. "
               "Confirm with a local soil-testing lab or agricultural extension office.")

# ---------------- Climate stress test ----------------
with tab_climate:
    st.subheader("Will your shortlist still work in a warmer, drier or wetter year?")
    st.caption("Re-scores your top 5 crops in 12 scenarios: warming of +0 to +3 °C combined "
               "with 20% less, the same, or 20% more rainfall. Pick a crop that stays suitable "
               "across scenarios, not just the best one today.")
    st.caption(FIELD_LINE)
    shortlist = list(ranking.crop.head(5))
    probs, ranks = cached_stress_test(version, tuple(inputs[f] for f in FEATURES),
                                      tuple(shortlist))
    n_first = (ranks == 1).sum(axis=1)          # scenarios where the crop is the top pick
    worst = probs.min(axis=1)
    resilient = max(shortlist, key=lambda c: (n_first[c], worst[c]))
    today_best = ranking.crop.iloc[0]
    n = probs.shape[1]

    k1, k2 = st.columns(2)
    k1.metric("Best today", crop_label(today_best),
              f"#1 in {n_first[today_best]} of {n} scenarios · worst case {worst[today_best]:.0%}",
              delta_color="off")
    k2.metric("Most resilient", crop_label(resilient),
              f"#1 in {n_first[resilient]} of {n} scenarios · worst case {worst[resilient]:.0%}",
              delta_color="off")
    if resilient != today_best:
        st.info(f"{crop_name(resilient)} holds up better than {crop_name(today_best)} if the "
                "climate shifts. Worth considering if you expect hotter or more erratic seasons.",
                icon="💡")
    lost = [s for s in probs.columns if ranks.loc[today_best, s] != 1]
    if lost:
        st.warning(f"{crop_name(today_best)} stops being the top pick in {len(lost)} of {n} "
                   f"scenarios: {', '.join(lost)}.", icon="⚠️")

    st.markdown("**Model confidence for each crop under each scenario**")
    show = probs.T.copy()
    show.columns = [crop_label(c) for c in show.columns]
    show.insert(0, "Top crop", [crop_label(c) for c in ranks.idxmin(axis=0)])
    show.index.name = "Scenario"
    st.dataframe(show.style.format("{:.0%}", subset=show.columns[1:])
                 .background_gradient(cmap="Greens", vmin=0, vmax=1, subset=show.columns[1:]),
                 width="stretch", height=38 * (n + 1))
    st.caption("Top crop = highest-ranked crop of your shortlist in that scenario. Humidity is "
               "kept the same. Warming beyond the dataset's range is extrapolated by the model "
               "and is less reliable.")

# ---------------- Crop families (unsupervised learning) ----------------
with tab_fam:
    st.subheader("Which crops need similar conditions? (unsupervised learning)")
    if not families:
        st.info("Run `python train.py` to build the crop families.")
    else:
        fm = metrics.get("families", {})
        st.caption("Each crop's average soil and climate profile is standardised, then grouped "
                   "with Ward hierarchical clustering, without using the crop labels' meaning. "
                   "The number of families is chosen automatically by the silhouette score. "
                   "PCA (principal component analysis) then squeezes the 7 inputs into 2 "
                   "dimensions so the crops can be drawn on a map.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Families found", families["k"])
        m2.metric("Silhouette score", f"{families['silhouette'][families['k']]:.2f}")
        if metrics.get("family_accuracy") is not None:
            m3.metric("Right family, even when the crop is wrong",
                      f"{metrics['family_accuracy']:.0%}",
                      f"vs {best_res['test_accuracy']:.0%} exact crop", delta_color="off")

        cent = families["centroids"].copy()
        cent["crop"] = [crop_name(c) for c in cent.index]
        cent["family"] = [family_name(c) for c in cent.index]
        field_z = families["scaler"].transform(pd.DataFrame([inputs])[FEATURES])
        fx, fy = families["pca"].transform(pd.DataFrame(field_z, columns=FEATURES))[0]
        you = pd.DataFrame({"pc1": [fx], "pc2": [fy], "crop": ["Your field"]})
        ev = families["explained_variance"]
        load = families["loadings"]

        def axis_words(pc):
            top = load[pc].abs().sort_values(ascending=False).index[:2]
            return ", ".join(f"{FEATURE_INFO[f][0].lower()} {'+' if load.loc[f, pc] > 0 else '−'}"
                             for f in top)

        base_chart = alt.Chart(cent).encode(
            x=alt.X("pc1:Q", title=f"PC1 ({ev[0]:.0%} of variation: {axis_words('pc1')})"),
            y=alt.Y("pc2:Q", title=f"PC2 ({ev[1]:.0%} of variation: {axis_words('pc2')})"))
        fam_scale = alt.Scale(scheme="tableau10")
        points = base_chart.mark_circle(size=110, opacity=0.85).encode(
            color=alt.Color("family:N", title="Family", scale=fam_scale,
                            legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=["crop", "family"])
        labels = base_chart.mark_text(dx=7, align="left", fontSize=10).encode(
            text="crop", color=alt.Color("family:N", scale=fam_scale, legend=None))
        star = alt.Chart(you).mark_point(shape="diamond", size=320, filled=True,
                                         color="#ff6f00", stroke="white",
                                         strokeWidth=1.5).encode(
            x="pc1:Q", y="pc2:Q", tooltip=["crop"])
        star_lbl = alt.Chart(you).mark_text(dy=-16, fontWeight="bold", fontSize=13,
                                            color="#ff6f00").encode(
            x="pc1:Q", y="pc2:Q", text="crop")
        st.altair_chart((points + labels + star + star_lbl).properties(height=560)
                        .interactive(), width="stretch")
        st.caption("Orange ◆ = your field from the Recommend tab. Crops close together need similar "
                   "conditions, which is why the model confuses them. Scroll to zoom, drag to pan.")

        rows = []
        for f_id, name in families["names"].items():
            members = [c for c, g in families["family_of"].items() if g == f_id]
            rows.append({"Family": name,
                         "Stands out for": families.get("descriptions", {}).get(f_id, ""),
                         "Crops": len(members),
                         "Members": ", ".join(crop_name(c) for c in members)})
        st.dataframe(pd.DataFrame(rows).sort_values("Crops", ascending=False),
                     hide_index=True, width="stretch")
        st.caption("Each family is named after its most typical crop (the one closest to the "
                   "family's centre). 'Stands out for' lists the two inputs that most set the "
                   f"family apart from the average of all {metrics['n_crops']} crops.")
        if (FIG_DIR / "crop_dendrogram.png").exists():
            with st.expander("Dendrogram (how the clustering merged crops step by step)"):
                st.image(str(FIG_DIR / "crop_dendrogram.png"), width="stretch")

# ---------------- Data explorer ----------------
with tab_data:
    st.subheader("Dataset overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Records", f"{len(df):,}")
    m2.metric("Crops", df[TARGET].nunique())
    m3.metric("Missing values", int(df.isna().sum().sum()))
    st.dataframe(df.head(50), width="stretch", height=250)

    st.subheader("Descriptive statistics")
    st.dataframe(df[FEATURES].describe().round(2), width="stretch")

    st.subheader("Statistical feature analysis (ANOVA F-test)")
    st.caption("A higher F-score means the feature differs more between crops, "
               "so it is more useful for telling crops apart.")
    anova = pd.DataFrame(metrics["anova"])
    st.bar_chart(anova.set_index("feature")["F_score"])

    st.subheader("Feature by crop")
    feat = st.selectbox("Feature", FEATURES, format_func=lambda f: FEATURE_INFO[f][0])
    st.bar_chart(df.groupby(TARGET)[feat].mean().sort_values(), horizontal=True,
                 height=max(400, 14 * df[TARGET].nunique()))

    g1, g2 = st.columns(2)
    for col, name in ((g1, "correlation.png"), (g2, "crop_profiles.png")):
        if (FIG_DIR / name).exists():
            col.image(str(FIG_DIR / name), width="stretch")
    if (FIG_DIR / "distributions.png").exists():
        st.image(str(FIG_DIR / "distributions.png"), width="stretch")

# ---------------- Model performance ----------------
with tab_model:
    st.subheader("Algorithm comparison")
    order = ["model", "test_accuracy", "test_top3", "test_top5", "test_f1_macro",
             "cv_mean", "cv_std", "train_seconds"]
    res = pd.DataFrame(metrics["results"])
    res = res[[c for c in order if c in res.columns]].rename(columns={
        "model": "Model", "cv_mean": "CV accuracy (mean)", "cv_std": "CV std",
        "test_accuracy": "Top-1 accuracy", "test_top3": "Top-3 accuracy",
        "test_top5": "Top-5 accuracy", "test_f1_macro": "F1 (macro)",
        "train_seconds": "Time (s)"})
    score_cols = [c for c in res.columns if c not in ("Model", "Time (s)")]
    st.dataframe(res.style.format({**{c: "{:.4f}" for c in score_cols}, "Time (s)": "{:.1f}"})
                 .highlight_max(subset=["Top-1 accuracy"],
                                props="background-color:#2e7d32;color:white;font-weight:bold"),
                 hide_index=True, width="stretch")
    st.caption("5-fold stratified cross-validation on the 80% training split; other scores on "
               "the held-out 20% test split. Top-3 / top-5 = the true crop is among the "
               "model's 3 / 5 highest-ranked crops.")

    if metrics.get("tuning"):
        st.subheader("Hyperparameter tuning and ensembles")
        st.caption("RandomizedSearchCV tunes three tree ensembles (3-fold CV inside the search), "
                   "then a soft-voting ensemble averages their probabilities and a stacking "
                   "ensemble trains a logistic-regression 'meta-model' on their outputs. "
                   f"The app's model is chosen by **{metrics.get('selection', 'cross-validation')}**; "
                   "the test set is only used afterwards to report the final score.")
        tun = pd.DataFrame(metrics["tuning"])
        tun["selected"] = tun.model.eq(metrics["best_model"]).map({True: "✅", False: ""})
        cols = ["selected", "model", "cv_mean", "cv_std", "test_accuracy", "test_top3",
                "test_top5", "memory_mb", "seconds"]
        tun = tun[[c for c in cols if c in tun.columns]].rename(columns={
            "selected": "", "model": "Model", "cv_mean": "CV accuracy", "cv_std": "CV std",
            "test_accuracy": "Test top-1", "test_top3": "Test top-3", "test_top5": "Test top-5",
            "memory_mb": "Memory (MB)", "seconds": "Time (s)"})
        st.dataframe(tun.style.format({**{c: "{:.4f}" for c in tun.columns[2:7]},
                                       "Memory (MB)": "{:.0f}", "Time (s)": "{:.1f}"}),
                     hide_index=True, width="stretch")
        with st.expander("Best hyperparameters found"):
            for t in metrics["tuning"]:
                if t["params"]:
                    st.markdown(f"**{t['model']}**: " + ", ".join(
                        f"`{k}={v}`" for k, v in t["params"].items()))

    if metrics.get("learning_curve"):
        st.subheader("Learning curve: would more data help?")
        lc = metrics["learning_curve"]
        lc_df = pd.DataFrame({"Training records": lc["train_size"] * 2,
                              "Accuracy": lc["train_mean"] + lc["val_mean"],
                              "Score": ["Training"] * len(lc["train_size"])
                                       + ["Cross-validation"] * len(lc["train_size"])})
        st.altair_chart(alt.Chart(lc_df).mark_line(point=True).encode(
            x="Training records:Q", y=alt.Y("Accuracy:Q", scale=alt.Scale(zero=False),
                                            axis=alt.Axis(format="%")),
            color=alt.Color("Score:N", title=None), tooltip=["Score", "Training records",
                                                             alt.Tooltip("Accuracy:Q", format=".1%")])
            .properties(height=300), width="stretch")
        gain = lc["val_mean"][-1] - lc["val_mean"][-2]
        gap = lc["train_mean"][-1] - lc["val_mean"][-1]
        st.caption(
            f"Going from {lc['train_size'][-2]:,} to {lc['train_size'][-1]:,} training records "
            f"changed cross-validation accuracy by {gain:+.1%}"
            + (", so the curve is levelling off: more records of the same kind help only a "
               "little, and new information (season, region) would help more."
               if gain < 0.015 else ", so more data is still helping noticeably.")
            + (f" Training accuracy ({lc['train_mean'][-1]:.0%}) is far above validation "
               f"accuracy ({lc['val_mean'][-1]:.0%}): the model memorises its training data, "
               "and overlapping crops limit what it can generalise." if gap > 0.1 else ""))

    if metrics.get("baseline_original") and metrics.get("group_accuracy"):
        st.subheader("Why accuracy dropped with more data")
        g, b = metrics["group_accuracy"], metrics["baseline_original"]
        b_best = max(b["results"], key=lambda r: r["test_accuracy"])
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Original {b['n_crops']} crops only ({b['n_rows']:,} rows)",
                  f"{b_best['test_accuracy']:.1%}")
        k2.metric(f"Same {g['n_original_crops']} crops, inside the {metrics['n_crops']}-crop model",
                  f"{g['original_crops']:.1%}")
        k3.metric(f"The {g['n_new_crops']} newly added crops", f"{g['new_crops']:.1%}")
        st.markdown(
            "Many added crops need almost the same soil and climate, so seven measurements "
            "can't separate them. The most common mix-ups:")
        errs = pd.DataFrame(metrics["errors"]).head(10)
        errs["actual"] = errs.actual.map(crop_label)
        errs["predicted"] = errs.predicted.map(crop_label)
        st.dataframe(errs.rename(columns={"actual": "Actual crop", "predicted": "Predicted as",
                                          "count": "Test records"}),
                     hide_index=True, width="stretch")

    if metrics.get("soil_experiment"):
        s = metrics["soil_experiment"]
        st.info(f"**Soil type experiment:** adding the dataset's soil-type column raises Random "
                f"Forest accuracy from {s['rf_without_soil']:.1%} to {s['rf_with_soil']:.1%}. "
                f"It is left out because {s['crops_with_one_soil_type']} of "
                f"{metrics['n_crops']} crops have exactly one soil type, so the column "
                "partly gives away the answer rather than describing the field.", icon="🧪")

    p1, p2 = st.columns(2)
    for col, name in ((p1, "model_comparison.png"), (p2, "feature_importance.png")):
        if (FIG_DIR / name).exists():
            col.image(str(FIG_DIR / name), width="stretch")
    if (FIG_DIR / "confusion_matrix.png").exists():
        st.image(str(FIG_DIR / "confusion_matrix.png"), width="stretch")

# ---------------- About ----------------
with tab_about:
    st.markdown(f"""
### About AgriAI
AgriAI is a machine-learning decision-support tool that suggests suitable crops
for a field from seven inputs. It covers {metrics['n_crops']} crops.

| Input | Meaning |
|---|---|
| N, P, K | Nitrogen, Phosphorus and Potassium content of the soil (kg/ha) |
| Temperature | Average temperature (°C) |
| Humidity | Relative humidity (%) |
| pH | Soil acidity / alkalinity |
| Rainfall | Rainfall (mm) |

**Pipeline:** data loading → statistical analysis (descriptive stats, correlation,
ANOVA F-test) → standard scaling → training 7 classifiers → 5-fold cross-validation →
selection of the best model on a held-out test set → deployment in this Streamlit app.

**Tech stack:** Python · Pandas · NumPy · scikit-learn · Matplotlib / Seaborn · Streamlit

*This is an academic project. Recommendations are suggestions, not agronomic advice.*
""")
