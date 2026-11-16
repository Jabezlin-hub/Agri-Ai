"""Dataset loading for AgriAI.

Sources, in order of preference:
1. data/Crop_recommendation_data.csv - extended Kaggle dataset (7,950 rows, 64 crops,
   eniolaajala, Apache 2.0). It also has a soil_type column, which is only used in a
   side experiment because it is assigned per crop rather than measured.
2. data/Crop_recommendation.csv - the original Kaggle dataset (2,200 rows, 22 crops).
3. A synthetic stand-in with the original schema, generated if neither file exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTENDED_CSV = DATA_DIR / "Crop_recommendation_data.csv"
REAL_CSV = DATA_DIR / "Crop_recommendation.csv"
SYNTH_CSV = DATA_DIR / "crop_data_synthetic.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# Display name and unit for each feature. Input limits come from the loaded data.
FEATURE_INFO = {
    "N": ("Nitrogen", "kg/ha"),
    "P": ("Phosphorus", "kg/ha"),
    "K": ("Potassium", "kg/ha"),
    "temperature": ("Temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "ph": ("Soil pH", ""),
    "rainfall": ("Rainfall", "mm"),
}

# Typical growing ranges per crop: (N, P, K, temp, humidity, pH, rainfall)
CROP_RANGES = {
    "rice":        [(60, 99), (35, 60), (35, 45), (20, 27), (80, 85), (5.0, 7.9), (183, 299)],
    "maize":       [(60, 100), (35, 60), (15, 25), (18, 27), (55, 75), (5.5, 7.0), (60, 110)],
    "chickpea":    [(20, 60), (55, 80), (75, 85), (17, 21), (14, 20), (6.0, 9.0), (65, 95)],
    "kidneybeans": [(0, 40), (55, 80), (15, 25), (15, 25), (18, 25), (5.5, 6.0), (60, 150)],
    "pigeonpeas":  [(0, 40), (55, 80), (15, 25), (18, 37), (30, 70), (4.5, 7.5), (90, 200)],
    "mothbeans":   [(0, 40), (35, 60), (15, 25), (24, 32), (40, 65), (3.5, 10.0), (30, 75)],
    "mungbean":    [(0, 40), (35, 60), (15, 25), (27, 30), (80, 90), (6.2, 7.2), (36, 60)],
    "blackgram":   [(20, 60), (55, 80), (15, 25), (25, 35), (60, 70), (6.5, 7.8), (60, 75)],
    "lentil":      [(0, 40), (55, 80), (15, 25), (18, 30), (60, 70), (5.9, 7.8), (35, 55)],
    "pomegranate": [(0, 40), (5, 30), (35, 45), (18, 25), (85, 95), (5.5, 7.2), (102, 113)],
    "banana":      [(80, 120), (70, 95), (45, 55), (25, 30), (75, 85), (5.5, 6.5), (90, 120)],
    "mango":       [(0, 40), (15, 40), (25, 35), (27, 36), (45, 55), (4.5, 7.0), (89, 101)],
    "grapes":      [(0, 40), (120, 145), (195, 205), (8, 42), (80, 84), (5.5, 6.5), (65, 75)],
    "watermelon":  [(80, 120), (5, 30), (45, 55), (24, 27), (80, 90), (6.0, 7.0), (40, 60)],
    "muskmelon":   [(80, 120), (5, 30), (45, 55), (27, 30), (90, 95), (6.0, 6.8), (20, 30)],
    "apple":       [(0, 40), (120, 145), (195, 205), (21, 24), (90, 95), (5.5, 6.5), (100, 125)],
    "orange":      [(0, 40), (5, 30), (5, 15), (10, 35), (90, 95), (6.0, 8.0), (100, 120)],
    "papaya":      [(31, 70), (46, 70), (45, 55), (23, 44), (90, 95), (6.5, 7.0), (40, 250)],
    "coconut":     [(0, 40), (5, 30), (25, 35), (25, 30), (90, 100), (5.5, 6.5), (130, 225)],
    "cotton":      [(100, 140), (35, 60), (15, 25), (22, 26), (75, 85), (5.8, 8.0), (60, 100)],
    "jute":        [(60, 100), (35, 60), (35, 45), (23, 27), (70, 90), (6.0, 7.5), (150, 200)],
    "coffee":      [(80, 120), (15, 40), (25, 35), (23, 28), (50, 70), (6.0, 7.5), (115, 200)],
}


def generate_synthetic(n_per_crop: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a 22-crop dataset (100 rows each = 2,200 rows) from CROP_RANGES."""
    rng = np.random.default_rng(seed)
    rows = []
    for crop, ranges in CROP_RANGES.items():
        cols = []
        for i, (lo, hi) in enumerate(ranges):
            vals = rng.uniform(lo, hi, n_per_crop)
            if i < 3:  # N, P, K are whole numbers
                vals = np.round(vals).astype(int)
            cols.append(vals)
        block = pd.DataFrame(dict(zip(FEATURES, cols)))
        block[TARGET] = crop
        rows.append(block)
    df = pd.concat(rows, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_name). Prefers the extended Kaggle CSV."""
    if EXTENDED_CSV.exists():
        return pd.read_csv(EXTENDED_CSV), "Crop_recommendation_data.csv (extended dataset)"
    if REAL_CSV.exists():
        return pd.read_csv(REAL_CSV), "Crop_recommendation.csv (real dataset)"
    if not SYNTH_CSV.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_synthetic().to_csv(SYNTH_CSV, index=False)
    return pd.read_csv(SYNTH_CSV), "crop_data_synthetic.csv (synthetic stand-in)"
