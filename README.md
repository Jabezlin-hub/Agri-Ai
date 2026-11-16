# 🌱 AgriAI — Machine-Learning Crop Recommender

A machine-learning system that recommends suitable crops for a field from soil nutrients
(**N, P, K**), **soil pH**, **temperature**, **humidity** and **rainfall**, trained on
**7,950 records across 64 crops**. It combines three kinds of machine learning:

| ML technique | What it does in AgriAI |
|---|---|
| **Supervised learning** (classification) | 7 algorithms compared, 3 tuned with `RandomizedSearchCV`, combined into a **stacking ensemble** chosen by 5-fold cross-validation (within a 300 MB memory budget for deployment) |
| **Unsupervised learning** (clustering + PCA) | Ward hierarchical clustering groups the 64 crops into **7 crop families**; PCA draws them on a 2-D map |
| **Explainable AI** (Shapley values) | For every recommendation, **exact Shapley values** show how much each input raised or lowered each crop's score |

**Tech stack:** Python · Pandas · NumPy · SciPy · scikit-learn · Altair/Matplotlib/Seaborn · Streamlit

---

## Headline results (1,590 held-out test records)

| | Value |
|---|---|
| Final model | Stacking ensemble (tuned Random Forest + Hist Gradient Boosting → logistic regression) |
| Right crop is the top pick | **65.41%** (vs 64.15% for an untuned Random Forest) |
| Right crop in the top 3 / top 5 | 85.97% / **93.14%** |
| Right **crop family** (of 7) | **96.35%** |
| Same pipeline on the classic 22-crop dataset | 99.55% |
| Model size | 14 MB file, ~216 MB in memory |

## App features

- **🔮 Recommend**: enter the field's values (or auto-fill the climate from a town), get the
  top crop, a 5-crop shortlist, the crop's **family**, and a **🧠 Why this crop?** chart of
  Shapley values.
- **🛠️ Soil fix advisor**: counterfactual search for the smallest N, P, K, pH change that
  makes a chosen crop the model's confident top pick, turned into fertiliser / lime actions.
- **🌡️ Climate stress test**: the shortlist re-scored under 12 warming × rainfall scenarios.
- **🧬 Crop families**: PCA map of all crops coloured by family, with the user's field.
- **📊 Data explorer**: descriptive statistics, ANOVA F-test, correlations, distributions.
- **🧪 Model performance**: 7-algorithm comparison, tuning and ensemble results, best
  hyperparameters, learning curve, 22- vs 64-crop comparison, confusions, soil-type test.

## Project structure

```
AgriAI/
├── app.py                 # Streamlit web application
├── train.py               # Full ML pipeline: analysis, training, tuning, ensembles, clustering
├── requirements.txt
├── src/
│   ├── data.py            # Dataset loading (+ synthetic fallback)
│   ├── ml.py              # Hyperparameter tuning, ensembles, learning curve, crop families
│   ├── explain.py         # Exact Shapley values for one prediction
│   ├── advisor.py         # Soil fix advisor (counterfactual search) + climate stress test
│   └── weather.py         # Open-Meteo geocoding + growing-season climate lookup
├── data/
│   ├── Crop_recommendation_data.csv  # Extended dataset, 7,950 rows, 64 crops (used)
│   └── Crop_recommendation.csv       # Original dataset, 2,200 rows, 22 crops (baseline)
├── models/
│   ├── crop_model.joblib     # Selected model (compressed)
│   ├── crop_families.joblib  # Clustering, scaler and PCA for the families map
│   └── metrics.json          # Every score, parameter and comparison
└── reports/
    ├── feature_stats.csv  # ANOVA F-test results
    └── figures/           # EDA, evaluation, learning curve and dendrogram plots
```

## Setup and run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ML pipeline (creates models/ and reports/; about 4-5 minutes on 16 cores)
python train.py

# 4. Launch the app (opens http://localhost:8501)
streamlit run app.py
```

> **Windows tip:** if `import sklearn` fails with *"The filename or extension is too long"*,
> the project folder's path is too deep. Move the project (or the `.venv`) to a shorter
> path such as `C:\projects\AgriAI`.

## Deploying to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository (the trained model is included, ~38 MB).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose
   **Create app → Deploy a public app from GitHub**.
3. Pick the repository and branch, set **Main file path** to `app.py`.
4. Under **Advanced settings**, choose **Python 3.12** (or 3.13). `requirements.txt` pins
   exact versions because the saved model only loads with the scikit-learn version it was
   trained with.
5. Deploy. The first build takes a few minutes; the app then gets a public
   `*.streamlit.app` URL. Free apps sleep after a period without visitors and wake when opened.

The model is chosen to fit the free tier's memory: Streamlit guarantees about 690 MB per app,
so `train.py` selects the best model by cross-validation **among models needing at most
300 MB**. If you retrain, commit the new `models/` files before redeploying.

## Datasets

| File | Rows | Crops | Source | Use |
|---|---|---|---|---|
| `Crop_recommendation_data.csv` | 7,950 | 64 | Kaggle, *Crop Recommendation Dataset* by Eniola Ajala (Apache 2.0) | Training the app's model |
| `Crop_recommendation.csv` | 2,200 | 22 | Kaggle, *Crop Recommendation Dataset* (Atharva Ingle) | Baseline comparison |

Both have N, P, K (kg/ha), temperature (°C), humidity (%), ph, rainfall (mm) and label;
the extended dataset also has `soil_type`. Checks on the extended dataset: no missing
values or duplicates; 100–250 records per crop; it contains all 22 original crops (1,471
original rows unchanged, 729 slightly altered); `soil_type` is **not** used as a feature
because 62 of 64 crops have exactly one soil type, so it partly encodes the answer.

## The machine-learning pipeline (`train.py`)

1. **Data checks and statistical analysis**: descriptive statistics, correlation matrix,
   one-way ANOVA F-test of each feature against the crop label.
2. **Split**: stratified 80/20 train/test split (6,360 / 1,590 records). The test set is not
   touched until the final model has been chosen.
3. **Baseline comparison**: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree,
   SVM, Random Forest, Gradient Boosting, each in a `StandardScaler` pipeline, scored with
   5-fold stratified cross-validation.
4. **Hyperparameter tuning**: `RandomizedSearchCV` (3-fold) for Random Forest (20 settings),
   Extra Trees (12) and Histogram Gradient Boosting (12). Forest sizes are capped (≤ 300
   trees, leaves ≥ 2 records) to keep the saved model small.
5. **Ensembles**: soft voting (average of the three tuned models' probabilities) and
   stacking (a logistic-regression meta-model trained on their out-of-fold probabilities).
6. **Model selection by cross-validation**: the candidate with the highest 5-fold CV
   accuracy on the training set is chosen; only then is it scored once on the test set.
7. **Learning curve**: CV accuracy when training on 10%–100% of the training data.
8. **Crop families**: each crop's mean profile is standardised; Ward hierarchical
   clustering groups them; the number of families (5–12) is chosen by silhouette score;
   each family is named after its medoid crop; PCA projects everything to 2-D.
9. **Comparisons**: accuracy on original vs new crops, the 22-crop dataset as a baseline,
   and a with/without `soil_type` experiment.

### Results

**Baseline algorithms** (5-fold CV on the training set, then the test set)

| Model | CV accuracy | Test top-1 | Top-3 | Top-5 |
|---|---|---|---|---|
| Random Forest | 0.6223 ± 0.0101 | 0.6415 | 0.8660 | 0.9352 |
| Gradient Boosting | 0.6091 ± 0.0110 | 0.6208 | 0.8516 | 0.9308 |
| Gaussian Naive Bayes | 0.5917 ± 0.0071 | 0.5962 | 0.8396 | 0.9239 |
| Support Vector Machine | 0.5660 ± 0.0101 | 0.5824 | 0.8264 | 0.9170 |
| Logistic Regression | 0.5358 ± 0.0054 | 0.5447 | 0.7887 | 0.8899 |
| Decision Tree | 0.5263 ± 0.0084 | 0.5472 | 0.5528 | 0.5805 |
| K-Nearest Neighbors | 0.5115 ± 0.0076 | 0.5182 | 0.7220 | 0.7862 |

**Tuning and ensembles** (selection by CV; test scores shown for transparency)

| Model | CV accuracy | Memory | Test top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| Stacking ensemble (RF + ET + HGB) | 0.6412 ± 0.0093 | 658 MB | 0.6597 | 0.8648 | 0.9321 |
| Soft-voting ensemble (RF + ET + HGB) | 0.6409 ± 0.0121 | 658 MB | 0.6604 | 0.8805 | 0.9472 |
| **Stacking ensemble (RF + HGB)** (selected) | **0.6404 ± 0.0101** | **216 MB** | **0.6541** | 0.8597 | 0.9314 |
| Hist Gradient Boosting (tuned) | 0.6303 ± 0.0117 | 4 MB | 0.6465 | 0.8780 | 0.9503 |
| Random Forest (tuned) | 0.6277 ± 0.0096 | 212 MB | 0.6434 | 0.8629 | 0.9365 |
| Random Forest (default settings) | 0.6223 ± 0.0101 | 378 MB | 0.6415 | 0.8660 | 0.9352 |
| Extra Trees (tuned) | 0.6140 ± 0.0048 | 442 MB | 0.6239 | 0.8535 | 0.9352 |

The three-model ensembles score highest on CV but need ~660 MB of memory, close to the
~690 MB that Streamlit Community Cloud guarantees. The selection rule is therefore *best CV
accuracy among models needing ≤ 300 MB*, which picks the RF + HGB stacking ensemble: only
0.08 points of CV accuracy behind, at a third of the memory. Tuning plus stacking adds 1.8
points of CV accuracy and 1.3 points of test top-1 accuracy over the default Random Forest.
The top ensembles are within one standard deviation of each other, so choosing among them
on test scores would only fit noise in the test set; selecting by **cross-validated top-5
accuracy**, since the app shows a shortlist, is a sensible next step.

**Learning curve.** CV accuracy rises from 50.1% (508 training records) to 62.7% (5,088), but
the last step adds only +0.6 points, while training accuracy stays around 99%. More records of
the same kind would help a little; new information (season, region) would help more.

**Why accuracy is lower than on the 22-crop dataset.** The same pipeline scores 99.55% there.
Inside the 64-crop model the original 22 crops still score 92.6%, but the 42 added crops score
51.8%: many need almost the same conditions. The most common confusions are quinoa → lentil,
parsnip → swede, cranberry → blueberry and peas ↔ faba bean.

**Crop families.** 7 families (silhouette 0.30): Jute group (13 warm-climate crops incl.
rice, maize, cotton), Mothbeans group (7 pulses), Flax group (9 cool-season grains), Cabbage
group (6 brassicas + potato), Beetroot group (26 high-N, high-K vegetables and fruit),
Cranberry group (2 berries) and Grapes (alone). **The exact crop is right 65% of the time,
but the right family 96.4% of the time**: most mistakes are between close relatives.

**Soil-type experiment.** Adding `soil_type` raises Random Forest accuracy from 64.2% to
77.1%, but only because it is assigned per crop, so it is excluded.

### Explainable AI: exact Shapley values (`src/explain.py`)

With 7 inputs there are only 2⁷ = 128 subsets of inputs, so Shapley values are computed
exactly. For each subset, the model's average probability is taken over 64 background
profiles (each crop's median field) with the field's own values for the inputs in the
subset. Each input's Shapley value is its weighted average marginal effect over all subsets,
so **base value + all contributions = the prediction** (verified to 10⁻¹⁵). Example: for a
typical rice field, rice rises from 1.6% (average profile) to 95.8%, mainly from rainfall
(+29 points), humidity (+25), potassium (+18) and temperature (+17).

### Soil fix advisor (`src/advisor.py`)

Climate is held fixed; candidate soils move the field's N, P, K, pH 25–100% of the way toward
each real profile of the target crop (up to ~1,000 candidates). The model scores them all;
the cheapest change (in standard deviations, decreases counted double) that makes the crop #1
with at least 25% confidence is chosen. Example: on a typical chickpea field, adding about
16 kg/ha of nitrogen lifts lentil from 12% to 77%.

## Limitations and future work

- The data looks partly generated (sharp, evenly filled per-crop ranges) and its rainfall
  period is undocumented, so weather auto-fill is approximate.
- Seven inputs cannot separate closely related crops; season, region and growing period
  are missing.
- Fertiliser amounts are rules of thumb; confirm with a soil lab.
- Possible extensions: an anomaly detector that flags fields unlike the training data,
  probability calibration, season/region features, regional languages, cloud deployment.

---
*Academic project for the Machine Learning course. Recommendations are suggestions,
not agronomic advice.*
