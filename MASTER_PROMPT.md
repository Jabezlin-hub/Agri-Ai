# AgriAI — Master Prompt for the Project Report

> **How to use:** copy everything below the line into ChatGPT, Claude or Gemini. First paste
> your faculty's report format (chapter list, page limit, citation style) into the
> `[FACULTY FORMAT]` slot. Every number below comes from the trained model's `metrics.json`,
> so the report will match the app exactly.

---

You are helping me write the academic project report for my **Machine Learning** course
project, **"AgriAI — Machine-Learning Crop Recommender"**. Write the report using **only the
facts below**. Do not invent numbers, datasets, citations or features. If my faculty format
asks for something not covered here (for example a name, roll number, guide name or
college), leave a clearly marked placeholder like `[STUDENT NAME]`.

**[FACULTY FORMAT]:** _<paste your faculty's required chapters / structure / length / style here>_

If no format is given, use: Title page → Certificate → Acknowledgement → Abstract → Table of
contents → 1 Introduction → 2 Literature review → 3 Problem statement & objectives →
4 Dataset → 5 Methodology → 6 System design & implementation → 7 Results & discussion →
8 Application walkthrough → 9 Limitations & future work → 10 Conclusion → References →
Appendix. Write in formal academic English, past tense for work done, with tables where
numbers are compared. Suggest where each figure (listed in section M) should go.

## A. Project summary

- **Title:** AgriAI — Machine-Learning Crop Recommender
- **One line:** a web app that recommends suitable crops for a field from 7 measurements:
  soil nitrogen (N), phosphorus (P), potassium (K) in kg/ha, soil pH, temperature (°C),
  relative humidity (%) and rainfall (mm).
- **Problem:** farmers often choose crops by habit or guesswork. A wrong choice for the soil
  and climate lowers yield and wastes fertiliser and water. Soil-test values exist but are
  hard to turn into a decision.
- **Solution:** a supervised multi-class classifier (64 crops) that gives a ranked shortlist
  with probabilities. It is combined with unsupervised crop families, explainable AI (exact
  Shapley values), a counterfactual "soil fix" advisor, a climate-change stress test and
  weather auto-fill by location.
- **ML concepts covered:** supervised classification, 7-algorithm comparison, k-fold
  cross-validation, hyperparameter tuning (RandomizedSearchCV), ensemble learning (soft
  voting, stacking), model selection under a resource constraint, learning curves,
  hierarchical clustering, silhouette analysis, PCA, explainable AI (Shapley values),
  counterfactual search, top-k accuracy, statistical analysis (ANOVA F-test).
- **Live app:** https://agriai-crop-recommender.streamlit.app
- **Source code:** https://github.com/Jabezlin-hub/Agri-Ai
- **Tech stack:** Python 3.12 (cloud) / 3.14 (local), pandas 3.1.1, NumPy 2.5.4, SciPy 1.18.1,
  scikit-learn 1.9.1, joblib 1.6.0, Matplotlib 3.11.2, Seaborn 0.13.2, Altair 6.3.0,
  Streamlit 1.67.0, Open-Meteo API (weather), Git + GitHub, Streamlit Community Cloud.

## B. Objectives

1. Build an accurate multi-class crop classifier from soil and climate data.
2. Compare several ML algorithms fairly with cross-validation and pick the best one.
3. Improve on the baseline with hyperparameter tuning and ensemble learning.
4. Discover natural crop groups with unsupervised learning.
5. Make each prediction explainable to a non-expert.
6. Give actionable advice (soil corrections, climate risk), not just a label.
7. Deploy as a free, public web application.

## C. Datasets

| | Extended dataset (used for the model) | Original dataset (baseline) |
|---|---|---|
| File | `Crop_recommendation_data.csv` | `Crop_recommendation.csv` |
| Source | Kaggle, "Crop Recommendation Dataset" by Eniola Ajala (Apache 2.0) | Kaggle, "Crop Recommendation Dataset" by Atharva Ingle |
| Rows / crops | **7,950 / 64** | 2,200 / 22 |
| Columns | N, P, K, temperature, humidity, ph, rainfall, label, soil_type | N, P, K, temperature, humidity, ph, rainfall, label |

- Data quality: no missing values and no duplicates; 100–250 records per crop.
- The extended dataset contains all 22 original crops (1,471 original rows unchanged, 729
  slightly altered) plus 42 new crops.
- `soil_type` is **excluded** as a feature: 62 of the 64 crops have exactly one soil type, so
  it partly encodes the answer (data leakage). Experiment: Random Forest accuracy rises from
  64.2% to 77.1% with it, but only because of this leakage.
- The 64 crops: apple, artichoke, asparagus, banana, barley, beetroot, blackgram, blueberry,
  broccoli, brussels sprouts, buckwheat, cabbage, canola, carrot, cauliflower, celery,
  cherry, chickpea, coconut, coffee, cotton, cranberry, faba bean, flax, garlic, grapes,
  hazelnut, horseradish, jute, kale, kidney beans, leek, lentil, lettuce, maize, mango, moth
  beans, mung bean, muskmelon, mustard, oats, onion, orange, papaya, parsnip, pear, peas,
  pigeon peas, plum, pomegranate, potato, quinoa, radish, raspberry, rice, rye, spinach,
  strawberry, sugar beet, swede, turnip, walnut, watermelon, wheat.

**Statistical analysis.** One-way ANOVA F-test of each feature against the crop label (all
p ≈ 0). All features are significant. In ranked order: K (F = 1498.3), N (841.4),
rainfall (505.1), P (258.9), temperature (213.0), humidity (137.1), pH (123.8).

**Feature importance** (tuned Random Forest, mean decrease in impurity): rainfall 0.195,
K 0.164, humidity 0.154, N 0.138, P 0.121, pH 0.116, temperature 0.113.

## D. Methodology (ML pipeline in `train.py`)

1. **Data checks and EDA:** descriptive statistics, correlation matrix, distributions, crop
   profiles, ANOVA F-test.
2. **Split:** stratified 80/20 train/test split, giving 6,360 training and 1,590 test
   records. The test set is untouched until the final model is chosen.
3. **Baseline comparison:** 7 algorithms, each in a `StandardScaler` pipeline, scored with
   5-fold stratified cross-validation: Logistic Regression, K-Nearest Neighbors, Gaussian
   Naive Bayes, Decision Tree, Support Vector Machine, Random Forest, Gradient Boosting.
4. **Hyperparameter tuning:** `RandomizedSearchCV` (3-fold) on Random Forest (20 settings),
   Extra Trees (12) and Histogram Gradient Boosting (12). Forests are capped at ≤ 300 trees
   with min_samples_leaf ≥ 2 to keep the model small.
   - Best Random Forest: n_estimators = 200, max_depth = 20, min_samples_leaf = 2,
     max_features = "sqrt", class_weight = "balanced".
   - Best Extra Trees: n_estimators = 200, max_depth = None, min_samples_leaf = 2,
     max_features = 0.5.
   - Best Hist Gradient Boosting: max_iter = 200, learning_rate = 0.1, max_leaf_nodes = 15,
     min_samples_leaf = 10, l2_regularization = 1.0.
5. **Ensembles:**
   - **Soft voting** averages the predicted probabilities of the tuned models.
   - **Stacking** trains a logistic-regression meta-model on the members' out-of-fold
     probabilities.
   - Three ensembles were built: soft voting (RF + ET + HGB), stacking (RF + ET + HGB) and
     stacking (RF + HGB).
6. **Model selection:** pick the highest 5-fold CV accuracy **among models that need ≤ 300 MB
   of memory**. Streamlit Community Cloud guarantees only ~690 MB of RAM per app. Only after
   this choice is the model scored once on the test set.
7. **Learning curve:** CV accuracy when training on 10%–100% of the training data.
8. **Unsupervised crop families:**
   - Standardise each crop's mean profile.
   - Run Ward hierarchical (agglomerative) clustering.
   - Choose k from 5–12 by silhouette score.
   - Name each family after its medoid crop.
   - Project the crops to 2-D with PCA.
9. **Explainability:** exact Shapley values per prediction (section G).
10. **Extra comparisons:** accuracy on the original 22 vs the 42 new crops, the same
    pipeline on the 22-crop dataset, and the with/without `soil_type` experiment.
11. **Saving:** `models/crop_model.joblib` (compressed), `models/crop_families.joblib`,
    `models/metrics.json`, and plots in `reports/figures/`.

## E. Results

**Final model:** Stacking ensemble (tuned Random Forest + tuned Hist Gradient Boosting →
logistic-regression meta-model). Results on the 1,590 held-out test records:

| Metric | Value |
|---|---|
| 5-fold CV accuracy | 64.04% ± 1.01 |
| Test accuracy (top-1) | **65.41%** |
| Macro F1-score | 0.668 |
| Right crop in top 3 | 85.97% |
| Right crop in top 5 | **93.14%** |
| Right crop **family** (of 7) | **96.35%** |
| Memory in RAM / file size | ~216 MB / ~14 MB |

**Baseline algorithms** (5-fold CV on the training set, then the test set)

| Model | CV accuracy | Test top-1 | Top-3 | Top-5 | Train time (s) |
|---|---|---|---|---|---|
| Random Forest | 0.6223 ± 0.0101 | 0.6415 | 0.8660 | 0.9352 | 5.8 |
| Gradient Boosting | 0.6091 ± 0.0110 | 0.6208 | 0.8516 | 0.9308 | 107.4 |
| Gaussian Naive Bayes | 0.5917 ± 0.0071 | 0.5962 | 0.8396 | 0.9239 | 0.09 |
| Support Vector Machine | 0.5660 ± 0.0101 | 0.5824 | 0.8264 | 0.9170 | 11.0 |
| Logistic Regression | 0.5358 ± 0.0054 | 0.5447 | 0.7887 | 0.8899 | 3.5 |
| Decision Tree | 0.5263 ± 0.0084 | 0.5472 | 0.5528 | 0.5805 | 0.2 |
| K-Nearest Neighbors | 0.5115 ± 0.0076 | 0.5182 | 0.7220 | 0.7862 | — |

**Tuning and ensembles**

| Model | CV accuracy | Memory | Test top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| Stacking ensemble (RF + ET + HGB) | 0.6412 ± 0.0093 | 658 MB | 0.6597 | 0.8648 | 0.9321 |
| Soft-voting ensemble (RF + ET + HGB) | 0.6409 ± 0.0121 | 658 MB | 0.6604 | 0.8805 | 0.9472 |
| **Stacking ensemble (RF + HGB) — selected** | **0.6404 ± 0.0101** | **216 MB** | **0.6541** | 0.8597 | 0.9314 |
| Hist Gradient Boosting (tuned) | 0.6303 ± 0.0117 | 4 MB | 0.6465 | 0.8780 | 0.9503 |
| Random Forest (tuned) | 0.6277 ± 0.0096 | 212 MB | 0.6434 | 0.8629 | 0.9365 |
| Random Forest (default) | 0.6223 ± 0.0101 | 378 MB | 0.6415 | 0.8660 | 0.9352 |
| Extra Trees (tuned) | 0.6140 ± 0.0048 | 442 MB | 0.6239 | 0.8535 | 0.9352 |

**Discussion points to include:**
- Tuning plus stacking adds +1.8 points of CV accuracy and +1.3 points of test top-1
  accuracy over the default Random Forest.
- The two 3-model ensembles score slightly higher on CV but need ~660 MB, too close to the
  cloud's memory limit. The selected model is only 0.08 points behind on CV at one-third of
  the memory. This is a deliberate accuracy-versus-resource trade-off.
- The top ensembles are within one standard deviation of each other. Choosing between them
  on test scores would only fit noise, which is why selection used CV only.
- **Learning curve:** validation accuracy rises from 50.1% (508 records) to 62.7% (5,088),
  with only +0.6 points in the last step. Training accuracy stays around 99%. More records of
  the same kind help little; new features (season, region) would help more.
- **Why accuracy is lower than on the 22-crop dataset:** the same pipeline scores 99.55% on
  the 22-crop dataset. Inside the 64-crop model, the original 22 crops still score 92.6%,
  but the 42 new crops score 51.8%, because many need almost identical conditions.
- **Most common confusions:** quinoa → lentil (8), parsnip → swede (8), chickpea → lentil
  (7), cranberry → blueberry (6), faba bean ↔ peas (6 / 5), cauliflower → broccoli, pear →
  apple. Most mistakes are between botanically or agronomically close crops, which is why
  family accuracy (96.35%) and top-5 accuracy (93.14%) are much higher than top-1.
- Crops with perfect test scores include banana and blackgram (F1 = 1.0). The weakest are
  close relatives such as broccoli, carrot and asparagus (F1 ≈ 0.30).

## F. Unsupervised learning — crop families

- Ward hierarchical clustering on the standardised crop mean profiles.
- Silhouette scores for k = 5…12: 0.287, 0.288, **0.300 (k = 7, chosen)**, 0.288, 0.290,
  0.232, 0.236, 0.249.
- PCA: the first two components explain 36.4% + 19.0% = 55.4% of the variance.
- The 7 families (each named after its medoid crop):
  1. **Jute group** (low rainfall, warm): jute, rice, maize, cotton, banana, coconut,
     coffee, mung bean, muskmelon, orange, papaya, pomegranate, watermelon (13 crops).
  2. **Mothbeans group** (low nitrogen, dry air): moth beans, blackgram, chickpea, kidney
     beans, lentil, mango, pigeon peas (7).
  3. **Flax group** (low potassium, cool): flax, barley, buckwheat, faba bean, oats, peas,
     quinoa, rye, wheat (9).
  4. **Cabbage group** (high nitrogen, high potassium): cabbage, broccoli, brussels sprouts,
     cauliflower, celery, potato (6).
  5. **Beetroot group** (high potassium, high nitrogen): 26 vegetables and fruit, including
     beetroot, carrot, onion, garlic, apple, pear, plum, cherry, strawberry, spinach,
     lettuce, kale and walnut.
  6. **Cranberry group** (acidic soil, high rainfall): cranberry, blueberry (2).
  7. **Grapes group** (high phosphorus, high potassium): grapes alone (1).
- **Key finding:** the exact crop is right 65% of the time, but the right family is right
  96.4% of the time, so the classifier's errors are mostly "near misses".

## G. Explainable AI — exact Shapley values (`src/explain.py`)

- With 7 inputs there are only 2⁷ = 128 feature subsets, so Shapley values are computed
  **exactly**, not approximated as in sampling-based SHAP.
- For each subset, the model's average probability is computed over 64 background profiles
  (each crop's median field). Inputs in the subset take the user's values; the others take
  the background values (interventional Shapley).
- Each input's Shapley value is its weighted average marginal contribution over all subsets.
  **Base value + sum of contributions = predicted probability**, verified to about 10⁻¹⁵.
- **Worked example:** on a typical rice field, rice's probability rises from 1.6% (average
  field) to 95.8%. The main contributions are rainfall +29 points, humidity +25, potassium
  +18 and temperature +17.
- In the app it is shown as a "🧠 Why this crop?" bar chart, behind a toggle so it is only
  computed on request.

## H. Innovative features

1. **Soil fix advisor (counterfactual search, `src/advisor.py`):**
   - Climate is held fixed. Candidate soils move the field's N, P, K and pH 25–100% of the
     way toward each real profile of the target crop (up to ~1,000 candidates).
   - The model scores every candidate. It picks the cheapest change (measured in standard
     deviations, with decreases counted double because removing nutrients is harder) that
     makes the crop #1 with at least 25% confidence.
   - The change is turned into fertiliser and lime actions. If the crop can't reach the top
     because of climate, the advisor says so.
   - **Example:** on a typical chickpea field, adding about 16 kg/ha of nitrogen lifts lentil
     from 12% to 77%.
2. **Climate-change stress test:** the shortlist is re-scored under 12 scenarios (warming
   steps × rainfall changes), showing which crops stay suitable as the climate shifts.
3. **Weather auto-fill by location:** the user types a town. The Open-Meteo geocoding and
   historical-archive APIs fetch last season's average temperature, humidity and total
   rainfall to fill the form.

## I. Application (Streamlit web app, `app.py`) — 7 tabs

1. **🔮 Recommend:**
   - Form with 7 inputs, bounded by the data range, plus an optional weather auto-fill.
   - Shows the top crop with its confidence and a ranked top-5 shortlist with probabilities.
   - Shows the crop's family, the "Why this crop?" Shapley chart, and a table comparing the
     user's values with the crop's typical values.
2. **🛠️ Soil fix advisor:** choose a target crop and get the smallest soil change plus
   fertiliser and lime advice.
3. **🌡️ Climate stress test:** heat-map style table of the shortlist under 12 scenarios.
4. **🧬 Crop families:** PCA 2-D map of all crops coloured by family, with the user's field
   plotted.
5. **📊 Data explorer:** statistics, ANOVA, correlations, distributions.
6. **🧪 Model performance:** algorithm comparison, tuning and ensemble table (with memory),
   best hyperparameters, learning curve, 22- vs 64-crop comparison, common confusions, and
   the soil-type experiment.
7. **ℹ️ About:** project description.

## J. System design / architecture

- **Code modules:**
  - `src/data.py` (loading)
  - `src/ml.py` (tuning, ensembles, learning curve, clustering)
  - `src/explain.py` (Shapley values)
  - `src/advisor.py` (soil fix, fertiliser rules, stress test)
  - `src/weather.py` (Open-Meteo)
  - `train.py` (offline training pipeline)
  - `app.py` (UI)
- **Flow:** dataset → `train.py` (EDA → split → baselines → tuning → ensembles → selection →
  evaluation → clustering) → saved model files → `app.py` loads them → user input →
  prediction + explanation + advice.
- **Performance engineering:**
  - Streamlit caching keyed by model version.
  - Single-threaded inference on the cloud, to avoid CPU oversubscription in the container.
  - On-demand Shapley computation.
  - Compressed model file.
  - The 300 MB memory budget during model selection.
- **Deployment:**
  - GitHub repository → Streamlit Community Cloud (Python 3.12), with pinned
    `requirements.txt` because the saved model needs the same scikit-learn version.
  - Local one-click launcher `run_app.bat`: creates a virtual environment, installs the
    packages and opens http://localhost:8501.

## K. Challenges faced and solutions (good for a "Challenges" section)

| Challenge | Fix |
|---|---|
| 2,200 records / 22 crops felt too small | Moved to the extended 7,950-record, 64-crop dataset and kept the old one as a baseline |
| Accuracy dropped from 99.5% to ~64% | Analysed why (similar crops); added top-k accuracy, crop families, tuning and ensembles |
| `soil_type` gave a fake accuracy boost | Detected the leakage experimentally and excluded the feature |
| The first ensemble model was 4.8 GB | Cloned members, compressed, capped forest sizes |
| The 658 MB model was too big for the free cloud tier | Added the 300 MB memory budget to model selection |
| Cloud app lagged, tabs were blank | Thread oversubscription; forced single-thread inference, caching, lazy Shapley |
| Misleading soil-fix advice | Added a minimum confidence of 25% and climate-mismatch warnings |
| Windows path-length error installing scikit-learn | Virtual environment at a short path outside OneDrive |

## L. Limitations and future work

- **Limitations:**
  - The dataset looks partly synthetic (sharp, evenly filled per-crop ranges), and its
    rainfall period is undocumented, so weather auto-fill is approximate.
  - Seven inputs cannot fully separate closely related crops; season, region, irrigation
    and growing period are missing.
  - Fertiliser amounts are rules of thumb; confirm them with a soil-testing lab.
- **Future work:**
  - Season, region and market-price features.
  - An anomaly detector for fields unlike the training data.
  - Probability calibration.
  - Model selection by top-5 accuracy.
  - Regional-language interface.
  - Mobile app and IoT soil-sensor input.

## M. Figures available (in `reports/figures/`) and app screenshots

`model_comparison.png`, `confusion_matrix.png`, `feature_importance.png`,
`learning_curve.png`, `correlation.png`, `distributions.png`, `crop_profiles.png`,
`crop_dendrogram.png`. App screenshots can be taken of each of the 7 tabs.

## N. Conclusion points

- AgriAI shows a complete ML workflow: data analysis, fair model comparison, tuning,
  ensembles, resource-aware model selection, unsupervised discovery, explainable AI and
  deployment.
- On a hard 64-crop problem it puts the right crop in a 5-crop shortlist 93% of the time and
  in the right family 96% of the time.
- On the classic 22-crop benchmark it reaches 99.55%.
- Explanations and soil and climate advice turn the prediction into a decision aid that a
  farmer can act on.
- The recommendations are suggestions, not certified agronomic advice.

## O. References to cite (verify formatting for your style)

1. Ajala, E. *Crop Recommendation Dataset*, Kaggle (Apache 2.0).
2. Ingle, A. *Crop Recommendation Dataset*, Kaggle.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
4. Breiman, L. (2001). Random Forests. *Machine Learning* 45, 5–32.
5. Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks* 5(2), 241–259.
6. Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games II*.
7. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
8. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA* 58(301), 236–244.
9. Rousseeuw, P. J. (1987). Silhouettes. *J. Computational and Applied Mathematics* 20, 53–65.
10. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* (basis of histogram gradient boosting).
11. Open-Meteo, free weather API, https://open-meteo.com.
12. Streamlit documentation, https://docs.streamlit.io.

Now write the full report following the faculty format.
