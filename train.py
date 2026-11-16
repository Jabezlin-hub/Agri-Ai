"""AgriAI - train and compare crop recommendation models.

Run:  python train.py
Outputs:
  models/crop_model.joblib     best model pipeline (scaler + classifier)
  models/metrics.json          accuracy / CV scores for every model
  reports/figures/*.png        EDA and evaluation plots
  reports/feature_stats.csv    ANOVA F-test per feature
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from src.data import EXTENDED_CSV, FEATURES, REAL_CSV, TARGET, load_data  # noqa: E402
from src.ml import crop_families, learning_curve_data, memory_mb, tune_and_ensemble  # noqa: E402

MEMORY_BUDGET_MB = 300  # max in-memory model size for free cloud hosting
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42


def statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats + one-way ANOVA F-test of each feature against crop label."""
    print("\n=== Descriptive statistics ===")
    print(df[FEATURES].describe().round(2).to_string())

    f_vals, p_vals = f_classif(df[FEATURES], df[TARGET])
    stats = pd.DataFrame({"feature": FEATURES, "F_score": f_vals, "p_value": p_vals})
    stats = stats.sort_values("F_score", ascending=False).reset_index(drop=True)
    print("\n=== ANOVA F-test (feature vs crop) ===")
    print(stats.round(4).to_string(index=False))
    stats.to_csv(ROOT / "reports" / "feature_stats.csv", index=False)
    return stats


def eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#3a7d44")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distributions.png", dpi=110)
    plt.close()

    means = df.groupby(TARGET)[FEATURES].mean()
    norm = (means - means.min()) / (means.max() - means.min())
    plt.figure(figsize=(8, max(9, 0.22 * len(norm))))
    sns.heatmap(norm, cmap="YlGn", cbar_kws={"label": "relative mean (0-1)"})
    plt.title("Average soil & climate needs per crop")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "crop_profiles.png", dpi=120)
    plt.close()


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }


def topk(pipe, X_test, y_test) -> dict:
    """Share of test records whose true crop is among the model's top 3 / top 5."""
    proba = pipe.predict_proba(X_test)
    classes = pipe.classes_
    return {f"test_top{k}": float(top_k_accuracy_score(y_test, proba, k=k, labels=classes))
            for k in (3, 5)}


def original_dataset_baseline() -> dict | None:
    """Train the two strongest models on the original 22-crop dataset for comparison."""
    if not REAL_CSV.exists():
        return None
    df = pd.read_csv(REAL_CSV)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=SEED)
    out = {"n_rows": len(df), "n_crops": int(df[TARGET].nunique()), "results": []}
    for name in ("Random Forest", "Gaussian Naive Bayes"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", build_models()[name])])
        pipe.fit(X_tr, y_tr)
        out["results"].append({"model": name,
                               "test_accuracy": float(accuracy_score(y_te, pipe.predict(X_te))),
                               **topk(pipe, X_te, y_te)})
    return out


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, source = load_data()
    print(f"Loaded {len(df)} rows, {df[TARGET].nunique()} crops from {source}")
    print(f"Missing values: {int(df.isna().sum().sum())}")

    stats = statistical_analysis(df)
    eda_plots(df)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = [], {}
    print("\n=== Model comparison ===")
    for name, clf in build_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        t0 = time.time()
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=5)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        res = {
            "model": name,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            **topk(pipe, X_test, y_test),
            "train_seconds": round(time.time() - t0, 2),
        }
        results.append(res)
        fitted[name] = pipe
        print(f"{name:24s} CV {res['cv_mean']:.4f} ± {res['cv_std']:.4f} | "
              f"test acc {res['test_accuracy']:.4f} | top-3 {res['test_top3']:.4f} | "
              f"top-5 {res['test_top5']:.4f} | F1 {res['test_f1_macro']:.4f}")

    # Model selection uses cross-validation only; the test set is kept for the final check.
    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    baseline_best = results[0]

    print("\n=== Hyperparameter tuning and ensembles (5-fold CV on the training set) ===")
    candidates = tune_and_ensemble(X_train, y_train, cv)
    candidates.append({"model": f"{baseline_best['model']} (default settings)",
                       "estimator": fitted[baseline_best["model"]], "params": {},
                       "cv_mean": baseline_best["cv_mean"], "cv_std": baseline_best["cv_std"],
                       "seconds": baseline_best["train_seconds"]})
    for c in candidates:  # test scores are reported for transparency, not used to choose
        p = c["estimator"].predict(X_test)
        c.update({"test_accuracy": float(accuracy_score(y_test, p)),
                  "test_f1_macro": float(f1_score(y_test, p, average="macro")),
                  **topk(c["estimator"], X_test, y_test),
                  "memory_mb": memory_mb(c["estimator"])})
    candidates.sort(key=lambda c: c["cv_mean"], reverse=True)
    for c in candidates:
        print(f"  {c['model']:32s} CV {c['cv_mean']:.4f}  memory {c['memory_mb']:.0f} MB")
    # Deployment constraint: Streamlit Community Cloud guarantees only ~690 MB of RAM per
    # app, so the app's model must fit a memory budget. Best CV accuracy within the budget.
    eligible = [c for c in candidates if c["memory_mb"] <= MEMORY_BUDGET_MB] or candidates
    final = eligible[0]
    best_name, best = final["model"], final["estimator"]
    print(f"\nSelected by cross-validation within {MEMORY_BUDGET_MB} MB: {best_name} "
          f"(CV {final['cv_mean']:.4f}) -> "
          f"test acc {final['test_accuracy']:.4f}, top-3 {final['test_top3']:.4f}, "
          f"top-5 {final['test_top5']:.4f}")
    tuned_rf = next(c["estimator"] for c in candidates if c["model"] == "Random Forest (tuned)")

    print("\nLearning curve (tuned Random Forest)...")
    curve = learning_curve_data(tuned_rf, X_train, y_train, cv)
    for n, v in zip(curve["train_size"], curve["val_mean"]):
        print(f"  {n:5d} training records -> CV accuracy {v:.4f}")
    plt.figure(figsize=(7, 4))
    plt.plot(curve["train_size"], curve["train_mean"], "o-", label="Training accuracy")
    plt.plot(curve["train_size"], curve["val_mean"], "o-", label="Cross-validation accuracy")
    plt.xlabel("Training records")
    plt.ylabel("Accuracy")
    plt.title("Learning curve - Random Forest (tuned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curve.png", dpi=130)
    plt.close()

    pred = best.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    print(classification_report(y_test, pred, digits=3))

    # Evaluation plots
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=res_df, y="model", x="test_accuracy", color="#3a7d44")
    plt.xlim(max(0, res_df.test_accuracy.min() - 0.05), 1.0)
    plt.title("Test accuracy by model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=130)
    plt.close()

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    size = max(11, 0.22 * len(labels))
    plt.figure(figsize=(size, size * 0.82))
    sns.heatmap(cm, annot=len(labels) <= 30, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # Importances come from the selected model if it has them, else the tuned Random Forest
    if hasattr(best, "feature_importances_"):
        imp_name, imp_values = best_name, best.feature_importances_
    else:
        imp_name, imp_values = "Random Forest (tuned)", tuned_rf.feature_importances_
    importances = dict(zip(FEATURES, imp_values.tolist()))
    imp = pd.Series(importances).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot.barh(color="#3a7d44")
    plt.title(f"Feature importance - {imp_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=130)
    plt.close()

    # Misclassified test records, as (actual, predicted, count)
    errors = (pd.DataFrame({"actual": y_test.values, "predicted": pred})
              .query("actual != predicted").value_counts().reset_index(name="count"))
    print(f"\nMisclassified test records: {int(errors['count'].sum()) if len(errors) else 0}. "
          "Most common confusions:\n" + (errors.head(10).to_string(index=False) if len(errors) else "none"))

    # Side experiment: does adding soil_type help? It is assigned per crop in the
    # extended dataset, so a big jump would mean it leaks the label.
    soil_experiment = None
    if "soil_type" in df.columns:
        Xs = pd.get_dummies(df[FEATURES + ["soil_type"]], columns=["soil_type"], dtype=float)
        Xs_tr, Xs_te = Xs.loc[X_train.index], Xs.loc[X_test.index]
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xs_tr, y_train)
        with_soil = float(accuracy_score(y_test, rf.predict(Xs_te)))
        without_soil = next(r["test_accuracy"] for r in results if r["model"] == "Random Forest")
        soil_experiment = {"rf_without_soil": without_soil, "rf_with_soil": with_soil,
                           "crops_with_one_soil_type":
                               int((df.groupby(TARGET)["soil_type"].nunique() == 1).sum())}
        print(f"\nSoil-type experiment (Random Forest): without {without_soil:.4f}, "
              f"with soil_type {with_soil:.4f}")

    # Accuracy on the original 22 crops vs the crops added by the extended dataset
    group_accuracy, baseline = None, None
    if source.startswith(EXTENDED_CSV.name) and REAL_CSV.exists():
        orig_crops = set(pd.read_csv(REAL_CSV)[TARGET])
        is_orig = y_test.isin(orig_crops).to_numpy()
        group_accuracy = {
            "original_crops": float(accuracy_score(y_test[is_orig], pred[is_orig])),
            "new_crops": float(accuracy_score(y_test[~is_orig], pred[~is_orig])),
            "n_original_crops": len(orig_crops),
            "n_new_crops": int(y.nunique() - len(orig_crops)),
        }
        print(f"\nBest-model accuracy: original {len(orig_crops)} crops "
              f"{group_accuracy['original_crops']:.4f}, new crops {group_accuracy['new_crops']:.4f}")
        baseline = original_dataset_baseline()
        for r in baseline["results"]:
            print(f"Original 2,200-row dataset, {r['model']}: acc {r['test_accuracy']:.4f}, "
                  f"top-3 {r['test_top3']:.4f}, top-5 {r['test_top5']:.4f}")

    # Unsupervised learning: crop families + PCA map
    print("\nCrop families (hierarchical clustering)...")
    fam = crop_families(df)
    fam_of = fam["family_of"]
    family_accuracy = float(np.mean([fam_of[a] == fam_of[p] for a, p in zip(y_test, pred)]))
    print(f"  {fam['k']} families (silhouette {fam['silhouette'][fam['k']]:.3f}); "
          f"predicted crop is in the right family for {family_accuracy:.4f} of test records")
    for f_id, name in fam["names"].items():
        members = [c for c, f in fam_of.items() if f == f_id]
        print(f"  {name} ({fam['descriptions'][f_id]}): {', '.join(members)}")
    plt.figure(figsize=(9, max(9, 0.2 * len(fam["crop_order"]))))
    dendrogram(fam["linkage"], labels=fam["crop_order"], orientation="right",
               color_threshold=fam["linkage"][-(fam["k"] - 1), 2], leaf_font_size=8)
    plt.title(f"Crop families - Ward hierarchical clustering ({fam['k']} families)")
    plt.xlabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "crop_dendrogram.png", dpi=120)
    plt.close()
    joblib.dump(fam, MODEL_DIR / "crop_families.joblib")

    joblib.dump(best, MODEL_DIR / "crop_model.joblib", compress=3)
    print(f"Model file: {(MODEL_DIR / 'crop_model.joblib').stat().st_size / 1e6:.1f} MB")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({
            "data_source": source,
            "n_rows": len(df),
            "n_crops": int(y.nunique()),
            "best_model": best_name,
            "selection": ("highest 5-fold cross-validation accuracy on the training set, among "
                          f"models that need at most {MEMORY_BUDGET_MB} MB of memory"),
            "memory_budget_mb": MEMORY_BUDGET_MB,
            "final": {k: v for k, v in final.items() if k != "estimator"},
            "results": results,
            "tuning": [{k: v for k, v in c.items() if k != "estimator"} for c in candidates],
            "learning_curve": curve,
            "family_accuracy": family_accuracy,
            "families": {"k": fam["k"], "silhouette": fam["silhouette"], "names": fam["names"],
                         "descriptions": fam["descriptions"],
                         "family_of": fam["family_of"],
                         "explained_variance": fam["explained_variance"]},
            "feature_importance": importances,
            "feature_importance_model": imp_name,
            "anova": stats.to_dict(orient="records"),
            "errors": errors.to_dict(orient="records"),
            "soil_experiment": soil_experiment,
            "group_accuracy": group_accuracy,
            "baseline_original": baseline,
            "per_class": {k: v for k, v in report.items() if k in labels},
        }, f, indent=2)
    print(f"Saved model and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
