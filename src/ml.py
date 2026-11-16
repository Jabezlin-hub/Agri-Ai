"""Advanced machine-learning steps for AgriAI.

tune_and_ensemble  - hyperparameter search (RandomizedSearchCV) for three tree ensembles,
                     then soft-voting and stacking ensembles built from the tuned models.
learning_curve_data- accuracy vs. amount of training data (is more data still helping?).
crop_families      - unsupervised learning: hierarchical clustering of the crops' average
                     soil/climate profiles into families, plus a 2-D PCA map.
"""
from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, TARGET

SEED = 42

# Forest sizes are capped (<= 300 trees, leaves of >= 2 samples): with 64 classes every
# tree node stores 64 probabilities, and bigger forests made the saved model hundreds of
# MB for a gain of ~0.2 accuracy points.
SEARCH_SPACES = {
    "Random Forest (tuned)": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [150, 200, 300], "max_depth": [None, 20, 30],
         "min_samples_leaf": [2, 3, 5], "max_features": ["sqrt", "log2", 0.5],
         "class_weight": [None, "balanced"]},
        20),
    "Extra Trees (tuned)": (
        ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [150, 200, 300], "max_depth": [None, 20, 30],
         "min_samples_leaf": [2, 3, 5], "max_features": ["sqrt", 0.5, None]},
        12),
    "Hist Gradient Boosting (tuned)": (
        HistGradientBoostingClassifier(random_state=SEED, early_stopping=True),
        {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
         "min_samples_leaf": [10, 20, 40], "l2_regularization": [0.0, 0.1, 1.0],
         "max_iter": [200, 400]},
        12),
}


def tune_and_ensemble(X_train, y_train, cv) -> list[dict]:
    """Tune three tree ensembles, then combine them. Every candidate is scored with the same
    5-fold cross-validation so they can be compared fairly; the test set is not used here."""
    out, tuned = [], {}
    for name, (est, space, n_iter) in SEARCH_SPACES.items():
        t0 = time.time()
        search = RandomizedSearchCV(est, space, n_iter=n_iter, cv=3, scoring="accuracy",
                                    random_state=SEED, n_jobs=4)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=5)
        tuned[name] = best
        out.append({"model": name, "estimator": best, "params": search.best_params_,
                    "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
                    "seconds": round(time.time() - t0, 1)})
        print(f"  {name:32s} CV {scores.mean():.4f} ± {scores.std():.4f}  "
              f"{search.best_params_}  ({time.time() - t0:.0f}s)")

    # Unfitted copies: an ensemble refits its members itself, and passing the fitted ones
    # would make it store every forest twice.
    members = [(n.split(" (")[0].lower().replace(" ", "_"), clone(m)) for n, m in tuned.items()]
    light = [(n, clone(m)) for n, m in members if n != "extra_trees"]
    ensembles = {
        "Soft-voting ensemble": VotingClassifier(members, voting="soft", n_jobs=1),
        "Stacking ensemble": StackingClassifier(
            members, final_estimator=LogisticRegression(max_iter=3000, C=1.0),
            stack_method="predict_proba", cv=3, n_jobs=1),
        # Without Extra Trees (the largest member), for deployment on limited memory.
        "Stacking ensemble (RF + HGB)": StackingClassifier(
            light, final_estimator=LogisticRegression(max_iter=3000, C=1.0),
            stack_method="predict_proba", cv=3, n_jobs=1),
    }
    for name, est in ensembles.items():
        t0 = time.time()
        scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=5)
        est.fit(X_train, y_train)
        out.append({"model": name, "estimator": est,
                    "params": {"members": [n for n, _ in members]},
                    "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
                    "seconds": round(time.time() - t0, 1)})
        print(f"  {name:32s} CV {scores.mean():.4f} ± {scores.std():.4f}  "
              f"({time.time() - t0:.0f}s)")
    return out


def memory_mb(estimator) -> float:
    """Approximate in-memory size: the model's uncompressed serialised size in MB."""
    buf = io.BytesIO()
    joblib.dump(estimator, buf)
    return round(buf.tell() / 1e6, 1)


def learning_curve_data(estimator, X, y, cv) -> dict:
    """Cross-validated accuracy when training on 10%..100% of the training data."""
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 6), cv=cv,
        scoring="accuracy", n_jobs=5, shuffle=True, random_state=SEED)
    return {"train_size": sizes.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "val_mean": val_scores.mean(axis=1).tolist(),
            "val_std": val_scores.std(axis=1).tolist()}


WORDS = {  # (word if below average, word if above average)
    "N": ("low nitrogen", "high nitrogen"), "P": ("low phosphorus", "high phosphorus"),
    "K": ("low potassium", "high potassium"), "temperature": ("cool", "warm"),
    "humidity": ("dry air", "humid"), "ph": ("acidic soil", "alkaline soil"),
    "rainfall": ("low rainfall", "high rainfall"),
}


def crop_families(df: pd.DataFrame, k_range=range(5, 13)) -> dict:
    """Group crops with similar needs (Ward hierarchical clustering of standardised crop
    means; the number of families is chosen by silhouette score) and build a PCA map."""
    scaler = StandardScaler().fit(df[FEATURES])
    Z = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES, index=df.index)
    means = Z.groupby(df[TARGET]).mean()
    link = linkage(means.to_numpy(), method="ward")

    sil = {k: float(silhouette_score(means, fcluster(link, k, "maxclust"))) for k in k_range}
    k = max(sil, key=sil.get)
    labels = fcluster(link, k, "maxclust")
    family_of = {crop: int(f) for crop, f in zip(means.index, labels)}

    # Name each family after its most typical crop (the member closest to the family's
    # centre) and describe it by the two inputs that most set it apart from the average crop.
    names, descriptions = {}, {}
    for fam in sorted(set(labels)):
        members = means[labels == fam]
        centre = members.mean()
        medoid = ((members - centre) ** 2).sum(axis=1).idxmin()
        names[int(fam)] = f"{medoid.replace('_', ' ').title()} group"
        top = centre.abs().sort_values(ascending=False).index[:2]
        descriptions[int(fam)] = ", ".join(WORDS[f][int(centre[f] > 0)] for f in top)

    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    centroids = pd.DataFrame(pca.transform(means), index=means.index, columns=["pc1", "pc2"])
    return {
        "scaler": scaler, "pca": pca, "linkage": link, "k": k, "silhouette": sil,
        "family_of": family_of, "names": names, "descriptions": descriptions,
        "crop_order": list(means.index),
        "centroids": centroids, "explained_variance": pca.explained_variance_ratio_.tolist(),
        "loadings": pd.DataFrame(pca.components_.T, index=FEATURES, columns=["pc1", "pc2"]),
    }
