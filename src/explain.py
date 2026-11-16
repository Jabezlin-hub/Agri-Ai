"""Exact Shapley-value explanations for a single prediction (explainable AI).

With only 7 features there are 2^7 = 128 feature subsets, so Shapley values can be
computed exactly instead of approximated (as SHAP's KernelExplainer would). For each
subset S, the value v(S) is the model's average predicted probability when the field's
own values are used for the features in S and the background profiles supply the rest
("interventional" Shapley values). Each feature's Shapley value is its average marginal
contribution over all subsets, so  base + sum(contributions) = the model's prediction.
"""
from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np
import pandas as pd

from src.data import FEATURES


def shapley_values(model, x: dict, background: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (contributions, base): contributions has one row per feature and one column
    per crop (probability points); base is the average prediction over the background."""
    n = len(FEATURES)
    x_vec = pd.Series(x)[FEATURES].to_numpy(float)
    bg = background[FEATURES].to_numpy(float)
    masks = np.array(list(product([0, 1], repeat=n)), dtype=bool)        # 128 x 7

    # Every subset x every background row, predicted in one batch.
    rows = np.where(masks[:, None, :], x_vec[None, None, :], bg[None, :, :])
    proba = model.predict_proba(pd.DataFrame(rows.reshape(-1, n), columns=FEATURES))
    v = proba.reshape(len(masks), len(bg), -1).mean(axis=1)             # 128 x classes

    index = {tuple(m): i for i, m in enumerate(masks)}
    sizes = masks.sum(axis=1)
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros((n, v.shape[1]))
    for i in range(n):
        without = np.flatnonzero(~masks[:, i])
        for j in without:
            m = masks[j].copy()
            m[i] = True
            phi[i] += weight[sizes[j]] * (v[index[tuple(m)]] - v[j])
    base = v[index[(False,) * n]]
    return pd.DataFrame(phi, index=FEATURES, columns=model.classes_), base
