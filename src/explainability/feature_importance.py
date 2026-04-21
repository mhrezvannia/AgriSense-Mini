from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.inspection import permutation_importance


def extract_model_feature_importance(model: Any, feature_names: Sequence[str]) -> dict[str, Any]:
    if hasattr(model, "coef_"):
        raw_importance = np.abs(model.coef_)
        if raw_importance.ndim > 1:
            importance = raw_importance.mean(axis=0)
        else:
            importance = raw_importance
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        raise ValueError("Model does not expose coefficient-based or tree-based feature importance.")

    order = np.argsort(importance)[::-1]
    return {
        "feature_names": np.asarray(feature_names)[order].tolist(),
        "importances": importance[order].astype(float).tolist(),
    }


def permutation_importance_summary(
    model: Any,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: Sequence[str],
    random_state: int = 42,
) -> dict[str, Any]:
    result = permutation_importance(model, x_eval, y_eval, n_repeats=10, random_state=random_state)
    order = np.argsort(result.importances_mean)[::-1]
    return {
        "feature_names": np.asarray(feature_names)[order].tolist(),
        "importances": result.importances_mean[order].astype(float).tolist(),
        "importances_std": result.importances_std[order].astype(float).tolist(),
    }
