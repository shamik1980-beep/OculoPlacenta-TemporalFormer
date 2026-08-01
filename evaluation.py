"""Cross-validation, uncertainty estimates, and classification metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


def repeated_oof_probabilities(estimator, x, y: np.ndarray, cv) -> np.ndarray:
    """Average out-of-fold probabilities across repeated stratified folds."""
    probability_sum = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    for train_index, test_index in cv.split(x, y):
        fitted = clone(estimator)
        fitted.fit(x.iloc[train_index], y[train_index])
        probabilities = fitted.predict_proba(x.iloc[test_index])[:, 1]
        probability_sum[test_index] += probabilities
        counts[test_index] += 1
    if np.any(counts == 0):
        raise RuntimeError("At least one participant received no out-of-fold prediction.")
    return probability_sum / counts


def bootstrap_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Participant-level nonparametric bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indexes = rng.integers(0, n, n)
        y_sample = y_true[indexes]
        p_sample = probabilities[indexes]
        if len(np.unique(y_sample)) < 2:
            continue
        values.append(float(metric(y_sample, p_sample)))
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate threshold-dependent metrics at a prespecified threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int,
    seed: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return discrimination, calibration, uncertainty, and threshold metrics."""
    auroc = float(roc_auc_score(y_true, probabilities))
    auprc = float(average_precision_score(y_true, probabilities))
    auroc_low, auroc_high = bootstrap_ci(
        y_true, probabilities, roc_auc_score, n_bootstrap, seed
    )
    auprc_low, auprc_high = bootstrap_ci(
        y_true, probabilities, average_precision_score, n_bootstrap, seed
    )
    return {
        "events": int(y_true.sum()),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": auroc,
        "auroc_ci_low": auroc_low,
        "auroc_ci_high": auroc_high,
        "auprc": auprc,
        "auprc_ci_low": auprc_low,
        "auprc_ci_high": auprc_high,
        "brier": float(brier_score_loss(y_true, probabilities)),
        **threshold_metrics(y_true, probabilities, threshold),
    }


def make_cv(n_splits: int, n_repeats: int, seed: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
