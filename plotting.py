"""Publication-oriented diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def safe_slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def plot_roc(y, predictions: dict[str, np.ndarray], outcome: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for name, probabilities in predictions.items():
        fpr, tpr, _ = roc_curve(y, probabilities)
        auc = roc_auc_score(y, probabilities)
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", linewidth=1, label="Chance")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=f"ROC curves: {outcome}",
    )
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_precision_recall(
    y, predictions: dict[str, np.ndarray], outcome: str, output: Path
) -> None:
    prevalence = float(np.mean(y))
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for name, probabilities in predictions.items():
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ap = average_precision_score(y, probabilities)
        axis.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    axis.axhline(prevalence, linestyle="--", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall: {outcome}")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_calibration(
    y, predictions: dict[str, np.ndarray], outcome: str, output: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for name, probabilities in predictions.items():
        observed, mean_predicted = calibration_curve(
            y, probabilities, n_bins=6, strategy="quantile"
        )
        axis.plot(mean_predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "--", linewidth=1, label="Ideal")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event proportion",
        title=f"Calibration: {outcome}",
    )
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, outcome: str, output: Path) -> None:
    top = importance.head(12).sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(7.6, 5.8))
    axis.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"])
    axis.set(
        xlabel="Decrease in AUROC after permutation",
        title=f"Exploratory feature importance: {outcome}",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
