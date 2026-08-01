"""Reproducible clinical benchmark for the Bogotá pre-eclampsia cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .constants import DEFAULT_SEED, SELECTED_CHARACTERISTIC_COLUMNS
from .data import cohort_audit, load_bogota_dataset
from .evaluation import evaluate_probabilities, make_cv, repeated_oof_probabilities
from .plotting import (
    plot_calibration,
    plot_importance,
    plot_precision_recall,
    plot_roc,
    safe_slug,
)
from .preprocessing import build_preprocessor, infer_feature_schema


def build_models(seed: int) -> dict[str, object]:
    """Return low-complexity prespecified models appropriate for a small cohort."""
    return {
        "Penalised logistic regression": LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        ),
        "Shrinkage LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Random forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    }


def _selected_characteristics(frame: pd.DataFrame, outcomes: dict[str, np.ndarray]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    selected = [column for column in SELECTED_CHARACTERISTIC_COLUMNS if column in frame]
    for column in selected:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        for outcome_name, y in outcomes.items():
            event = numeric[y == 1]
            nonevent = numeric[y == 0]
            records.append(
                {
                    "outcome": outcome_name,
                    "variable": column,
                    "event_median": float(event.median()),
                    "event_q1": float(event.quantile(0.25)),
                    "event_q3": float(event.quantile(0.75)),
                    "nonevent_median": float(nonevent.median()),
                    "nonevent_q1": float(nonevent.quantile(0.25)),
                    "nonevent_q3": float(nonevent.quantile(0.75)),
                    "event_mean": float(event.mean()),
                    "nonevent_mean": float(nonevent.mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def run_benchmark(
    data_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    n_splits: int = 5,
    n_repeats: int = 2,
    n_bootstrap: int = 500,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the complete leakage-resistant benchmark and write all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_bogota_dataset(data_path)
    schema = infer_feature_schema(frame)
    x = frame[schema.features].copy()
    preprocessor = build_preprocessor(schema)
    models = build_models(seed)
    outcomes = {
        "Early-onset pre-eclampsia": 1 - frame["preeclampsia_onset"].to_numpy(),
        "Intrauterine growth restriction": frame["iugr"].to_numpy(),
    }
    cv = make_cv(n_splits, n_repeats, seed)

    rows: list[dict[str, object]] = []
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    best_by_outcome: dict[str, str] = {}

    for outcome_name, y in outcomes.items():
        all_predictions[outcome_name] = {}
        for model_name, model in models.items():
            estimator = Pipeline(
                [("preprocessor", clone(preprocessor)), ("model", clone(model))]
            )
            probabilities = repeated_oof_probabilities(estimator, x, y, cv)
            all_predictions[outcome_name][model_name] = probabilities
            rows.append(
                {
                    "outcome": outcome_name,
                    "model": model_name,
                    **evaluate_probabilities(
                        y, probabilities, n_bootstrap, seed, threshold
                    ),
                }
            )
        best_by_outcome[outcome_name] = max(
            models,
            key=lambda name: roc_auc_score(y, all_predictions[outcome_name][name]),
        )

    performance = pd.DataFrame(rows)
    performance.to_csv(output_dir / "model_performance.csv", index=False)

    audit = cohort_audit(frame, len(schema.features))
    audit["best_model_by_outcome"] = best_by_outcome
    audit["analysis_parameters"] = {
        "seed": seed,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_bootstrap": n_bootstrap,
        "threshold": threshold,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    _selected_characteristics(frame, outcomes).to_csv(
        output_dir / "selected_characteristics.csv", index=False
    )

    for outcome_name, y in outcomes.items():
        best_name = best_by_outcome[outcome_name]
        estimator = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("model", clone(models[best_name])),
            ]
        )
        estimator.fit(x, y)
        importance_result = permutation_importance(
            estimator,
            x,
            y,
            scoring="roc_auc",
            n_repeats=5,
            random_state=seed,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": schema.features,
                "importance_mean": importance_result.importances_mean,
                "importance_sd": importance_result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        slug = safe_slug(outcome_name)
        importance.to_csv(output_dir / f"importance_{slug}.csv", index=False)
        plot_importance(importance, outcome_name, output_dir / f"importance_{slug}.png")
        plot_roc(y, all_predictions[outcome_name], outcome_name, output_dir / f"roc_{slug}.png")
        plot_precision_recall(
            y,
            all_predictions[outcome_name],
            outcome_name,
            output_dir / f"pr_{slug}.png",
        )
        plot_calibration(
            y,
            all_predictions[outcome_name],
            outcome_name,
            output_dir / f"calibration_{slug}.png",
        )

    prediction_frame = pd.DataFrame({"id": frame["id"]})
    for outcome_name, model_predictions in all_predictions.items():
        for model_name, probabilities in model_predictions.items():
            column = safe_slug(f"{outcome_name}_{model_name}")
            prediction_frame[column] = probabilities
    prediction_frame.to_csv(
        output_dir / "participant_oof_predictions.csv", index=False
    )
    return performance, audit
