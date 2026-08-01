"""Leakage-resistant feature selection and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import CATEGORICAL_COLUMNS, LEAKAGE_COLUMNS


@dataclass(frozen=True)
class FeatureSchema:
    features: list[str]
    categorical: list[str]
    numerical: list[str]


def infer_feature_schema(frame: pd.DataFrame) -> FeatureSchema:
    """Infer usable predictors while excluding post-outcome leakage variables."""
    features = [column for column in frame.columns if column not in LEAKAGE_COLUMNS]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    numerical = [column for column in features if column not in categorical]
    if not features:
        raise ValueError("No predictors remain after leakage exclusion.")
    return FeatureSchema(features, categorical, numerical)


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """Create median/mode imputation, scaling, and one-hot encoding pipeline."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if schema.numerical:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                schema.numerical,
            )
        )
    if schema.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                schema.categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
