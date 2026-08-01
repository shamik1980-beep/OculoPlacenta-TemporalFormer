"""Data loading, validation, checksum verification, and cohort summaries."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .constants import EXPECTED_BOGOTA_SHA256, REQUIRED_BOGOTA_COLUMNS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bogota_checksum(path: Path, strict: bool = True) -> bool:
    """Verify the dataset version used in the manuscript."""
    actual = sha256_file(path)
    matches = actual == EXPECTED_BOGOTA_SHA256
    if strict and not matches:
        raise ValueError(
            "Dataset checksum mismatch. "
            f"Expected {EXPECTED_BOGOTA_SHA256}, received {actual}."
        )
    return matches


def load_bogota_dataset(path: Path) -> pd.DataFrame:
    """Load the Bogotá cohort and validate minimum required fields.

    ``sep=None`` supports comma- or semicolon-separated versions of the source file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, sep=None, engine="python")
    validate_columns(frame.columns, REQUIRED_BOGOTA_COLUMNS)
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    return frame


def validate_columns(columns: Iterable[str], required: set[str]) -> None:
    """Raise a helpful error when required columns are absent."""
    available = set(columns)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def cohort_audit(frame: pd.DataFrame, predictor_count: int) -> dict[str, object]:
    """Create a compact machine-readable cohort audit."""
    return {
        "n": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "early_onset_n": int((frame["preeclampsia_onset"] == 0).sum()),
        "late_onset_n": int((frame["preeclampsia_onset"] == 1).sum()),
        "iugr_n": int(frame["iugr"].sum()),
        "predictor_count_after_leakage_exclusion": int(predictor_count),
    }
