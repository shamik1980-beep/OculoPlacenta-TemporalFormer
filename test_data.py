from pathlib import Path

import pandas as pd
import pytest

from oculoplacenta_temporalformer.data import load_bogota_dataset


def test_load_minimum_valid_dataset(tmp_path: Path) -> None:
    path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {"id": [1, 2], "preeclampsia_onset": [0, 1], "iugr": [1, 0]}
    ).to_csv(path, index=False)
    loaded = load_bogota_dataset(path)
    assert loaded.shape == (2, 3)


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "cohort.csv"
    pd.DataFrame({"id": [1], "iugr": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_bogota_dataset(path)
