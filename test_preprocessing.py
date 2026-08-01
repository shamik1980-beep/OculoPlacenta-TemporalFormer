import pandas as pd

from oculoplacenta_temporalformer.preprocessing import infer_feature_schema


def test_leakage_columns_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "preeclampsia_onset": [0],
            "iugr": [1],
            "gestational_age_delivery": [31],
            "maternal_age": [33],
        }
    )
    schema = infer_feature_schema(frame)
    assert schema.features == ["maternal_age"]
