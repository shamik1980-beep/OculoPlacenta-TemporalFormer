"""Project-wide constants and schema definitions."""

from __future__ import annotations

__version__ = "0.1.0"
DEFAULT_SEED = 20260731
EXPECTED_BOGOTA_SHA256 = (
    "319e79453531aa2bb48cdf3c05099601e965dfce3c15f4e924a25c7909fd69fe"
)

REQUIRED_BOGOTA_COLUMNS = {
    "id",
    "preeclampsia_onset",
    "iugr",
}

# Variables unavailable before diagnosis or delivery are intentionally excluded.
LEAKAGE_COLUMNS = {
    "id",
    "preeclampsia_onset",
    "gestational_age_delivery",
    "delivery_type",
    "newborn_weight",
    "newborn_vital_status",
    "newborn_malformations",
    "eclampsia_hellp",
    "iugr",
    "newborn_sex",
}

CATEGORICAL_COLUMNS = [
    "marital_status",
    "education_level",
    "occupation",
    "socioeconomic_level",
]

SELECTED_CHARACTERISTIC_COLUMNS = [
    "maternal_age",
    "bmi",
    "education_level",
    "time_relationship_partner",
    "pe_history_personal",
    "pe_history_family",
    "hipertension_history_personal",
    "alergy_history_personal",
    "obit_history_family",
    "hypertension_history_family",
    "n_abortions",
    "n_pregnancies",
    "primigravidity",
    "primipaternity",
    "n_sexual_partners",
]
