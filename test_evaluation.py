import numpy as np

from oculoplacenta_temporalformer.evaluation import threshold_metrics


def test_threshold_metrics_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = threshold_metrics(y, probabilities)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
