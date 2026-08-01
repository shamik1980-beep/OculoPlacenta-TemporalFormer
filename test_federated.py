import pytest

torch = pytest.importorskip("torch")

from oculoplacenta_temporalformer.federated import weighted_fedavg  # noqa: E402


def test_weighted_fedavg() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0])},
        {"weight": torch.tensor([5.0, 7.0])},
    ]
    result = weighted_fedavg(states, [1, 3])
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))
