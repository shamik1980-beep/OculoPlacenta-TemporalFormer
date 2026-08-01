import pytest

torch = pytest.importorskip("torch")

from oculoplacenta_temporalformer.temporalformer import (  # noqa: E402
    OculoPlacentaTemporalFormer,
    TemporalFormerConfig,
)


def test_temporalformer_output_shapes() -> None:
    config = TemporalFormerConfig(
        retinal_dim=16,
        biomarker_dim=4,
        doppler_dim=5,
        clinical_dim=6,
        model_dim=16,
        n_heads=4,
        n_layers=1,
        max_visits=3,
    )
    model = OculoPlacentaTemporalFormer(config)
    batch, visits = 2, 3
    output = model(
        retinal=torch.randn(batch, visits, 16),
        biomarkers=torch.randn(batch, visits, 4),
        doppler=torch.randn(batch, visits, 5),
        clinical=torch.randn(batch, visits, 6),
        modality_mask=torch.ones(batch, visits, 4, dtype=torch.bool),
        visit_mask=torch.ones(batch, visits, dtype=torch.bool),
    )
    assert output["logits"].shape == (batch, 2)
    assert output["time_to_event"].shape == (batch, 2)
