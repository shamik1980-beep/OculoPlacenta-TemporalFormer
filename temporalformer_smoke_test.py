"""Run one forward pass through the proposed TemporalFormer using synthetic data."""

import torch

from oculoplacenta_temporalformer.temporalformer import (
    OculoPlacentaTemporalFormer,
    TemporalFormerConfig,
)


def main() -> None:
    config = TemporalFormerConfig()
    model = OculoPlacentaTemporalFormer(config)
    batch, visits = 3, 3
    inputs = {
        "retinal": torch.randn(batch, visits, config.retinal_dim),
        "biomarkers": torch.randn(batch, visits, config.biomarker_dim),
        "doppler": torch.randn(batch, visits, config.doppler_dim),
        "clinical": torch.randn(batch, visits, config.clinical_dim),
        "modality_mask": torch.tensor(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],
                [[1, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            ],
            dtype=torch.bool,
        ),
        "visit_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.bool
        ),
    }
    outputs = model(**inputs)
    print("Outcome logits:", outputs["logits"].shape)
    print("Time-to-event outputs:", outputs["time_to_event"].shape)


if __name__ == "__main__":
    main()
