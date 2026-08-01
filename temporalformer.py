"""PyTorch prototype of the proposed longitudinal multimodal transformer.

This module is a research scaffold. It is not trained or validated by the Bogotá
cohort, which contains no paired retinal, biomarker, Doppler, or serial data.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The TemporalFormer prototype requires PyTorch. Install with "
        "`pip install -e .[deep-learning]`."
    ) from exc


@dataclass(frozen=True)
class TemporalFormerConfig:
    retinal_dim: int = 768
    biomarker_dim: int = 8
    doppler_dim: int = 12
    clinical_dim: int = 20
    model_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    max_visits: int = 4
    n_outcomes: int = 2


class ModalityProjector(nn.Module):
    """Project one modality into a shared embedding space."""

    def __init__(self, input_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class OculoPlacentaTemporalFormer(nn.Module):
    """Missing-modality-aware, visit-aware multimodal transformer.

    Inputs have shape ``[batch, visits, feature_dim]``. ``modality_mask`` has
    shape ``[batch, visits, 4]`` and uses ``True`` for available modalities.
    ``visit_mask`` has shape ``[batch, visits]`` and uses ``True`` for visits
    that exist. The output contains logits for pre-eclampsia and FGR.
    """

    def __init__(self, config: TemporalFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.projectors = nn.ModuleList(
            [
                ModalityProjector(config.retinal_dim, config.model_dim, config.dropout),
                ModalityProjector(config.biomarker_dim, config.model_dim, config.dropout),
                ModalityProjector(config.doppler_dim, config.model_dim, config.dropout),
                ModalityProjector(config.clinical_dim, config.model_dim, config.dropout),
            ]
        )
        self.modality_embedding = nn.Embedding(4, config.model_dim)
        self.visit_embedding = nn.Embedding(config.max_visits, config.model_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, config.model_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.n_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.normalisation = nn.LayerNorm(config.model_dim)
        self.outcome_head = nn.Linear(config.model_dim, config.n_outcomes)
        self.time_to_event_head = nn.Linear(config.model_dim, config.n_outcomes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(
        self,
        retinal: Tensor,
        biomarkers: Tensor,
        doppler: Tensor,
        clinical: Tensor,
        modality_mask: Tensor,
        visit_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch, visits, _ = retinal.shape
        if visits > self.config.max_visits:
            raise ValueError(
                f"Received {visits} visits; maximum is {self.config.max_visits}."
            )
        inputs = [retinal, biomarkers, doppler, clinical]
        tokens = torch.stack(
            [projector(values) for projector, values in zip(self.projectors, inputs, strict=True)],
            dim=2,
        )  # [B, V, M, D]

        modality_ids = torch.arange(4, device=tokens.device)
        tokens = tokens + self.modality_embedding(modality_ids)[None, None, :, :]
        visit_ids = torch.arange(visits, device=tokens.device)
        tokens = tokens + self.visit_embedding(visit_ids)[None, :, None, :]

        available = modality_mask.bool().unsqueeze(-1)
        missing = self.missing_token.expand(batch, visits, 4, -1)
        tokens = torch.where(available, tokens, missing)
        tokens = tokens.reshape(batch, visits * 4, self.config.model_dim)

        token_valid = (visit_mask.bool().unsqueeze(-1) & modality_mask.bool()).reshape(
            batch, visits * 4
        )
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device),
                ~token_valid,
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        representation = self.normalisation(encoded[:, 0])
        return {
            "logits": self.outcome_head(representation),
            "time_to_event": self.time_to_event_head(representation),
            "representation": representation,
        }
