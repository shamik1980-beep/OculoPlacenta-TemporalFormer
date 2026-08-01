"""Framework-agnostic federated averaging utilities for PyTorch models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

try:
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("Federated utilities require PyTorch.") from exc


def weighted_fedavg(
    client_states: Sequence[dict[str, Tensor]], client_sizes: Sequence[int]
) -> OrderedDict[str, Tensor]:
    """Compute sample-size-weighted FedAvg over compatible state dictionaries."""
    if not client_states:
        raise ValueError("At least one client state is required.")
    if len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes must have equal length.")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("All client sample sizes must be positive.")
    keys = list(client_states[0].keys())
    if any(list(state.keys()) != keys for state in client_states[1:]):
        raise ValueError("All client state dictionaries must have identical keys.")

    total = float(sum(client_sizes))
    averaged: OrderedDict[str, Tensor] = OrderedDict()
    for key in keys:
        accumulator = client_states[0][key].detach().clone() * (client_sizes[0] / total)
        for state, size in zip(client_states[1:], client_sizes[1:], strict=True):
            accumulator.add_(state[key].detach(), alpha=size / total)
        averaged[key] = accumulator
    return averaged
