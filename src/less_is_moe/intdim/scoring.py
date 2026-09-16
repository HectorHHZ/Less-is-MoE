"""Compatibility helpers delegating to the unified :mod:`.prune` implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .discover import MoeLayerHandle, discover

Scores = dict[int, dict[int, torch.Tensor]]


def collect_scores(
    model: torch.nn.Module,
    calib_batches: Sequence[torch.Tensor],
    handles: list[MoeLayerHandle] | None = None,
) -> Scores:
    """Score routed gate/up/down weights; keep the legacy reduction order.

    Calibration uses labels=input_ids and averages per-sample mean(abs(grad)).
    Router, shared-expert, attention and bias parameters remain frozen. As in
    the released collectors, the model ends in eval mode with gradients off.
    """
    from .prune import collect_neuron_gradient_scores

    handles = discover(model) if handles is None else handles
    indices = [h.layer_index for h in handles]
    if None in indices or len(set(indices)) != len(indices):
        raise ValueError("Scoring requires one MoE handle per numbered decoder layer")
    return collect_neuron_gradient_scores(model, handles, list(calib_batches))


def select_expert_units(scores: Scores, drop_ratio: float) -> dict[int, torch.Tensor]:
    """Return ascending kept indices; preserve legacy bottom-k tie handling."""
    from .prune import decide_neurons_to_drop

    if not math.isfinite(drop_ratio) or not 0 <= drop_ratio < 1:
        raise ValueError("Structural IntDim-E requires 0 <= drop_ratio < 1")
    dropped = decide_neurons_to_drop(scores, drop_ratio)
    kept = {}
    for layer, experts in scores.items():
        rows = []
        for e in range(len(experts)):
            values = experts[e]
            mask = torch.ones(len(values), dtype=torch.bool, device=values.device)
            mask[dropped[layer][e]] = False
            rows.append(torch.nonzero(mask, as_tuple=False).flatten())
        kept[layer] = torch.stack(rows)
    return kept
