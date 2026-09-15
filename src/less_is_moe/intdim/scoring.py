"""IntDim-E scoring/selection with the released mean-absolute-gradient rule."""

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
    handles = discover(model) if handles is None else handles
    indices = [h.layer_index for h in handles]
    if None in indices or len(set(indices)) != len(indices):
        raise ValueError("Scoring requires one MoE handle per numbered decoder layer")
    if not calib_batches:
        raise ValueError("Calibration batches must not be empty")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for h in handles:
        if h.kind == "fused":
            h.gate_up.requires_grad_(True)
            h.down.requires_grad_(True)
        else:
            for e in range(h.num_experts):
                for weight in h.expert_weights(e):
                    weight.requires_grad_(True)
    scores = {
        h.layer_index: {e: torch.zeros(h.intermediate_size, dtype=torch.float32) for e in range(h.num_experts)}
        for h in handles
    }
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        for batch in calib_batches:
            tokens = batch.to(device)
            model(tokens, labels=tokens).loss.backward()
            for h in handles:
                if h.kind == "fused":
                    gu, down = h.gate_up.grad, h.down.grad
                    if gu is None or down is None:
                        continue
                    if h.fused.gate_up_unit_axis == 2:
                        gu = gu.transpose(1, 2)
                    if h.fused.pairing == "interleaved":
                        gate, up = gu[:, 0::2], gu[:, 1::2]
                    else:
                        gate, up = gu[:, :h.intermediate_size], gu[:, h.intermediate_size:]
                    gate_imp = gate.float().abs().sum(dim=2)
                    up_imp = up.float().abs().sum(dim=2)
                    hidden_axis = 2 if h.fused.down_unit_axis == 1 else 1
                    down_imp = down.float().abs().sum(dim=hidden_axis)
                    values = ((gate_imp + up_imp + down_imp) / (3 * h.hidden_size)).cpu()
                    for e in range(h.num_experts):
                        scores[h.layer_index][e] += values[e]
                else:
                    for e in range(h.num_experts):
                        importance = torch.zeros(h.intermediate_size, dtype=torch.float32)
                        elements = 0
                        for weight, axis in zip(h.expert_weights(e), (1, 1, 0)):
                            if weight.grad is not None:
                                importance += weight.grad.float().abs().sum(dim=axis).cpu()
                                elements += weight.shape[axis]
                        if elements:
                            importance /= elements
                        scores[h.layer_index][e] += importance
            model.zero_grad(set_to_none=True)
        for experts in scores.values():
            for score in experts.values():
                score /= len(calib_batches)
        return scores
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def select_expert_units(scores: Scores, drop_ratio: float) -> dict[int, torch.Tensor]:
    """Return ascending kept indices; preserve legacy bottom-k tie handling."""
    if not math.isfinite(drop_ratio) or not 0 <= drop_ratio < 1:
        raise ValueError("Structural IntDim-E requires 0 <= drop_ratio < 1")
    kept = {}
    for layer, experts in scores.items():
        rows = []
        for e in range(len(experts)):
            values = experts[e]
            count = int(len(values) * drop_ratio)
            dropped = torch.topk(values, count, largest=False).indices if count else []
            mask = torch.ones(len(values), dtype=torch.bool, device=values.device)
            mask[dropped] = False
            rows.append(torch.nonzero(mask, as_tuple=False).flatten())
        kept[layer] = torch.stack(rows)
    return kept
