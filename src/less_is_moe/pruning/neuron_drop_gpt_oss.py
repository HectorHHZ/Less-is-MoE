"""Explicit GPT-OSS port of the released IntDim-E zero-mask method.

This reference uses the fixed Transformers GPT-OSS layout, not intdim
discovery/handles/scoring. It preserves mean(abs(gradient)) over gate/up/down
weights and reuses the released per-expert bottom-k selector. Biases do not
contribute to the score. Floating-point expert weights are required (no MXFP4).
"""

from collections.abc import Sequence

import torch

from .neuron_drop_qwen15_moe import decide_neurons_to_drop

__all__ = ["collect_neuron_gradient_scores", "decide_neurons_to_drop", "zero_dropped_neurons"]


def _experts(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    if model.config.model_type != "gpt_oss":
        raise ValueError("GPT-OSS reference requires a stock GPT-OSS causal LM")
    experts = model.model.layers[layer].mlp.experts
    count, hidden, width = (model.config.num_local_experts, model.config.hidden_size,
                            model.config.intermediate_size)
    expected = {
        "gate_up_proj": (count, hidden, 2 * width),
        "down_proj": (count, width, hidden),
        "gate_up_proj_bias": (count, 2 * width),
        "down_proj_bias": (count, hidden),
    }
    for name, shape in expected.items():
        parameter = getattr(experts, name)
        if tuple(parameter.shape) != shape or not parameter.is_floating_point():
            raise ValueError(f"Unsupported GPT-OSS {name}: expected floating-point {shape}")
    return experts


def collect_neuron_gradient_scores(
    model: torch.nn.Module,
    calib_batches: Sequence[torch.Tensor],
    moe_layer_indices: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Port the released per-expert gradient reduction to interleaved columns."""
    if not calib_batches:
        raise ValueError("Calibration batches must not be empty")
    layers = {layer: _experts(model, layer) for layer in moe_layer_indices}
    width, hidden = model.config.intermediate_size, model.config.hidden_size
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for experts in layers.values():
        experts.gate_up_proj.requires_grad_(True)
        experts.down_proj.requires_grad_(True)
    scores = {layer: {e: torch.zeros(width, dtype=torch.float32)
                      for e in range(model.config.num_local_experts)} for layer in layers}
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        for batch in calib_batches:
            tokens = batch.to(next(model.parameters()).device)
            model(tokens, labels=tokens).loss.backward()
            for layer, experts in layers.items():
                for e in scores[layer]:
                    gu, down = experts.gate_up_proj.grad, experts.down_proj.grad
                    if gu is None or down is None:
                        raise RuntimeError(f"No expert weight gradients in GPT-OSS layer {layer}")
                    # Explicit (I,H) gate/up and (H,I) down, as in the old
                    # Linear implementation; no detected axis/pairing is used.
                    projections = (gu[e, :, 0::2].T, gu[e, :, 1::2].T, down[e].T)
                    importance = torch.zeros(width, dtype=torch.float32)
                    for gradient, axis in zip(projections, (1, 1, 0)):
                        importance += gradient.float().abs().sum(dim=axis).cpu()
                    scores[layer][e] += importance / (3 * hidden)
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


@torch.no_grad()
def zero_dropped_neurons(
    model: torch.nn.Module,
    drop_per_layer: dict[int, dict[int, list[int]]],
    moe_layer_indices: Sequence[int],
) -> None:
    """Zero gate/up columns and biases plus down rows; preserve output biases.

    Tensor shapes and config widths stay unchanged. GPT-OSS's gate/up columns
    are interleaved: neuron j owns columns 2*j and 2*j+1.
    """
    for layer in moe_layer_indices:
        experts = _experts(model, layer)
        for e, dropped in drop_per_layer.get(layer, {}).items():
            ids = torch.tensor(dropped, dtype=torch.long, device=experts.gate_up_proj.device)
            if ids.numel() == 0:
                continue
            if ids.min() < 0 or ids.max() >= model.config.intermediate_size:
                raise ValueError("Dropped GPT-OSS neuron index is out of range")
            experts.gate_up_proj[e, :, 2 * ids] = 0
            experts.gate_up_proj[e, :, 2 * ids + 1] = 0
            experts.gate_up_proj_bias[e, 2 * ids] = 0
            experts.gate_up_proj_bias[e, 2 * ids + 1] = 0
            experts.down_proj[e, ids, :] = 0
