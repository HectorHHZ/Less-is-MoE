"""Explicit Gemma 4 port of the released IntDim-E zero-mask method.

The stock text model stores routed experts directly on each decoder layer.
Fixed concatenated gate/up rows are used throughout; this module never calls
intdim discovery, scoring, selection or structural pruning. The dense MLP,
router and all other parameters remain unchanged.
"""

from collections.abc import Sequence

import torch

from .neuron_drop_qwen15_moe import decide_neurons_to_drop

__all__ = ["collect_neuron_gradient_scores", "decide_neurons_to_drop", "zero_dropped_neurons"]


def _experts(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    if model.config.model_type != "gemma4_text":
        raise ValueError("Gemma 4 reference requires a stock Gemma4ForCausalLM text model")
    experts = model.model.layers[layer].experts
    count, hidden, width = (model.config.num_experts, model.config.hidden_size,
                            model.config.moe_intermediate_size)
    expected = {"gate_up_proj": (count, 2 * width, hidden), "down_proj": (count, hidden, width)}
    for name, shape in expected.items():
        parameter = getattr(experts, name)
        if tuple(parameter.shape) != shape or not parameter.is_floating_point():
            raise ValueError(f"Unsupported Gemma 4 {name}: expected floating-point {shape}")
    return experts


def collect_neuron_gradient_scores(
    model: torch.nn.Module,
    calib_batches: Sequence[torch.Tensor],
    moe_layer_indices: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    """Port the released per-expert gradient reduction to concatenated rows."""
    if not calib_batches:
        raise ValueError("Calibration batches must not be empty")
    layers = {layer: _experts(model, layer) for layer in moe_layer_indices}
    width, hidden = model.config.moe_intermediate_size, model.config.hidden_size
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for experts in layers.values():
        experts.gate_up_proj.requires_grad_(True)
        experts.down_proj.requires_grad_(True)
    scores = {layer: {e: torch.zeros(width, dtype=torch.float32)
                      for e in range(model.config.num_experts)} for layer in layers}
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
                        raise RuntimeError(f"No expert weight gradients in Gemma 4 layer {layer}")
                    projections = (gu[e, :width, :], gu[e, width:, :], down[e])
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
    """Zero gate/up rows and down columns while keeping all original shapes."""
    width = model.config.moe_intermediate_size
    for layer in moe_layer_indices:
        experts = _experts(model, layer)
        for e, dropped in drop_per_layer.get(layer, {}).items():
            ids = torch.tensor(dropped, dtype=torch.long, device=experts.gate_up_proj.device)
            if ids.numel() == 0:
                continue
            if ids.min() < 0 or ids.max() >= width:
                raise ValueError("Dropped Gemma 4 neuron index is out of range")
            experts.gate_up_proj[e, ids, :] = 0
            experts.gate_up_proj[e, ids + width, :] = 0
            experts.down_proj[e, :, ids] = 0
