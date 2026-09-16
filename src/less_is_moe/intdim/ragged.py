"""Version 1 compact, non-uniform Qwen3-MoE checkpoints (GPU inference).

Each layer stores two flat tensors, in original expert-ID order. For expert e,
gate/up is row-major [2*I_e, H], gate rows first; down is [H, I_e]. Widths in
config determine every offset. Zero-width experts retain their router slot.
The ordinary scalar moe_intermediate_size retains its original meaning.
"""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "RaggedQwen3MoeForCausalLM"
LAYOUT = "flat_gate_up_down_v1"


def validate_metadata(config) -> dict[str, list[int]]:
    if config.model_type != "qwen3_moe" or config.hidden_act != "silu":
        raise ValueError("Ragged v1 supports only SiLU Qwen3-MoE")
    if getattr(config, "quantization_config", None):
        raise ValueError("Ragged v1 does not support quantized checkpoints")
    if getattr(config, "mlp_only_layers", []) or config.decoder_sparse_step != 1:
        raise ValueError("Ragged v1 requires an MoE block in every decoder layer")
    meta = getattr(config, "less_is_moe", None)
    if not isinstance(meta, dict) or meta.get("format_version") != 1 or meta.get("weight_layout") != LAYOUT:
        raise ValueError("Missing or unsupported less_is_moe checkpoint format")
    widths = meta.get("expert_intermediate_sizes")
    if not isinstance(widths, dict) or set(widths) != {str(i) for i in range(config.num_hidden_layers)}:
        raise ValueError("Expert widths must describe every decoder layer exactly once")
    for layer, values in widths.items():
        if not isinstance(values, list) or len(values) != config.num_experts:
            raise ValueError(f"Layer {layer}: width count must match num_experts")
        if any(type(v) is not int or not 0 <= v <= config.moe_intermediate_size for v in values):
            raise ValueError(f"Layer {layer}: invalid expert width")
    return widths


class PackedExperts(nn.Module):
    """Compact storage and an explicit HF GPU reference implementation."""

    def __init__(self, widths, hidden_size, *, device=None, dtype=None):
        super().__init__()
        self.widths = tuple(widths)
        self.hidden_size = hidden_size
        self.offsets = tuple(accumulate((0, *widths)))
        self.gate_up_proj = nn.Parameter(torch.empty(2 * sum(widths) * hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(sum(widths) * hidden_size, device=device, dtype=dtype), requires_grad=False)

    def expert_weights(self, expert):
        width, hidden = self.widths[expert], self.hidden_size
        start = self.offsets[expert] * hidden
        gu = self.gate_up_proj[2 * start:2 * start + 2 * width * hidden].view(2 * width, hidden)
        down = self.down_proj[start:start + width * hidden].view(hidden, width)
        return gu[:width], gu[width:], down

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if not hidden_states.is_cuda:
            raise RuntimeError("Ragged expert execution requires a GPU")
        result = torch.zeros_like(hidden_states)
        for expert, width in enumerate(self.widths):
            if width == 0:
                continue
            tokens, slots = torch.where(top_k_index == expert)
            if tokens.numel() == 0:
                continue
            gate, up, down = self.expert_weights(expert)
            # Match the BF16 rounding boundaries of the stock HF eager experts.
            gu = F.linear(hidden_states[tokens], torch.cat((gate, up)))
            g, u = gu.chunk(2, dim=-1)
            values = F.linear(F.silu(g) * u, down)
            result.index_add_(0, tokens, (values * top_k_weights[tokens, slots, None]).to(result.dtype))
        return result


@torch.no_grad()
def compact_model(model, handles, drop_plan):
    """Apply an existing L/G drop plan without changing its selected neurons."""
    widths = {}
    selections = {}
    # Validate the entire plan before mutating any weights.
    for h in handles:
        if h.layer_index is None or h.name != f"model.layers.{h.layer_index}.mlp.experts":
            raise ValueError("Unexpected Qwen3-MoE expert path")
        if h.kind == "fused" and (h.gate_up_bias is not None or h.down_bias is not None):
            raise ValueError("Ragged v1 supports bias-free experts only")
        keep = []
        for e in range(h.num_experts):
            dropped = drop_plan[h.layer_index][e]
            if len(set(dropped)) != len(dropped) or any(type(i) is not int or not 0 <= i < h.intermediate_size for i in dropped):
                raise ValueError("Invalid dropped-neuron indices")
            ids = set(dropped)
            keep.append([i for i in range(h.intermediate_size) if i not in ids])
        widths[str(h.layer_index)] = [len(ids) for ids in keep]
        selections[h.layer_index] = keep
    import copy
    config = copy.deepcopy(model.config)
    config.less_is_moe = dict(format_version=1, weight_layout=LAYOUT, expert_intermediate_sizes=widths)
    validate_metadata(config)
    original_count = sum(p.numel() for p in model.parameters())
    for h in handles:
        sample = h.expert_weights(0)[0]
        packed = PackedExperts(widths[str(h.layer_index)], h.hidden_size, device=sample.device, dtype=sample.dtype)
        for e, ids in enumerate(selections[h.layer_index]):
            idx = torch.tensor(ids, device=sample.device, dtype=torch.long)
            old_gate, old_up, old_down = h.expert_weights(e)
            new_gate, new_up, new_down = packed.expert_weights(e)
            new_gate.copy_(old_gate.index_select(0, idx))
            new_up.copy_(old_up.index_select(0, idx))
            new_down.copy_(old_down.index_select(1, idx))
        parent, name = h.name.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, packed)
    model.config.less_is_moe = config.less_is_moe
    model.config.architectures = [ARCHITECTURE]
    return dict(format_version=1, original_parameters=original_count,
                compact_parameters=sum(p.numel() for p in model.parameters()),
                expert_intermediate_sizes=widths)


def save_checkpoint(model, directory, tokenizer=None):
    validate_metadata(model.config)
    model.save_pretrained(directory)
    # save_pretrained uses the live Python class, which may still be the stock
    # class immediately after in-place compaction. Explicitly mark the format.
    model.config.architectures = [ARCHITECTURE]
    model.config.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)


def load_checkpoint(directory, *, dtype=torch.bfloat16, **kwargs):
    from .ragged_hf import RaggedQwen3MoeForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("Ragged checkpoint verification requires a GPU")
    return RaggedQwen3MoeForCausalLM.from_pretrained(
        Path(directory), dtype=dtype, device_map="cuda", **kwargs).eval()
