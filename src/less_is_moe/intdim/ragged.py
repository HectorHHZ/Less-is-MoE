"""Compact, non-uniform MoE checkpoints for GPU inference.

Each layer stores two flat tensors, in original expert-ID order. For expert e,
gate/up is row-major [2*I_e, H], gate rows first; down is [H, I_e]. Widths in
config determine every offset. Zero-width experts retain their router slot.
The ordinary scalar moe_intermediate_size retains its original meaning.
Version 1 remains unchanged for SiLU; version 2 adds explicit activation/bias
metadata for Gemma4 and GPT-OSS, including down bias for zero-width experts.
"""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "RaggedQwen3MoeForCausalLM"
ARCHITECTURES = {
    "qwen2_moe": "RaggedQwen2MoeForCausalLM",
    "olmoe": "RaggedOlmoeForCausalLM",
    "qwen3_moe": ARCHITECTURE,
    "qwen3_5_moe_text": "RaggedQwen3_5MoeForCausalLM",
    "gpt_oss": "RaggedGptOssForCausalLM",
    "gemma4_text": "RaggedGemma4ForCausalLM",
}
LAYOUT = "flat_gate_up_down_v1"
EXTENDED_LAYOUT = "flat_gate_up_down_v2"


def expert_count(config):
    return config.num_local_experts if config.model_type == "gpt_oss" else config.num_experts


def expert_path(config, layer):
    suffix = "experts" if config.model_type == "gemma4_text" else "mlp.experts"
    return f"model.layers.{layer}.{suffix}"


def expert_options(config):
    if config.model_type == "gpt_oss":
        return dict(activation="swigluoai", bias=True)
    if config.model_type == "gemma4_text":
        return dict(activation="gelu_tanh", bias=False)
    return dict(activation="silu", bias=False)


def validate_family(config):
    """Only audited causal-LM expert semantics may use these formats."""
    if config.model_type not in ARCHITECTURES:
        raise ValueError("Unsupported ragged model family")
    if config.model_type == "gemma4_text":
        if config.hidden_activation != "gelu_pytorch_tanh" or not config.enable_moe_block:
            raise ValueError("Ragged Gemma4 requires GELU-tanh and MoE in every layer")
    elif config.hidden_act != "silu":
        raise ValueError("Unsupported routed expert activation")
    if config.model_type == "gpt_oss" and getattr(config, "swiglu_limit", 7.0) != 7.0:
        raise ValueError("Ragged GPT-OSS requires the audited SwiGLU limit of 7")
    if getattr(config, "quantization_config", None):
        raise ValueError("Ragged inference does not support quantized checkpoints")
    if getattr(config, "mlp_only_layers", []) or getattr(config, "decoder_sparse_step", 1) != 1:
        raise ValueError("Ragged inference requires an MoE block in every decoder layer")


def original_width(config):
    return config.intermediate_size if config.model_type in ("olmoe", "gpt_oss") else config.moe_intermediate_size


def validate_metadata(config) -> dict[str, list[int]]:
    validate_family(config)
    meta = getattr(config, "less_is_moe", None)
    extended = config.model_type in ("gpt_oss", "gemma4_text")
    version, layout = (2, EXTENDED_LAYOUT) if extended else (1, LAYOUT)
    if not isinstance(meta, dict) or meta.get("format_version") != version or meta.get("weight_layout") != layout:
        raise ValueError("Missing or unsupported less_is_moe checkpoint format")
    if extended and meta.get("expert_options") != expert_options(config):
        raise ValueError("Checkpoint expert activation/bias metadata does not match its model family")
    widths = meta.get("expert_intermediate_sizes")
    if not isinstance(widths, dict) or set(widths) != {str(i) for i in range(config.num_hidden_layers)}:
        raise ValueError("Expert widths must describe every decoder layer exactly once")
    for layer, values in widths.items():
        if not isinstance(values, list) or len(values) != expert_count(config):
            raise ValueError(f"Layer {layer}: width count must match num_experts")
        if any(type(v) is not int or not 0 <= v <= original_width(config) for v in values):
            raise ValueError(f"Layer {layer}: invalid expert width")
    return widths


class PackedExperts(nn.Module):
    """Compact storage and an explicit HF GPU reference implementation."""

    def __init__(self, widths, hidden_size, *, device=None, dtype=None, activation="silu", bias=False):
        super().__init__()
        self.widths = tuple(widths)
        self.hidden_size = hidden_size
        self.activation = activation
        if activation not in ("silu", "gelu_tanh", "swigluoai"):
            raise ValueError("Unknown ragged activation")
        self.offsets = tuple(accumulate((0, *widths)))
        self.gate_up_proj = nn.Parameter(torch.empty(2 * sum(widths) * hidden_size, device=device, dtype=dtype), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(sum(widths) * hidden_size, device=device, dtype=dtype), requires_grad=False)
        if bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(2 * sum(widths), device=device, dtype=dtype), requires_grad=False)
            self.down_proj_bias = nn.Parameter(torch.empty(len(widths), hidden_size, device=device, dtype=dtype), requires_grad=False)

    def expert_biases(self, expert):
        if not hasattr(self, "gate_up_proj_bias"):
            return None, None, None
        start, width = 2 * self.offsets[expert], self.widths[expert]
        gu = self.gate_up_proj_bias[start:start + 2 * width]
        return gu[:width], gu[width:], self.down_proj_bias[expert]

    def activate(self, gate, up):
        if self.activation == "swigluoai":
            gate, up = gate.clamp(max=7.0), up.clamp(min=-7.0, max=7.0)
            return (up + 1) * (gate * torch.sigmoid(gate * 1.702))
        if self.activation == "gelu_tanh":
            return F.gelu(gate, approximate="tanh") * up
        return F.silu(gate) * up

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
            if width == 0 and not hasattr(self, "down_proj_bias"):
                continue
            tokens, slots = torch.where(top_k_index == expert)
            if tokens.numel() == 0:
                continue
            gate, up, down = self.expert_weights(expert)
            # Match the BF16 rounding boundaries of the stock HF eager experts.
            gu = F.linear(hidden_states[tokens], torch.cat((gate, up)))
            gb, ub, db = self.expert_biases(expert)
            if gb is not None:
                gu = gu + torch.cat((gb, ub))
            g, u = gu.chunk(2, dim=-1)
            values = F.linear(self.activate(g, u), down)
            if db is not None:
                values = values + db
            result.index_add_(0, tokens, (values * top_k_weights[tokens, slots, None]).to(result.dtype))
        return result


@torch.no_grad()
def compact_model(model, handles, drop_plan):
    """Apply an existing L/G drop plan without changing its selected neurons."""
    widths = {}
    selections = {}
    # Validate the entire plan before mutating any weights.
    for h in handles:
        if h.layer_index is None or h.name != expert_path(model.config, h.layer_index):
            raise ValueError("Unexpected causal-LM expert path")
        if model.config.model_type != "gpt_oss" and h.kind == "fused" and (h.gate_up_bias is not None or h.down_bias is not None):
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
    extended = config.model_type in ("gpt_oss", "gemma4_text")
    config.less_is_moe = dict(format_version=2 if extended else 1,
                              weight_layout=EXTENDED_LAYOUT if extended else LAYOUT,
                              expert_intermediate_sizes=widths)
    if extended:
        config.less_is_moe["expert_options"] = expert_options(config)
    validate_metadata(config)
    original_count = sum(p.numel() for p in model.parameters())
    for h in handles:
        sample = h.expert_weights(0)[0]
        packed = PackedExperts(widths[str(h.layer_index)], h.hidden_size, device=sample.device, dtype=sample.dtype,
                               **expert_options(config))
        for e, ids in enumerate(selections[h.layer_index]):
            idx = torch.tensor(ids, device=sample.device, dtype=torch.long)
            old_gate, old_up, old_down = h.expert_weights(e)
            new_gate, new_up, new_down = packed.expert_weights(e)
            new_gate.copy_(old_gate.index_select(0, idx))
            new_up.copy_(old_up.index_select(0, idx))
            new_down.copy_(old_down.index_select(1, idx))
            if hasattr(packed, "gate_up_proj_bias"):
                old_bias = h.gate_up_bias[e]
                gb, ub = (old_bias[::2], old_bias[1::2]) if h.fused.pairing == "interleaved" else old_bias.chunk(2)
                new_gb, new_ub, new_db = packed.expert_biases(e)
                new_gb.copy_(gb.index_select(0, idx))
                new_ub.copy_(ub.index_select(0, idx))
                new_db.copy_(h.down_bias[e])
        parent, name = h.name.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, packed)
    model.config.less_is_moe = config.less_is_moe
    model.config.architectures = [ARCHITECTURES[model.config.model_type]]
    return dict(format_version=config.less_is_moe["format_version"], original_parameters=original_count,
                compact_parameters=sum(p.numel() for p in model.parameters()),
                expert_intermediate_sizes=widths)


def save_checkpoint(model, directory, tokenizer=None):
    validate_metadata(model.config)
    # HF 5 normally reverses its fused-expert conversion when saving Qwen.
    # This checkpoint deliberately stores our flat parameters as they are.
    model.save_pretrained(directory, save_original_format=False, max_shard_size="4GB")
    # save_pretrained uses the live Python class, which may still be the stock
    # class immediately after in-place compaction. Explicitly mark the format.
    model.config.architectures = [ARCHITECTURES[model.config.model_type]]
    model.config.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)


def load_checkpoint(directory, *, dtype=torch.bfloat16, **kwargs):
    from transformers import AutoConfig
    from . import ragged_hf
    if not torch.cuda.is_available():
        raise RuntimeError("Ragged checkpoint verification requires a GPU")
    config = AutoConfig.from_pretrained(directory)
    validate_metadata(config)
    if config.model_type == "gpt_oss":
        kwargs["attn_implementation"] = "eager"
    cls = getattr(ragged_hf, ARCHITECTURES[config.model_type])
    from .ragged_hf import gpu_device_map
    device_map = kwargs.pop("device_map", gpu_device_map())
    model = cls.from_pretrained(Path(directory), dtype=dtype, device_map=device_map, **kwargs).eval()
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("The complete checkpoint must fit on the visible GPUs; CPU/disk offload is unsupported")
    return model
