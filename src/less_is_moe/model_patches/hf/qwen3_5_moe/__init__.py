"""Hugging Face patch for pruned Qwen3.5-MoE checkpoints.

The HF batched-expert implementation requires one scalar ``num_experts`` for
all layers.  Uneven/global expert-count checkpoints are rejected by
``validate_hf_patch_config``; uniform expert pruning, router masking, and
neuron-structured checkpoints are supported.
"""

from .modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
)

__all__ = [
    "Qwen3_5MoeDecoderLayer",
    "Qwen3_5MoeSparseMoeBlock",
    "Qwen3_5MoeTopKRouter",
]
