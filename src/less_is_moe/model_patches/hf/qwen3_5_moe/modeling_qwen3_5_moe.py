"""
Pruned Qwen3.5-MoE SparseMoeBlock with router-logits masking.

Differences from qwen3_moe patch:
- Experts are stored as 3D tensors in Qwen3_5MoeExperts (gate_up_proj, down_proj),
  not as an nn.ModuleList. Pruning still uses router-logits masking (soft prune);
  full expert removal from the 3D tensor can be done at save time.
- Router is Qwen3_5MoeTopKRouter (self.weight directly, no self.gate.weight).
- Block has a shared expert + shared_expert_gate (always on, not routed).
- The block is present on EVERY decoder layer (no mlp_only_layers / decoder_sparse_step).

Strategy: we only need to intercept the router to apply a mask on router_logits,
and keep everything else upstream. Cleanest approach: subclass Qwen3_5MoeTopKRouter
and Qwen3_5MoeSparseMoeBlock to carry layer_idx + router_logits_mask, and expose
the same API (_set_router_logits_mask, disable/enable) used by the trainer.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer as _UpstreamQwen3_5MoeDecoderLayer,
)


class Qwen3_5MoeTopKRouter(nn.Module):
    """Drop-in for upstream Qwen3_5MoeTopKRouter with router_logits_mask support."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

        # Pruning mask (additive, 0 or -inf per expert)
        self.router_logits_mask: Optional[torch.Tensor]
        self.register_buffer("router_logits_mask", None, persistent=False)
        self.use_router_mask = True

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)

        if self.use_router_mask and self.router_logits_mask is not None:
            router_logits = router_logits + self.router_logits_mask.to(router_logits.dtype)

        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Drop-in for upstream Qwen3_5MoeSparseMoeBlock. Adds layer_idx + mask API."""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = 0 if layer_idx is None else layer_idx

        # Use upstream MLP + Experts unchanged
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeExperts,
            Qwen3_5MoeMLP,
        )

        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        # Return (hidden, router_logits) when trainer wants it (patched to match qwen3_moe API)
        return expert_output, router_logits

    # -- Pruning API (mirrors qwen3_moe patch) --

    def _set_router_logits_mask(self, router_logits_mask: torch.Tensor):
        if router_logits_mask.dim() != 1 or router_logits_mask.shape[0] != self.gate.num_experts:
            raise ValueError(f"router_logits_mask must be 1D of length {self.gate.num_experts}")
        keep = (router_logits_mask > float("-inf")).sum().item()
        self.gate.top_k = min(self.gate.top_k, int(keep)) if keep > 0 else 0
        self.gate.router_logits_mask = router_logits_mask

    def disable_router_mask(self):
        self.gate.use_router_mask = False

    def enable_router_mask(self):
        self.gate.use_router_mask = True


class Qwen3_5MoeDecoderLayer(_UpstreamQwen3_5MoeDecoderLayer):
    """Upstream decoder with its layer index forwarded to the patched MoE."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Upstream constructs Qwen3_5MoeSparseMoeBlock(config) without the
        # index.  Recording it here keeps per-layer router masks addressable
        # without changing the decoder forward graph.
        if hasattr(self.mlp, "layer_idx"):
            self.mlp.layer_idx = layer_idx
