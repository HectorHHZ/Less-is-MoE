"""
Pruned OLMoE SparseMoeBlock and DecoderLayer with per-layer expert counts.

Drop-in replacement for ``transformers.models.olmoe.modeling_olmoe`` classes.
Handles checkpoints saved by ``scripts/expert_drop_olmoe.py`` /
``scripts/neuron_structure_drop_olmoe.py`` whose ``config.num_experts`` is a
**per-layer list** (e.g. ``[32, 32, 33, 32, ...]``) instead of a single int.

Key differences from upstream:
  - ``OlmoeSparseMoeBlock.__init__`` accepts an optional ``layer_idx`` and reads
    ``config.num_experts[layer_idx]`` when ``num_experts`` is a list.
  - ``top_k`` is clamped by the per-layer expert count so a model pruned below
    the original ``num_experts_per_tok`` (e.g. 8) still loads/forwards correctly.
  - ``OlmoeDecoderLayer.__init__`` forwards ``layer_idx`` into the MoE block.
  - All other behavior (forward graph, output shape, router-logits return) is
    bit-identical to the upstream implementation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeMLP,
    OlmoeRMSNorm,
    OLMOE_ATTENTION_CLASSES,
)
from transformers.cache_utils import Cache


def _resolve_per_layer_num_experts(config, layer_idx: int) -> int:
    """Return the routed-expert count for ``layer_idx``.

    Pruning scripts may save the count as a per-layer list. This helper
    centralizes the lookup so both the MoE block and the decoder layer agree.
    """
    n = getattr(config, "num_experts", None)
    if isinstance(n, int):
        return n
    if isinstance(n, (list, tuple)):
        idx = layer_idx if 0 <= layer_idx < len(n) else 0
        return int(n[idx])
    raise TypeError(
        f"OLMoE config.num_experts must be int or list[int], got {type(n).__name__}"
    )


class OlmoeSparseMoeBlock(nn.Module):
    """Pruning-aware OLMoE MoE block.

    Forward semantics match the upstream class. The only changes are in
    ``__init__`` to handle per-layer expert counts.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = 0 if layer_idx is None else layer_idx
        self.num_experts = _resolve_per_layer_num_experts(config, self.layer_idx)
        # Pruning may bring a layer below the original top-k — clamp.
        self.top_k = min(int(config.num_experts_per_tok), self.num_experts)
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class OlmoeDecoderLayer(nn.Module):
    """Same as the upstream layer, but threads ``layer_idx`` to the MoE block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OLMOE_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx,
        )
        self.mlp = OlmoeSparseMoeBlock(config, layer_idx=layer_idx)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected — MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs
