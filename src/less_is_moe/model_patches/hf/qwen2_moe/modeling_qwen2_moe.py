"""
Pruned Qwen2-MoE SparseMoeBlock and DecoderLayer with router-logits masking.

Drop-in replacement for transformers.models.qwen2_moe.modeling_qwen2_moe classes.
Adds:
  - ``router_logits_mask`` buffer (1-D, shape ``[num_experts]``, 0 or -inf)
  - ``_set_router_logits_mask`` / ``disable_router_mask`` / ``enable_router_mask``
  - ``layer_idx`` attribute on the MoE block for per-layer pruning
  - Support for per-layer ``num_experts_list`` / ``gate_num_experts`` in config
  - Support for ``layer_experts_idx`` based pruning decisions
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


# ---------------------------------------------------------------------------
# MLP (identical to upstream, kept here so the module is self-contained)
# ---------------------------------------------------------------------------
class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Pruned SparseMoeBlock
# ---------------------------------------------------------------------------
class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.norm_topk_prob = config.norm_topk_prob
        self.layer_idx = 0 if layer_idx is None else layer_idx

        # Per-layer expert count (for pruned checkpoints)
        if getattr(config, "num_experts_list", None):
            idx = self.layer_idx if self.layer_idx < len(config.num_experts_list) else 0
            self.num_experts = config.num_experts_list[idx]
        elif isinstance(config.num_experts, int):
            self.num_experts = config.num_experts
        else:
            idx = self.layer_idx if self.layer_idx < len(config.num_experts) else 0
            self.num_experts = config.num_experts[idx]

        # gate_num_experts: gate output dimension may differ from actual expert count
        if hasattr(config, "gate_num_experts") and config.gate_num_experts is not None:
            if isinstance(config.gate_num_experts, list):
                gate_idx = self.layer_idx if self.layer_idx < len(config.gate_num_experts) else 0
                self.gate_num_experts = config.gate_num_experts[gate_idx]
            else:
                self.gate_num_experts = config.gate_num_experts
            self.top_k = config.num_experts_per_tok
        else:
            self.gate_num_experts = self.num_experts
            self.top_k = min(config.num_experts_per_tok, self.num_experts)

        # Router mask for pruning
        self.router_logits_mask: Optional[torch.Tensor]
        self.register_buffer("router_logits_mask", None, persistent=False)

        # Gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        # Qwen2-MoE specific: shared expert
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        # Toggle to skip router mask (used for teacher model)
        self.use_router_mask = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        # Apply pruning mask
        if self.use_router_mask and self.router_logits_mask is not None:
            router_logits = router_logits + self.router_logits_mask.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.gate_num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            scatter_idx = top_x.to(final_hidden_states.device).unsqueeze(-1).expand_as(current_hidden_states)
            final_hidden_states.scatter_add_(0, scatter_idx, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    # -- Pruning API --

    def _set_router_logits_mask(self, router_logits_mask: torch.Tensor):
        """Set -inf/0 mask to disable selected experts."""
        if router_logits_mask.dim() != 1 or router_logits_mask.shape[0] != self.gate.out_features:
            raise ValueError(f"router_logits_mask must be 1D of length {self.gate.out_features}")
        keep = (router_logits_mask > float("-inf")).sum().item()
        self.top_k = min(self.top_k, int(keep)) if keep > 0 else 0
        self.router_logits_mask = router_logits_mask

    def disable_router_mask(self):
        """Ignore router_logits_mask (for teacher forward)."""
        self.use_router_mask = False

    def enable_router_mask(self):
        """Re-enable router_logits_mask usage."""
        self.use_router_mask = True


# ---------------------------------------------------------------------------
# Pruned DecoderLayer
# ---------------------------------------------------------------------------
class Qwen2MoeDecoderLayer(nn.Module):
    """Drop-in for transformers Qwen2MoeDecoderLayer with pruning support."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Import attention from the upstream module (we don't modify attention)
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            QWEN2MOE_ATTENTION_CLASSES,
            Qwen2MoeRMSNorm,
        )

        self.self_attn = QWEN2MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if (layer_idx not in config.mlp_only_layers) and (
            (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # Determine per-layer expert count
            if hasattr(config, "layer_experts_idx"):
                num_experts = -1 if config.layer_experts_idx[layer_idx] is None else len(config.layer_experts_idx[layer_idx])
            elif getattr(config, "num_experts_list", None):
                num_experts = config.num_experts_list[layer_idx if layer_idx < len(config.num_experts_list) else 0]
            else:
                num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]

            if num_experts < 0:  # no MoE or Norm
                self.mlp = None
                self.post_attention_layernorm = None
            elif num_experts == 0:  # no MoE
                self.mlp = None
                self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            else:
                self.mlp = Qwen2MoeSparseMoeBlock(config, layer_idx)
                self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)
            self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
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
        )
        hidden_states = residual + hidden_states

        # MLP / MoE
        router_logits = None
        if self.post_attention_layernorm is None and self.mlp is None:
            pass
        elif self.mlp is None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states
        else:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)
            if isinstance(mlp_output, tuple):
                hidden_states, router_logits = mlp_output[0], mlp_output[1] if len(mlp_output) >= 2 else None
            else:
                hidden_states = mlp_output
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)

        return outputs
