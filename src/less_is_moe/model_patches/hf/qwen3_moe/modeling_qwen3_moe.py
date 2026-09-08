"""
Pruned Qwen3-MoE SparseMoeBlock and DecoderLayer with router-logits masking.

Drop-in replacement for transformers.models.qwen3_moe.modeling_qwen3_moe classes.
Adds:
  - ``router_logits_mask`` buffer (1-D, shape ``[num_experts]``, 0 or -inf)
  - ``_set_router_logits_mask`` / ``disable_router_mask`` / ``enable_router_mask``
  - ``layer_idx`` attribute on the MoE block for per-layer pruning
  - Support for per-layer ``num_experts_list`` in config (post-pruning save/load)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack


# ---------------------------------------------------------------------------
# MLP (identical to upstream, kept here so the module is self-contained)
# ---------------------------------------------------------------------------
class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Pruned SparseMoeBlock
# ---------------------------------------------------------------------------
class Qwen3MoeSparseMoeBlock(nn.Module):
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

        self.top_k = min(config.num_experts_per_tok, self.num_experts)

        # Router mask for pruning
        self.router_logits_mask: Optional[torch.Tensor]
        self.register_buffer("router_logits_mask", None, persistent=False)

        # Gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

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

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    # -- Pruning API (same interface as Qwen2 version) --

    def _set_router_logits_mask(self, router_logits_mask: torch.Tensor):
        if router_logits_mask.dim() != 1 or router_logits_mask.shape[0] != self.gate.out_features:
            raise ValueError(f"router_logits_mask must be 1D of length {self.gate.out_features}")
        keep = (router_logits_mask > float("-inf")).sum().item()
        self.top_k = min(self.top_k, int(keep)) if keep > 0 else 0
        self.router_logits_mask = router_logits_mask

    def disable_router_mask(self):
        self.use_router_mask = False

    def enable_router_mask(self):
        self.use_router_mask = True


# ---------------------------------------------------------------------------
# Pruned DecoderLayer
# ---------------------------------------------------------------------------
class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    """Drop-in for transformers Qwen3MoeDecoderLayer with pruning support."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Import attention from the upstream module (we don't modify attention)
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        is_sparse_layer = (layer_idx not in config.mlp_only_layers) and (
            (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_sparse_layer:
            # Resolve the current layer before testing the count: expert-drop
            # checkpoints store ``config.num_experts`` as a list whose dense
            # positions are None.
            if getattr(config, "num_experts_list", None):
                num_exp = config.num_experts_list[layer_idx if layer_idx < len(config.num_experts_list) else 0]
            else:
                num_exp = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]

            if num_exp is None or num_exp <= 0:
                self.mlp = None
                self.post_attention_layernorm = None
            else:
                self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx)
                from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
                self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
            self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
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

        # MLP / MoE
        router_logits = None
        if self.post_attention_layernorm is None and self.mlp is None:
            pass
        elif self.mlp is None:
            pass
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
            outputs += (past_key_value,)
        if output_router_logits:
            outputs += (router_logits,)

        return outputs
