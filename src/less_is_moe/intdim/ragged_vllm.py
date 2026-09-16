"""Qwen3-MoE ragged inference on vLLM 0.29.0, BF16 / TP=PP=DP=1 / eager.

Reuse upstream attention, decoder forward, model scheduling and weight mapping.
Only construct our own MLP and its packed expert tensors. No stock FusedMoE
instances or rectangular expert allocations are created.
"""

import torch
from torch import nn
import vllm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM, Qwen3MoeModel,
)
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix

from .ragged import PackedExperts, validate_metadata
from .ragged_triton import ragged_experts


class RaggedMLP(nn.Module):
    def __init__(self, config, widths, prefix):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False,
                                     quant_config=None, prefix=f"{prefix}.gate")
        self.experts = PackedExperts(widths, config.hidden_size)
        self.register_buffer("widths", torch.tensor(widths, dtype=torch.int32), persistent=False)
        self.register_buffer("offsets", torch.tensor(self.experts.offsets, dtype=torch.int64), persistent=False)
        self.max_width = max(widths)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        logits, _ = self.gate(hidden)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(probabilities, self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        result = ragged_experts(hidden, self.experts.gate_up_proj, self.experts.down_proj,
                                self.widths, self.offsets, self.max_width, ids, weights.to(hidden.dtype))
        return result.reshape(shape)


class RaggedDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, rope_parameters=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings, rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False), head_dim=getattr(config, "head_dim", None),
            cache_config=vllm_config.cache_config, quant_config=None, prefix=f"{prefix}.self_attn")
        layer = extract_layer_index(prefix)
        self.mlp = RaggedMLP(config, config.less_is_moe["expert_intermediate_sizes"][str(layer)], f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class RaggedQwen3MoeForCausalLM(nn.Module):
    # Intentionally do not advertise LoRA, quantization, EP or speculative modes.
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if vllm.__version__ != "0.29.0":
            raise ValueError("Ragged v1 is pinned to vLLM 0.29.0")
        pc, mc = vllm_config.parallel_config, vllm_config.model_config
        if any(getattr(pc, key, 1) != 1 for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")):
            raise ValueError("Ragged v1 requires TP=PP=DP=1")
        if pc.enable_expert_parallel or pc.enable_eplb or getattr(pc, "use_sequence_parallel_moe", False):
            raise ValueError("Ragged v1 does not support expert/sequence parallelism or EPLB")
        if mc.dtype != torch.bfloat16 or not mc.enforce_eager:
            raise ValueError("Ragged v1 requires BF16 and enforce_eager=True")
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Ragged v1 does not support quantization, LoRA or speculative decoding")
        config = mc.hf_text_config
        validate_metadata(config)
        if getattr(config, "shared_expert_intermediate_size", 0) or getattr(config, "dual_chunk_attention_config", None):
            raise ValueError("This Qwen3-MoE variant is not supported by ragged v1")
        self.config = config
        self.model = Qwen3MoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
                                   decoder_layer_type=RaggedDecoderLayer)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=None,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    embed_input_ids = Qwen3MoeForCausalLM.embed_input_ids
    forward = Qwen3MoeForCausalLM.forward
    compute_logits = Qwen3MoeForCausalLM.compute_logits
    load_weights = Qwen3MoeForCausalLM.load_weights
