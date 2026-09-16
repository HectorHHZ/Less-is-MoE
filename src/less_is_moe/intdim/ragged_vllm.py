"""Shared ragged SiLU experts and four thin vLLM 0.29.0 model adapters.

GPU BF16, TP=PP=DP=1, eager. Upstream attention, decoder/model forwards,
weight mappings and hybrid-cache methods are reused without global patches.
No stock FusedMoE or rectangular routed-expert weights are instantiated.
"""

import torch
from torch import nn
import vllm
from vllm.model_executor.layers.layernorm import RMSNorm, GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models import qwen2_moe as q2, qwen3_moe as q3, olmoe as ol
from vllm.model_executor.models import qwen3_5 as q35
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.utils import (
    extract_layer_index, maybe_prefix, make_layers, make_empty_intermediate_tensors_factory,
)

from .ragged import PackedExperts, validate_metadata
from .ragged_triton import ragged_experts


def _validate(vllm_config):
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
    if getattr(config, "dual_chunk_attention_config", None):
        raise ValueError("Dual-chunk attention is not supported by ragged v1")
    return config


class RaggedMLP(nn.Module):
    """Shared routed experts; retain each family's routing and shared branch."""
    def __init__(self, config, widths, prefix):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False,
                                     quant_config=None, prefix=f"{prefix}.gate")
        self.experts = PackedExperts(widths, config.hidden_size)
        self.register_buffer("widths", torch.tensor(widths, dtype=torch.int32), persistent=False)
        self.register_buffer("offsets", torch.tensor(self.experts.offsets, dtype=torch.int64), persistent=False)
        self.max_width = max(widths)
        shared_width = getattr(config, "shared_expert_intermediate_size", 0)
        self.shared_expert = None
        if shared_width:
            self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False,
                                                       quant_config=None, prefix=f"{prefix}.shared_expert_gate")
            self.shared_expert = q2.Qwen2MoeMLP(
                hidden_size=config.hidden_size, intermediate_size=shared_width,
                hidden_act=config.hidden_act, quant_config=None,
                expert_gate=self.shared_expert_gate, prefix=f"{prefix}.shared_expert")

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
        if self.shared_expert is not None:
            result = result + self.shared_expert(hidden)
        return result.reshape(shape)


def _mlp(config, prefix):
    layer = extract_layer_index(prefix)
    return RaggedMLP(config, config.less_is_moe["expert_intermediate_sizes"][str(layer)], f"{prefix}.mlp")


class RaggedDecoderLayer(q3.Qwen3MoeDecoderLayer):
    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = q3.Qwen3MoeAttention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, rope_parameters=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings, rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False), head_dim=getattr(config, "head_dim", None),
            cache_config=vllm_config.cache_config, quant_config=None, prefix=f"{prefix}.self_attn")
        self.mlp = _mlp(config, prefix)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class RaggedQwen2Decoder(q2.Qwen2MoeDecoderLayer):
    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = q2.Qwen2MoeAttention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, rope_parameters=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config, quant_config=None, prefix=f"{prefix}.self_attn")
        self.mlp = _mlp(config, prefix)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class RaggedOlmoeDecoder(ol.OlmoeDecoderLayer):
    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = ol.OlmoeAttention(vllm_config=vllm_config, prefix=f"{prefix}.self_attn")
        self.mlp = _mlp(config, prefix)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)


class RaggedQwen35Decoder(q35.Qwen3_5DecoderLayer):
    def __init__(self, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.layer_idx = extract_layer_index(prefix)
        self.layer_type = config.layer_types[self.layer_idx]
        self.use_attn_reduce_scatter_for_moe = False
        if self.layer_type == "linear_attention":
            self.linear_attn = q35.QwenGatedDeltaNetAttention(
                config=config, vllm_config=vllm_config, prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False, reduce_results=True)
        elif self.layer_type == "full_attention":
            self.self_attn = q35.Qwen3NextAttention(
                config, model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config, quant_config=None,
                prefix=f"{prefix}.self_attn", reduce_results=True)
        else:
            raise ValueError(f"Unknown Qwen3.5 layer type: {self.layer_type}")
        self.mlp = _mlp(config, prefix)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))


def _init_model(module, vllm_config, prefix, decoder, *, hybrid=False):
    nn.Module.__init__(module)
    # These upstream classes wrap __call__ with support_torch_compile. Their
    # stock initializer normally sets this flag; our backend requires eager.
    module.do_not_compile = True
    config = vllm_config.model_config.hf_text_config
    module.config = config
    module.quant_config = None
    module.vocab_size = config.vocab_size
    module.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                 quant_config=None, prefix=f"{prefix}.embed_tokens")
    module.start_layer, module.end_layer, module.layers = make_layers(
        config.num_hidden_layers, lambda prefix: decoder(vllm_config, prefix), prefix=f"{prefix}.layers")
    norm = GemmaRMSNorm if hybrid else RMSNorm
    module.norm = norm(config.hidden_size, eps=config.rms_norm_eps)
    module.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], config.hidden_size)
    if hybrid:
        module.is_fused_shared_expert_enabled = False
        module.num_redundant_experts = 0
        module.aux_hidden_state_layers = ()


class RaggedQwen2Model(q2.Qwen2MoeModel):
    def __init__(self, *, vllm_config, prefix=""):
        _init_model(self, vllm_config, prefix, RaggedQwen2Decoder)


class RaggedQwen35Model(q35.Qwen3_5Model):
    def __init__(self, *, vllm_config, prefix=""):
        _init_model(self, vllm_config, prefix, RaggedQwen35Decoder, hybrid=True)


class _RaggedCausalLM(nn.Module):
    # No LoRA/quantization/parallelism advertised; reject them before allocation.
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}

    def _init(self, vllm_config, prefix, model_type, **model_kwargs):
        nn.Module.__init__(self)
        config = _validate(vllm_config)
        self.config = config
        self.quant_config = None
        self.model = model_type(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"), **model_kwargs)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=None,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    embed_input_ids = q3.Qwen3MoeForCausalLM.embed_input_ids
    forward = q3.Qwen3MoeForCausalLM.forward
    compute_logits = q3.Qwen3MoeForCausalLM.compute_logits


class RaggedQwen3MoeForCausalLM(_RaggedCausalLM):
    hf_to_vllm_mapper = q3.Qwen3MoeForCausalLM.hf_to_vllm_mapper
    load_weights = q3.Qwen3MoeForCausalLM.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        self._init(vllm_config, prefix, q3.Qwen3MoeModel, decoder_layer_type=RaggedDecoderLayer)


class RaggedQwen2MoeForCausalLM(_RaggedCausalLM):
    hf_to_vllm_mapper = q2.Qwen2MoeForCausalLM.hf_to_vllm_mapper
    load_weights = q2.Qwen2MoeForCausalLM.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        self._init(vllm_config, prefix, RaggedQwen2Model)


class RaggedOlmoeForCausalLM(_RaggedCausalLM):
    hf_to_vllm_mapper = ol.OlmoeForCausalLM.hf_to_vllm_mapper
    load_weights = ol.OlmoeForCausalLM.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        self._init(vllm_config, prefix, ol.OlmoeModel, layer_type=RaggedOlmoeDecoder)


class RaggedQwen3_5MoeForCausalLM(_RaggedCausalLM, HasInnerState, IsHybrid, SupportsMRoPE):
    hf_to_vllm_mapper = q35.Qwen3_5ForCausalLMBase.hf_to_vllm_mapper
    load_weights = q35.Qwen3_5ForCausalLMBase.load_weights
    forward = q35.Qwen3_5ForCausalLMBase.forward
    get_mrope_input_positions = q35.Qwen3_5ForCausalLMBase.get_mrope_input_positions
    get_mamba_state_dtype_from_config = q35.Qwen3_5ForCausalLMBase.__dict__["get_mamba_state_dtype_from_config"]
    get_mamba_state_shape_from_config = q35.Qwen3_5ForCausalLMBase.__dict__["get_mamba_state_shape_from_config"]
    get_mamba_state_copy_func = q35.Qwen3_5ForCausalLMBase.__dict__["get_mamba_state_copy_func"]

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise ValueError("Qwen3.5 ragged requires mamba_cache_mode=align or none")
        self._init(vllm_config, prefix, RaggedQwen35Model)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
