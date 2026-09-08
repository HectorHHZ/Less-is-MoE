# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series (pruned-compatible).

Adapted from vllm/model_executor/models/qwen3_5.py with per-layer expert count
and router_logits_mask support, mirroring the pruning hooks in
less_is_moe.model_patches.vllm.qwen3_moe.Qwen3MoeSparseMoeBlock.
"""

import typing
from collections.abc import Callable, Iterable
from typing import Optional

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock as _UpstreamQwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
# Keep the upstream class name bound in this module so upstream code that
# imports `Qwen3NextSparseMoeBlock` from us still sees the original.
Qwen3NextSparseMoeBlock = _UpstreamQwen3NextSparseMoeBlock
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)

logger = init_logger(__name__)


class _PrunedQwen3NextSparseMoeBlock(_UpstreamQwen3NextSparseMoeBlock):
    """Qwen3NextSparseMoeBlock variant with per-layer expert count and an
    optional router mask for pruned Qwen3.5-MoE checkpoints.

    Inherits from the upstream block so `isinstance(..., Qwen3NextSparseMoeBlock)`
    checks used by `QwenNextMixtureOfExperts.set_moe_parameters` still match.
    ``__init__`` is fully overridden (calls ``nn.Module.__init__`` directly,
    not ``super().__init__``) so we don't double-initialize submodules.

    Mirrors the pruning hooks in
    ``less_is_moe.model_patches.vllm.qwen3_moe.Qwen3MoeSparseMoeBlock``:

    * Reads ``config.num_experts_list`` (or a list-valued ``config.num_experts``)
      per layer, falling back to the scalar ``config.num_experts``.
    * Exposes ``_set_router_logits_mask`` / ``enable_router_mask`` /
      ``disable_router_mask`` so external callers can disable a subset of
      experts via a -inf mask added to the router logits.
    * Adjusts ``top_k`` down when experts are masked out.

    The upstream forward path with an internal router (fused gate inside
    SharedFusedMoE) does not surface router logits here, so the mask has no
    effect in that mode — a warning is emitted.
    """

    _internal_router_warned = False

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # Skip the upstream Qwen3NextSparseMoeBlock.__init__; we re-implement
        # it below with per-layer expert count + router mask hooks.
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()

        # --- Per-layer expert count (pruned / variable MoE) ---------------
        self.layer_idx = extract_layer_index(prefix)
        if getattr(config, "num_experts_list", None) is not None:
            idx = (
                self.layer_idx
                if self.layer_idx < len(config.num_experts_list)
                else 0
            )
            self.n_routed_experts = config.num_experts_list[idx]
        elif isinstance(getattr(config, "num_experts", None), list):
            idx = (
                self.layer_idx
                if self.layer_idx < len(config.num_experts)
                else 0
            )
            self.n_routed_experts = config.num_experts[idx]
        else:
            self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        # Load-balancing settings (identical to upstream).
        _cur_vllm_config = get_current_vllm_config()
        eplb_config = _cur_vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = (
            self.n_logical_experts + self.n_redundant_experts
        )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.top_k = min(config.num_experts_per_tok, self.n_routed_experts)

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

        # Router mask buffer; skipped entirely when use_router_mask=False.
        self.router_logits_mask: Optional[torch.Tensor]
        self.register_buffer("router_logits_mask", None, persistent=False)
        self.use_router_mask = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            if (self.use_router_mask
                    and self.router_logits_mask is not None
                    and not _PrunedQwen3NextSparseMoeBlock._internal_router_warned):
                logger.warning(
                    "[Pruned Qwen3.5-MoE] Router mask is set but the fused-moe "
                    "backend uses an internal router; the mask will be ignored "
                    "for this forward path."
                )
                _PrunedQwen3NextSparseMoeBlock._internal_router_warned = True
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            if self.use_router_mask and self.router_logits_mask is not None:
                router_logits = router_logits + self.router_logits_mask.to(
                    router_logits.dtype
                )
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa: E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)

    def _set_router_logits_mask(self, router_logits_mask: torch.Tensor) -> None:
        """Disable selected experts via a -inf mask, shrinking top_k."""
        if (router_logits_mask.dim() != 1
                or router_logits_mask.shape[0] != self.n_routed_experts):
            raise ValueError(
                "router_logits_mask must be 1D of length "
                f"{self.n_routed_experts}"
            )
        keep = int((router_logits_mask > float("-inf")).sum().item())
        new_top_k = min(self.top_k, keep) if keep > 0 else 0
        self.experts.top_k = new_top_k
        self.router_logits_mask = router_logits_mask

    def disable_router_mask(self) -> None:
        self.use_router_mask = False

    def enable_router_mask(self) -> None:
        self.use_router_mask = True


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                create_in_proj_qkvz=vllm_config.lora_config is None,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = _PrunedQwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.enable_lora = vllm_config.lora_config is not None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        if self.enable_lora:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkv", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_z", "in_proj_z", 0),
                ]
            )
        else:
            stacked_params_mapping.extend(
                [
                    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
                    ("in_proj_qkvz", "in_proj_z", 3),
                ]
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if param_name == "in_proj_z" and self.enable_lora:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # When LoRA is enabled, GDN uses separate in_proj_qkv and in_proj_z
        # instead of merged in_proj_qkvz; pack mapping must match.
        if vllm_config.lora_config:
            base = getattr(Qwen3_5ForCausalLMBase, "packed_modules_mapping", {})
            self.packed_modules_mapping = {k: list(v) for k, v in base.items()}
            self.packed_modules_mapping.pop("in_proj_qkvz", None)
            self.packed_modules_mapping["in_proj_qkv"] = ["in_proj_qkv"]
            self.packed_modules_mapping["in_proj_z"] = ["in_proj_z"]

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Text-only pruned/SFT checkpoints (arch=Qwen3_5MoeForCausalLM) are
        # saved from a structure where `self.language_model = Qwen3_5Model(...)`
        # is attached directly — so weights are named
        #   language_model.embed_tokens.weight
        #   language_model.layers.<i>.<rest>
        #   language_model.norm.weight
        # but our class has `self.model = Qwen3_5Model(...)`, not
        # `self.language_model`. Remap the top-level prefix so the auto-loader
        # routes them into `self.model` correctly.
        # This checkpoint was saved with an extra `language_model` wrapper level
        # at the torch-state-dict top, so keys look like
        #   model.language_model.embed_tokens.weight
        #   model.language_model.layers.<i>.<rest>
        # while our class expects model.<rest>. Normalize the prefix here.
        def _remap(n: str) -> str:
            # Some pruned VL-SFT checkpoints (e.g. jhhj25/...-Prune-SFT-gsm8k-v2)
            # have *doubled* language_model prefixes after save, e.g.:
            #   model.language_model.language_model.layers.0.mlp.experts.down_proj
            # Keep stripping until no language_model prefix remains at the top
            # level; the resulting name is what our Qwen3_5MoeForCausalLM
            # params_dict (which lives under "model.<rest>") expects.
            while True:
                if n.startswith("model.language_model."):
                    n = "model." + n[len("model.language_model."):]
                    continue
                if n.startswith("language_model."):
                    n = "model." + n[len("language_model."):]
                    continue
                break
            return n

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp.", "visual."],
        )
        return loader.load_weights(((_remap(n), w) for n, w in weights))


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts, IsHybrid):
    # IsHybrid marks this arch as a hybrid (full-attn + GDN/Mamba) model so
    # that vLLM's config machinery invokes
    # ``HybridAttentionMambaModelConfig.verify_and_update_config`` — which
    # auto-computes ``mamba_block_size``, bumps the attention ``block_size``
    # to align with the mamba page size, and pads the mamba page to match.
    # Upstream registers this arch without IsHybrid (only the VL variant
    # gets IsHybrid), so the text-only CausalLM path otherwise skips the
    # hybrid setup and blows up in ``unify_kv_cache_spec_page_size`` when
    # gate+value-head geometry doesn't naturally divide.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    # The hybrid config path and KV cache layout both look up three classmethods
    # on the model class. The pruned file already defines them on the VL
    # Qwen3_5ForConditionalGeneration class; mirror them here so MRO resolves
    # correctly for the text-only arch.
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        self.update_packed_mapping(enable_lora=vllm_config.lora_config is not None)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def update_packed_mapping(self, enable_lora: bool):
        # When LoRA is enabled, GDN uses separate in_proj_qkv and in_proj_z
        if enable_lora:
            base = getattr(
                Qwen3_5ForConditionalGeneration, "packed_modules_mapping", {}
            )
            self.packed_modules_mapping = {k: list(v) for k, v in base.items()}
            self.packed_modules_mapping.pop("in_proj_qkvz", None)
            self.packed_modules_mapping["in_proj_qkv"] = ["in_proj_qkv"]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        self.update_packed_mapping(enable_lora=vllm_config.lora_config is not None)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
