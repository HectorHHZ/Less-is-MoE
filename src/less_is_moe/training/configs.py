# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional, List

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )

    # Distillation / teacher settings
    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Optional teacher model id or path for distillation."}
    )
    teacher_model_revision: Optional[str] = field(
        default=None, metadata={"help": "Revision to use when loading the teacher model."}
    )
    teacher_torch_dtype: Optional[str] = field(
        default="auto", metadata={"help": "torch dtype for the teacher model (e.g., float16, bfloat16, auto)."}
    )
    teacher_attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation for the teacher model (falls back to student if None)."},
    )
    disable_teacher_dropout: bool = field(
        default=True, metadata={"help": "Disable dropout modules in the teacher (keeps inference deterministic)."}
    )
    layer_entropy_l1_weight: float = field(
        default=0.0, metadata={"help": "Weight for layer-wise entropy L1 loss (set 0 to disable)."}
    )
    layer_entropy_l1_layers: Optional[list[int]] = field(
        default=None,
        metadata={"help": "List of 0-based layer ids to include in layer L1 loss; None means all."},
    )
    last_entropy_weight: float = field(
        default=0.0, metadata={"help": "Weight for last-layer entropy loss (set 0 to disable)."}
    )
    attn_kl_weight: float = field(
        default=0.0, metadata={"help": "Weight for attention KL loss (set 0 to disable)."}
    )

    # Router pruning (score-based expert masking)
    router_prune_enable: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run additional online router pruning during SFT. "
                "The release defaults to False because sft_pruned trains an "
                "already-pruned checkpoint."
            )
        },
    )
    router_prune_start_step: Optional[int] = field(
        default=None,
        metadata={"help": "Start step for router score-based pruning (None to disable)."},
    )
    router_prune_interval: int = field(
        default=5, metadata={"help": "Prune one expert per layer every N steps."}
    )
    router_prune_expert_per_layer: Optional[int] = field(
        default=None,
        metadata={"help": "Target experts to prune per layer; total budget = this value * MoE layer count."},
    )
    router_prune_min_keep: int = field(
        default=1, metadata={"help": "Minimum experts to keep unmasked per layer."}
    )
    router_prune_step_size: int = field(
        default=32, metadata={"help": "How many experts to prune globally per pruning step."}
    )
    cluster_prune_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "是否按簇分配剪枝：保留字段（当前使用 router_prune_use_plan 作为开关），留空即可。"
            )
        },
    )
    cluster_prune_tau: float = field(
        default=1.0,
        metadata={
            "help": "簇级剪枝分配温度 tau，越小越偏向低激活簇。"
        },
    )
    router_prune_use_plan: bool = field(
        default=True,
        metadata={
            "help": (
                "是否在裁剪开始时按层生成固定的裁剪计划。"
                "True=沿用原逻辑，预先为每层分配需裁剪的 expert 数；"
                "False=每个裁剪 step 直接按分数全局选择并满足最小保留数。"
            )
        },
    )
    router_manual_mask: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "固定路由 mask，禁用指定层的指定专家；训练全程生效。"
                "支持 JSON/YAML 文件路径或内联格式，例如 \"0:1,3;2:0\" 表示"
                "第 0 层屏蔽 1/3 号专家，第 2 层屏蔽 0 号专家。"
            )
        },
    )

    # Merging / clustering (expert grouping for logging/analysis)
    cluster_num_groups: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "专家聚类的目标簇数（上限）。为空则使用回退值 8；"
                "实际每层簇数 = min(cluster_num_groups, num_experts)。"
            )
        },
    )
    cluster_mode: str = field(
        default="hierarchical-dynamic",
        metadata={
            "help": (
                "专家聚类算法模式：kmeans | hierarchical | hierarchical-dynamic，"
                "默认 hierarchical-dynamic（基于 silhouette 动态确定簇数上限以内的最佳簇数）。"
            )
        },
    )
    weight_feature_rank: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "weight 视角的低秩功能表示的主成分数 r（SVD/eigen 取前 r）。"
                "留空默认 64，过大时会被 clamp 到 d_model。"
            )
        },
    )
    merging_metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "用于专家聚类的指标列表。支持：weight, router-weight, expert-output 以及它们的组合 "
                "(例如 weight+router-weight, weight+expert-output, router-weight+expert-output, weight+router-weight+expert-output)。"
                "留空则使用默认全量组合。"
            )
        },
    )

    # Layer entropy slope loss (student only, before router pruning starts)
    entropy_slope_alpha: float = field(
        default=1.0, metadata={"help": "Positive-slope penalty scale for layer entropy deltas."}
    )
    entropy_slope_beta: float = field(
        default=1.0, metadata={"help": "Negative-slope penalty scale for layer entropy deltas."}
    )
