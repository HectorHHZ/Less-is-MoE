from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeForCausalLM,
    Qwen2MoeSparseMoeBlock,
)
from .base import BaseAWQForCausalLM


class Qwen2MoeAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen2MoeDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate", "shared_expert_gate"]

    @staticmethod
    def fuse_layers(model: Qwen2MoeForCausalLM):
        pass

    @staticmethod
    def get_model_layers(model: Qwen2MoeForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: Qwen2MoeDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: Qwen2MoeForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module: Qwen2MoeDecoderLayer, input_feat, module_kwargs
    ):
        layers = []

        # attention input
        if "self_attn.q_proj" in input_feat:
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

        # attention out
        if "self_attn.o_proj" in input_feat:
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

        if isinstance(module.mlp, Qwen2MoeSparseMoeBlock):
            # Sparse MoE: all experts gate_proj/up_proj together
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                        w
                        for expert in module.mlp.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ] + [
                        module.mlp.shared_expert.gate_proj,
                        module.mlp.shared_expert.up_proj,
                    ],
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # down_proj per sparse expert
            for i, expert in enumerate(module.mlp.experts):
                if f"mlp.experts.{i}.down_proj" in input_feat:
                    layers.append(
                        dict(
                            prev_op=expert.up_proj,
                            layers=[expert.down_proj],
                            inp=input_feat[f"mlp.experts.{i}.down_proj"],
                        )
                    )

            # shared expert down_proj
            if "mlp.shared_expert.down_proj" in input_feat:
                layers.append(
                    dict(
                        prev_op=module.mlp.shared_expert.up_proj,
                        layers=[module.mlp.shared_expert.down_proj],
                        inp=input_feat["mlp.shared_expert.down_proj"],
                    )
                )
        else:
            # Dense MLP layer (mlp_only_layers)
            if "mlp.gate_proj" in input_feat:
                layers.append(
                    dict(
                        prev_op=module.post_attention_layernorm,
                        layers=[module.mlp.gate_proj, module.mlp.up_proj],
                        inp=input_feat["mlp.gate_proj"],
                        module2inspect=module.mlp,
                    )
                )
            if "mlp.down_proj" in input_feat:
                layers.append(
                    dict(
                        prev_op=module.mlp.up_proj,
                        layers=[module.mlp.down_proj],
                        inp=input_feat["mlp.down_proj"],
                    )
                )

        return layers
