import torch

from .base import BaseAWQForCausalLM


def _patch_transformers_qwen2_moe():
    """Inject pruned Qwen2-MoE classes into the transformers namespace so that
    AutoModelForCausalLM.from_pretrained picks up our custom modeling file."""
    from less_is_moe.model_patches.registry import apply_hf_patch

    apply_hf_patch("qwen2_moe")


class Qwen2MoePrunedAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen2MoeDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate", "shared_expert_gate"]

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        model_type,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
        download_kwargs=None,
        **model_init_kwargs,
    ):
        _patch_transformers_qwen2_moe()
        return super().from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            download_kwargs=download_kwargs,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        cls,
        model_path,
        model_type,
        model_filename="",
        max_seq_len=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        safetensors=True,
        fuse_layers=True,
        use_exllama=False,
        use_exllama_v2=False,
        use_ipex=False,
        device_map="balanced",
        max_memory=None,
        offload_folder=None,
        download_kwargs=None,
        **config_kwargs,
    ):
        _patch_transformers_qwen2_moe()
        return super().from_quantized(
            model_path,
            model_type,
            model_filename,
            max_seq_len,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            fuse_layers=fuse_layers,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
            use_ipex=use_ipex,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )

    @staticmethod
    def fuse_layers(model):
        pass

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        from less_is_moe.model_patches.hf.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeSparseMoeBlock,
        )

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

        if module.mlp is None:
            # pruned-out layer
            pass
        elif isinstance(module.mlp, Qwen2MoeSparseMoeBlock):
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
            # Dense MLP layer
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
