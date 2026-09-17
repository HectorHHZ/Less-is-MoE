"""Explicit Transformers reload class; no global AutoModel monkeypatch."""

from transformers import (Qwen2MoeForCausalLM, OlmoeForCausalLM,
                          Qwen3MoeForCausalLM, Qwen3_5MoeForCausalLM,
                          GptOssForCausalLM, Gemma4ForCausalLM)

from .ragged import PackedExperts, validate_metadata, expert_path, expert_options


class _RaggedExpertsMixin:
    def __init__(self, config):
        widths = validate_metadata(config)
        super().__init__(config)
        for layer_id in range(len(self.model.layers)):
            path = expert_path(config, layer_id)
            original = self.get_submodule(path).gate_up_proj
            parent, name = path.rsplit(".", 1)
            setattr(self.get_submodule(parent), name,
                    PackedExperts(widths[str(layer_id)], config.hidden_size,
                                  device=original.device, dtype=original.dtype, **expert_options(config)))


class RaggedQwen2MoeForCausalLM(_RaggedExpertsMixin, Qwen2MoeForCausalLM):
    pass


class RaggedOlmoeForCausalLM(_RaggedExpertsMixin, OlmoeForCausalLM):
    pass


class RaggedQwen3MoeForCausalLM(_RaggedExpertsMixin, Qwen3MoeForCausalLM):
    pass


class RaggedQwen3_5MoeForCausalLM(_RaggedExpertsMixin, Qwen3_5MoeForCausalLM):
    pass


class RaggedGptOssForCausalLM(_RaggedExpertsMixin, GptOssForCausalLM):
    pass


class RaggedGemma4ForCausalLM(_RaggedExpertsMixin, Gemma4ForCausalLM):
    pass


def gpu_device_map():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Ragged models require GPU execution")
    return "balanced" if torch.cuda.device_count() > 1 else "cuda"


def load_source_model(path, *, dtype, **kwargs):
    """Load every language layer of the original checkpoint directly on GPU.

    Qwen3.5 ships a multimodal config. Its stock causal-LM loader extracts the
    full text config and maps language_model weights; vision/MTP are excluded.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    import copy
    from .ragged import validate_family
    config = AutoConfig.from_pretrained(path)
    text = config.get_text_config()
    # The published GPT-OSS weights are MXFP4. Explicitly dequantize once to
    # BF16 for *both* comparisons; never compare quantized and BF16 models.
    quant = getattr(text, "quantization_config", None)
    if text.model_type == "gpt_oss" and quant and quant.get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        text = copy.deepcopy(text)
        del text.quantization_config
    validate_family(text)
    if text.model_type == "gpt_oss":
        kwargs["attn_implementation"] = "eager"  # HF SDPA does not implement attention sinks.
    device_map = kwargs.pop("device_map", gpu_device_map())
    loader = Gemma4ForCausalLM if text.model_type == "gemma4_text" else AutoModelForCausalLM
    if config.model_type == "gemma4":
        kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
    model, loading = loader.from_pretrained(
        path, dtype=dtype, device_map=device_map, output_loading_info=True, **kwargs)
    model.eval()
    missing = loading.get("missing_keys", [])
    unexpected_text = [key for key in loading.get("unexpected_keys", [])
                       if not any(part in key for part in ("vision", "visual", "audio", "mtp"))]
    if missing or loading.get("mismatched_keys") or unexpected_text:
        raise RuntimeError(f"Incomplete pretrained language weights: missing={missing}, unexpected={unexpected_text}, "
                           f"mismatched={loading.get('mismatched_keys')}")
    if model.config.model_type != text.model_type:
        raise RuntimeError("The source loader did not select the complete causal language-model tower")
    if not all(p.is_cuda for p in model.parameters()):
        raise RuntimeError("The full model must fit on the visible GPUs; CPU/disk offload is unsupported")
    if quant and text.model_type == "gpt_oss":
        from . import discover
        if any(h.gate_up.dtype != dtype or h.down.dtype != dtype for h in discover(model)):
            raise RuntimeError("GPT-OSS was not completely dequantized to the requested dtype")
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        model.hf_quantizer = None
        model.is_quantized = False
    return model
