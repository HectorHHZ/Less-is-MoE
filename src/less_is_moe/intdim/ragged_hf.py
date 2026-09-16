"""Explicit Transformers reload class; no global AutoModel monkeypatch."""

from transformers import (Qwen2MoeForCausalLM, OlmoeForCausalLM,
                          Qwen3MoeForCausalLM, Qwen3_5MoeForCausalLM)

from .ragged import PackedExperts, validate_metadata


class _RaggedExpertsMixin:
    def __init__(self, config):
        widths = validate_metadata(config)
        super().__init__(config)
        for layer_id, layer in enumerate(self.model.layers):
            original = layer.mlp.experts.gate_up_proj
            layer.mlp.experts = PackedExperts(widths[str(layer_id)], config.hidden_size,
                                             device=original.device, dtype=original.dtype)


class RaggedQwen2MoeForCausalLM(_RaggedExpertsMixin, Qwen2MoeForCausalLM):
    pass


class RaggedOlmoeForCausalLM(_RaggedExpertsMixin, OlmoeForCausalLM):
    pass


class RaggedQwen3MoeForCausalLM(_RaggedExpertsMixin, Qwen3MoeForCausalLM):
    pass


class RaggedQwen3_5MoeForCausalLM(_RaggedExpertsMixin, Qwen3_5MoeForCausalLM):
    pass


def load_source_model(path, *, dtype, **kwargs):
    """Load every language layer of the original checkpoint directly on GPU.

    Qwen3.5 ships a multimodal config. Its stock causal-LM loader extracts the
    full text config and maps language_model weights; vision/MTP are excluded.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from .ragged import validate_family
    config = AutoConfig.from_pretrained(path)
    text = config.get_text_config()
    validate_family(text)
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="cuda", **kwargs).eval()
