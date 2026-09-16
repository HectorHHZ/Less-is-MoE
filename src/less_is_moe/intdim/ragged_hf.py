"""Explicit Transformers reload class; no global AutoModel monkeypatch."""

from transformers import Qwen3MoeForCausalLM

from .ragged import PackedExperts, validate_metadata


class RaggedQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, config):
        widths = validate_metadata(config)
        super().__init__(config)
        for layer_id, layer in enumerate(self.model.layers):
            original = layer.mlp.experts.gate_up_proj
            layer.mlp.experts = PackedExperts(widths[str(layer_id)], config.hidden_size,
                                             device=original.device, dtype=original.dtype)
