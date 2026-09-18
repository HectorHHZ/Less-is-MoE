"""Opt-in vLLM registration; never replaces a stock model architecture."""


def register():
    from vllm import ModelRegistry, __version__
    # General plugins are auto-discovered in legacy environments too. These
    # adapters and config classes are only supported by the pinned runtime.
    if __version__ != "0.29.0":
        return
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP, Gemma4Config, Qwen3_5ForCausalLMConfig,
    )
    from .ragged import ARCHITECTURES
    # vLLM dispatches these fixups by architecture name, independently of the
    # model class. Reuse the text-only Qwen3.5 contract before cache sizing:
    # preserve mamba_ssm_dtype (including explicit user overrides) and strip
    # inherited multimodal RoPE fields. No stock architecture is replaced.
    MODELS_CONFIG_MAP[ARCHITECTURES["qwen3_5_moe_text"]] = Qwen3_5ForCausalLMConfig
    # Gemma4 mixes head dimensions across sliding and full attention layers
    # (256 and 512 in gemma-4-26B-A4B). Upstream's hook forces one attention
    # backend for every layer; without it the mixed FA3/FA4 selection it exists
    # to prevent diverges numerically. Attention is upstream's, so is this fix.
    MODELS_CONFIG_MAP[ARCHITECTURES["gemma4_text"]] = Gemma4Config
    for architecture in ARCHITECTURES.values():
        ModelRegistry.register_model(
            architecture, f"less_is_moe.intdim.ragged_vllm:{architecture}")
