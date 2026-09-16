"""Opt-in vLLM registration; never replaces a stock model architecture."""


def register():
    from vllm import ModelRegistry
    from .ragged import ARCHITECTURES
    for architecture in ARCHITECTURES.values():
        ModelRegistry.register_model(
            architecture, f"less_is_moe.intdim.ragged_vllm:{architecture}")
