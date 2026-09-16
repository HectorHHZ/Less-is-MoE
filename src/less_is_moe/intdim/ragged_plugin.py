"""Opt-in vLLM registration; never replaces a stock model architecture."""


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "RaggedQwen3MoeForCausalLM",
        "less_is_moe.intdim.ragged_vllm:RaggedQwen3MoeForCausalLM",
    )
