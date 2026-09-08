"""Source modules loaded in place of vLLM's built-in model modules."""

# Do not import the sibling modules here.  They intentionally use relative
# imports from ``vllm.model_executor.models`` and are loaded under that module
# namespace by :func:`less_is_moe.model_patches.apply_vllm_patch`.
