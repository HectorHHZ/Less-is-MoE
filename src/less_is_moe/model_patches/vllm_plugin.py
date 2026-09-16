"""vLLM general plugin entry point for Less-is-MoE.

Registered under the ``vllm.general_plugins`` entry-point group in
``pyproject.toml`` so that vLLM auto-loads it in **every** process
(driver + EngineCore subprocess) via ``vllm.plugins.load_general_plugins()``.

This is the only reliable way to make our Qwen3.5-MoE patch survive the
spawn boundary introduced by vLLM v1's EngineCore — patches applied only
in the driver process are invisible to the subprocess.

The entry point is intentionally import-light.  On older vLLM releases that
do not contain Qwen3.5 support it returns without importing the patch, making
one installed Less-is-MoE package usable by both the legacy and Qwen3.5
environments.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_PLUGIN_LABEL = "[less_is_moe.model_patches.vllm_plugin]"


def _qwen3_5_is_available() -> bool:
    """Return whether this vLLM release has all Qwen3.5 dependencies."""
    try:
        required_modules = (
            "vllm.model_executor.models.qwen3_5",
            "vllm.transformers_utils.configs.qwen3_5",
            "vllm.transformers_utils.configs.qwen3_5_moe",
        )
        return all(
            importlib.util.find_spec(module_name) is not None
            for module_name in required_modules
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _patch_vllm_qwen3_5_moe() -> bool:
    """Swap in the pruned Qwen3.5-MoE module and register the text-only arch."""
    if not _qwen3_5_is_available():
        return False

    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vllm",
        "qwen3_5.py",
    )
    if not os.path.exists(local_path):
        print(f"{_PLUGIN_LABEL} patch file missing, skipping: {local_path}")
        return False

    full_module = "vllm.model_executor.models.qwen3_5"
    previous_module = sys.modules.get(full_module)
    try:
        spec = importlib.util.spec_from_file_location(full_module, local_path)
        if spec is None or spec.loader is None:
            raise ImportError("spec_from_file_location failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_module] = module
        spec.loader.exec_module(module)
        try:
            import vllm.model_executor.models as models_pkg
            setattr(models_pkg, "qwen3_5", module)
        except Exception:
            pass
        print(f"{_PLUGIN_LABEL} replaced {full_module} with pruned impl.")
    except Exception as e:  # pragma: no cover
        if previous_module is None:
            sys.modules.pop(full_module, None)
        else:
            sys.modules[full_module] = previous_module
        print(f"{_PLUGIN_LABEL} failed to swap qwen3_5 module: {e}")
        return False

    try:
        from vllm import ModelRegistry
        # Register with the class *object*, not a lazy "module:class" string.
        # The lazy path makes vllm re-import the module in a fresh subprocess
        # to compute _ModelInfo — that subprocess does NOT see our sys.modules
        # swap above, so it inspects the upstream class (no IsHybrid), which
        # then causes HybridAttentionMambaModelConfig.verify_and_update_config
        # to be skipped and unify_kv_cache_spec_page_size to crash.
        for arch in (
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            cls = getattr(module, arch, None)
            if cls is not None:
                ModelRegistry.register_model(arch, cls)
                print(
                    f"{_PLUGIN_LABEL} registered {arch} in ModelRegistry "
                    f"(is_hybrid={getattr(cls, 'is_hybrid', False)})."
                )

        # ---- Dense Qwen3.5 (e.g. Qwen3.5-0.8B-Base / 2B-Base) ----
        # vLLM 0.19.1 does NOT register the text-only ``Qwen3_5ForCausalLM``
        # in its dispatch table (only the VL ConditionalGeneration variants
        # are registered). Without registration vLLM normalizes the arch to
        # ``Qwen3_5ForConditionalGeneration`` (multimodal), which then trips
        # over the ``Qwen3_5TextConfig`` vs ``Qwen3_5Config`` mismatch in
        # the MM data-parser path. Bare ``Qwen3_5ForCausalLM`` from the
        # patched module also lacks IsHybrid + the mamba state classmethods,
        # so we synthesize a subclass that mirrors what
        # ``Qwen3_5MoeForCausalLM`` does for hybrid setup.
        dense_cls = getattr(module, "Qwen3_5ForCausalLM", None)
        moe_cls = getattr(module, "Qwen3_5MoeForCausalLM", None)
        if dense_cls is not None and moe_cls is not None:
            try:
                from vllm.model_executor.models.interfaces import IsHybrid
                attrs = {
                    "__module__": dense_cls.__module__,
                    "get_mamba_state_dtype_from_config":
                        moe_cls.__dict__["get_mamba_state_dtype_from_config"],
                    "get_mamba_state_shape_from_config":
                        moe_cls.__dict__["get_mamba_state_shape_from_config"],
                    "get_mamba_state_copy_func":
                        moe_cls.__dict__["get_mamba_state_copy_func"],
                }
                dense_hybrid_cls = type(
                    "Qwen3_5ForCausalLM",
                    (dense_cls, IsHybrid),
                    attrs,
                )
                ModelRegistry.register_model("Qwen3_5ForCausalLM", dense_hybrid_cls)
                print(
                    f"{_PLUGIN_LABEL} registered Qwen3_5ForCausalLM "
                    f"(dense, is_hybrid=True, with mamba classmethods)."
                )
            except Exception as e:
                print(
                    f"{_PLUGIN_LABEL} failed to register dense "
                    f"Qwen3_5ForCausalLM: {e}"
                )
    except Exception as e:  # pragma: no cover
        print(f"{_PLUGIN_LABEL} ModelRegistry.register_model failed: {e}")
    return True


def _register_qwen3_5_moe_causal_lm_config() -> None:
    """Inject a MODELS_CONFIG_MAP entry for ``Qwen3_5MoeForCausalLM``.

    Strip ``mrope_section`` / ``mrope_interleaved`` from
    ``hf_config.rope_parameters`` so vLLM's ``uses_mrope(config)`` returns
    ``False``. These keys are vestigial leftovers from the multimodal/VL
    parent config; the text-only CausalLM arch does not implement M-RoPE,
    and when ``uses_mrope`` is ``True`` vLLM tries to build M-RoPE positions
    during generation and hits ``assert supports_mrope(model)``.
    """
    try:
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForConditionalGenerationConfig,
        )
    except Exception as e:  # pragma: no cover
        print(f"{_PLUGIN_LABEL} cannot import MODELS_CONFIG_MAP: {e}")
        return

    arch = "Qwen3_5MoeForCausalLM"
    if arch in MODELS_CONFIG_MAP and getattr(
        MODELS_CONFIG_MAP[arch], "__less_is_moe_injected__", False
    ):
        return

    class Qwen3_5MoeForCausalLMConfig(Qwen3_5ForConditionalGenerationConfig):
        __less_is_moe_injected__ = True

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            # Reuse upstream's mamba_ssm_cache_dtype resolution.
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(
                vllm_config
            )
            hf_cfg = vllm_config.model_config.hf_config
            hf_text = getattr(vllm_config.model_config, "hf_text_config", hf_cfg)
            for cfg in (hf_cfg, hf_text):
                rp = getattr(cfg, "rope_parameters", None)
                if isinstance(rp, dict):
                    for k in ("mrope_section", "mrope_interleaved"):
                        rp.pop(k, None)

    MODELS_CONFIG_MAP[arch] = Qwen3_5MoeForCausalLMConfig
    print(
        f"{_PLUGIN_LABEL} injected MODELS_CONFIG_MAP[{arch!r}] "
        f"(mrope strip + mamba_ssm_cache_dtype)."
    )

    # Mirror the same fixup for the dense ``Qwen3_5ForCausalLM`` arch
    # (Qwen3.5-0.8B-Base / 2B-Base). It also has the vestigial mrope keys
    # in rope_parameters that need stripping, and needs the hybrid
    # mamba_ssm_cache_dtype path.
    dense_arch = "Qwen3_5ForCausalLM"
    if dense_arch in MODELS_CONFIG_MAP and getattr(
        MODELS_CONFIG_MAP[dense_arch], "__less_is_moe_injected__", False
    ):
        return
    MODELS_CONFIG_MAP[dense_arch] = Qwen3_5MoeForCausalLMConfig
    print(
        f"{_PLUGIN_LABEL} injected MODELS_CONFIG_MAP[{dense_arch!r}] "
        f"(mrope strip + mamba_ssm_cache_dtype, dense)."
    )


def register() -> None:
    """Entry point invoked by vllm.plugins.load_general_plugins()."""
    if os.environ.get("LESS_IS_MOE_RUNTIME_PATCH") == "stock":
        return
    if _patch_vllm_qwen3_5_moe():
        _register_qwen3_5_moe_causal_lm_config()
