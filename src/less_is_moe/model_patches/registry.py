"""Model-family detection and runtime patch registration.

The patches in this package replace only the model classes that need to know
about a pruned checkpoint.  Family selection is deliberately based on an
explicit family name or Hugging Face configuration metadata; checkpoint paths
are not used as the source of truth.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES = (
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_5_moe",
    "olmoe",
)

_FAMILY_ALIASES = {
    "qwen1_5_moe": "qwen2_moe",
    "qwen15_moe": "qwen2_moe",
    "qwen2_moe": "qwen2_moe",
    "qwen2moe": "qwen2_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3moe": "qwen3_moe",
    "qwen3_5": "qwen3_5_moe",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen35_moe": "qwen3_5_moe",
    "qwen3_5moe": "qwen3_5_moe",
    "olmoe": "olmoe",
    "olmoe_moe": "olmoe",
}

_MODEL_TYPE_TO_FAMILY = {
    "qwen2_moe": "qwen2_moe",
    "qwen3_moe": "qwen3_moe",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
    "olmoe": "olmoe",
}

_HF_PATCHES = {
    "qwen2_moe": (
        "less_is_moe.model_patches.hf.qwen2_moe.modeling_qwen2_moe",
        "transformers.models.qwen2_moe.modeling_qwen2_moe",
        ("Qwen2MoeSparseMoeBlock", "Qwen2MoeDecoderLayer"),
    ),
    "qwen3_moe": (
        "less_is_moe.model_patches.hf.qwen3_moe.modeling_qwen3_moe",
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        ("Qwen3MoeSparseMoeBlock", "Qwen3MoeDecoderLayer"),
    ),
    "qwen3_5_moe": (
        "less_is_moe.model_patches.hf.qwen3_5_moe.modeling_qwen3_5_moe",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        (
            "Qwen3_5MoeTopKRouter",
            "Qwen3_5MoeSparseMoeBlock",
            "Qwen3_5MoeDecoderLayer",
        ),
    ),
    "olmoe": (
        "less_is_moe.model_patches.hf.olmoe.modeling_olmoe",
        "transformers.models.olmoe.modeling_olmoe",
        ("OlmoeSparseMoeBlock", "OlmoeDecoderLayer"),
    ),
}

_VLLM_PATCHES = {
    "qwen2_moe": ("qwen2_moe", "qwen2_moe.py"),
    "qwen3_moe": ("qwen3_moe", "qwen3_moe.py"),
    "qwen3_5_moe": ("qwen3_5", "qwen3_5.py"),
    "olmoe": ("olmoe", "olmoe.py"),
}


def _normalise_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


def normalise_model_family(family: str) -> str:
    """Return the canonical family name for a supported explicit alias."""
    key = _normalise_name(family)
    try:
        return _FAMILY_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_FAMILIES)
        raise ValueError(
            f"Unsupported model family {family!r}; expected one of: {supported}"
        ) from exc


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _family_from_architecture(architecture: str) -> str | None:
    compact = _normalise_name(architecture).replace("_", "")
    # Qwen3.5 must be checked before Qwen3.
    if compact.startswith("qwen35moe"):
        return "qwen3_5_moe"
    if compact.startswith("qwen3moe"):
        return "qwen3_moe"
    if compact.startswith("qwen2moe"):
        return "qwen2_moe"
    if compact.startswith("olmoe"):
        return "olmoe"
    return None


def family_from_config(config: Any) -> str:
    """Resolve a supported family from ``model_type`` or ``architectures``.

    Composite Qwen3.5 configurations may hold the language-model metadata in
    ``text_config``.  Both the outer and nested configs are therefore checked.
    """
    candidates = [config]
    text_config = _config_value(config, "text_config")
    if text_config is not None and text_config is not config:
        candidates.append(text_config)

    seen_model_types: list[str] = []
    seen_architectures: list[str] = []
    for candidate in candidates:
        model_type = _config_value(candidate, "model_type")
        if isinstance(model_type, str):
            normalised = _normalise_name(model_type)
            seen_model_types.append(model_type)
            family = _MODEL_TYPE_TO_FAMILY.get(normalised)
            if family is not None:
                return family

        architectures = _config_value(candidate, "architectures", ()) or ()
        if isinstance(architectures, str):
            architectures = (architectures,)
        for architecture in architectures:
            if not isinstance(architecture, str):
                continue
            seen_architectures.append(architecture)
            family = _family_from_architecture(architecture)
            if family is not None:
                return family

    raise ValueError(
        "Unsupported model configuration: "
        f"model_type={seen_model_types or None}, "
        f"architectures={seen_architectures or None}. "
        f"Supported families: {', '.join(SUPPORTED_FAMILIES)}"
    )


def validate_hf_patch_config(
    config: Any,
    family: str | None = None,
) -> str:
    """Fail fast for checkpoint layouts that the HF patch cannot reload.

    Qwen3.5 stores all experts in batched 3-D parameters, and its upstream HF
    constructor requires one scalar expert count.  A list-valued count is
    emitted only by uneven/global expert pruning and cannot be reconstructed
    safely by the available experiment patch.
    """
    resolved = family_from_config(config) if family is None else normalise_model_family(family)
    if resolved != "qwen3_5_moe":
        return resolved

    text_config = _config_value(config, "text_config")
    if text_config is None:
        text_config = config
    num_experts = _config_value(text_config, "num_experts")
    if isinstance(num_experts, (list, tuple)):
        raise ValueError(
            "Qwen3.5-MoE Hugging Face checkpoints require a scalar "
            "config.num_experts. A list-valued count indicates uneven/global "
            "expert pruning, which the batched Qwen3_5MoeExperts layout cannot "
            "reload. Use uniform per-layer expert pruning, router masking, or "
            "a neuron-structured checkpoint instead."
        )
    return resolved


def detect_model_family(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    revision: str | None = None,
    *,
    validate_for_hf: bool = False,
) -> str:
    """Load only the Hugging Face config and return its canonical family.

    Set ``validate_for_hf`` for training/loading paths that will use the HF
    patch.  It is off by default because vLLM supports some per-layer layouts
    (notably uneven Qwen3.5 expert counts) that the HF implementation cannot.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "transformers is required to detect a model family from a checkpoint"
        ) from exc

    config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        config_kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    family = family_from_config(config)
    if validate_for_hf:
        validate_hf_patch_config(config, family)
    return family


def _resolve_family(family_or_config: str | Any) -> str:
    if isinstance(family_or_config, str):
        return normalise_model_family(family_or_config)
    return family_from_config(family_or_config)


def _refresh_qwen3_5_output_recorder(
    upstream_module: ModuleType,
    patch_module: ModuleType,
) -> None:
    """Teach Transformers 5.x output capture about the replaced router class."""
    recorder_class = getattr(upstream_module, "OutputRecorder", None)
    pretrained_class = getattr(upstream_module, "Qwen3_5MoePreTrainedModel", None)
    router_class = getattr(patch_module, "Qwen3_5MoeTopKRouter", None)
    if recorder_class is None or pretrained_class is None or router_class is None:
        logger.debug("Qwen3.5 output recorder API is unavailable in this Transformers version")
        return

    recorders = dict(getattr(pretrained_class, "_can_record_outputs", {}))
    recorders["router_logits"] = recorder_class(router_class, index=0)
    pretrained_class._can_record_outputs = recorders


def apply_hf_patch(family: str | Any) -> bool:
    """Install the Hugging Face class replacements for ``family``.

    ``family`` may be a canonical/aliased family string or an already-loaded
    config object.  The function is safe to call more than once.
    """
    resolved = _resolve_family(family)
    if not isinstance(family, str):
        validate_hf_patch_config(family, resolved)
    patch_module_name, upstream_module_name, symbols = _HF_PATCHES[resolved]

    patch_module = importlib.import_module(patch_module_name)
    upstream_module = importlib.import_module(upstream_module_name)
    upstream_package_name = upstream_module_name.rpartition(".")[0]
    upstream_package = importlib.import_module(upstream_package_name)

    for symbol in symbols:
        replacement = getattr(patch_module, symbol)
        setattr(upstream_module, symbol, replacement)
        # Transformers exposes model classes from a LazyModule package too.
        setattr(upstream_package, symbol, replacement)

    if resolved == "qwen3_5_moe":
        _refresh_qwen3_5_output_recorder(upstream_module, patch_module)

    logger.info("Applied Hugging Face %s pruning patch", resolved)
    return True


def _module_is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_vllm_replacement(module_name: str, patch_path: Path) -> ModuleType:
    previous = sys.modules.get(module_name)
    spec = importlib.util.spec_from_file_location(module_name, patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {patch_path}")

    replacement = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = replacement
    try:
        spec.loader.exec_module(replacement)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise

    models_package = importlib.import_module("vllm.model_executor.models")
    setattr(models_package, module_name.rsplit(".", 1)[-1], replacement)
    return replacement


def _register_qwen3_5_architectures(module: ModuleType) -> None:
    """Register text/VL Qwen3.5 classes with vLLM's process-safe registry."""
    try:
        from vllm import ModelRegistry
    except (ImportError, AttributeError):  # pragma: no cover - version specific
        return

    for architecture in (
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        model_class = getattr(module, architecture, None)
        if model_class is not None:
            ModelRegistry.register_model(architecture, model_class)

    # vLLM 0.19.1 does not register the text-only dense architecture.  Keep
    # the compatibility shim from the experiment environment; it is harmless
    # for MoE checkpoints and lets the same Qwen3.5 module be inspected safely.
    dense_class = getattr(module, "Qwen3_5ForCausalLM", None)
    moe_class = getattr(module, "Qwen3_5MoeForCausalLM", None)
    if dense_class is None or moe_class is None:
        return
    try:
        from vllm.model_executor.models.interfaces import IsHybrid

        attributes = {
            "__module__": dense_class.__module__,
            "get_mamba_state_dtype_from_config": moe_class.__dict__[
                "get_mamba_state_dtype_from_config"
            ],
            "get_mamba_state_shape_from_config": moe_class.__dict__[
                "get_mamba_state_shape_from_config"
            ],
            "get_mamba_state_copy_func": moe_class.__dict__[
                "get_mamba_state_copy_func"
            ],
        }
        dense_hybrid_class = type(
            "Qwen3_5ForCausalLM",
            (dense_class, IsHybrid),
            attributes,
        )
        ModelRegistry.register_model("Qwen3_5ForCausalLM", dense_hybrid_class)
    except (ImportError, KeyError, TypeError, AttributeError) as exc:
        logger.debug("Could not register dense Qwen3.5 compatibility class: %s", exc)


def apply_vllm_patch(family: str | Any) -> bool:
    """Replace vLLM's model module with the pruning-aware implementation.

    Returns ``False`` when vLLM (or its Qwen3.5 implementation in an older
    vLLM release) is not installed.  Other import failures are allowed to
    surface because they usually indicate a mismatched supported environment.
    """
    resolved = _resolve_family(family)
    upstream_leaf, filename = _VLLM_PATCHES[resolved]
    upstream_module_name = f"vllm.model_executor.models.{upstream_leaf}"

    if not _module_is_available(upstream_module_name):
        logger.info(
            "Skipping vLLM %s patch because %s is unavailable",
            resolved,
            upstream_module_name,
        )
        return False

    patch_path = Path(__file__).with_name("vllm") / filename
    replacement = _load_vllm_replacement(upstream_module_name, patch_path)
    if resolved == "qwen3_5_moe":
        _register_qwen3_5_architectures(replacement)

    logger.info("Applied vLLM %s pruning patch from %s", resolved, patch_path)
    return True


__all__ = [
    "SUPPORTED_FAMILIES",
    "apply_hf_patch",
    "apply_vllm_patch",
    "detect_model_family",
    "family_from_config",
    "normalise_model_family",
    "validate_hf_patch_config",
]
