from __future__ import annotations

import pytest

from less_is_moe.evaluation.common import (
    OLMOE_MULTISHOT,
    ZERO_SHOT,
    normalize_protocol,
)
from less_is_moe.model_patches.registry import (
    family_from_config,
    normalise_model_family,
    validate_hf_patch_config,
)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("qwen15_moe", "qwen2_moe"),
        ("Qwen3-MoE", "qwen3_moe"),
        ("qwen3.5_moe", "qwen3_5_moe"),
        ("OLMoE", "olmoe"),
    ],
)
def test_model_family_aliases(alias: str, expected: str) -> None:
    assert normalise_model_family(alias) == expected


def test_nested_qwen35_config_detection() -> None:
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_experts": 64,
        },
    }
    assert family_from_config(config) == "qwen3_5_moe"
    assert validate_hf_patch_config(config) == "qwen3_5_moe"


def test_hf_rejects_uneven_qwen35_expert_counts() -> None:
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_experts": [32, 31],
        },
    }
    with pytest.raises(ValueError, match="scalar"):
        validate_hf_patch_config(config)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("zero-shot", ZERO_SHOT),
        ("qwen_zero_shot", ZERO_SHOT),
        ("olmoe", OLMOE_MULTISHOT),
        ("multi-shot", OLMOE_MULTISHOT),
    ],
)
def test_protocol_aliases(name: str, expected: str) -> None:
    assert normalize_protocol(name) == expected


def test_unknown_protocol_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown evaluation protocol"):
        normalize_protocol("auto")
