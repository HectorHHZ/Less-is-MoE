"""Stable access to the repository's two evaluation protocols.

The original experiment scripts intentionally used different prompts for
OLMoE and the Qwen MoE models.  Callers such as quantized-model evaluators
should select a protocol explicitly instead of silently mixing prompt sets.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Literal, cast


ProtocolName = Literal["zero-shot", "olmoe-multishot"]

ZERO_SHOT: ProtocolName = "zero-shot"
OLMOE_MULTISHOT: ProtocolName = "olmoe-multishot"

_PROTOCOL_MODULES = {
    ZERO_SHOT: "less_is_moe.evaluation.vllm_zero_shot",
    OLMOE_MULTISHOT: "less_is_moe.evaluation.vllm_olmoe_multishot",
}
_PROTOCOL_ALIASES = {
    "zero_shot": ZERO_SHOT,
    "qwen-zero-shot": ZERO_SHOT,
    "qwen_zero_shot": ZERO_SHOT,
    "olmoe": OLMOE_MULTISHOT,
    "olmoe_multishot": OLMOE_MULTISHOT,
    "multi-shot": OLMOE_MULTISHOT,
    "multishot": OLMOE_MULTISHOT,
}


def normalize_protocol(protocol: str) -> ProtocolName:
    """Return the canonical protocol name or raise for an ambiguous value."""
    normalized = protocol.strip().lower()
    normalized = _PROTOCOL_ALIASES.get(normalized, normalized)
    if normalized not in _PROTOCOL_MODULES:
        valid = ", ".join(_PROTOCOL_MODULES)
        raise ValueError(f"Unknown evaluation protocol {protocol!r}; choose {valid}.")
    return cast(ProtocolName, normalized)


@lru_cache(maxsize=2)
def get_protocol_module(protocol: str) -> ModuleType:
    """Load the exact experiment module for *protocol*.

    Import is lazy so lightweight callers can inspect protocol names without
    importing vLLM.
    """
    canonical = normalize_protocol(protocol)
    return import_module(_PROTOCOL_MODULES[canonical])


def build_math_prompt(problem: str, *, protocol: str = ZERO_SHOT) -> str:
    return get_protocol_module(protocol)._build_math_prompt(problem)


def build_gsm_prompt(question: str, *, protocol: str = ZERO_SHOT) -> str:
    module = get_protocol_module(protocol)
    if normalize_protocol(protocol) == ZERO_SHOT:
        return question
    return module._build_gsm_paper_prompt(question)


def build_multiarith_prompt(
    instruction: str, *, protocol: str = ZERO_SHOT
) -> str:
    module = get_protocol_module(protocol)
    if normalize_protocol(protocol) == ZERO_SHOT:
        return instruction
    return module._build_multiarith_paper_prompt(instruction)


def build_mbpp_prompt(
    prompt: str,
    *,
    protocol: str = ZERO_SHOT,
    n_shots: int = 3,
) -> str:
    module = get_protocol_module(protocol)
    if normalize_protocol(protocol) == ZERO_SHOT:
        return prompt
    return module._build_mbpp_paper_prompt(prompt, n_shots=n_shots)


def build_chinese_mcq_prompt(
    subject: str,
    question: str,
    a: str,
    b: str,
    c: str,
    d: str,
    *,
    protocol: str = ZERO_SHOT,
) -> str:
    return get_protocol_module(protocol)._build_mcq_prompt_zh(
        subject, question, a, b, c, d
    )


def build_bbh_prompt(question: str, *, protocol: str = ZERO_SHOT) -> str:
    return get_protocol_module(protocol)._build_bbh_prompt(question)


def extract_last_boxed(text: str, *, protocol: str = ZERO_SHOT) -> str:
    return get_protocol_module(protocol)._extract_last_boxed(text)


def normalize_math_answer(answer: str, *, protocol: str = ZERO_SHOT) -> str:
    return get_protocol_module(protocol)._normalize_math_answer(answer)


def normalize_bbh_answer(answer: str, *, protocol: str = ZERO_SHOT) -> str:
    return get_protocol_module(protocol)._normalize_bbh_answer(answer)


def extract_chinese_mcq_letter(
    text: str, *, protocol: str = ZERO_SHOT
) -> str:
    return get_protocol_module(protocol).extract_chinese_mcq_letter(text)


def preprocess_humaneval_completion(
    completion: str,
    *,
    language: str = "python",
    protocol: str = ZERO_SHOT,
) -> str:
    return get_protocol_module(protocol).preprocess_humaneval_completion(
        completion, language=language
    )


def build_humaneval_check(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    *,
    protocol: str = ZERO_SHOT,
) -> str:
    return get_protocol_module(protocol).build_humaneval_check(
        prompt, completion, test, entry_point
    )
