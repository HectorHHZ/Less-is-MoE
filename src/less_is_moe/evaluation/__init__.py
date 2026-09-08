"""Evaluation entry points for Less-is-MoE."""

from .common import (
    OLMOE_MULTISHOT,
    ZERO_SHOT,
    get_protocol_module,
    normalize_protocol,
)

__all__ = [
    "OLMOE_MULTISHOT",
    "ZERO_SHOT",
    "get_protocol_module",
    "normalize_protocol",
]
