"""Known names and per-family overrides for structural MoE discovery.

Discovery works from parameter shapes, so this table only records what shapes
cannot tell apart. It is expected to stay short: a new family should need zero
entries, or one line when its gate/up rows are interleaved instead of
concatenated.
"""

from __future__ import annotations

# Config attributes that may hold the routed-expert count, expert intermediate
# size, and top-k. Discovery picks the attribute whose value matches the
# discovered tensor shapes, so order does not encode priority.
EXPERT_COUNT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
)
INTERMEDIATE_SIZE_KEYS = (
    "moe_intermediate_size",
    "intermediate_size",
    "expert_intermediate_size",
    "moe_ffn_hidden_size",
)
TOP_K_KEYS = (
    "num_experts_per_tok",
    "top_k_experts",
    "experts_per_token",
    "moe_top_k",
    "num_selected_experts",
)

# Per-expert ``nn.Linear`` attribute names, in (gate, up, down) order, tried in
# turn on the children of an expert ``ModuleList``.
LINEAR_NAME_SETS = (
    ("gate_proj", "up_proj", "down_proj"),
    ("w1", "w3", "w2"),
)

# Module attributes that mirror the expert intermediate size and must follow a
# structural shrink.
INTERMEDIATE_SIZE_ATTRS = ("intermediate_size", "intermediate_dim", "ffn_dim")

# How the fused ``gate_up`` tensor pairs gate and up rows. ``concat`` means
# rows ``[0, I)`` are gate and ``[I, 2I)`` are up; ``interleaved`` means even
# rows are gate and odd rows are up. The layout probe verifies the choice, and
# resolves it when a family is missing here, so an entry only records a known
# answer and skips the probe's fallback search.
LAYOUT_OVERRIDES: dict[str, dict[str, object]] = {
    "gpt_oss": {"pairing": "interleaved"},
}
DEFAULT_PAIRING = "concat"
