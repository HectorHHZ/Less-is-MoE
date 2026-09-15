"""Regression tests for generic IntDim-E discovery and verification.

Every family is exercised on a tiny randomly initialised model built from its
Transformers config class, so the suite runs on CPU in seconds. Families whose
config class is missing from the installed Transformers are skipped, not
failed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from less_is_moe.intdim import DiscoveryError, discover, probe_fused_layout, verify_checkpoint  # noqa: E402
from less_is_moe.intdim import registry  # noqa: E402
from less_is_moe.intdim.discover import FusedLayout, _consistent  # noqa: E402

HIDDEN, INTER, EXPERTS, TOP_K, LAYERS, VOCAB = 32, 12, 4, 2, 2, 64
COMMON = dict(
    hidden_size=HIDDEN, num_hidden_layers=LAYERS, vocab_size=VOCAB,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
)


# name -> builder returning a config, plus the expected intermediate-size key
FAMILIES = {
    "qwen2_moe": (
        lambda: transformers.Qwen2MoeConfig(**COMMON, intermediate_size=48, moe_intermediate_size=INTER,
                                            shared_expert_intermediate_size=16, num_experts=EXPERTS,
                                            num_experts_per_tok=TOP_K, decoder_sparse_step=1),
        "moe_intermediate_size",
    ),
    "qwen3_moe": (
        lambda: transformers.Qwen3MoeConfig(**COMMON, intermediate_size=48, moe_intermediate_size=INTER,
                                            num_experts=EXPERTS, num_experts_per_tok=TOP_K, mlp_only_layers=[]),
        "moe_intermediate_size",
    ),
    "olmoe": (
        lambda: transformers.OlmoeConfig(**COMMON, intermediate_size=INTER, num_experts=EXPERTS,
                                         num_experts_per_tok=TOP_K),
        "intermediate_size",
    ),
    "qwen3_5_moe": (
        # Released Qwen3.5 checkpoints nest the text config, so exercise the nested key path.
        lambda: transformers.Qwen3_5MoeConfig(text_config=dict(
            **COMMON, intermediate_size=48, moe_intermediate_size=INTER, shared_expert_intermediate_size=16,
            num_experts=EXPERTS, num_experts_per_tok=TOP_K, linear_num_value_heads=2, linear_num_key_heads=1,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=2)),
        "text_config.moe_intermediate_size",
    ),
    "gpt_oss": (
        lambda: transformers.GptOssConfig(**COMMON, intermediate_size=HIDDEN, num_local_experts=EXPERTS,
                                          num_experts_per_tok=TOP_K, sliding_window=16),
        "intermediate_size",
    ),
    "gemma4": (
        # The text config selects Gemma4ForCausalLM; the multimodal wrapper would also build a vision tower.
        lambda: transformers.Gemma4TextConfig(**COMMON, intermediate_size=48, moe_intermediate_size=INTER,
                                              num_experts=EXPERTS, top_k_experts=TOP_K, enable_moe_block=True,
                                              sliding_window=16, hidden_size_per_layer_input=0,
                                              vocab_size_per_layer_input=VOCAB),
        "moe_intermediate_size",
    ),
}
FUSED_ONLY = ("gpt_oss", "gemma4")  # families that are fused on every Transformers version


def _build(family):
    builder, key = FAMILIES[family]
    try:
        config = builder()
    except AttributeError as exc:  # config class not in this Transformers release
        pytest.skip(f"{family}: {exc}")
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(config).float().eval()
    # Guard against a config silently falling back to full-size defaults.
    size = sum(p.numel() for p in model.parameters())
    assert size < 5_000_000, f"{family}: built a {size:,}-parameter model; the tiny config was not applied"
    return model, config, key


@pytest.fixture(params=list(FAMILIES))
def family(request):
    return request.param


def test_discover_every_layer(family):
    model, config, key = _build(family)
    handles = discover(model, config)
    assert [h.layer_index for h in handles] == list(range(LAYERS))
    expected_inter = HIDDEN if family == "gpt_oss" else INTER
    for h in handles:
        assert (h.num_experts, h.intermediate_size, h.hidden_size) == (EXPERTS, expected_inter, HIDDEN)
        assert str(h.intermediate_size_key) == key
        assert h.expert_count_key is not None and h.expert_count_key.get(config) == EXPERTS
        assert h.top_k_key is not None and h.top_k_key.get(config) == TOP_K
        gate, up, down = h.expert_weights(0)
        assert gate.shape == up.shape == (expected_inter, HIDDEN) and down.shape == (HIDDEN, expected_inter)
    if family in FUSED_ONLY:
        assert all(h.kind == "fused" for h in handles)


def test_gpt_oss_layout_is_interleaved_with_biases():
    model, config, _ = _build("gpt_oss")
    h = discover(model, config)[0]
    assert h.kind == "fused" and h.fused.pairing == "interleaved"
    assert h.gate_up_bias is not None and h.down_bias is not None
    # I == H for gpt-oss: the down axis is ambiguous by shape and must come from the probe.
    assert h.fused.down_unit_axis == 1  # GptOssExperts.down_proj is (E, I, H)


def test_gpt_oss_half_width_is_square_and_resolved():
    """Pruning gpt-oss to I = H/2 (2880 -> 1440 in the real model) makes gate_up square."""
    model, config, _ = _build("gpt_oss")
    handles = discover(model, config)
    for h in handles:
        h.apply_units(torch.arange(HIDDEN // 2))
        assert tuple(h.gate_up.shape[1:]) == (HIDDEN, HIDDEN)
        assert h.gate_up_bias is not h.down_bias  # both are (E, H) now; resolved by name
    handles[0].intermediate_size_key.set(config, HIDDEN // 2)
    again = discover(model, config)
    assert [(a.intermediate_size, a.fused) for a in again] == [(HIDDEN // 2, h.fused) for h in handles]

    # Before the config is updated, tensors and config disagree: discovery must refuse, not guess.
    handles[0].intermediate_size_key.set(config, HIDDEN)
    with pytest.raises(DiscoveryError, match="no config attribute equals"):
        discover(model, config)


def test_probe_rejects_wrong_pairing():
    for fam in FUSED_ONLY:
        model, config, _ = _build(fam)
        h = discover(model, config)[0]
        right = h.fused
        wrong = FusedLayout(right.gate_up_unit_axis, right.down_unit_axis,
                            "interleaved" if right.pairing == "concat" else "concat")
        assert _consistent(h, right, num_tokens=4, seed=0)
        assert not _consistent(h, wrong, num_tokens=4, seed=0)


def test_probe_resolves_layout_without_registry(monkeypatch):
    """A family with no registry entry must still be resolved, not guessed."""
    import importlib

    monkeypatch.setattr(registry, "LAYOUT_OVERRIDES", {})
    monkeypatch.setattr(importlib.import_module("less_is_moe.intdim.discover"), "LAYOUT_OVERRIDES", {})
    model, config, _ = _build("gpt_oss")
    h = discover(model, config)[0]
    assert h.fused.pairing == "interleaved" and h.fused.down_unit_axis == 1


@pytest.mark.parametrize("fam", FUSED_ONLY)
def test_probe_survives_dead_experts_and_units(fam):
    """Zero masks remove units, and with the layer or global scope whole experts.

    Expert 0 is fully zeroed and every other expert loses its even units; the
    probe must still find live units and resolve the same layout.
    """
    model, config, _ = _build(fam)
    h = discover(model, config)[0]  # probed, so the layout used for zeroing is the real one
    expected = h.fused
    with torch.no_grad():
        h.gate_up[0].zero_()
        h.down[0].zero_()
        if h.gate_up_bias is not None:
            h.gate_up_bias[0].zero_()
        dead = torch.arange(0, h.intermediate_size, 2)
        for e in range(1, h.num_experts):
            rows = h.gate_up_indices(dead)
            h.gate_up[e].index_fill_(expected.gate_up_unit_axis - 1, rows, 0)
            if h.gate_up_bias is not None:
                h.gate_up_bias[e].index_fill_(0, rows, 0)
            h.down[e].index_fill_(expected.down_unit_axis - 1, dead, 0)
    assert discover(model, config)[0].fused == expected


def test_probe_reports_ambiguity_instead_of_guessing():
    model, config, _ = _build("gemma4")
    h = discover(model, config, probe=False)[0]
    # Zero every expert weight: both hypotheses now leave the output unchanged, so none passes.
    with torch.no_grad():
        h.gate_up.zero_(); h.down.zero_()
    with pytest.raises(DiscoveryError, match="consistent layouts"):
        probe_fused_layout(h)


def _prune_half(model, config):
    """Structural IntDim-E shrink: each expert keeps a different half of its units."""
    handles = discover(model, config)
    gen = torch.Generator().manual_seed(1)
    for h in handles:
        keep = torch.stack([torch.randperm(h.intermediate_size, generator=gen)[: h.intermediate_size // 2].sort().values
                            for _ in range(h.num_experts)])
        before = [h.expert_weights(e) for e in range(h.num_experts)]
        h.apply_units(keep)
        for e, (gate, up, down) in enumerate(before):
            g2, u2, d2 = h.expert_weights(e)
            assert torch.equal(g2, gate[keep[e]]) and torch.equal(u2, up[keep[e]]) and torch.equal(d2, down[:, keep[e]])
    h0 = handles[0]
    h0.intermediate_size_key.set(config, h0.intermediate_size)
    return handles


def test_verify_roundtrip(family, tmp_path):
    model, config, _ = _build(family)
    model.save_pretrained(tmp_path / "base")
    report = verify_checkpoint(tmp_path / "base")
    assert report.ok, str(report)

    handles = _prune_half(model, config)
    model.config = config
    model.save_pretrained(tmp_path / "pruned")
    report = verify_checkpoint(tmp_path / "pruned")
    assert report.ok, str(report)
    # Re-discovery on the reloaded checkpoint must find the same layout and the same kept weights.
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "pruned").float().eval()
    rebuilt = discover(reloaded)
    assert len(rebuilt) == len(handles)
    for before, after in zip(handles, rebuilt):
        assert (after.kind, after.fused, after.intermediate_size) == (before.kind, before.fused, before.intermediate_size)
        for e in range(after.num_experts):
            for w_before, w_after in zip(before.expert_weights(e), after.expert_weights(e)):
                torch.testing.assert_close(w_after, w_before)

    # A config that disagrees with the tensors must be caught.
    handles[0].intermediate_size_key.set(config, handles[0].intermediate_size * 2)
    config.save_pretrained(tmp_path / "pruned")
    report = verify_checkpoint(tmp_path / "pruned")
    assert not report.ok and report.mismatched, str(report)


class _LegacyExpert(torch.nn.Module):
    """Per-expert SwiGLU MLP as stored by Transformers 4.x Qwen-MoE and OLMoE."""

    def __init__(self, hidden, inter):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)
        self.intermediate_size = inter

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _LegacyMoeModel(torch.nn.Module):
    """Minimal stand-in for a Transformers 4.x MoE: layers.N.mlp.experts is a ModuleList."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(LAYERS):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = torch.nn.Linear(HIDDEN, EXPERTS, bias=False)
            layer.mlp.experts = torch.nn.ModuleList(_LegacyExpert(HIDDEN, INTER) for _ in range(EXPERTS))
            layer.mlp.shared_expert = _LegacyExpert(HIDDEN, 16)  # a single MLP, not an expert list
            self.layers.append(layer)


def test_discover_modulelist_layout():
    """The per-expert ``nn.Linear`` layout used by the legacy and qwen3 environments."""
    config = transformers.PretrainedConfig(hidden_size=HIDDEN, moe_intermediate_size=INTER, intermediate_size=48,
                                           num_experts=EXPERTS, num_experts_per_tok=TOP_K)
    torch.manual_seed(0)
    model = _LegacyMoeModel(config)
    handles = discover(model, config)
    assert [h.name for h in handles] == [f"layers.{i}.mlp.experts" for i in range(LAYERS)]
    h = handles[0]
    assert h.kind == "modulelist" and h.linear_names == ("gate_proj", "up_proj", "down_proj")
    assert (h.num_experts, h.intermediate_size) == (EXPERTS, INTER)
    assert str(h.intermediate_size_key) == "moe_intermediate_size"  # not the dense intermediate_size=48

    x = torch.randn(3, HIDDEN)
    keep = torch.stack([torch.randperm(INTER)[: INTER // 2].sort().values for _ in range(EXPERTS)])
    # Removing units must equal zeroing them: the kept units' contribution is unchanged.
    expected = []
    with torch.no_grad():
        for e in range(EXPERTS):
            expert = model.layers[0].mlp.experts[e]
            mask = torch.zeros(INTER); mask[keep[e]] = 1
            gate, up = expert.gate_proj(x), expert.up_proj(x)
            expected.append(expert.down_proj(torch.nn.functional.silu(gate) * up * mask))
    h.apply_units(keep)
    with torch.no_grad():
        for e in range(EXPERTS):
            expert = model.layers[0].mlp.experts[e]
            assert expert.gate_proj.weight.shape == (INTER // 2, HIDDEN) and expert.intermediate_size == INTER // 2
            torch.testing.assert_close(expert(x), expected[e])


def test_verify_cli_exit_code(tmp_path):
    model, config, _ = _build("olmoe")
    model.save_pretrained(tmp_path)
    from less_is_moe.intdim.verify import main

    assert main([str(tmp_path)]) == 0
