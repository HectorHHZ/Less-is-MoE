"""GPU comparisons against the unchanged released per-family functions."""

from __future__ import annotations

import copy
import importlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_intdim import _build
from test_intdim_runtime import eager_model
from less_is_moe.intdim import discover
from less_is_moe.intdim.scoring import collect_scores, select_expert_units

LEGACY = {
    "qwen2_moe": "qwen15_moe",
    "qwen3_moe": "qwen3",
    "olmoe": "olmoe",
    "qwen3_5_moe": "qwen3_5",
}


class LegacyExperts(torch.nn.ModuleList):
    """Expose the original Linear layout to both algorithms in the same HF model.

    HF 5.x fuses these parameters. The old Qwen2/Qwen3/OLMoE functions require
    ModuleList, so both runs use this identical routing/weight representation.
    This compares algorithms within one runtime, not results across HF versions.
    """

    def forward(self, x, top_k_index, top_k_weights):
        result = torch.zeros_like(x)
        for e, expert in enumerate(self):
            tokens, slots = torch.where(top_k_index == e)
            if tokens.numel():
                values = expert.down_proj(torch.nn.functional.silu(expert.gate_proj(x[tokens])) * expert.up_proj(x[tokens]))
                result.index_add_(0, tokens, values * top_k_weights[tokens, slots, None])
        return result


def _legacy_layout(model):
    for h in discover(model):
        experts = LegacyExperts()
        for e in range(h.num_experts):
            expert = torch.nn.Module()
            expert.intermediate_size = h.intermediate_size
            for name, weight in zip(("gate_proj", "up_proj", "down_proj"), h.expert_weights(e)):
                linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
                linear.weight = torch.nn.Parameter(weight.detach().clone())
                setattr(expert, name, linear)
            experts.append(expert)
        parent_name, child_name = h.name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, experts)
    return model


@pytest.mark.parametrize("family,native", [(family, False) for family in LEGACY] + [(family, True) for family in LEGACY if family != "qwen3_5_moe"])
def test_legacy_intdim_e_equivalence(family, native, runtime):
    model, config, _ = _build(family)
    model.to(device=runtime[0], dtype=runtime[1])
    eager_model(model)
    new = copy.deepcopy(model)
    if family != "qwen3_5_moe":
        model = _legacy_layout(model)
        if not native:
            new = copy.deepcopy(model)
    old = copy.deepcopy(model)
    suffix = LEGACY[family]
    legacy_score = importlib.import_module(f"less_is_moe.pruning.neuron_drop_{suffix}")
    legacy_shrink = importlib.import_module(f"less_is_moe.pruning.neuron_structure_drop_{suffix}")
    handles = discover(new)
    layers = [h.layer_index for h in handles]
    batches = [torch.tensor([[1, 3, 5, 7, 9, 11]], device=runtime[0]), torch.tensor([[2, 4, 6, 8, 10, 12]], device=runtime[0])]
    torch.manual_seed(123)
    old_scores = legacy_score.collect_neuron_gradient_scores(old, batches, layers)
    torch.manual_seed(123)
    new_scores = collect_scores(new, batches, handles)
    for layer in layers:
        for e in old_scores[layer]:
            if native:
                # Fused gate/up GEMMs and separate Linear GEMMs can round differently.
                torch.testing.assert_close(old_scores[layer][e], new_scores[layer][e],
                                           rtol=0.02 if runtime[1] == torch.bfloat16 else 1e-5, atol=1e-8)
            else:
                assert torch.equal(old_scores[layer][e], new_scores[layer][e]), (family, layer, e, "score drift")
    score_max_abs = max((old_scores[l][e] - new_scores[l][e]).abs().max().item()
                        for l in layers for e in old_scores[l])
    drops = legacy_score.decide_neurons_to_drop(old_scores, 0.5)
    keep = select_expert_units(new_scores, 0.5)
    for h in handles:
        for e in range(h.num_experts):
            expected = [i for i in range(h.intermediate_size) if i not in drops[h.layer_index][e]]
            assert keep[h.layer_index][e].tolist() == expected
    legacy_shrink.structurally_remove_neurons(old, drops, layers)
    for h in handles:
        h.apply_units(keep[h.layer_index])
    handles[0].intermediate_size_key.set(new.config, handles[0].intermediate_size)
    handles[0].intermediate_size_key.set(old.config, handles[0].intermediate_size)
    # Compare canonical expert tensors across the old Linear and native fused layouts.
    old_handles = discover(old)
    for old_h, new_h in zip(old_handles, handles):
        for e in range(new_h.num_experts):
            for a, b in zip(old_h.expert_weights(e), new_h.expert_weights(e)):
                assert torch.equal(a, b), (family, e, "expert tensor drift")
    expert_prefixes = tuple(h.name + "." for h in handles)
    for key, value in new.state_dict().items():
        if not key.startswith(expert_prefixes) or not native:
            assert torch.equal(value, old.state_dict()[key]), (family, key, "tensor drift")
    with torch.inference_mode():
        a = old(batches[0], use_cache=False).logits
        b = new(batches[0], use_cache=False).logits
    if native:
        torch.testing.assert_close(a, b, rtol=0.02 if runtime[1] == torch.bfloat16 else 1e-5,
                                   atol=0.002 if runtime[1] == torch.bfloat16 else 1e-6)
    else:
        assert torch.equal(a, b), (family, "forward drift")
    print(json.dumps({"family": family, "dtype": str(runtime[1]), "native_fused": native,
                      "indices_and_weights_equal": True, "score_max_abs": score_max_abs,
                      "logits_max_abs": (a - b).abs().max().item(), "logits_bitwise": torch.equal(a, b)}))
