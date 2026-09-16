"""GPU-only non-uniform checkpoint and kernel regressions."""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM

from docker.intdim_vllm_smoke import make_config
from less_is_moe.intdim import discover
from less_is_moe.intdim import prune as P
from less_is_moe.intdim.ragged import PackedExperts, compact_model, load_checkpoint, save_checkpoint, validate_metadata


@pytest.fixture
def model():
    if not torch.cuda.is_available():
        pytest.fail("Ragged validation requires a GPU")
    torch.manual_seed(7)
    config = make_config("qwen3_moe")
    config._experts_implementation = "eager"
    return AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()


@pytest.mark.parametrize("scope", ["layer", "global"])
def test_calibrated_roundtrip(model, scope, tmp_path):
    calibration = [torch.tensor([[1, 3, 5, 7, 9, 11]], device="cuda"),
                   torch.tensor([[2, 4, 6, 8, 10, 12]], device="cuda")]
    handles = discover(model)
    scores = P.collect_neuron_gradient_scores(model, handles, calibration)
    plan = P.pick_neurons_to_drop(scores, 0.5, scope)
    reference = copy.deepcopy(model)
    P.zero_dropped_neurons(discover(reference), plan)
    summary = compact_model(model, handles, plan)
    assert summary["compact_parameters"] < summary["original_parameters"]
    assert len({w for ws in summary["expert_intermediate_sizes"].values() for w in ws}) > 1
    tokens = torch.tensor([[1, 13, 15, 17]], device="cuda")
    with torch.inference_mode():
        expected = reference(tokens).logits
        actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.015)
    save_checkpoint(model, tmp_path)
    reloaded = load_checkpoint(tmp_path, attn_implementation="eager")
    for name, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name]), name
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(tokens).logits, actual, rtol=0.03, atol=0.015)
    wrong = copy.deepcopy(reloaded.config)
    wrong.less_is_moe["expert_intermediate_sizes"]["0"].pop()
    with pytest.raises(ValueError, match="width count"):
        validate_metadata(wrong)


@pytest.mark.parametrize("tokens", [1, 17, 65])
@pytest.mark.parametrize("widths", [[0, 33, 127, 256], [17, 64, 191, 255], [0, 0, 0, 0]])
def test_ragged_kernel(tokens, widths):
    from less_is_moe.intdim.ragged_triton import ragged_experts
    if not torch.cuda.is_available():
        pytest.fail("Ragged validation requires a GPU")
    torch.manual_seed(17)
    experts = PackedExperts(widths, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.04)
        experts.down_proj.normal_(std=0.04)
    hidden = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.randn(tokens, 4, device="cuda").softmax(-1)
    weights, ids = probabilities.topk(2)
    weights = (weights / weights.sum(-1, keepdim=True)).bfloat16()
    sizes = torch.tensor(widths, dtype=torch.int32, device="cuda")
    offsets = torch.tensor(experts.offsets, dtype=torch.int64, device="cuda")
    expected = experts(hidden, ids, weights)
    actual = ragged_experts(hidden, experts.gate_up_proj, experts.down_proj,
                            sizes, offsets, max(widths), ids, weights)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.003)


def test_invalid_export_preserves_model(model):
    handles = discover(model)
    plan = {h.layer_index: {e: [0] for e in range(h.num_experts)} for h in handles}
    plan[handles[-1].layer_index][0] = [-1]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="Invalid dropped"):
        compact_model(model, handles, plan)
    assert not hasattr(model.config, "less_is_moe")
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
