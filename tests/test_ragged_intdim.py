"""GPU-only non-uniform checkpoint and kernel regressions."""

import copy
import json
import os

import pytest

torch = pytest.importorskip("torch")
AutoModelForCausalLM = pytest.importorskip("transformers").AutoModelForCausalLM

from docker.intdim_vllm_smoke import make_config
from less_is_moe.intdim import discover
from less_is_moe.intdim import prune as P
from less_is_moe.intdim.ragged import PackedExperts, compact_model, load_checkpoint, save_checkpoint, validate_metadata


def require_gpu():
    if not torch.cuda.is_available():
        if os.environ.get("INTDIM_REQUIRE_GPU") == "1":
            pytest.fail("Ragged validation requires a GPU")
        pytest.skip("GPU-only ragged validation")


def test_probe_finds_sparse_signal_among_tiny_nonzero_weights():
    """The full 122B checkpoint has units that are nonzero but uninformative."""
    require_gpu()
    from less_is_moe.intdim.discover import FusedLayout, _consistent
    config = make_config("qwen3_5_moe_122b").get_text_config()
    config._experts_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
    handle = discover(model, probe=False)[0]
    with torch.no_grad():
        handle.gate_up.fill_(1e-20)
        handle.down.fill_(1e-20)
        # Neither this expert nor its active unit was chosen by the old
        # maximum-nonzero-count / evenly-spaced-row sampling.
        gate, up, down = handle.expert_weights(1)
        gate[55].normal_(std=0.1)
        up[55].normal_(std=0.1)
        down[:, 55].normal_(std=0.1)
    before = {k: v.clone() for k, v in handle.experts.state_dict().items()}
    assert _consistent(handle, FusedLayout(1, 2, "concat"), 4, 0)
    assert not _consistent(handle, FusedLayout(1, 2, "interleaved"), 4, 0)
    for name, value in before.items():
        assert torch.equal(value, handle.experts.state_dict()[name])


@pytest.fixture(params=["qwen2_moe", "olmoe", "qwen3_moe", "qwen3_5_moe_35b", "qwen3_5_moe_122b", "gpt_oss", "gemma4"])
def model(request):
    require_gpu()
    torch.manual_seed(7)
    config = make_config(request.param).get_text_config()
    config._experts_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
    if request.param == "gpt_oss":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".experts." in name and "bias" in name:
                    parameter.normal_(std=0.02)
    return model


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
    from docker.ragged_qwen_smoke import verify_zero_mask_compaction
    verified = verify_zero_mask_compaction(model, discover(reference), plan)
    assert verified["experts_checked"] == sum(h.num_experts for h in handles)
    assert verified["removed_zero_neurons"] == sum(len(ids) for layer in plan.values() for ids in layer.values())
    # In particular, shared experts and their gates must not be pruned.
    for name, value in reference.state_dict().items():
        if ".experts." not in name:
            assert torch.equal(value, model.state_dict()[name]), name
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


def test_exact_compaction_check_rejects_changed_weights_and_nonzero_removals(model):
    from docker.ragged_qwen_smoke import verify_zero_mask_compaction
    handles = discover(model)
    plan = {h.layer_index: {e: list(range(h.intermediate_size // 2)) for e in range(h.num_experts)}
            for h in handles}
    P.zero_dropped_neurons(handles, plan)
    compact_model(model, handles, plan)
    verify_zero_mask_compaction(model, handles, plan)
    gate = model.get_submodule(handles[0].name).expert_weights(0)[0]
    with torch.no_grad():
        original = gate[0, 0].clone()
        gate[0, 0] = original + 1
    with pytest.raises(AssertionError, match="Retained weight differs"):
        verify_zero_mask_compaction(model, handles, plan)
    with torch.no_grad():
        gate[0, 0] = original
        handles[0].expert_weights(0)[2][0, 0] = 1
    with pytest.raises(AssertionError, match="Removed columns must be zero-masked"):
        verify_zero_mask_compaction(model, handles, plan)


@pytest.mark.parametrize("tokens", [1, 17, 65])
@pytest.mark.parametrize("widths", [[0, 33, 127, 256], [17, 64, 191, 255], [0, 0, 0, 0]])
@pytest.mark.parametrize("activation,bias", [("silu", False), ("gelu_tanh", False), ("swigluoai", True)])
def test_ragged_kernel(tokens, widths, activation, bias):
    from less_is_moe.intdim.ragged_triton import ragged_experts
    require_gpu()
    torch.manual_seed(17)
    experts = PackedExperts(widths, 128, device="cuda", dtype=torch.bfloat16, activation=activation, bias=bias)
    with torch.no_grad():
        experts.gate_up_proj.normal_(std=0.04)
        experts.down_proj.normal_(std=0.04)
        if bias:
            experts.gate_up_proj_bias.normal_(std=0.1)
            experts.down_proj_bias.normal_(std=0.1)
    hidden = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    probabilities = torch.randn(tokens, 4, device="cuda").softmax(-1)
    weights, ids = probabilities.topk(2)
    weights = (weights / weights.sum(-1, keepdim=True)).bfloat16()
    sizes = torch.tensor(widths, dtype=torch.int32, device="cuda")
    offsets = torch.tensor(experts.offsets, dtype=torch.int64, device="cuda")
    expected = experts(hidden, ids, weights)
    actual = ragged_experts(hidden, experts.gate_up_proj, experts.down_proj,
                            sizes, offsets, max(widths), ids, weights, activation=activation,
                            gate_up_bias=getattr(experts, "gate_up_proj_bias", None),
                            down_bias=getattr(experts, "down_proj_bias", None))
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


@pytest.mark.parametrize("scope", ["layer", "global"])
def test_ragged_cli(model, scope, tmp_path):
    from test_intdim_prune import _save_tokenizer
    base = tmp_path / "base"
    model.save_pretrained(base)
    _save_tokenizer(base)
    calib = tmp_path / "calib.jsonl"
    calib.write_text(json.dumps({"text": "t1 t3 t5 t7 t9 t11"}) + "\n")
    out = tmp_path / "compact"
    assert P.main(["--model_name_or_path", str(base), "--output_dir", str(out),
                   "--mode", "ragged", "--prune_mode", scope, "--drop_ratio", "0.5",
                   "--calib_data", str(calib), "--n_samples", "1", "--seq_len", "6", "--dtype", "bf16"]) == 0
    summary = json.loads((out / P.STRUCTURAL_SUMMARY_FILE).read_text())
    assert summary["ragged_load_verified"] is True
    assert summary["prune_mode"] == scope


def test_zero_layer_roundtrip(model, tmp_path):
    handles = discover(model)
    plan = {h.layer_index: {e: list(range(h.intermediate_size)) if h.layer_index == 0 else [0, 3, 9]
                            for e in range(h.num_experts)} for h in handles}
    compact_model(model, handles, plan)
    save_checkpoint(model, tmp_path)
    restored = load_checkpoint(tmp_path)
    tokens = torch.tensor([[1, 3, 5]], device="cuda")
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)


@pytest.mark.parametrize("missing", [False, True])
def test_gemma_multimodal_prefix_and_missing_weight_guard(tmp_path, missing):
    from safetensors.torch import load_file, save_file
    from transformers import Gemma4Config
    from less_is_moe.intdim.ragged_hf import load_source_model
    require_gpu()
    config = make_config("gemma4")
    original = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
    original.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    renamed = {name.replace("model.", "model.language_model.", 1): tensor for name, tensor in weights.items()}
    if missing:
        del renamed["model.language_model.layers.0.experts.down_proj"]
    save_file(renamed, tmp_path / "model.safetensors")
    Gemma4Config(text_config=config.to_dict()).save_pretrained(tmp_path)
    if missing:
        with pytest.raises(RuntimeError, match="Incomplete pretrained language weights"):
            load_source_model(tmp_path, dtype=torch.bfloat16)
    else:
        restored = load_source_model(tmp_path, dtype=torch.bfloat16)
        assert restored.config.model_type == "gemma4_text"
        for name, value in original.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name]), name
