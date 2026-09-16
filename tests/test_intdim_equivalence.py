"""One GPU equivalence matrix for all models, plus native-layout regressions."""

from __future__ import annotations

import copy
import importlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
hf = pytest.importorskip("transformers")

from docker.intdim_vllm_smoke import make_config
from test_intdim import _build
from test_intdim_runtime import eager_model
from less_is_moe.intdim import discover, verify_checkpoint
from less_is_moe.intdim import prune as pruning
from less_is_moe.intdim.scoring import collect_scores, select_expert_units

LEGACY = {
    "qwen2_moe": "qwen15_moe",
    "qwen3_moe": "qwen3",
    "olmoe": "olmoe",
    "qwen3_5_moe": "qwen3_5",
}


@dataclass(frozen=True)
class ModelCase:
    family: str
    reference: str
    width: int
    linear_reference: bool = False
    ported_reference: bool = False


CASES = (
    ModelCase("qwen2_moe", "qwen15_moe", 256, linear_reference=True),
    ModelCase("qwen3_moe", "qwen3", 256, linear_reference=True),
    ModelCase("olmoe", "olmoe", 256, linear_reference=True),
    ModelCase("qwen3_5_moe_35b", "qwen3_5", 512),
    ModelCase("qwen3_5_moe_122b", "qwen3_5", 1024),
    ModelCase("gpt_oss", "gpt_oss", 2880, ported_reference=True),
    ModelCase("gemma4", "gemma4", 704, ported_reference=True),
)


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


def _case_layers(model, case):
    if case.reference == "qwen3_5":
        from less_is_moe.pruning.neuron_drop_qwen3_5 import _get_decoder_layers
        return _get_decoder_layers(model)
    return model.model.layers


def _case_parent(model, case, layer):
    block = _case_layers(model, case)[layer]
    return block if case.family == "gemma4" else block.mlp


def _case_experts(model, case, layer):
    return _case_parent(model, case, layer).experts


def _fixed_tensors(experts, case, expert):
    """Explicit reference layouts; never use discovery to read oracle tensors."""
    if isinstance(experts, torch.nn.ModuleList):
        mlp = experts[expert]
        return mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    gu, down = experts.gate_up_proj[expert], experts.down_proj[expert]
    if case.family == "gpt_oss":
        return gu[:, ::2].T, gu[:, 1::2].T, down.T
    width = gu.shape[0] // 2
    return gu[:width], gu[width:], down


def _reference_layout(model, case):
    if case.linear_reference:
        for layer in range(len(_case_layers(model, case))):
            original = _case_experts(model, case, layer)
            experts = LegacyExperts()
            for e in range(original.num_experts):
                expert = torch.nn.Module()
                expert.intermediate_size = case.width
                for name, weight in zip(("gate_proj", "up_proj", "down_proj"), _fixed_tensors(original, case, e)):
                    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                             device=weight.device, dtype=weight.dtype)
                    linear.weight = torch.nn.Parameter(weight.detach().clone())
                    setattr(expert, name, linear)
                experts.append(expert)
            _case_parent(model, case, layer).experts = experts
    return model


def _set_reference_width(model, case, width):
    config = getattr(model.config, "text_config", model.config)
    key = "intermediate_size" if case.family in ("olmoe", "gpt_oss") else "moe_intermediate_size"
    setattr(config, key, width)


def _stock_checkpoint_model(model, case):
    """Repack old Linear checkpoints for stock HF, without using autodetect.

    The uniform scoring comparison uses identical layouts in both branches.
    Separate native-layout tests above cover changed GEMM reduction rounding.
    """
    if not case.linear_reference:
        return model
    parameter = next(model.parameters())
    stock = hf.AutoModelForCausalLM.from_config(copy.deepcopy(model.config), dtype=parameter.dtype).to(parameter.device).eval()
    eager_model(stock)
    state = dict(model.state_dict())
    for layer in range(len(_case_layers(model, case))):
        experts = _case_experts(model, case, layer)
        prefix = next(name for name, module in model.named_modules() if module is experts) + "."
        for name in list(state):
            if name.startswith(prefix):
                del state[name]
        projections = [_fixed_tensors(experts, case, e) for e in range(len(experts))]
        state[prefix + "gate_up_proj"] = torch.stack([torch.cat((gate, up)) for gate, up, _ in projections])
        state[prefix + "down_proj"] = torch.stack([down for _, _, down in projections])
    stock.load_state_dict(state, strict=True)
    return stock


def _assert_state_equal(actual, expected):
    left, right = actual.state_dict(), expected.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _manual_mask(model, case, dropped):
    with torch.no_grad():
        for layer, experts_dropped in dropped.items():
            experts = _case_experts(model, case, layer)
            for e, ids in experts_dropped.items():
                gate, up, down = _fixed_tensors(experts, case, e)
                gate[ids] = 0
                up[ids] = 0
                down[:, ids] = 0
                if case.family == "gpt_oss":
                    bias_ids = [index for j in ids for index in (2 * j, 2 * j + 1)]
                    experts.gate_up_proj_bias[e, bias_ids] = 0


@contextmanager
def _forbid_autodetect(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The reference must compute scores and masks independently of autodetect")
    with monkeypatch.context() as guard:
        discovery = importlib.import_module("less_is_moe.intdim.discover")
        scoring = importlib.import_module("less_is_moe.intdim.scoring")
        guard.setattr(importlib.import_module("less_is_moe.intdim"), "discover", forbidden)
        guard.setattr(discovery, "discover", forbidden)
        for name in ("discover", "collect_scores", "select_expert_units"):
            guard.setattr(scoring, name, forbidden)
        for name in ("discover", "collect_neuron_gradient_scores", "pick_neurons_to_drop", "zero_dropped_neurons", "prune"):
            guard.setattr(pruning, name, forbidden)
        for name in ("expert_weights", "apply_units", "select_units"):
            guard.setattr(discovery.MoeLayerHandle, name, forbidden)
        yield


def _logits(model, batches):
    with torch.inference_mode():
        return torch.cat([model(tokens, use_cache=False).logits for tokens in batches], dim=1)


def _assert_logits_close(actual, expected, dtype):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.002 if dtype == torch.bfloat16 else 1e-5)


def _vllm_tokens(checkpoint):
    script = Path(__file__).resolve().parents[1] / "docker" / "intdim_vllm_smoke.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint)],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"checkpoint":')]
    assert len(records) == 1 and records[0]["stock_vllm"]
    return records[0]["tokens"]


def _prune_args(mode="structural", scope="expert", from_zeroed=False):
    argv = ["--model_name_or_path", "gpu-fixture", "--output_dir", "unused",
            "--mode", mode, "--prune_mode", scope, "--drop_ratio", "0.5",
            "--n_samples", "2", "--seq_len", "6"]
    if from_zeroed:
        argv.append("--from_zeroed_model")
    args = pruning.build_parser().parse_args(argv)
    pruning.validate_args(args)
    return args


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.family)
def test_uniform_intdim_e(case, runtime, tmp_path, monkeypatch):
    """All seven cases follow the same independent scoring -> mask -> HF/vLLM checks."""
    device, dtype = runtime
    torch.manual_seed(0)
    base = hf.AutoModelForCausalLM.from_config(make_config(case.family), dtype=dtype).to(device).eval()
    assert sum(p.numel() for p in base.parameters()) < 20_000_000
    eager_model(base)
    layers = list(range(len(_case_layers(base, case))))
    assert layers == [0, 1]
    if case.family == "gpt_oss":
        with torch.no_grad():
            for layer in layers:
                experts = _case_experts(base, case, layer)
                for parameter in (experts.gate_up_proj_bias, experts.down_proj_bias):
                    parameter.copy_(torch.linspace(0.01, 0.05, parameter.numel(), device=device).reshape(parameter.shape))
    reference = _reference_layout(copy.deepcopy(base), case)
    automatic, expected_mask = copy.deepcopy(reference), copy.deepcopy(reference)
    original_state = {name: value.detach().clone() for name, value in automatic.state_dict().items()}
    original_config = reference.config.to_dict()
    calib = [torch.tensor([row], device=device) for row in ([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12])]
    evaluation = calib + [torch.tensor([[13, 15, 17, 19]], device=device)]
    old_score = importlib.import_module(f"less_is_moe.pruning.neuron_drop_{case.reference}")
    old_compact = None

    with _forbid_autodetect(monkeypatch):
        torch.manual_seed(123)
        reference_scores = old_score.collect_neuron_gradient_scores(reference, calib, layers)
        dropped = old_score.decide_neurons_to_drop(reference_scores, 0.5)
        _assert_state_equal(reference, expected_mask)
        if not case.ported_reference:
            old_compact = copy.deepcopy(reference)
            shrink = importlib.import_module(f"less_is_moe.pruning.neuron_structure_drop_{case.reference}")
            shrink.structurally_remove_neurons(old_compact, dropped, layers)
            _set_reference_width(old_compact, case, case.width // 2)
        old_score.zero_dropped_neurons(reference, dropped, layers)
    _manual_mask(expected_mask, case, dropped)
    _assert_state_equal(reference, expected_mask)
    assert reference.config.to_dict() == original_config

    handles = discover(automatic)
    _assert_state_equal(automatic, _reference_layout(copy.deepcopy(base), case))
    torch.manual_seed(123)
    automatic_scores = pruning.collect_neuron_gradient_scores(automatic, handles, calib)
    for name, value in automatic.state_dict().items():
        assert torch.equal(value, original_state[name]), (case.family, name, "scoring modified weights")
    automatic_plan = pruning.pick_neurons_to_drop(automatic_scores, 0.5, "expert")
    assert automatic_plan == dropped
    # All scopes use an independent legacy selector and mask audit. Exercise
    # the real public pipeline, including its own scoring, rather than a mock.
    selector = importlib.import_module("less_is_moe.pruning.neuron_drop_qwen15_moe")
    for scope in pruning.PRUNE_MODES:
        with _forbid_autodetect(monkeypatch):
            expected_plan = selector.pick_neurons_to_drop(reference_scores, 0.5, scope)
        assert pruning.pick_neurons_to_drop(automatic_scores, 0.5, scope) == expected_plan
        masked = _reference_layout(copy.deepcopy(base), case)
        wanted = copy.deepcopy(masked)
        _manual_mask(wanted, case, expected_plan)
        torch.manual_seed(123)
        mask_summary = pruning.prune(masked, _prune_args("mask", scope), calib)
        _assert_state_equal(masked, wanted)
        assert masked.config.to_dict() == original_config
        assert mask_summary["total_dropped"] == sum(len(ids) for experts in expected_plan.values() for ids in experts.values())
        assert torch.equal(_logits(masked, evaluation), _logits(wanted.eval(), evaluation))
        del masked, wanted

    torch.manual_seed(123)
    structural_summary = pruning.prune(automatic, _prune_args(), calib)
    assert structural_summary["new_d_ffn"] == case.width // 2
    handles = discover(automatic)
    score_max_abs, scores_bitwise = 0.0, True
    for h in handles:
        assert (h.num_experts, h.hidden_size, h.intermediate_size) == (4, 128, case.width // 2)
        selected = {}
        for e in range(h.num_experts):
            actual, expected = automatic_scores[h.layer_index][e], reference_scores[h.layer_index][e]
            assert torch.isfinite(actual).all() and (actual >= 0).all()
            if case.ported_reference:
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
            else:
                assert torch.equal(actual, expected), "same-layout score drift"
            score_max_abs = max(score_max_abs, (actual - expected).abs().max().item())
            scores_bitwise &= torch.equal(actual, expected)
            ids = dropped[h.layer_index][e]
            assert len(ids) == case.width // 2
            selected[e] = torch.tensor([j for j in range(case.width) if j not in ids], device=device)
        assert sum(score.sum().item() for score in automatic_scores[h.layer_index].values()) > 0
        experts = _case_experts(reference, case, h.layer_index)
        for e, ids in selected.items():
            gate, up, down = _fixed_tensors(experts, case, e)
            for actual, expected in zip(h.expert_weights(e), (gate[ids], up[ids], down[:, ids])):
                assert torch.equal(actual, expected), "kept tensor drift"
            if case.family == "gpt_oss":
                bias_ids = torch.stack((2 * ids, 2 * ids + 1), dim=-1).flatten()
                assert torch.equal(h.gate_up_bias[e], experts.gate_up_proj_bias[e, bias_ids])
                assert torch.equal(h.down_bias[e], experts.down_proj_bias[e])
    from_mask = copy.deepcopy(reference)
    from_mask_summary = pruning.prune(from_mask, _prune_args(from_zeroed=True), None)
    assert from_mask_summary["new_d_ffn"] == case.width // 2
    _assert_state_equal(from_mask, automatic)
    assert from_mask.config.to_dict() == automatic.config.to_dict()
    prefixes = tuple(h.name + "." for h in handles)
    for name, value in automatic.state_dict().items():
        if not name.startswith(prefixes):
            assert torch.equal(value, original_state[name]), (case.family, name, "non-expert tensor drift")
    reference_logits, automatic_logits = _logits(reference, evaluation), _logits(automatic, evaluation)
    _assert_logits_close(automatic_logits, reference_logits, dtype)
    if old_compact is not None:
        _assert_state_equal(automatic, old_compact)
        assert torch.equal(automatic_logits, _logits(old_compact, evaluation))
    metrics = {"suite": "uniform", "implementation": "intdim.prune (PR #26)",
               "mask_scopes": list(pruning.PRUNE_MODES), "from_zeroed_equal": True, "family": case.family, "dtype": str(dtype),
               "reference_kind": "ported" if case.ported_reference else "released",
               "score_max_abs": score_max_abs, "scores_bitwise": scores_bitwise,
               "indices_and_kept_tensors_equal": True,
               "logits_max_abs": (automatic_logits - reference_logits).abs().max().item(),
               "legacy_structural_checked": old_compact is not None}

    for name, model, logits in (("zero_mask", reference, reference_logits), ("structural", automatic, automatic_logits)):
        stock = _stock_checkpoint_model(model, case)
        stock_logits = _logits(stock, evaluation)
        _assert_logits_close(stock_logits, logits, dtype)
        checkpoint = tmp_path / name
        stock.save_pretrained(checkpoint)
        assert verify_checkpoint(checkpoint).ok
        loaded = hf.AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype).to(device).eval()
        eager_model(loaded)
        _assert_state_equal(loaded, stock)
        assert torch.equal(stock_logits, _logits(loaded, evaluation))
        if name == "structural":
            assert [h.intermediate_size for h in discover(loaded)] == [case.width // 2] * 2
        del loaded, stock
    metrics["both_checkpoints_reload_exactly"] = True
    if dtype == torch.bfloat16 and os.environ.get("INTDIM_TEST_VLLM") == "1":
        base.save_pretrained(tmp_path / "base")
        metrics["stock_vllm_base_tokens"] = _vllm_tokens(tmp_path / "base")
        zero_tokens, structural_tokens = _vllm_tokens(tmp_path / "zero_mask"), _vllm_tokens(tmp_path / "structural")
        assert zero_tokens == structural_tokens, (zero_tokens, structural_tokens)
        metrics["stock_vllm_mask_structural_tokens_equal"] = zero_tokens
    print(json.dumps(metrics), flush=True)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.family)
def test_uniform_prune_cli(case, runtime, tmp_path, monkeypatch):
    """Actual CLI loaders, calibration, save and verify on all seven GPU models."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    device, dtype = runtime
    torch.manual_seed(0)
    model = hf.AutoModelForCausalLM.from_config(make_config(case.family), dtype=dtype).to(device).eval()
    eager_model(model)
    source = tmp_path / "base"
    model.save_pretrained(source)
    del model
    backend = Tokenizer(models.WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    hf.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0").save_pretrained(source)
    calib = tmp_path / "calib.jsonl"
    calib.write_text('\n'.join(json.dumps({"text": text}) for text in
                               ("t1 t3 t5 t7 t9 t11", "t2 t4 t6 t8 t10 t12")))
    dtype_arg = "fp32" if dtype == torch.float32 else "bf16"
    original_load = pruning.load_model

    def gpu_load(path, dtype):
        loaded = original_load(path, dtype)
        assert all(p.is_cuda for p in loaded.parameters()), "CLI must load all model parameters on GPU"
        return eager_model(loaded)

    monkeypatch.setattr(pruning, "load_model", gpu_load)
    common = ["--dtype", dtype_arg, "--n_samples", "2", "--seq_len", "6", "--calib_data", str(calib)]
    for mode in ("mask", "structural"):
        torch.manual_seed(123)
        assert pruning.main(["--model_name_or_path", str(source), "--output_dir", str(tmp_path / mode),
                             "--mode", mode, "--drop_ratio", "0.5", *common]) == 0
    assert pruning.main(["--model_name_or_path", str(tmp_path / "mask"),
                         "--output_dir", str(tmp_path / "from_zeroed"), "--mode", "structural",
                         "--from_zeroed_model", "--dtype", dtype_arg]) == 0
    loaded = {}
    tokens = torch.tensor([[13, 15, 17, 19]], device=device)
    for name in ("mask", "structural", "from_zeroed"):
        loaded[name] = gpu_load(tmp_path / name, dtype)
        assert verify_checkpoint(tmp_path / name).ok
        if name != "mask":
            summary = json.loads((tmp_path / name / pruning.STRUCTURAL_SUMMARY_FILE).read_text())
            assert summary["stock_load_verified"] is True
            assert summary["new_d_ffn"] == case.width // 2
    _assert_state_equal(loaded["from_zeroed"], loaded["structural"])
    _assert_logits_close(_logits(loaded["structural"], [tokens]), _logits(loaded["mask"], [tokens]), dtype)
    assert torch.equal(_logits(loaded["from_zeroed"], [tokens]), _logits(loaded["structural"], [tokens]))
    print(json.dumps({"suite": "prune_cli", "family": case.family, "dtype": str(dtype),
                      "mask_structural_from_zeroed": True, "all_model_parameters_cuda": True}), flush=True)
