"""Independent, model-specific old-method ports vs autodetected IntDim-E."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
hf = pytest.importorskip("transformers")

from docker.intdim_vllm_smoke import make_config
from less_is_moe.intdim import discover, verify_checkpoint
from less_is_moe.intdim.scoring import collect_scores, select_expert_units
from test_intdim_runtime import eager_model


def _explicit_experts(model: torch.nn.Module, family: str, layer: int) -> torch.nn.Module:
    block = model.model.layers[layer]
    return block.mlp.experts if family == "gpt_oss" else block.experts


def _explicit_tensors(experts: torch.nn.Module, family: str, expert: int) -> tuple:
    """Read canonical gate/up/down tensors without using a discovered handle."""
    gu, down = experts.gate_up_proj[expert], experts.down_proj[expert]
    if family == "gpt_oss":
        return gu[:, ::2].T, gu[:, 1::2].T, down.T
    width = gu.shape[0] // 2
    return gu[:width], gu[width:], down


def _logits(model: torch.nn.Module, batches: list[torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return torch.cat([model(tokens, use_cache=False).logits for tokens in batches], dim=1)


def _vllm_tokens(checkpoint: Path) -> list[int]:
    script = Path(__file__).resolve().parents[1] / "docker" / "intdim_vllm_smoke.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint)],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"checkpoint":')]
    assert len(records) == 1 and records[0]["stock_vllm"]
    return records[0]["tokens"]


@pytest.mark.parametrize("family", ("gpt_oss", "gemma4"))
def test_explicit_zero_mask_vs_autodetect(family: str, runtime: tuple, tmp_path: Path, monkeypatch) -> None:
    device, dtype = runtime
    width = 2880 if family == "gpt_oss" else 704
    torch.manual_seed(0)
    explicit = hf.AutoModelForCausalLM.from_config(make_config(family), dtype=dtype).to(device).eval()
    eager_model(explicit)
    layers = list(range(explicit.config.num_hidden_layers))
    assert layers == [0, 1]
    if family == "gpt_oss":
        with torch.no_grad():
            for layer in layers:
                experts = _explicit_experts(explicit, family, layer)
                # Nonzero biases ensure masking just weights cannot hide a bias bug.
                for parameter in (experts.gate_up_proj_bias, experts.down_proj_bias):
                    parameter.copy_(torch.linspace(0.01, 0.05, parameter.numel(), device=device).reshape(parameter.shape))
    automatic = copy.deepcopy(explicit)
    original_state = {name: value.detach().clone() for name, value in explicit.state_dict().items()}
    original_config = explicit.config.to_dict()
    calib = [torch.tensor([row], device=device) for row in ([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12])]
    evaluation = calib + [torch.tensor([[13, 15, 17, 19]], device=device)]
    reference = importlib.import_module(f"less_is_moe.pruning.neuron_drop_{family}")

    def forbidden(*args, **kwargs):
        raise AssertionError("The explicit zero-mask reference must not call autodetect")

    # Enforce independence while BOTH reference scores and reference masks are
    # computed. In particular, no keep/drop IDs from the automatic path exist yet.
    with monkeypatch.context() as guard:
        discovery = importlib.import_module("less_is_moe.intdim.discover")
        scoring = importlib.import_module("less_is_moe.intdim.scoring")
        guard.setattr(importlib.import_module("less_is_moe.intdim"), "discover", forbidden)
        guard.setattr(discovery, "discover", forbidden)
        for name in ("discover", "collect_scores", "select_expert_units"):
            guard.setattr(scoring, name, forbidden)
        for name in ("expert_weights", "apply_units", "select_units"):
            guard.setattr(discovery.MoeLayerHandle, name, forbidden)
        torch.manual_seed(123)
        reference_scores = reference.collect_neuron_gradient_scores(explicit, calib, layers)
        dropped = reference.decide_neurons_to_drop(reference_scores, 0.5)
        for name, value in explicit.state_dict().items():
            assert torch.equal(value, original_state[name]), (family, name, "reference scoring changed weights")
        reference.zero_dropped_neurons(explicit, dropped, layers)

    # Audit the zero-mask checkpoint itself: only intended expert weights and
    # GPT-OSS gate/up biases may change; all tensor shapes and config stay intact.
    expected_state = copy.deepcopy(original_state)
    kept = {}
    for layer in layers:
        prefix = f"model.layers.{layer}." + ("mlp.experts." if family == "gpt_oss" else "experts.")
        kept[layer] = {}
        for e, ids in dropped[layer].items():
            assert len(ids) == width // 2
            kept[layer][e] = torch.tensor([j for j in range(width) if j not in ids], device=device)
            if family == "gpt_oss":
                gu_ids = [index for j in ids for index in (2 * j, 2 * j + 1)]
                expected_state[prefix + "gate_up_proj"][e, :, gu_ids] = 0
                expected_state[prefix + "gate_up_proj_bias"][e, gu_ids] = 0
                expected_state[prefix + "down_proj"][e, ids, :] = 0
            else:
                gu_ids = ids + [j + width for j in ids]
                expected_state[prefix + "gate_up_proj"][e, gu_ids, :] = 0
                expected_state[prefix + "down_proj"][e, :, ids] = 0
    assert explicit.config.to_dict() == original_config
    for name, expected in expected_state.items():
        assert torch.equal(explicit.state_dict()[name], expected), (family, name, "incorrect zero mask")

    handles = discover(automatic)
    torch.manual_seed(123)
    automatic_scores = collect_scores(automatic, calib, handles)
    automatic_keep = select_expert_units(automatic_scores, 0.5)
    score_max_abs = 0.0
    scores_bitwise = True
    for h in handles:
        assert (h.num_experts, h.hidden_size, h.intermediate_size) == (4, 128, width)
        for e in range(h.num_experts):
            expected, actual = reference_scores[h.layer_index][e], automatic_scores[h.layer_index][e]
            # The old collector adds projection reductions on CPU; the fused
            # generic collector adds them on GPU. Allow only FP32 roundoff here.
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)
            score_max_abs = max(score_max_abs, (actual - expected).abs().max().item())
            scores_bitwise &= torch.equal(actual, expected)
            assert torch.equal(automatic_keep[h.layer_index][e].to(device), kept[h.layer_index][e])
        h.apply_units(automatic_keep[h.layer_index])
        experts = _explicit_experts(explicit, family, h.layer_index)
        for e in range(h.num_experts):
            ids = kept[h.layer_index][e]
            gate, up, down = _explicit_tensors(experts, family, e)
            for actual, expected in zip(h.expert_weights(e), (gate[ids], up[ids], down[:, ids])):
                assert torch.equal(actual, expected), (family, h.layer_index, e, "kept tensor drift")
            if family == "gpt_oss":
                bias_ids = torch.stack((2 * ids, 2 * ids + 1), dim=-1).flatten()
                assert torch.equal(h.gate_up_bias[e], experts.gate_up_proj_bias[e, bias_ids])
                assert torch.equal(h.down_bias[e], experts.down_proj_bias[e])
    handles[0].intermediate_size_key.set(automatic.config, width // 2)
    prefixes = tuple(h.name + "." for h in handles)
    for name, value in automatic.state_dict().items():
        if not name.startswith(prefixes):
            assert torch.equal(value, original_state[name]), (family, name, "non-expert tensor drift")
    expected_logits, actual_logits = _logits(explicit, evaluation), _logits(automatic, evaluation)
    torch.testing.assert_close(actual_logits, expected_logits,
                               rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.002 if dtype == torch.bfloat16 else 1e-5)
    metrics = {"family": family, "dtype": str(dtype), "independent_reference": True,
               "score_max_abs": score_max_abs, "scores_bitwise": scores_bitwise,
               "indices_and_kept_tensors_equal": True,
               "logits_max_abs": (actual_logits - expected_logits).abs().max().item(),
               "logits_bitwise": torch.equal(actual_logits, expected_logits)}

    for name, model, logits in (("zero_mask", explicit, expected_logits), ("structural", automatic, actual_logits)):
        path = tmp_path / name
        model.save_pretrained(path)
        assert verify_checkpoint(path).ok
        loaded = hf.AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        eager_model(loaded)
        for key, value in model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])
        assert torch.equal(logits, _logits(loaded, evaluation))
        del loaded
    metrics["both_checkpoints_reload_exactly"] = True
    if dtype == torch.bfloat16 and os.environ.get("INTDIM_TEST_VLLM") == "1":
        zero_tokens = _vllm_tokens(tmp_path / "zero_mask")
        structural_tokens = _vllm_tokens(tmp_path / "structural")
        assert zero_tokens == structural_tokens, (zero_tokens, structural_tokens)
        metrics["stock_vllm_tokens_equal"] = zero_tokens
    print(json.dumps(metrics), flush=True)
