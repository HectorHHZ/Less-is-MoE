"""GPU gradient-driven IntDim-E at the four new models' expert widths.

The random fixtures share the vLLM smoke configurations: two layers, hidden
size 128, four experts and top-2 routing. They are not pretrained checkpoints.
"""

from __future__ import annotations

import copy
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

TARGETS = {
    "qwen3_5_moe_35b": 512,
    "qwen3_5_moe_122b": 1024,
    "gpt_oss": 2880,
    "gemma4": 704,
}


def _logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(tokens, use_cache=False).logits


@pytest.mark.parametrize("target", TARGETS)
def test_new_model_intdim_e(target: str, runtime: tuple, tmp_path: Path) -> None:
    device, dtype = runtime
    torch.manual_seed(0)
    model = hf.AutoModelForCausalLM.from_config(make_config(target), dtype=dtype).to(device).eval()
    assert sum(p.numel() for p in model.parameters()) < 20_000_000
    eager_model(model)
    if target == "gpt_oss":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "bias" in name and parameter.ndim == 2:
                    parameter.copy_(torch.linspace(0.01, 0.05, parameter.numel(), device=device).reshape(parameter.shape))
    model.save_pretrained(tmp_path / "base")
    reference = copy.deepcopy(model)
    handles = discover(model)
    assert len(handles) == 2
    assert all((h.hidden_size, h.num_experts, h.intermediate_size) == (128, 4, TARGETS[target]) for h in handles)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batches = [torch.tensor([row], device=device) for row in ([1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12])]
    torch.manual_seed(123)
    scores = collect_scores(model, batches, handles)
    for layer in scores.values():
        values = torch.stack(list(layer.values()))
        assert values.shape == (4, TARGETS[target])
        assert torch.isfinite(values).all() and (values >= 0).all() and values.sum() > 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), (target, name, "scoring modified weights")
    keep = select_expert_units(scores, 0.5)
    metrics = {"target": target, "dtype": str(dtype), "width_before": TARGETS[target],
               "width_after": TARGETS[target] // 2, "selection": "mean_abs_gradient",
               "legacy_comparison": target.startswith("qwen3_5")}

    # Both Qwen3.5 widths have a released algorithm to compare against.
    # GPT-OSS/Gemma4 have no legacy implementation; use a zero-mask oracle below.
    legacy = None
    if target.startswith("qwen3_5"):
        from less_is_moe.pruning import neuron_drop_qwen3_5 as legacy_score
        from less_is_moe.pruning import neuron_structure_drop_qwen3_5 as legacy_shrink

        legacy = copy.deepcopy(reference)
        layers = [h.layer_index for h in handles]
        torch.manual_seed(123)
        expected_scores = legacy_score.collect_neuron_gradient_scores(legacy, batches, layers)
        dropped = legacy_score.decide_neurons_to_drop(expected_scores, 0.5)
        for h in handles:
            for expert in range(h.num_experts):
                assert torch.equal(scores[h.layer_index][expert], expected_scores[h.layer_index][expert])
                expected_keep = [j for j in range(h.intermediate_size) if j not in dropped[h.layer_index][expert]]
                assert keep[h.layer_index][expert].tolist() == expected_keep
        legacy_shrink.structurally_remove_neurons(legacy, dropped, layers)
        handles[0].intermediate_size_key.set(legacy.config, TARGETS[target] // 2)

    # A full-size model with dropped down-projection columns zeroed supplies an
    # independent functional reference without calling structural apply_units.
    reference_handles = discover(reference)
    expected_experts = {}
    for h, ref in zip(handles, reference_handles):
        selected = keep[h.layer_index].to(device)
        with torch.no_grad():
            for expert in range(h.num_experts):
                indices = selected[expert]
                assert indices.numel() == TARGETS[target] // 2
                assert torch.unique(indices).numel() == indices.numel()
                gate, up, down = ref.expert_weights(expert)
                expected_experts[h.layer_index, expert] = (
                    gate[indices].clone(), up[indices].clone(), down[:, indices].clone())
                mask = torch.ones(TARGETS[target], dtype=torch.bool, device=device)
                mask[indices] = False
                down[:, mask] = 0
        # Exercise canonicalization with descending rather than pre-sorted IDs.
        h.apply_units(selected.flip(-1))
        for expert in range(h.num_experts):
            for actual, expected in zip(h.expert_weights(expert), expected_experts[h.layer_index, expert]):
                assert torch.equal(actual, expected), (target, h.layer_index, expert, "kept weight drift")
    handles[0].intermediate_size_key.set(model.config, TARGETS[target] // 2)
    expert_prefixes = tuple(h.name + "." for h in handles)
    for name, value in model.state_dict().items():
        if not name.startswith(expert_prefixes):
            assert torch.equal(value, before[name]), (target, name, "non-expert weight drift")

    tokens = batches[0]
    actual_logits = _logits(model, tokens)
    masked_logits = _logits(reference, tokens)
    assert torch.isfinite(actual_logits).all()
    torch.testing.assert_close(actual_logits, masked_logits,
                               rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.002 if dtype == torch.bfloat16 else 1e-5)
    metrics["masked_logits_max_abs"] = (actual_logits - masked_logits).abs().max().item()
    metrics["kept_weights_equal"] = True
    if legacy is not None:
        for name, value in model.state_dict().items():
            assert torch.equal(value, legacy.state_dict()[name]), (target, name, "legacy weight drift")
        legacy_logits = _logits(legacy, tokens)
        assert torch.equal(actual_logits, legacy_logits)
        metrics["legacy_scores_weights_logits_bitwise"] = True

    checkpoint = tmp_path / "pruned"
    model.save_pretrained(checkpoint)
    report = verify_checkpoint(checkpoint)
    assert report.ok, str(report)
    reloaded = hf.AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype).to(device).eval()
    eager_model(reloaded)
    for name, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name]), (target, name, "reload weight drift")
    reloaded_handles = discover(reloaded)
    assert [h.intermediate_size for h in reloaded_handles] == [TARGETS[target] // 2] * 2
    reloaded_logits = _logits(reloaded, tokens)
    assert torch.equal(actual_logits, reloaded_logits)
    metrics["reload_weights_logits_bitwise"] = True

    if dtype == torch.bfloat16 and os.environ.get("INTDIM_TEST_VLLM") == "1":
        script = Path(__file__).resolve().parents[1] / "docker" / "intdim_vllm_smoke.py"
        for stage in ("base", "pruned"):
            subprocess.run([sys.executable, str(script), "--checkpoint", str(tmp_path / stage)],
                           check=True, timeout=600)
        metrics["stock_vllm_base_and_gradient_pruned"] = "passed"
    print(json.dumps(metrics), flush=True)
