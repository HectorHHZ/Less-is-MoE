"""Strict GPU family coverage in FP32 and BF16; no CPU model execution."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_intdim import FAMILIES, _build
from less_is_moe.intdim import discover, verify_checkpoint
from less_is_moe.intdim.discover import _eager_experts


def _model(family, runtime):
    try:
        model, config, _ = _build(family)
    except pytest.skip.Exception as exc:
        pytest.fail(f"The unified runtime must support {family}: {exc}")
    eager_model(model)
    return model.to(device=runtime[0], dtype=runtime[1]), model.config


def eager_model(model):
    # These tiny fixture widths are not aligned for grouped CUDA GEMMs.
    # Both equivalence branches use the same stock eager GPU implementation.
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._experts_implementation = "eager"
    return model


def _expert_output(experts, x, count):
    index = torch.arange(count, device=x.device).repeat_interleave(2)[:, None]
    weights = torch.ones_like(index, dtype=x.dtype)
    with torch.no_grad(), _eager_experts(experts):
        out = experts(x, index, weights)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("family", FAMILIES)
def test_family_runtime_roundtrip(family, runtime, tmp_path):
    model, config = _model(family, runtime)
    if family == "gpt_oss":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "bias" in name and parameter.ndim == 2:
                    parameter.copy_(torch.linspace(0.01, 0.05, parameter.numel(), device=parameter.device).reshape(parameter.shape))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    handles = discover(model)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)
    gen = torch.Generator().manual_seed(9)
    for h in handles:
        reference = copy.deepcopy(h.experts)
        ref_handle = copy.copy(h)
        ref_handle.experts = reference
        keep = torch.stack([
            torch.randperm(h.intermediate_size, generator=gen)[: h.intermediate_size // 2]
            for _ in range(h.num_experts)
        ])
        x = torch.randn(h.num_experts * 2, h.hidden_size, generator=gen).to(*runtime)
        with torch.no_grad():
            for e in range(h.num_experts):
                dropped = torch.ones(h.intermediate_size, dtype=torch.bool)
                dropped[keep[e]] = False
                ref_handle.expert_weights(e)[2][:, dropped] = 0
        expected = _expert_output(reference, x, h.num_experts)
        h.apply_units(keep)
        actual = _expert_output(h.experts, x, h.num_experts)
        tol = 2e-5 if runtime[1] == torch.bfloat16 else 1e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=0.02 if runtime[1] == torch.bfloat16 else 1e-5)
    handles[0].intermediate_size_key.set(config, handles[0].intermediate_size)
    checkpoint = tmp_path / family
    model.save_pretrained(checkpoint)
    report = verify_checkpoint(checkpoint)
    assert report.ok, str(report)
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=runtime[1]).to(runtime[0]).eval()
    eager_model(reloaded)
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), key
    after = discover(reloaded)
    assert len(after) == len(handles)
    for original, loaded in zip(handles, after):
        assert original.fused == loaded.fused
        for e in range(original.num_experts):
            for a, b in zip(original.expert_weights(e), loaded.expert_weights(e)):
                assert torch.equal(a, b)
    tokens = torch.tensor([[1, 3, 5, 7]], device=runtime[0])
    with torch.inference_mode():
        expected_logits = model(tokens, use_cache=False).logits
        actual_logits = reloaded(tokens, use_cache=False).logits
    assert torch.isfinite(actual_logits).all()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0.02, atol=0.02)


def test_probe_failure_preserves_live_weights(runtime, monkeypatch):
    import importlib
    module = importlib.import_module("less_is_moe.intdim.discover")
    model, _ = _model("gpt_oss", runtime)
    handle = discover(model)[0]
    before = {k: v.detach().clone() for k, v in handle.experts.state_dict().items()}
    original = module._run_expert
    calls = 0
    def fail_after_mutation(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected forward failure")
        return original(*args)
    monkeypatch.setattr(module, "_run_expert", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        module.probe_fused_layout(handle)
    for key, value in before.items():
        assert torch.equal(handle.experts.state_dict()[key], value)
