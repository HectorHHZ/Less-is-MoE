"""Equivalence and behavior tests for the generic IntDim pruner.

The per-family scripts in ``less_is_moe.pruning`` are the reference: on tiny
models the generic implementation must reproduce their scores, drop plans,
masked weights, zero-unit detection, and structural weights exactly. Fused
Transformers 5.x models are compared with the Qwen3.5 scripts (the only ones
with a fused code path); per-expert ``nn.Linear`` models on Transformers 4.x
are compared with their own family's scripts. Families without a script
(gpt-oss, Gemma-4) are checked for internal consistency instead.
"""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tqdm")  # imported at module level by the per-family scripts

from test_intdim import EXPERTS, FAMILIES, HIDDEN, INTER, VOCAB, _build  # noqa: E402

from less_is_moe import calibration  # noqa: E402
from less_is_moe.intdim import discover  # noqa: E402
from less_is_moe.intdim import prune as P  # noqa: E402
from less_is_moe.intdim.discover import _eager_experts  # noqa: E402

FUSED_REFERENCE_FAMILIES = {"qwen2_moe", "qwen3_moe", "olmoe", "qwen3_5_moe"}
MODULELIST_SCRIPT = {"qwen2_moe": "qwen15_moe", "qwen3_moe": "qwen3", "olmoe": "olmoe"}


def _reference_scripts(model, family):
    """(mask module, structural module) of the per-family implementation to compare against."""
    kind = discover(model)[0].kind
    if kind == "fused":
        if family not in FUSED_REFERENCE_FAMILIES:
            return None
        suffix = "qwen3_5"
    else:
        suffix = MODULELIST_SCRIPT.get(family)
        if suffix is None:
            return None
    return (
        importlib.import_module(f"less_is_moe.pruning.neuron_drop_{suffix}"),
        importlib.import_module(f"less_is_moe.pruning.neuron_structure_drop_{suffix}"),
    )


def _batches():
    gen = torch.Generator().manual_seed(123)
    return [torch.randint(0, VOCAB, (1, n), generator=gen) for n in (9, 12, 7)]


def _assert_same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for name in sa:
        assert sa[name].shape == sb[name].shape, name
        assert torch.equal(sa[name], sb[name]), name


def _assert_same_scores(old, new):
    assert list(old) == list(new)
    for layer in old:
        assert list(old[layer]) == list(new[layer])
        for e in old[layer]:
            assert torch.equal(old[layer][e], new[layer][e]), (layer, e)


# ------------------------------------------------------------ equivalence

@pytest.mark.parametrize("family", list(FAMILIES))
def test_matches_per_family_scripts(family):
    base, _, _ = _build(family)
    refs = _reference_scripts(base, family)
    if refs is None:
        pytest.skip(f"{family}: no per-family script to compare against")
    old_mask, old_struct = refs
    batches = _batches()

    old_model, new_model = copy.deepcopy(base), copy.deepcopy(base)
    indices, _ = old_mask.get_moe_layer_info(old_model)
    old_scores = old_mask.collect_neuron_gradient_scores(old_model, batches, indices)
    new_scores = P.collect_neuron_gradient_scores(new_model, discover(new_model), batches)
    _assert_same_scores(old_scores, new_scores)

    for mode in P.PRUNE_MODES:
        plan = P.pick_neurons_to_drop(new_scores, 0.5, mode)
        assert plan == old_mask.pick_neurons_to_drop(old_scores, 0.5, mode), mode

        old_masked, new_masked = copy.deepcopy(base), copy.deepcopy(base)
        old_summary = old_mask.zero_dropped_neurons(old_masked, plan, indices)
        new_summary = P.zero_dropped_neurons(discover(new_masked), plan)
        assert old_summary == new_summary, mode
        _assert_same_weights(old_masked, new_masked)
        # Discovery must still work on a half-zeroed model (from_zeroed_model reads one).
        assert old_struct.find_zeroed_neurons(old_masked, indices) == P.find_zeroed_neurons(discover(new_masked)), mode

    plan = P.pick_neurons_to_drop(new_scores, 0.5, "expert")
    old_pruned, new_pruned = copy.deepcopy(base), copy.deepcopy(base)
    old_result = old_struct.structurally_remove_neurons(old_pruned, plan, indices)
    new_result = P.structurally_remove_neurons(discover(new_pruned), plan)
    assert old_result == new_result
    _assert_same_weights(old_pruned, new_pruned)


def test_structural_rejects_non_uniform_widths():
    base, _, _ = _build("olmoe")
    handles = discover(base)
    scores = P.collect_neuron_gradient_scores(base, handles, _batches())
    plan = P.pick_neurons_to_drop(scores, 0.5, "global")
    if len({len(ids) for layer in plan.values() for ids in layer.values()}) == 1:
        pytest.skip("this random plan happens to be uniform")
    with pytest.raises(ValueError, match="uniform surviving d_ffn"):
        P.structurally_remove_neurons(handles, plan)


# ------------------------------------------------------ fused consistency

def _expert_outputs(handle, x):
    outs = []
    with _eager_experts(handle.experts):
        for e in range(handle.num_experts):
            index = torch.full((x.shape[0], 1), e, dtype=torch.long)
            weights = torch.ones(x.shape[0], 1, dtype=x.dtype)
            out = handle.experts(x, index, weights)
            outs.append(out[0] if isinstance(out, tuple) else out)
    return outs


@pytest.mark.parametrize("family", list(FAMILIES))
def test_structural_equals_mask_on_fused_experts(family):
    """Removing units must compute exactly what zeroing them computes, for every expert."""
    base, _, _ = _build(family)
    if discover(base)[0].kind != "fused":
        pytest.skip("covered for per-expert Linear layers by the equivalence test")
    scores = P.collect_neuron_gradient_scores(base, discover(base), _batches())
    plan = P.pick_neurons_to_drop(scores, 0.5, "expert")

    masked, pruned = copy.deepcopy(base), copy.deepcopy(base)
    P.zero_dropped_neurons(discover(masked), plan)
    pruned_before = discover(pruned)
    _, _, _, new_width = P.structurally_remove_neurons(pruned_before, plan)
    pruned_before[0].intermediate_size_key.set(pruned.config, new_width)  # as prune() does

    x = torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(7))
    masked_handles, pruned_handles = discover(masked), discover(pruned)
    expected_width = (HIDDEN if family == "gpt_oss" else INTER) // 2
    for mh, ph in zip(masked_handles, pruned_handles):
        assert ph.intermediate_size == expected_width and mh.fused == ph.fused
        for got, want in zip(_expert_outputs(ph, x), _expert_outputs(mh, x)):
            torch.testing.assert_close(got, want)


# ------------------------------------------------------------ calibration

class _FakeTokenizer:
    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [ord(c) % 97 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return _FakeDataset(rows)


def _assert_same_batches(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


CHAT_ROWS = [
    {"prompt": [{"role": "user", "content": "what is two plus two"}], "completion": "four"},
    {"prompt": "plain prompt text", "completion": {"role": "assistant", "content": "a reply"}},
    {"prompt": "short", "completion": ""},
    {"prompt": [{"role": "user", "content": "x" * 40}], "completion": "done"},
]


@pytest.mark.parametrize("shuffle_seed", [None, 3])
def test_hf_loader_matches_both_script_variants(monkeypatch, shuffle_seed):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: _FakeDataset(CHAT_ROWS)
    fake.Features = fake.Value = object
    monkeypatch.setitem(sys.modules, "datasets", fake)
    qwen15 = importlib.import_module("less_is_moe.pruning.expert_drop_qwen15_moe")
    qwen3 = importlib.import_module("less_is_moe.pruning.expert_drop_qwen3")
    tok = _FakeTokenizer()
    call = ("ds", None, "train", 3, 24, "prompt,completion")

    old15 = qwen15.load_calib_data_hf(tok, *call, shuffle_seed=shuffle_seed)
    old3 = qwen3.load_calib_data_hf(tok, *call, shuffle_seed=shuffle_seed)
    _assert_same_batches(calibration.load_calib_data_hf(tok, *call, shuffle_seed=shuffle_seed, unwrap_message_content=False), old15)
    _assert_same_batches(calibration.load_calib_data_hf(tok, *call, shuffle_seed=shuffle_seed, unwrap_message_content=True), old3)
    assert any(not torch.equal(a, b) for a, b in zip(old15, old3)), "rows must exercise the unwrap difference"


def test_file_loader_matches_script(tmp_path):
    qwen15 = importlib.import_module("less_is_moe.pruning.expert_drop_qwen15_moe")
    rows = [{"text": "alpha beta"}, {"prompt": "p", "completion": "c"}, {"instruction": "i", "output": "o"}, {"other": 1}]
    json_path, jsonl_path = tmp_path / "calib.json", tmp_path / "calib.jsonl"
    json_path.write_text(json.dumps(rows))
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows))
    tok = _FakeTokenizer()
    for path in (json_path, jsonl_path):
        _assert_same_batches(calibration.load_calib_data(tok, str(path), 5, 6), qwen15.load_calib_data(tok, str(path), 5, 6))


def test_preset_loader_matches_script(monkeypatch):
    qwen15 = importlib.import_module("less_is_moe.pruning.expert_drop_qwen15_moe")
    rows = [{"question": f"q{i}", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A", "explanation": "e" * i} for i in range(6)]
    rows.append({"question": "", "A": "skipped"})
    iterate = lambda dataset_name, split: iter([(None, r) for r in rows])  # noqa: E731
    monkeypatch.setattr(qwen15, "_iter_preset_rows", iterate)
    monkeypatch.setattr(calibration, "_iter_preset_rows", iterate)
    tok = _FakeTokenizer()
    for seed in (None, 5):
        _assert_same_batches(
            calibration.load_calib_data_preset(tok, "ceval", 4, 30, shuffle_seed=seed),
            qwen15.load_calib_data_preset(tok, "ceval", 4, 30, shuffle_seed=seed),
        )


# -------------------------------------------------------------------- CLI

def _save_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"t{i}": i for i in range(VOCAB - 1)}
    vocab["[UNK]"] = VOCAB - 1
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(path)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_cli_mask_then_structural(family, tmp_path):
    base, _, _ = _build(family)
    base.save_pretrained(tmp_path / "base")
    # The key is resolved on the model the CLI loads: AutoModelForCausalLM on a Qwen3.5 wrapper config
    # exposes the text config as model.config, exactly as the per-family script saw it.
    key = str(discover(transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "base"))[0].intermediate_size_key)
    _save_tokenizer(tmp_path / "base")
    calib = tmp_path / "calib.jsonl"
    calib.write_text("\n".join(json.dumps({"text": " ".join(f"t{(i * 7 + j) % 60}" for j in range(12))}) for i in range(4)))
    common = ["--n_samples", "3", "--seq_len", "16", "--dtype", "fp32", "--calib_data", str(calib)]
    width = HIDDEN if family == "gpt_oss" else INTER

    # One-shot structural prune.
    out = tmp_path / "structural"
    assert P.main(["--model_name_or_path", str(tmp_path / "base"), "--output_dir", str(out), "--mode", "structural", "--drop_ratio", "0.5", *common]) == 0
    summary = json.loads((out / P.STRUCTURAL_SUMMARY_FILE).read_text())
    assert summary["new_d_ffn"] == width // 2 and summary["stock_load_verified"] is True
    assert summary["intermediate_size_key"] == key
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert {h.intermediate_size for h in discover(reloaded)} == {width // 2}

    # Mask, then compact the masked checkpoint without calibration data.
    masked = tmp_path / "masked"
    assert P.main(["--model_name_or_path", str(tmp_path / "base"), "--output_dir", str(masked), "--mode", "mask", "--drop_ratio", "0.5", *common]) == 0
    mask_summary = json.loads((masked / P.MASK_SUMMARY_FILE).read_text())
    assert mask_summary["method"] == "neuron_drop_pure_gradient_expert" and mask_summary["total_dropped"] == mask_summary["total_neurons"] // 2
    compact = tmp_path / "compact"
    assert P.main(["--model_name_or_path", str(masked), "--output_dir", str(compact), "--mode", "structural", "--from_zeroed_model", "--dtype", "fp32"]) == 0
    compact_summary = json.loads((compact / P.STRUCTURAL_SUMMARY_FILE).read_text())
    assert compact_summary["method"] == "neuron_structure_drop_from_zeroed_model"
    assert compact_summary["new_d_ffn"] == width // 2 and compact_summary["stock_load_verified"] is True
