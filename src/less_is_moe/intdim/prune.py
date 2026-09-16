"""Model-agnostic IntDim pruning built on :mod:`less_is_moe.intdim.discover`.

Replaces the per-family ``neuron_drop_*`` (zero-mask) and
``neuron_structure_drop_*`` (structural) scripts with one implementation that
works on any model :func:`discover` understands.

The importance criterion, selection scopes, and outputs are unchanged:

* **Score.** For intermediate unit ``j`` of every routed expert, the mean absolute
  gradient of the language-modeling loss over the ``gate`` row, ``up`` row, and
  ``down`` column of that unit, averaged over calibration samples. This is not
  squared-gradient Fisher information. Bias gradients are not used.
* **Select.** ``expert`` (IntDim-E) drops the same number of lowest-scoring units
  in every expert; ``layer`` (IntDim-L) pools units across the experts of a layer;
  ``global`` (IntDim-G) pools units across all layers.
* **Apply.** ``mask`` keeps tensor shapes; ``structural`` removes equal counts
  with stock loaders; ``ragged`` exports unequal Qwen3-MoE widths for our plugin.

The arithmetic follows the per-family scripts operation for operation, including
where each reduction runs, so scores and outputs match them exactly.

Usage::

    python -m less_is_moe.intdim.prune --model_name_or_path MODEL --output_dir OUT \\
        --mode structural --drop_ratio 0.5 --calib_data calib.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable

import torch
from torch import nn

from .discover import MoeLayerHandle, describe, discover

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

Scores = dict[Any, dict[int, torch.Tensor]]
DropPlan = dict[Any, dict[int, list[int]]]

MASK_SUMMARY_FILE = "neuron_drop_summary.json"
STRUCTURAL_SUMMARY_FILE = "neuron_structure_drop_summary.json"
PRUNE_MODES = ("expert", "layer", "global")


def _layer_key(handle: MoeLayerHandle) -> Any:
    return handle.layer_index if handle.layer_index is not None else handle.name


# ---------------------------------------------------------------------- score

def collect_neuron_gradient_scores(model: nn.Module, handles: list[MoeLayerHandle], calib_batches: list[torch.Tensor]) -> Scores:
    """Per-unit mean absolute gradient for every routed expert.

    Returns ``{layer: {expert: Tensor(I)}}`` in float32 on the CPU.
    """
    if not calib_batches:
        raise ValueError("Calibration batches must not be empty")
    device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad_(False)

    scores: Scores = {}
    for h in handles:
        if h.kind == "fused":
            h.gate_up.requires_grad_(True)
            h.down.requires_grad_(True)
        else:
            for e in range(h.num_experts):
                for linear in h.expert_linears(e):
                    linear.weight.requires_grad_(True)
        scores[_layer_key(h)] = {e: torch.zeros(h.intermediate_size, dtype=torch.float32) for e in range(h.num_experts)}

    model.zero_grad(set_to_none=True)
    try:
        model.train()
        for batch in tqdm(calib_batches, desc="Collecting neuron gradient scores"):
            input_ids = batch.to(device)
            outputs = model(input_ids, labels=input_ids)
            outputs.loss.backward()
            for h in handles:
                layer = scores[_layer_key(h)]
                if h.kind == "fused":
                    _accumulate_fused(h, layer)
                else:
                    _accumulate_modulelist(h, layer)
            model.zero_grad(set_to_none=True)

        n = len(calib_batches)
        for layer in scores.values():
            for e in layer:
                layer[e] /= n

        return scores
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)


def _accumulate_fused(h: MoeLayerHandle, layer: dict[int, torch.Tensor]) -> None:
    """Batched reduction over all experts on the parameter's device, as in the Qwen3.5 script."""
    gu_grad, dp_grad = h.gate_up.grad, h.down.grad
    if gu_grad is None or dp_grad is None:
        return
    inter, hidden, layout = h.intermediate_size, h.hidden_size, h.fused
    if layout.gate_up_unit_axis == 1:  # (E, 2I, H)
        gate_rows = gu_grad[:, :inter, :] if layout.pairing == "concat" else gu_grad[:, 0::2, :]
        up_rows = gu_grad[:, inter:2 * inter, :] if layout.pairing == "concat" else gu_grad[:, 1::2, :]
        gate_imp = gate_rows.float().abs().sum(dim=2)
        up_imp = up_rows.float().abs().sum(dim=2)
    else:  # (E, H, 2I)
        gate_cols = gu_grad[:, :, :inter] if layout.pairing == "concat" else gu_grad[:, :, 0::2]
        up_cols = gu_grad[:, :, inter:2 * inter] if layout.pairing == "concat" else gu_grad[:, :, 1::2]
        gate_imp = gate_cols.float().abs().sum(dim=1)
        up_imp = up_cols.float().abs().sum(dim=1)
    down_imp = dp_grad.float().abs().sum(dim=1 if layout.down_unit_axis == 2 else 2)
    per_neuron = ((gate_imp + up_imp + down_imp) / (3 * hidden)).cpu()
    for e in range(h.num_experts):
        layer[e] += per_neuron[e]


def _accumulate_modulelist(h: MoeLayerHandle, layer: dict[int, torch.Tensor]) -> None:
    """Per-expert reduction, as in the Qwen1.5-MoE, Qwen3-MoE, and OLMoE scripts."""
    for e in range(h.num_experts):
        gate_w, up_w, down_w = (linear.weight for linear in h.expert_linears(e))
        importance = torch.zeros(gate_w.shape[0], dtype=torch.float32)
        n_el_per_neuron = 0
        if gate_w.grad is not None:
            importance += gate_w.grad.float().abs().sum(dim=1).cpu()
            n_el_per_neuron += gate_w.shape[1]
        if up_w.grad is not None:
            importance += up_w.grad.float().abs().sum(dim=1).cpu()
            n_el_per_neuron += up_w.shape[1]
        if down_w.grad is not None:
            importance += down_w.grad.float().abs().sum(dim=0).cpu()
            n_el_per_neuron += down_w.shape[0]
        if n_el_per_neuron > 0:
            importance /= n_el_per_neuron
        layer[e] += importance


# --------------------------------------------------------------------- select

def decide_neurons_to_drop(scores: Scores, drop_ratio: float) -> DropPlan:
    """IntDim-E: drop the lowest ``drop_ratio`` of units in every expert independently."""
    drop_per_layer: DropPlan = {}
    for layer_idx, expert_scores in scores.items():
        drop_per_layer[layer_idx] = {}
        for eid, neuron_scores in expert_scores.items():
            d_ffn = len(neuron_scores)
            n_drop = int(d_ffn * drop_ratio)
            if n_drop <= 0:
                drop_per_layer[layer_idx][eid] = []
            elif n_drop >= d_ffn:
                drop_per_layer[layer_idx][eid] = list(range(d_ffn))
            else:
                _, worst_ids = torch.topk(neuron_scores, n_drop, largest=False)
                drop_per_layer[layer_idx][eid] = sorted(worst_ids.tolist())
    return drop_per_layer


def decide_neurons_to_drop_layerwise(scores: Scores, drop_ratio: float) -> DropPlan:
    """IntDim-L: pool the units of all experts in a layer and drop the lowest ``drop_ratio``."""
    drop_per_layer: DropPlan = {}
    for layer_idx, expert_scores in scores.items():
        flat = [(float(s), eid, j) for eid, neuron_scores in expert_scores.items() for j, s in enumerate(neuron_scores.tolist())]
        n_total = len(flat)
        n_drop = int(n_total * drop_ratio)
        drops: dict[int, list[int]] = {eid: [] for eid in expert_scores}
        if n_drop > 0:
            n_drop = min(n_drop, n_total)
            flat.sort(key=lambda t: t[0])
            for _, eid, j in flat[:n_drop]:
                drops[eid].append(j)
            for eid in drops:
                drops[eid].sort()
        drop_per_layer[layer_idx] = drops
    return drop_per_layer


def decide_neurons_to_drop_global(scores: Scores, drop_ratio: float) -> DropPlan:
    """IntDim-G: pool the units of every expert in every layer and drop the lowest ``drop_ratio``."""
    flat = [
        (float(s), layer_idx, eid, j)
        for layer_idx, expert_scores in scores.items()
        for eid, neuron_scores in expert_scores.items()
        for j, s in enumerate(neuron_scores.tolist())
    ]
    n_total = len(flat)
    n_drop = int(n_total * drop_ratio)
    drop_per_layer: DropPlan = {layer_idx: {eid: [] for eid in expert_scores} for layer_idx, expert_scores in scores.items()}
    if n_drop > 0:
        n_drop = min(n_drop, n_total)
        flat.sort(key=lambda t: t[0])
        for _, layer_idx, eid, j in flat[:n_drop]:
            drop_per_layer[layer_idx][eid].append(j)
        for layer_idx in drop_per_layer:
            for eid in drop_per_layer[layer_idx]:
                drop_per_layer[layer_idx][eid].sort()
    return drop_per_layer


_SELECTORS = {
    "expert": decide_neurons_to_drop,
    "layer": decide_neurons_to_drop_layerwise,
    "global": decide_neurons_to_drop_global,
}


def pick_neurons_to_drop(scores: Scores, drop_ratio: float, mode: str) -> DropPlan:
    try:
        selector = _SELECTORS[mode]
    except KeyError:
        raise ValueError(f"Unknown prune_mode {mode!r}; expected one of {sorted(_SELECTORS)}.") from None
    return selector(scores, drop_ratio)


def summarize_drop_distribution(drop_per_layer: DropPlan) -> list[dict[str, Any]]:
    """Per-layer min/max/mean drop count across experts."""
    rows = []
    for layer_idx in sorted(drop_per_layer, key=str):
        counts = [len(ids) for ids in drop_per_layer[layer_idx].values()]
        if not counts:
            continue
        rows.append({
            "layer": layer_idx,
            "n_experts": len(counts),
            "min_drop": min(counts),
            "max_drop": max(counts),
            "mean_drop": sum(counts) / len(counts),
            "zero_experts": sum(1 for c in counts if c == 0),
        })
    return rows


# ---------------------------------------------------------------------- apply

@torch.no_grad()
def zero_dropped_neurons(handles: list[MoeLayerHandle], drop_per_layer: DropPlan) -> tuple[list[dict], int, int]:
    """Zero the gate/up rows and down columns of dropped units; shapes are preserved.

    A ``gate_up`` bias entry is zeroed with its row, otherwise the unit would
    stay active through the bias.
    """
    summary, total_dropped, total_neurons = [], 0, 0
    for h in handles:
        key = _layer_key(h)
        layer_info = {"layer": key, "experts": []}
        for e in range(h.num_experts):
            drop_ids = drop_per_layer.get(key, {}).get(e, [])
            if drop_ids:
                if h.kind == "modulelist":
                    gate, up, down = h.expert_linears(e)
                    idx = torch.tensor(drop_ids, dtype=torch.long, device=gate.weight.device)
                    gate.weight.data[idx, :] = 0
                    up.weight.data[idx, :] = 0
                    down.weight.data[:, idx] = 0
                else:
                    gu, dp = h.gate_up, h.down
                    idx = torch.tensor(drop_ids, dtype=torch.long, device=gu.device)
                    rows = h.gate_up_indices(idx).to(gu.device)
                    if h.fused.gate_up_unit_axis == 1:
                        gu.data[e, rows, :] = 0
                    else:
                        gu.data[e, :, rows] = 0
                    if (bias := h.gate_up_bias) is not None:
                        bias.data[e, rows.to(bias.device)] = 0
                    if h.fused.down_unit_axis == 2:
                        dp.data[e, :, idx.to(dp.device)] = 0
                    else:
                        dp.data[e, idx.to(dp.device), :] = 0
            layer_info["experts"].append({
                "expert_id": e,
                "total_neurons": h.intermediate_size,
                "dropped_count": len(drop_ids),
                "dropped_neurons": drop_ids,
            })
            total_dropped += len(drop_ids)
            total_neurons += h.intermediate_size
        summary.append(layer_info)
    return summary, total_dropped, total_neurons


@torch.no_grad()
def find_zeroed_neurons(handles: list[MoeLayerHandle]) -> DropPlan:
    """Units whose gate row is entirely zero, per expert (for ``--from_zeroed_model``)."""
    drop_per_layer: DropPlan = {}
    for h in handles:
        key = _layer_key(h)
        drop_per_layer[key] = {}
        for e in range(h.num_experts):
            gate, _, _ = h.expert_weights(e)
            zero_rows = gate.float().abs().sum(dim=1) == 0
            drop_per_layer[key][e] = sorted(torch.nonzero(zero_rows, as_tuple=False).flatten().tolist())
    return drop_per_layer


@torch.no_grad()
def structurally_remove_neurons(handles: list[MoeLayerHandle], drop_per_layer: DropPlan) -> tuple[list[dict], int, int, int | None]:
    """Remove dropped units. Every expert in every layer must keep the same count."""
    summary, total_dropped, total_neurons = [], 0, 0
    new_d_ffn: int | None = None
    for h in handles:
        key = _layer_key(h)
        orig = h.intermediate_size
        layer_info = {"layer": key, "experts": []}
        keep_per_expert = []
        for e in range(h.num_experts):
            drop_ids = drop_per_layer.get(key, {}).get(e, [])
            drop_set = set(drop_ids)
            keep = [j for j in range(orig) if j not in drop_set]
            if new_d_ffn is None:
                new_d_ffn = len(keep)
            elif len(keep) != new_d_ffn:
                raise ValueError(
                    "Structural removal requires a uniform surviving d_ffn across experts and layers, "
                    f"but got {new_d_ffn} and {len(keep)} (layer {key}, expert {e})."
                )
            keep_per_expert.append(keep)
            layer_info["experts"].append({
                "expert_id": e,
                "orig_neurons": orig,
                "kept_neurons": len(keep),
                "dropped_count": len(drop_ids),
                "dropped_neurons": drop_ids,
            })
            total_dropped += len(drop_ids)
            total_neurons += orig
        h.apply_units(torch.tensor(keep_per_expert, dtype=torch.long))
        summary.append(layer_info)
    return summary, total_dropped, total_neurons, new_d_ffn


# ------------------------------------------------------------------------ CLI

def load_model(path: str, dtype: torch.dtype) -> nn.Module:
    """Load like the per-family scripts: causal LM first, then the multimodal and base auto classes."""
    import transformers
    from transformers import AutoModelForCausalLM

    dtype_kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs = {dtype_kwarg: dtype, "device_map": "auto", "trust_remote_code": True}
    errors = []
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except (ValueError, KeyError, AttributeError) as exc:
        errors.append(f"AutoModelForCausalLM: {type(exc).__name__}: {exc}")
    for name in ("AutoModelForImageTextToText", "AutoModel"):
        try:
            return getattr(transformers, name).from_pretrained(path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - try the next auto class
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not load {path}. Tried:\n  " + "\n  ".join(errors))


def load_calibration(args: argparse.Namespace, tokenizer) -> list[torch.Tensor]:
    from less_is_moe import calibration

    if args.calib_preset:
        return calibration.load_calib_data_preset(
            tokenizer, args.calib_preset, args.n_samples, args.seq_len,
            split=args.calib_preset_split, shuffle_seed=args.shuffle_seed,
        )
    if args.dataset_name:
        return calibration.load_calib_data_hf(
            tokenizer, args.dataset_name, args.dataset_config, args.dataset_split,
            args.n_samples, args.seq_len, args.text_column,
            shuffle_seed=args.shuffle_seed, unwrap_message_content=args.unwrap_message_content,
        )
    if args.calib_data:
        return calibration.load_calib_data(tokenizer, args.calib_data, args.n_samples, args.seq_len)
    raise ValueError("Provide one of --calib_preset, --dataset_name, or --calib_data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("mask", "structural", "ragged"), required=True,
                        help="mask: zero units; structural: uniform stock-loader checkpoint; ragged: compact Qwen3-MoE plugin checkpoint")
    parser.add_argument("--prune_mode", choices=PRUNE_MODES, default="expert",
                        help="Selection scope: expert (IntDim-E), layer (IntDim-L), global (IntDim-G). Structural mode requires expert.")
    parser.add_argument("--drop_ratio", type=float, default=None, help="Fraction of units to drop, in (0, 1)")
    parser.add_argument("--from_zeroed_model", action="store_true",
                        help="Structural mode only: remove units that are already zero instead of scoring")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--calib_data", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text_column", default="prompt")
    parser.add_argument("--calib_preset", default=None, help="ceval, math, or cmmlu")
    parser.add_argument("--calib_preset_split", default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--unwrap_message_content", action=argparse.BooleanOptionalAction, default=True,
                        help="Use only the content of chat-message dict fields in HF datasets (Qwen3/Qwen3.5 script behavior). "
                             "The Qwen1.5-MoE and OLMoE scripts correspond to --no-unwrap_message_content.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--skip_verify", action="store_true", help="Skip checkpoint reload verification (stock for structural, custom GPU loader for ragged)")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.from_zeroed_model and args.mode != "structural":
        raise ValueError("--from_zeroed_model requires --mode structural")
    if args.mode == "structural" and args.prune_mode != "expert":
        raise ValueError("--mode structural requires --prune_mode expert: other scopes give non-uniform widths")
    if not args.from_zeroed_model and (args.drop_ratio is None or not 0.0 < args.drop_ratio < 1.0):
        raise ValueError(f"--drop_ratio must be in (0, 1), got {args.drop_ratio}")


def prune(model: nn.Module, args: argparse.Namespace, calib_batches: Iterable[torch.Tensor] | None) -> dict[str, Any]:
    """Run the full pipeline on a loaded model and return the summary (weights are modified in place)."""
    validate_args(args)
    if args.mode == "ragged":
        from .ragged import LAYOUT, validate_metadata
        import copy
        # Fail before expensive calibration for unsupported model families.
        config = copy.deepcopy(model.config)
        if config.model_type != "qwen3_moe":
            raise ValueError("Ragged v1 supports Qwen3-MoE only")
        config.less_is_moe = dict(format_version=1, weight_layout=LAYOUT,
                                 expert_intermediate_sizes={str(i): [config.moe_intermediate_size] * config.num_experts
                                                            for i in range(config.num_hidden_layers)})
        validate_metadata(config)
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("Ragged pruning requires the entire model on GPU")
    batches = None if args.from_zeroed_model else list(calib_batches or [])
    if batches is not None and not batches:
        raise ValueError("Calibration batches must not be empty")
    handles = discover(model)
    print(describe(handles))
    d_ffn = handles[0].intermediate_size
    common = {
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "dataset_name": args.dataset_name,
        "calib_preset": args.calib_preset,
        "calib_data": args.calib_data,
        "text_column": args.text_column,
    }

    if args.from_zeroed_model:
        drop_per_layer = find_zeroed_neurons(handles)
        counts = {len(ids) for layer in drop_per_layer.values() for ids in layer.values()}
        if len(counts) != 1:
            raise ValueError(f"--from_zeroed_model requires a uniform per-expert drop count; observed {sorted(counts)[:5]}")
        n_drop = next(iter(counts))
        args.drop_ratio = n_drop / d_ffn
    else:
        scores = collect_neuron_gradient_scores(model, handles, batches)
        drop_per_layer = pick_neurons_to_drop(scores, args.drop_ratio, args.prune_mode)
        n_drop = int(d_ffn * args.drop_ratio)

    if args.mode == "mask":
        per_layer, total_dropped, total_neurons = zero_dropped_neurons(handles, drop_per_layer)
        return {
            "method": f"neuron_drop_pure_gradient_{args.prune_mode}",
            "prune_mode": args.prune_mode,
            "drop_ratio": args.drop_ratio,
            "drop_pct": int(args.drop_ratio * 100),
            "d_ffn": d_ffn,
            "neurons_dropped_per_expert_target": n_drop,
            "total_neurons": total_neurons,
            "total_dropped": total_dropped,
            **common,
            "unwrap_message_content": args.unwrap_message_content,
            "drop_distribution": summarize_drop_distribution(drop_per_layer),
            "per_layer": per_layer,
        }

    if args.mode == "ragged":
        from .ragged import compact_model
        summary = compact_model(model, handles, drop_per_layer)
        return {**summary, **common, "method": "ragged_neuron_structure_drop",
                "prune_mode": args.prune_mode, "drop_ratio": args.drop_ratio,
                "source_model": args.model_name_or_path, "drop_plan": drop_per_layer}

    per_layer, total_dropped, total_neurons, new_d_ffn = structurally_remove_neurons(handles, drop_per_layer)
    handles[0].intermediate_size_key.set(model.config, new_d_ffn)
    return {
        "method": "neuron_structure_drop_from_zeroed_model" if args.from_zeroed_model else "neuron_structure_drop_pure_gradient",
        "source_model": args.model_name_or_path,
        "drop_ratio": args.drop_ratio,
        "drop_pct": int(args.drop_ratio * 100),
        "orig_d_ffn": d_ffn,
        "new_d_ffn": new_d_ffn,
        "neurons_removed_per_expert": n_drop,
        "total_neurons": total_neurons,
        "total_removed": total_dropped,
        **common,
        "unwrap_message_content": args.unwrap_message_content,
        "intermediate_size_key": str(handles[0].intermediate_size_key),
        "per_layer": per_layer,
    }


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    args = build_parser().parse_args(argv)
    validate_args(args)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_model(args.model_name_or_path, dtype)
    model.eval()
    batches = None if args.from_zeroed_model else load_calibration(args, tokenizer)
    if batches is not None:
        print(f"Loaded {len(batches)} calibration samples (seq_len={args.seq_len})")

    summary = prune(model, args, batches)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "ragged":
        from .ragged import save_checkpoint
        save_checkpoint(model, args.output_dir, tokenizer)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.mode == "structural" and not args.skip_verify:
        from .verify import verify_checkpoint

        report = verify_checkpoint(args.output_dir)
        print(report)
        summary["stock_load_verified"] = report.ok
        if not report.ok:
            raise RuntimeError(f"Pruned checkpoint does not load with stock classes:\n{report}")

    if args.mode == "ragged" and not args.skip_verify:
        import gc
        from .ragged import load_checkpoint
        # Release the pruning model before verifying a full-size GPU reload.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        restored = load_checkpoint(args.output_dir, dtype=dtype)
        with torch.inference_mode():
            token = restored.config.bos_token_id or 0
            output = restored(torch.tensor([[token, token]], device="cuda"), use_cache=False).logits
            if not torch.isfinite(output).all():
                raise RuntimeError("Ragged checkpoint reload produced non-finite logits")
        summary["ragged_load_verified"] = True

    summary_file = MASK_SUMMARY_FILE if args.mode == "mask" else STRUCTURAL_SUMMARY_FILE
    with open(os.path.join(args.output_dir, summary_file), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {os.path.join(args.output_dir, summary_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
