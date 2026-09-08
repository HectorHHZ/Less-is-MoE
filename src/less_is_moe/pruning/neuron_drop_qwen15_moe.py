"""
Neuron-level drop for Qwen1.5-MoE-A2.7B (and other Qwen2-MoE variants).

Strategy — runtime masking via weight zeroing (shapes preserved):
    For each routed expert in each MoE layer, we rank every FFN neuron by
    gradient importance and zero out the bottom `drop_ratio` fraction.

    For neuron index j the zeroed triplet is:
        gate_proj.weight[j, :] = 0      # shape (d_ffn, hidden_size)
        up_proj.weight[j, :]   = 0      # shape (d_ffn, hidden_size)
        down_proj.weight[:, j] = 0      # shape (hidden_size, d_ffn)

    Shared expert is NOT pruned — only routed experts are modified.

Importance metric (mirrors pure_gradient_pruning in expert_drop but at
neuron granularity):

    For neuron j of expert e in layer l,
        score_j = ( sum|∂L/∂gate_proj.weight[j,:]|
                  + sum|∂L/∂up_proj.weight[j,:]|
                  + sum|∂L/∂down_proj.weight[:,j]| )
                  / (gate_dim + up_dim + down_dim)

    averaged across calibration samples, where L is the standard LM
    cross-entropy loss (labels = input_ids).

Usage:
    scripts/prune/neuron_drop_qwen15_moe.sh \\
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \\
        --output_dir /path/to/pruned_model \\
        --drop_ratio 0.25 \\
        --n_samples 128 \\
        --seq_len 2048 \\
        --dataset_name RoxanneWsyw/gsm \\
        --dataset_split train \\
        --text_column prompt,completion \\
        --dtype bf16
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Reuse calibration loaders and MoE utilities from expert_drop_qwen15_moe.py
# ---------------------------------------------------------------------------

from .expert_drop_qwen15_moe import (
    get_moe_block,
    get_moe_layer_info,
    load_calib_data,
    load_calib_data_hf,
    load_calib_data_preset,
)


# ---------------------------------------------------------------------------
# Per-neuron gradient importance scoring
# ---------------------------------------------------------------------------

def collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices):
    """
    Compute per-neuron importance scores based on pure gradient magnitude
    across gate_proj, up_proj, down_proj for each routed expert.

    For neuron j in expert e of layer l:
        score_j = ( sum|∂L/∂gate_proj.W[j,:]|
                  + sum|∂L/∂up_proj.W[j,:]|
                  + sum|∂L/∂down_proj.W[:,j]| )
                  / (hidden_size + hidden_size + hidden_size)

    Returns
    -------
    dict  {layer_idx: {expert_idx: Tensor of shape (d_ffn,)}}
    """
    device = next(model.parameters()).device

    # Enable gradients only on routed-expert FFN projection weights
    for param in model.parameters():
        param.requires_grad_(False)

    expert_projs = {}  # {layer_idx: [(eid, gate_w, up_w, down_w), ...]}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is None:
            continue
        expert_projs[layer_idx] = []
        for eid, expert in enumerate(moe_block.experts):
            expert.gate_proj.weight.requires_grad_(True)
            expert.up_proj.weight.requires_grad_(True)
            expert.down_proj.weight.requires_grad_(True)
            expert_projs[layer_idx].append((
                eid,
                expert.gate_proj.weight,   # (d_ffn, hidden_size)
                expert.up_proj.weight,      # (d_ffn, hidden_size)
                expert.down_proj.weight,    # (hidden_size, d_ffn)
            ))

    # Initialise per-neuron accumulators
    scores = {}
    for layer_idx, experts in expert_projs.items():
        scores[layer_idx] = {}
        for eid, gate_w, up_w, _down_w in experts:
            d_ffn = gate_w.shape[0]
            scores[layer_idx][eid] = torch.zeros(d_ffn, dtype=torch.float32)

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting neuron gradient scores"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for layer_idx, experts in expert_projs.items():
            for eid, gate_w, up_w, down_w in experts:
                importance = torch.zeros(gate_w.shape[0], dtype=torch.float32)
                n_el_per_neuron = 0

                if gate_w.grad is not None:
                    # gate_w.grad: (d_ffn, hidden_size) -> sum over hidden_size
                    importance += gate_w.grad.float().abs().sum(dim=1).cpu()
                    n_el_per_neuron += gate_w.shape[1]
                if up_w.grad is not None:
                    # up_w.grad: (d_ffn, hidden_size) -> sum over hidden_size
                    importance += up_w.grad.float().abs().sum(dim=1).cpu()
                    n_el_per_neuron += up_w.shape[1]
                if down_w.grad is not None:
                    # down_w.grad: (hidden_size, d_ffn) -> sum over hidden_size
                    importance += down_w.grad.float().abs().sum(dim=0).cpu()
                    n_el_per_neuron += down_w.shape[0]

                if n_el_per_neuron > 0:
                    importance /= n_el_per_neuron

                scores[layer_idx][eid] += importance

        model.zero_grad(set_to_none=True)

    # Average over calibration samples
    n = max(len(calib_batches), 1)
    for layer_idx in scores:
        for eid in scores[layer_idx]:
            scores[layer_idx][eid] /= n

    # Restore
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return scores


# ---------------------------------------------------------------------------
# Neuron-drop decision
# ---------------------------------------------------------------------------

def decide_neurons_to_drop(scores, drop_ratio):
    """
    Expert-wise selection (default).

    For each (layer, expert) independently select the bottom ``drop_ratio``
    fraction of neurons by importance. Every expert drops the same count,
    so the resulting d_ffn is uniform across experts in a layer — the only
    selection mode that is compatible with the structural-shrink variant.

    Returns
    -------
    dict  {layer_idx: {expert_idx: sorted list of neuron indices to drop}}
    """
    drop_per_layer = {}
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


def decide_neurons_to_drop_layerwise(scores, drop_ratio):
    """
    Layer-wise selection.

    Pool every neuron in a given MoE layer across all routed experts, then
    drop the bottom ``drop_ratio`` fraction globally within that layer.
    Per-layer total drop count equals expert-wise's total in that layer
    (same budget), but allocation across experts is allowed to be uneven:
    an expert whose neurons all rank above the layer's threshold keeps
    everything, while an expert with many low-scoring neurons may lose more
    than ``drop_ratio × d_ffn``.

    Only meaningful under the zero-mask variant — the resulting d_ffn is
    no longer uniform, so the structural-shrink variant cannot use it.

    Returns
    -------
    dict  {layer_idx: {expert_idx: sorted list of neuron indices to drop}}
    """
    drop_per_layer = {}
    for layer_idx, expert_scores in scores.items():
        # (score, eid, local_neuron_idx) triples for the whole layer
        flat = [
            (float(s), eid, j)
            for eid, neuron_scores in expert_scores.items()
            for j, s in enumerate(neuron_scores.tolist())
        ]
        n_total = len(flat)
        n_drop = int(n_total * drop_ratio)

        drops = {eid: [] for eid in expert_scores}
        if n_drop > 0:
            n_drop = min(n_drop, n_total)
            flat.sort(key=lambda t: t[0])
            for _, eid, j in flat[:n_drop]:
                drops[eid].append(j)
            for eid in drops:
                drops[eid].sort()
        drop_per_layer[layer_idx] = drops
    return drop_per_layer


def decide_neurons_to_drop_global(scores, drop_ratio):
    """
    Global selection.

    Pool every neuron across **every MoE layer and every routed expert**,
    then drop the bottom ``drop_ratio`` fraction globally. Both per-layer
    and per-expert drop counts are allowed to vary: a layer whose neurons
    are uniformly important keeps more than (1-ratio)·d_ffn per expert,
    while a layer with many low-scoring neurons may lose more.

    Only meaningful under the zero-mask variant — the resulting d_ffn
    varies per expert and per layer, so the structural-shrink variant
    cannot use it.

    Returns
    -------
    dict  {layer_idx: {expert_idx: sorted list of neuron indices to drop}}
    """
    flat = [
        (float(s), layer_idx, eid, j)
        for layer_idx, expert_scores in scores.items()
        for eid, neuron_scores in expert_scores.items()
        for j, s in enumerate(neuron_scores.tolist())
    ]
    n_total = len(flat)
    n_drop = int(n_total * drop_ratio)

    drop_per_layer = {
        layer_idx: {eid: [] for eid in expert_scores}
        for layer_idx, expert_scores in scores.items()
    }
    if n_drop > 0:
        n_drop = min(n_drop, n_total)
        flat.sort(key=lambda t: t[0])
        for _, layer_idx, eid, j in flat[:n_drop]:
            drop_per_layer[layer_idx][eid].append(j)
        for layer_idx in drop_per_layer:
            for eid in drop_per_layer[layer_idx]:
                drop_per_layer[layer_idx][eid].sort()
    return drop_per_layer


_PRUNE_MODE_DISPATCH = {
    "expert": decide_neurons_to_drop,
    "layer": decide_neurons_to_drop_layerwise,
    "global": decide_neurons_to_drop_global,
}


def pick_neurons_to_drop(scores, drop_ratio, mode):
    """Dispatch to the selection function for ``mode``."""
    try:
        fn = _PRUNE_MODE_DISPATCH[mode]
    except KeyError:
        raise ValueError(
            f"Unknown prune_mode {mode!r}; expected one of "
            f"{sorted(_PRUNE_MODE_DISPATCH)}."
        )
    return fn(scores, drop_ratio)


def _summarize_drop_distribution(drop_per_layer):
    """Diagnostic: per-layer min/max/mean drop count across experts."""
    rows = []
    for layer_idx in sorted(drop_per_layer):
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


# ---------------------------------------------------------------------------
# In-place weight zeroing
# ---------------------------------------------------------------------------

@torch.no_grad()
def zero_dropped_neurons(model, drop_per_layer, moe_layer_indices):
    """
    Zero out the gate_proj / up_proj / down_proj slices for dropped neurons.
    Shapes are preserved so the checkpoint loads unchanged by vLLM / HF.
    Shared expert is NOT modified.
    """
    layers = model.model.layers
    summary = []
    total_dropped = 0
    total_neurons = 0

    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        layer_info = {"layer": layer_idx, "experts": []}

        for eid, expert in enumerate(moe_block.experts):
            d_ffn = expert.gate_proj.weight.shape[0]
            drop_ids = drop_per_layer.get(layer_idx, {}).get(eid, [])

            if drop_ids:
                idx = torch.tensor(drop_ids, dtype=torch.long,
                                   device=expert.gate_proj.weight.device)
                expert.gate_proj.weight.data[idx, :] = 0
                expert.up_proj.weight.data[idx, :] = 0
                expert.down_proj.weight.data[:, idx] = 0

            layer_info["experts"].append({
                "expert_id": eid,
                "total_neurons": d_ffn,
                "dropped_count": len(drop_ids),
                "dropped_neurons": drop_ids,
            })
            total_dropped += len(drop_ids)
            total_neurons += d_ffn

        n_experts = len(moe_block.experts)
        avg_drop = (sum(len(drop_per_layer.get(layer_idx, {}).get(e, []))
                        for e in range(n_experts))
                    / max(n_experts, 1))
        d_ffn = moe_block.experts[0].gate_proj.weight.shape[0]
        print(f"  Layer {layer_idx}: {n_experts} experts, "
              f"avg {avg_drop:.0f}/{d_ffn} neurons zeroed")

        summary.append(layer_info)

    return summary, total_dropped, total_neurons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neuron-level drop for Qwen1.5-MoE (runtime masking via weight zeroing)"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drop_ratio", type=float, required=True,
                        help="Fraction of neurons to drop per expert (e.g. 0.25 for 25%%)")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    # Data source (same args as expert_drop / attn_head_drop)
    parser.add_argument("--calib_data", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="prompt")
    parser.add_argument("--calib_preset", type=str, default=None,
                        choices=["ceval", "math", "cmmlu"])
    parser.add_argument("--calib_preset_split", type=str, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--prune_mode",
        type=str,
        default="expert",
        choices=["expert", "layer", "global"],
        help=(
            "Selection scope for ranking neurons. "
            "'expert' (default): each routed expert independently drops its "
            "own bottom drop_ratio, giving uniform d_ffn across experts. "
            "'layer': pool all neurons in a layer across experts, drop the "
            "bottom drop_ratio of the layer — per-expert drop counts vary. "
            "'global': pool every neuron in every MoE layer & expert, drop "
            "the bottom drop_ratio globally — per-layer and per-expert "
            "drop counts both vary."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.drop_ratio < 1.0):
        raise ValueError(f"--drop_ratio must be in (0, 1), got {args.drop_ratio}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # -------- calibration data --------
    if args.calib_preset:
        print(f"Loading calibration data from preset: {args.calib_preset} ...")
        calib_batches = load_calib_data_preset(
            tokenizer, args.calib_preset, args.n_samples, args.seq_len,
            split=args.calib_preset_split, shuffle_seed=args.shuffle_seed,
        )
    elif args.dataset_name:
        print(f"Loading calibration data from HF dataset: {args.dataset_name} ...")
        calib_batches = load_calib_data_hf(
            tokenizer, args.dataset_name, args.dataset_config, args.dataset_split,
            args.n_samples, args.seq_len, args.text_column,
            shuffle_seed=args.shuffle_seed,
        )
    else:
        if not args.calib_data:
            raise ValueError("Must provide one of --calib_preset / --dataset_name / --calib_data")
        print(f"Loading calibration data from {args.calib_data} ...")
        calib_batches = load_calib_data(tokenizer, args.calib_data, args.n_samples, args.seq_len)
    print(f"Loaded {len(calib_batches)} calibration samples (seq_len={args.seq_len})")

    # -------- MoE info --------
    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    # Quick sanity: show d_ffn for the first MoE layer
    first_moe = get_moe_block(model.model.layers[moe_layer_indices[0]])
    d_ffn = first_moe.experts[0].gate_proj.weight.shape[0]
    n_drop = int(d_ffn * args.drop_ratio)
    print(f"Expert FFN dimension (d_ffn): {d_ffn}")
    print(f"Neurons to drop per expert: {n_drop} ({args.drop_ratio*100:.0f}%)")

    # -------- score collection --------
    print(f"Collecting per-neuron gradient scores over {len(calib_batches)} samples ...")
    scores = collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices)

    # Print score stats for first few layers
    for lid in sorted(scores.keys())[:3]:
        all_s = torch.cat([scores[lid][eid] for eid in sorted(scores[lid].keys())])
        print(f"  Layer {lid} neuron scores: min={all_s.min():.6e}, "
              f"max={all_s.max():.6e}, mean={all_s.mean():.6e}")

    # -------- decide which neurons to drop --------
    scope_blurb = {
        "expert": "per-expert (uniform count)",
        "layer": "per-layer (pooled across experts in a layer)",
        "global": "global (pooled across all layers and experts)",
    }[args.prune_mode]
    print(f"Selecting bottom {args.drop_ratio*100:.0f}% neurons — scope: {scope_blurb}")
    drop_per_layer = pick_neurons_to_drop(scores, args.drop_ratio, args.prune_mode)

    if args.prune_mode != "expert":
        dist_rows = _summarize_drop_distribution(drop_per_layer)
        mins = [r["min_drop"] for r in dist_rows]
        maxs = [r["max_drop"] for r in dist_rows]
        zeros = sum(r["zero_experts"] for r in dist_rows)
        print(
            f"  Per-expert drop imbalance (layer-level min..max): "
            f"{min(mins)}..{max(maxs)}; experts left fully intact: {zeros}"
        )

    # -------- zero out dropped neurons --------
    print("Zeroing dropped-neuron weights in-place ...")
    drop_summary, total_dropped, total_neurons = zero_dropped_neurons(
        model, drop_per_layer, moe_layer_indices
    )
    print(f"Total neurons: {total_neurons} -> {total_neurons - total_dropped} active "
          f"({total_dropped} zeroed, {total_dropped/total_neurons*100:.1f}%)")

    # -------- save --------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving masked model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -------- summary JSON --------
    drop_pct = int(args.drop_ratio * 100)
    summary_path = os.path.join(args.output_dir, "neuron_drop_summary.json")
    summary_data = {
        "method": "neuron_drop_pure_gradient",
        "prune_mode": args.prune_mode,
        "drop_ratio": args.drop_ratio,
        "drop_pct": drop_pct,
        "d_ffn": d_ffn,
        # With non-expert modes the per-expert drop count is not uniform;
        # `neurons_dropped_per_expert` is the *nominal* per-expert budget
        # (drop_ratio × d_ffn). See `drop_distribution` for actual spread.
        "neurons_dropped_per_expert": n_drop,
        "total_neurons": total_neurons,
        "total_dropped": total_dropped,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "dataset_name": args.dataset_name,
        "calib_preset": args.calib_preset,
        "calib_data": args.calib_data,
        "text_column": args.text_column,
        "drop_distribution": _summarize_drop_distribution(drop_per_layer),
        "per_layer": drop_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print("Done!")


if __name__ == "__main__":
    main()
