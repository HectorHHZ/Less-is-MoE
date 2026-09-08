"""
Neuron-level drop for OLMoE-1B-7B.

Strategy — runtime masking via weight zeroing (shapes preserved):
    For each routed expert in each MoE layer, rank every FFN neuron by
    gradient importance and zero out the bottom `drop_ratio` fraction.

    For neuron index j the zeroed triplet is:
        gate_proj.weight[j, :] = 0      # shape (d_ffn, hidden_size)
        up_proj.weight[j, :]   = 0      # shape (d_ffn, hidden_size)
        down_proj.weight[:, j] = 0      # shape (hidden_size, d_ffn)

    OLMoE has no shared_expert; only routed experts exist and are modified.

Importance metric (mirrors pure_gradient_pruning at neuron granularity):

    For neuron j of expert e in layer l,
        score_j = ( sum|∂L/∂gate_proj.W[j,:]|
                  + sum|∂L/∂up_proj.W[j,:]|
                  + sum|∂L/∂down_proj.W[:,j]| )
                  / (gate_dim + up_dim + down_dim)

    averaged across calibration samples with L = LM cross-entropy loss.

Usage:
    scripts/prune/neuron_drop_olmoe.sh \\
        --model_name_or_path allenai/OLMoE-1B-7B \\
        --output_dir /path/to/pruned_model \\
        --drop_ratio 0.25 \\
        --n_samples 128 \\
        --seq_len 2048 \\
        --calib_data /path/to/c4_train_part_of_0.json \\
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
# Reuse calibration loaders & MoE utilities from expert_drop_olmoe.py
# ---------------------------------------------------------------------------

from .expert_drop_olmoe import (
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
    """Compute per-neuron pure-gradient importance for each routed expert.

    Returns
    -------
    dict  {layer_idx: {expert_idx: Tensor of shape (d_ffn,)}}
    """
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)

    expert_projs = {}
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

    scores = {}
    for layer_idx, experts in expert_projs.items():
        scores[layer_idx] = {}
        for eid, gate_w, _up_w, _down_w in experts:
            scores[layer_idx][eid] = torch.zeros(gate_w.shape[0], dtype=torch.float32)

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting neuron gradient scores"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for layer_idx, experts in expert_projs.items():
            for eid, gate_w, up_w, down_w in experts:
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

                scores[layer_idx][eid] += importance

        model.zero_grad(set_to_none=True)

    n = max(len(calib_batches), 1)
    for layer_idx in scores:
        for eid in scores[layer_idx]:
            scores[layer_idx][eid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return scores


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_neurons_to_drop(scores, drop_ratio):
    """Expert-wise selection (default).

    For each (layer, expert) independently drop the bottom ``drop_ratio``
    fraction. Every expert in a layer drops the same count, so the resulting
    d_ffn is uniform across experts — the only mode compatible with structural
    shrink.

    Returns dict  {layer_idx: {expert_idx: sorted list of neuron indices to drop}}.
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
    """Layer-wise selection.

    Pool every neuron in a given MoE layer across all routed experts, then
    drop the bottom ``drop_ratio`` fraction within that layer. Same total
    budget per layer as the expert-wise mode, but the per-expert split is
    free — an expert with uniformly-important neurons keeps everything,
    while a low-scoring expert may lose more than ``drop_ratio × d_ffn``.

    Only meaningful under the zero-mask variant: the resulting d_ffn is no
    longer uniform across experts, so structural shrink cannot use it.
    """
    drop_per_layer = {}
    for layer_idx, expert_scores in scores.items():
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
    """Global selection.

    Pool every neuron across **every MoE layer and every routed expert**, then
    drop the bottom ``drop_ratio`` fraction. Both per-layer and per-expert
    drop counts can vary. Only zero-mask compatible.
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
    "layer":  decide_neurons_to_drop_layerwise,
    "global": decide_neurons_to_drop_global,
}


def pick_neurons_to_drop(scores, drop_ratio, mode):
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
        description="Neuron-level drop for OLMoE (runtime masking via weight zeroing)"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drop_ratio", type=float, required=True,
                        help="Fraction of neurons to drop (e.g. 0.25)")
    parser.add_argument("--prune_mode", type=str, default="expert",
                        choices=["expert", "layer", "global"],
                        help="Selection scope for bottom-k neurons. "
                             "'expert' = uniform per-expert (structural-shrink compatible); "
                             "'layer'  = pool neurons across experts within each layer; "
                             "'global' = pool neurons across the whole model.")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
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

    # ---- calibration data ----
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

    # ---- MoE info ----
    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    first_moe = get_moe_block(model.model.layers[moe_layer_indices[0]])
    d_ffn = first_moe.experts[0].gate_proj.weight.shape[0]
    n_drop = int(d_ffn * args.drop_ratio)
    print(f"Expert FFN dimension (d_ffn): {d_ffn}")
    print(f"Neurons to drop per expert: {n_drop} ({args.drop_ratio*100:.0f}%)")

    # ---- score collection ----
    print(f"Collecting per-neuron gradient scores over {len(calib_batches)} samples ...")
    scores = collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices)

    for lid in sorted(scores.keys())[:3]:
        all_s = torch.cat([scores[lid][eid] for eid in sorted(scores[lid].keys())])
        print(f"  Layer {lid} neuron scores: min={all_s.min():.6e}, "
              f"max={all_s.max():.6e}, mean={all_s.mean():.6e}")

    # ---- decide ----
    print(f"Selecting bottom {args.drop_ratio*100:.0f}% neurons "
          f"(prune_mode={args.prune_mode}) ...")
    drop_per_layer = pick_neurons_to_drop(scores, args.drop_ratio, args.prune_mode)
    if args.prune_mode != "expert":
        diag = _summarize_drop_distribution(drop_per_layer)
        print("  Per-layer drop distribution (first 5):")
        for row in diag[:5]:
            print(f"    layer {row['layer']:>2}: experts={row['n_experts']} "
                  f"drop min/mean/max = {row['min_drop']}/{row['mean_drop']:.1f}/{row['max_drop']} "
                  f"(experts with 0 drops = {row['zero_experts']})")

    # ---- zero out ----
    print("Zeroing dropped-neuron weights in-place ...")
    drop_summary, total_dropped, total_neurons = zero_dropped_neurons(
        model, drop_per_layer, moe_layer_indices
    )
    print(f"Total neurons: {total_neurons} -> {total_neurons - total_dropped} active "
          f"({total_dropped} zeroed, {total_dropped/total_neurons*100:.1f}%)")

    # ---- save ----
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving masked model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    drop_pct = int(args.drop_ratio * 100)
    summary_path = os.path.join(args.output_dir, "neuron_drop_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "method": f"neuron_drop_pure_gradient_{args.prune_mode}",
            "prune_mode": args.prune_mode,
            "drop_ratio": args.drop_ratio,
            "drop_pct": drop_pct,
            "d_ffn": d_ffn,
            "neurons_dropped_per_expert_target": n_drop,
            "total_neurons": total_neurons,
            "total_dropped": total_dropped,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "dataset_name": args.dataset_name,
            "calib_preset": args.calib_preset,
            "calib_data": args.calib_data,
            "text_column": args.text_column,
            "per_layer": drop_summary,
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
