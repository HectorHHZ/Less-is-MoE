"""
Structural neuron-level drop for Qwen1.5-MoE-A2.7B (and other Qwen2-MoE variants).

Strategy — structural removal (shapes shrink):
    For each routed expert in each MoE layer, rank every FFN neuron by
    gradient importance and physically remove the bottom `drop_ratio`
    fraction by replacing gate_proj / up_proj / down_proj with smaller
    Linear modules.

    For surviving neuron set K ⊆ {0, ..., d_ffn - 1}:
        gate_proj: (d_ffn, hidden)  ->  (|K|, hidden)   rows[K] kept
        up_proj:   (d_ffn, hidden)  ->  (|K|, hidden)   rows[K] kept
        down_proj: (hidden, d_ffn)  ->  (hidden, |K|)   cols[K] kept

    Shared expert is NOT pruned — only routed experts are modified.

Equivalence to zero-masking (neuron_drop_qwen1.5_moe.py):
    In Qwen2-MoE experts all three projections use `bias=False`, so
        gate_proj.W[j, :] = 0  ->  gate_j(x) = 0      (bias-free)
        silu(0) = 0  ->  silu(gate)[j] * up[j] = 0
        down_proj.W[:, j] · 0  contributes nothing
    Removing the j-th row of gate/up and the j-th col of down produces
    identical outputs (modulo FP rounding) to zero-masking them.

Importance metric (same as neuron_drop_qwen1.5_moe.py):

    For neuron j of expert e in layer l,
        score_j = ( sum|∂L/∂gate_proj.weight[j,:]|
                  + sum|∂L/∂up_proj.weight[j,:]|
                  + sum|∂L/∂down_proj.weight[:,j]| )
                  / (gate_dim + up_dim + down_dim)

    averaged over calibration samples with L = LM cross-entropy.

Usage:
    scripts/prune/neuron_structure_drop_qwen15_moe.sh \\
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
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Reuse calibration loaders, MoE utilities, and scoring from neuron_drop
# ---------------------------------------------------------------------------

from .neuron_drop_qwen15_moe import (
    collect_neuron_gradient_scores,
    decide_neurons_to_drop,
    get_moe_block,
    get_moe_layer_info,
    load_calib_data,
    load_calib_data_hf,
    load_calib_data_preset,
)


# ---------------------------------------------------------------------------
# Detect pre-existing zero-masked neurons (from neuron_drop_qwen1.5_moe.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_zeroed_neurons(model, moe_layer_indices):
    """
    Scan each routed expert in each MoE layer for neurons whose gate_proj row
    is entirely zero. These are the neurons that neuron_drop_qwen1.5_moe.py
    zero-masked, and structurally removing them is bit-equivalent to the
    masked version (since Qwen2-MoE experts have bias=False).

    Returns
    -------
    dict  {layer_idx: {expert_idx: sorted list of zeroed-neuron indices}}
    """
    drop_per_layer = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is None:
            continue
        drop_per_layer[layer_idx] = {}
        for eid, expert in enumerate(moe_block.experts):
            gate_w = expert.gate_proj.weight.data
            zero_rows = (gate_w.float().abs().sum(dim=1) == 0)
            drop_ids = torch.nonzero(zero_rows, as_tuple=False).flatten().tolist()
            drop_per_layer[layer_idx][eid] = sorted(drop_ids)
    return drop_per_layer


# ---------------------------------------------------------------------------
# Structural neuron removal
# ---------------------------------------------------------------------------

def _replace_linear_rows(linear: nn.Linear, keep_ids: torch.Tensor) -> nn.Linear:
    """Return a new Linear keeping only `keep_ids` of the output dim (rows)."""
    new = nn.Linear(
        in_features=linear.in_features,
        out_features=keep_ids.numel(),
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data.index_select(0, keep_ids).clone()
    if linear.bias is not None:
        new.bias.data = linear.bias.data.index_select(0, keep_ids).clone()
    return new


def _replace_linear_cols(linear: nn.Linear, keep_ids: torch.Tensor) -> nn.Linear:
    """Return a new Linear keeping only `keep_ids` of the input dim (columns)."""
    new = nn.Linear(
        in_features=keep_ids.numel(),
        out_features=linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data.index_select(1, keep_ids).clone()
    if linear.bias is not None:
        new.bias.data = linear.bias.data.clone()
    return new


@torch.no_grad()
def structurally_remove_neurons(model, drop_per_layer, moe_layer_indices):
    """
    Replace each routed expert's gate/up/down projections with smaller Linears
    that only contain the surviving neurons. All experts drop the same count
    per layer (set by --drop_ratio), so the resulting d_ffn is uniform.

    Shared expert is NOT modified.

    Returns
    -------
    summary : list of per-layer dicts (kept/dropped indices per expert)
    total_dropped : int
    total_neurons : int
    new_moe_intermediate_size : int  (uniform new d_ffn of routed experts)
    """
    layers = model.model.layers
    summary = []
    total_dropped = 0
    total_neurons = 0
    new_d_ffn_seen = None

    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        layer_info = {"layer": layer_idx, "experts": []}

        for eid, expert in enumerate(moe_block.experts):
            orig_d_ffn = expert.gate_proj.weight.shape[0]
            drop_ids = drop_per_layer.get(layer_idx, {}).get(eid, [])
            drop_set = set(drop_ids)
            keep_ids_list = [i for i in range(orig_d_ffn) if i not in drop_set]
            n_kept = len(keep_ids_list)

            if drop_ids:
                keep_ids = torch.tensor(
                    keep_ids_list,
                    dtype=torch.long,
                    device=expert.gate_proj.weight.device,
                )
                expert.gate_proj = _replace_linear_rows(expert.gate_proj, keep_ids)
                expert.up_proj = _replace_linear_rows(expert.up_proj, keep_ids)
                expert.down_proj = _replace_linear_cols(expert.down_proj, keep_ids)
                if hasattr(expert, "intermediate_size"):
                    expert.intermediate_size = n_kept

            if new_d_ffn_seen is None:
                new_d_ffn_seen = n_kept
            elif new_d_ffn_seen != n_kept:
                raise ValueError(
                    "Structural removal requires a uniform surviving d_ffn "
                    f"across experts, but got {new_d_ffn_seen} and {n_kept} "
                    f"(layer {layer_idx}, expert {eid})."
                )

            layer_info["experts"].append({
                "expert_id": eid,
                "orig_neurons": orig_d_ffn,
                "kept_neurons": n_kept,
                "dropped_count": len(drop_ids),
                "dropped_neurons": drop_ids,
            })
            total_dropped += len(drop_ids)
            total_neurons += orig_d_ffn

        n_experts = len(moe_block.experts)
        avg_drop = (sum(len(drop_per_layer.get(layer_idx, {}).get(e, []))
                        for e in range(n_experts))
                    / max(n_experts, 1))
        print(f"  Layer {layer_idx}: {n_experts} experts, "
              f"avg {avg_drop:.0f} neurons removed -> d_ffn={new_d_ffn_seen}")

        summary.append(layer_info)

    return summary, total_dropped, total_neurons, new_d_ffn_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structural neuron-level drop for Qwen1.5-MoE"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drop_ratio", type=float, default=None,
                        help="Fraction of neurons to structurally remove per expert "
                             "(e.g. 0.25). Not needed when --from_zeroed_model is set.")
    parser.add_argument("--from_zeroed_model", action="store_true",
                        help="Skip calibration: detect already-zeroed neurons in "
                             "`--model_name_or_path` (a neuron_drop output) and "
                             "structurally shrink them. Produces a bit-equivalent "
                             "structural checkpoint.")
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

    if not args.from_zeroed_model:
        if args.drop_ratio is None or not (0.0 < args.drop_ratio < 1.0):
            raise ValueError(
                f"--drop_ratio must be in (0, 1) when calibrating, got {args.drop_ratio}"
            )

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

    # -------- MoE info --------
    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    first_moe = get_moe_block(model.model.layers[moe_layer_indices[0]])
    orig_d_ffn = first_moe.experts[0].gate_proj.weight.shape[0]
    print(f"Expert FFN dimension (d_ffn): {orig_d_ffn}")

    if args.from_zeroed_model:
        # -------- detect zeroed neurons from the input checkpoint --------
        print("Detecting already-zeroed neurons (from neuron_drop output) ...")
        drop_per_layer = find_zeroed_neurons(model, moe_layer_indices)
        # Sanity: all experts must have the same drop count for uniform d_ffn.
        counts = set()
        for lid in drop_per_layer:
            for eid, ids in drop_per_layer[lid].items():
                counts.add(len(ids))
        if len(counts) != 1:
            raise ValueError(
                f"--from_zeroed_model requires uniform per-expert drop count; "
                f"observed counts {sorted(counts)[:5]}..."
            )
        n_drop = next(iter(counts))
        args.drop_ratio = n_drop / orig_d_ffn
        print(f"Detected {n_drop} zeroed neurons per expert "
              f"(drop_ratio≈{args.drop_ratio*100:.1f}%)")
    else:
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

        n_drop = int(orig_d_ffn * args.drop_ratio)
        print(f"Neurons to remove per expert: {n_drop} ({args.drop_ratio*100:.0f}%)")
        print(f"Surviving d_ffn per expert: {orig_d_ffn - n_drop}")

        # -------- score collection --------
        print(f"Collecting per-neuron gradient scores over {len(calib_batches)} samples ...")
        scores = collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices)

        for lid in sorted(scores.keys())[:3]:
            all_s = torch.cat([scores[lid][eid] for eid in sorted(scores[lid].keys())])
            print(f"  Layer {lid} neuron scores: min={all_s.min():.6e}, "
                  f"max={all_s.max():.6e}, mean={all_s.mean():.6e}")

        # -------- decide which neurons to drop --------
        print(f"Selecting bottom {args.drop_ratio*100:.0f}% neurons per expert ...")
        drop_per_layer = decide_neurons_to_drop(scores, args.drop_ratio)

    # -------- structurally remove --------
    print("Structurally removing dropped-neuron slices ...")
    drop_summary, total_dropped, total_neurons, new_d_ffn = structurally_remove_neurons(
        model, drop_per_layer, moe_layer_indices
    )
    print(f"Total neurons: {total_neurons} -> {total_neurons - total_dropped} kept "
          f"({total_dropped} removed, {total_dropped/total_neurons*100:.1f}%)")

    # -------- update config so HF/vLLM reload with correct shapes --------
    old_moe_int = getattr(model.config, "moe_intermediate_size", None)
    model.config.moe_intermediate_size = new_d_ffn
    print(f"config.moe_intermediate_size: {old_moe_int} -> {new_d_ffn}")

    # -------- save --------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving structurally-pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -------- summary JSON --------
    drop_pct = int(args.drop_ratio * 100)
    summary_path = os.path.join(args.output_dir, "neuron_structure_drop_summary.json")
    method = ("neuron_structure_drop_from_zeroed_model"
              if args.from_zeroed_model
              else "neuron_structure_drop_pure_gradient")
    summary_data = {
        "method": method,
        "source_model": args.model_name_or_path,
        "drop_ratio": args.drop_ratio,
        "drop_pct": drop_pct,
        "orig_d_ffn": orig_d_ffn,
        "new_d_ffn": new_d_ffn,
        "neurons_removed_per_expert": n_drop,
        "total_neurons": total_neurons,
        "total_removed": total_dropped,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "dataset_name": args.dataset_name,
        "calib_preset": args.calib_preset,
        "calib_data": args.calib_data,
        "text_column": args.text_column,
        "per_layer": drop_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print("Done!")


if __name__ == "__main__":
    main()
