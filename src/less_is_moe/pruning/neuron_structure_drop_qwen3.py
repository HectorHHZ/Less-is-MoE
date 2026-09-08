"""
Structural neuron-level drop for Qwen3-30B-A3B (and other Qwen3-MoE variants).

Strategy — structural removal (shapes shrink):
    For each routed expert in each MoE layer, rank every FFN neuron by pure
    gradient magnitude and physically remove the bottom `drop_ratio` fraction
    by replacing gate_proj / up_proj / down_proj with smaller Linear modules.

    For surviving neuron set K ⊆ {0, ..., d_ffn - 1}:
        gate_proj: (d_ffn, hidden)  ->  (|K|, hidden)   rows[K] kept
        up_proj:   (d_ffn, hidden)  ->  (|K|, hidden)   rows[K] kept
        down_proj: (hidden, d_ffn)  ->  (hidden, |K|)   cols[K] kept

    Qwen3-MoE has no shared_expert, so only routed experts are modified.

Importance metric (same as Qwen1.5 port):
    For neuron j of expert e in layer l,
        score_j = ( sum|∂L/∂gate_proj.weight[j,:]|
                  + sum|∂L/∂up_proj.weight[j,:]|
                  + sum|∂L/∂down_proj.weight[:,j]| )
                  / (gate_dim + up_dim + down_dim)
    averaged over calibration samples with L = LM cross-entropy.

Usage:
    scripts/prune/neuron_structure_drop_qwen3.sh \
        --model_name_or_path Qwen/Qwen3-30B-A3B \
        --output_dir /path/to/pruned_model \
        --drop_ratio 0.25 \
        --n_samples 128 \
        --seq_len 2048 \
        --dataset_name HectorHe/math7k \
        --dataset_split train \
        --text_column instruction,output \
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
# Calibration data loaders (mirror of expert_drop_qwen3.py)
# ---------------------------------------------------------------------------

def load_calib_data(tokenizer, calib_data_path, n_samples, seq_len):
    """Load JSON/JSONL calibration file and tokenize into batches."""
    with open(calib_data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        raw = json.loads(content)
        if isinstance(raw, dict):
            raw = [raw]
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in content.splitlines() if line.strip()]

    def _row_text(item):
        if "text" in item and item["text"]:
            return item["text"]
        if "prompt" in item and "completion" in item:
            return f"{item['prompt']}\n{item['completion']}"
        if "prompt" in item:
            return item["prompt"]
        if "instruction" in item and "output" in item:
            return f"{item['instruction']}\n{item['output']}"
        return None

    texts = [t for t in (_row_text(item) for item in raw) if t][:n_samples * 4]

    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches


def load_calib_data_hf(tokenizer, dataset_name, dataset_config, dataset_split,
                       n_samples, seq_len, text_column="prompt",
                       shuffle_seed=None):
    """Load calibration data from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading HF dataset: {dataset_name} (config={dataset_config}, split={dataset_split})")

    try:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    except Exception as first_err:
        print(f"  [WARN] Standard load failed, trying raw JSON fallback...")
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
        candidates = [f for f in repo_files
                      if dataset_split in f
                      and any(f.endswith(ext) for ext in (".json", ".jsonl"))]
        if candidates:
            data_urls = [f"hf://datasets/{dataset_name}/{f}" for f in candidates]
            ds = load_dataset("json", data_files=data_urls, split="train")
        else:
            raise first_err

    if shuffle_seed is not None:
        print(f"  Shuffling dataset with seed={shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    columns = ds.column_names
    text_cols = [c.strip() for c in text_column.split(",")]
    print(f"  Dataset columns: {columns}, num rows: {len(ds)}")
    print(f"  Using columns {text_cols} as calibration text")

    def _to_str(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, (list, tuple)):
            return " ".join(_to_str(x) for x in val)
        if isinstance(val, dict):
            # Chat-style dicts (e.g. messages = [{"role":..., "content":...}, ...])
            # → use only the content field for calibration text.
            if "content" in val:
                return _to_str(val["content"])
            return " ".join(f"{k}: {_to_str(v)}" for k, v in val.items())
        return str(val)

    texts = []
    for row in ds:
        parts = [_to_str(row[c]) for c in text_cols if c in row]
        text = " ".join(p for p in parts if p)
        if not text:
            for c in columns:
                v = _to_str(row[c])
                if v:
                    text = v
                    break
        if text:
            texts.append(text)

    print(f"  Extracted {len(texts)} text samples from dataset")

    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if ids.shape[1] < seq_len:
            batches.append(ids)
        else:
            batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches


# ---------------------------------------------------------------------------
# Qwen3-MoE layer detection
# ---------------------------------------------------------------------------

def get_moe_layer_info(model):
    """Return (moe_layer_indices, num_experts_per_layer) for Qwen3-MoE."""
    config = model.config
    num_layers = config.num_hidden_layers
    decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
    mlp_only_layers = set(getattr(config, "mlp_only_layers", []))
    num_experts = config.num_experts

    moe_layer_indices = []
    num_experts_per_layer = {}

    for layer_idx in range(num_layers):
        is_moe = (
            layer_idx not in mlp_only_layers
            and (isinstance(num_experts, int) and num_experts > 0)
            and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if is_moe:
            n_exp = num_experts if isinstance(num_experts, int) else num_experts[layer_idx]
            moe_layer_indices.append(layer_idx)
            num_experts_per_layer[layer_idx] = n_exp

    return moe_layer_indices, num_experts_per_layer


def get_moe_block(layer):
    """Return the SparseMoeBlock from a decoder layer, or None."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
        return mlp
    return None


# ---------------------------------------------------------------------------
# Per-neuron gradient scoring
# ---------------------------------------------------------------------------

def collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices):
    """
    Per-neuron importance based on pure gradient magnitude.

    Uses full-model forward with LM loss (labels=input_ids), so no manual
    position_embeddings forwarding is needed (transformers handles it).

    Returns
    -------
    dict  {layer_idx: {expert_idx: Tensor of shape (d_ffn,)}}
    """
    device = next(model.parameters()).device

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
                expert.gate_proj.weight,   # (d_ffn, hidden)
                expert.up_proj.weight,     # (d_ffn, hidden)
                expert.down_proj.weight,   # (hidden, d_ffn)
            ))

    scores = {}
    for layer_idx, experts in expert_projs.items():
        scores[layer_idx] = {}
        for eid, gate_w, *_ in experts:
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


def decide_neurons_to_drop(scores, drop_ratio):
    """
    Expert-wise selection: each expert drops the same count of lowest-scoring
    neurons so the resulting d_ffn is uniform (required for structural shrink).

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


# ---------------------------------------------------------------------------
# Detect pre-existing zero-masked neurons (for --from_zeroed_model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_zeroed_neurons(model, moe_layer_indices):
    """Scan each expert for neurons whose gate_proj row is entirely zero."""
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
    """New Linear keeping only `keep_ids` rows (output dim)."""
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
    """New Linear keeping only `keep_ids` columns (input dim)."""
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
    Replace each routed expert's gate/up/down projections with smaller Linears.
    All experts drop the same count per layer (set by --drop_ratio) so the
    resulting d_ffn is uniform.
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
        description="Structural neuron-level drop for Qwen3-MoE"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--drop_ratio", type=float, default=None,
                        help="Fraction of neurons to structurally remove per expert "
                             "(e.g. 0.25). Not needed when --from_zeroed_model is set.")
    parser.add_argument("--from_zeroed_model", action="store_true",
                        help="Skip calibration: detect already-zeroed neurons in "
                             "the input checkpoint and structurally shrink them.")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--calib_data", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="prompt")
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

    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    first_moe = get_moe_block(model.model.layers[moe_layer_indices[0]])
    orig_d_ffn = first_moe.experts[0].gate_proj.weight.shape[0]
    print(f"Expert FFN dimension (d_ffn): {orig_d_ffn}")

    if args.from_zeroed_model:
        print("Detecting already-zeroed neurons ...")
        drop_per_layer = find_zeroed_neurons(model, moe_layer_indices)
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
        if args.dataset_name:
            print(f"Loading calibration data from HF dataset: {args.dataset_name} ...")
            calib_batches = load_calib_data_hf(
                tokenizer, args.dataset_name, args.dataset_config, args.dataset_split,
                args.n_samples, args.seq_len, args.text_column,
                shuffle_seed=args.shuffle_seed,
            )
        elif args.calib_data:
            print(f"Loading calibration data from {args.calib_data} ...")
            calib_batches = load_calib_data(tokenizer, args.calib_data, args.n_samples, args.seq_len)
        else:
            raise ValueError("Must provide one of --dataset_name / --calib_data")
        print(f"Loaded {len(calib_batches)} calibration samples (seq_len={args.seq_len})")

        n_drop = int(orig_d_ffn * args.drop_ratio)
        print(f"Neurons to remove per expert: {n_drop} ({args.drop_ratio*100:.0f}%)")
        print(f"Surviving d_ffn per expert: {orig_d_ffn - n_drop}")

        print(f"Collecting per-neuron gradient scores over {len(calib_batches)} samples ...")
        scores = collect_neuron_gradient_scores(model, calib_batches, moe_layer_indices)

        for lid in sorted(scores.keys())[:3]:
            all_s = torch.cat([scores[lid][eid] for eid in sorted(scores[lid].keys())])
            print(f"  Layer {lid} neuron scores: min={all_s.min():.6e}, "
                  f"max={all_s.max():.6e}, mean={all_s.mean():.6e}")

        print(f"Selecting bottom {args.drop_ratio*100:.0f}% neurons per expert ...")
        drop_per_layer = decide_neurons_to_drop(scores, args.drop_ratio)

    print("Structurally removing dropped-neuron slices ...")
    drop_summary, total_dropped, total_neurons, new_d_ffn = structurally_remove_neurons(
        model, drop_per_layer, moe_layer_indices
    )
    print(f"Total neurons: {total_neurons} -> {total_neurons - total_dropped} kept "
          f"({total_dropped} removed, {total_dropped/total_neurons*100:.1f}%)")

    old_moe_int = getattr(model.config, "moe_intermediate_size", None)
    model.config.moe_intermediate_size = new_d_ffn
    print(f"config.moe_intermediate_size: {old_moe_int} -> {new_d_ffn}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving structurally-pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
