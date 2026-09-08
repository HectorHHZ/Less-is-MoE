"""
Standalone expert-drop script for Qwen3-30B-A3B (and other Qwen3-MoE variants).

Supported methods (mirror of expert_drop_qwen1.5_moe.py minus densemixer):
    - layerwise_pruning          (router score -> topk per layer)
    - global_pruning             (router score -> global topk across layers)
    - bias_pruning               (aux-free bias simulation -> keep bias>=threshold)
    - weight_magnitude_pruning   (L2 norm of expert weights, data-free)
    - pure_gradient_pruning      (mean |grad| of router gate weight rows)
    - pure_expert_gradient_pruning (mean |grad| of expert gate/up/down params)

Architecture differences vs Qwen1.5-MoE:
    - No shared_expert / shared_expert_gate (routed experts only)
    - Qwen3MoeSparseMoeBlock instead of Qwen2MoeSparseMoeBlock
    - Only some layers are MoE (decoder_sparse_step / mlp_only_layers)
    - position_embeddings (cos, sin) computed *before* decoder layers, must be
      forwarded through kwargs when doing layer-by-layer manual forward passes.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Calibration data loaders
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
    """Load calibration data from a HuggingFace dataset.

    text_column: Comma-separated columns to concatenate (e.g. "prompt,completion").
    """
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
            # Chat-style entries (e.g. yentinglin/s1K-1.1-trl-format `messages`)
            # are {"role": ..., "content": ...} dicts; we only want the content.
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
# Identify Qwen3-MoE blocks
# ---------------------------------------------------------------------------

def get_moe_layer_info(model):
    """
    Return (moe_layer_indices, num_experts_per_layer) for Qwen3-MoE.
    Qwen3 uses decoder_sparse_step and mlp_only_layers to decide which layers are MoE.
    """
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
# Layer-by-layer input capture (shared by routing & bias score collectors)
# ---------------------------------------------------------------------------

def _capture_layer0_inputs(model, calib_batches):
    """Run calib batches through the model until layer[0] and capture its inputs+kwargs.

    Qwen3 computes position_embeddings (cos, sin) before entering the decoder layers,
    so those must be captured and forwarded when replaying layer-by-layer.
    Returns (hidden_states_list, kwargs_list).
    """
    device = next(model.parameters()).device
    layers = model.model.layers
    inputs = []
    kwargs_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inputs.append(hidden_states)
            kwargs_list.append(kwargs)
            raise ValueError  # early exit

    layers[0] = Catcher(layers[0])
    for batch in calib_batches:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    return inputs, kwargs_list


# ---------------------------------------------------------------------------
# Routing score collection (layerwise_pruning / global_pruning)
# ---------------------------------------------------------------------------

class RouterScoreCollector:
    """Collects routing scores for a single MoE block."""

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.scores = None
        self.nsamples = 0

    def hook_fn(self, module, input, output):
        """Hook on SparseMoeBlock. output = (hidden_states, router_logits)."""
        router_logits = output[1].reshape(-1, output[1].shape[-1])
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        batch_score = routing_weights.sum(0)
        if self.scores is None:
            self.scores = batch_score
        else:
            self.scores = self.scores + batch_score
        self.nsamples += 1

    def get_avg_scores(self):
        if self.scores is None:
            return None
        return self.scores / max(self.nsamples, 1)


@torch.no_grad()
def collect_routing_scores(model, calib_batches, moe_layer_indices):
    """Layer-by-layer replay with hooks on MoE blocks to collect router softmax sums."""
    layers = model.model.layers
    inputs, kwargs_list = _capture_layer0_inputs(model, calib_batches)
    num_samples = len(inputs)
    outputs = [None] * num_samples

    scores_per_layer = {}

    for i in tqdm(range(len(layers)), desc="Collecting routing scores"):
        layer = layers[i]

        collector = None
        handle = None
        if i in moe_layer_indices:
            moe_block = get_moe_block(layer)
            if moe_block is not None:
                n_exp = moe_block.gate.out_features
                collector = RouterScoreCollector(n_exp)
                handle = moe_block.register_forward_hook(collector.hook_fn)

        for j in range(num_samples):
            outputs[j] = layer(inputs[j], **kwargs_list[j])[0]

        if handle is not None:
            handle.remove()
        if collector is not None:
            scores_per_layer[i] = collector.get_avg_scores()

        inputs, outputs = outputs, inputs

    return scores_per_layer


# ---------------------------------------------------------------------------
# Bias score collection (simulates aux-free bias accumulation)
# ---------------------------------------------------------------------------

class BiasCollector:
    """Simulates the aux-free bias update for a single MoE block."""

    def __init__(self, num_experts, bias_update_speed, bias_criterion, bias_clip):
        self.num_experts = num_experts
        self.bias = torch.zeros(num_experts, dtype=torch.float32)
        self.bias_update_speed = bias_update_speed
        self.bias_criterion = bias_criterion
        self.bias_clip = bias_clip

    def hook_fn(self, module, input, output):
        router_logits = output[1].reshape(-1, self.num_experts)
        biased_logits = router_logits.float() + self.bias.to(router_logits.device)
        routing_weights = F.softmax(biased_logits, dim=1, dtype=torch.float32)
        expert_usage = routing_weights.sum(dim=0)

        with torch.no_grad():
            if self.bias_criterion == "median":
                avg_usage = expert_usage.median()
            else:
                avg_usage = expert_usage.mean()

            update = torch.zeros(self.num_experts, dtype=torch.float32,
                                 device=self.bias.device)
            update[expert_usage.cpu() > avg_usage.cpu()] = +self.bias_update_speed
            update[expert_usage.cpu() < avg_usage.cpu()] = -self.bias_update_speed

            if self.bias_clip > 0:
                update[self.bias >= self.bias_clip] = torch.clamp(
                    update[self.bias >= self.bias_clip], max=0)

            self.bias.add_(update)


@torch.no_grad()
def collect_bias_scores(model, calib_batches, moe_layer_indices,
                        bias_update_speed, bias_criterion, bias_clip, num_epochs):
    """Run multiple passes over calibration data to accumulate per-expert bias."""
    layers = model.model.layers
    inputs, kwargs_list = _capture_layer0_inputs(model, calib_batches)
    num_samples = len(inputs)

    collectors = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is not None:
            n_exp = moe_block.gate.out_features
            collectors[layer_idx] = BiasCollector(
                n_exp, bias_update_speed, bias_criterion, bias_clip)

    for epoch in range(num_epochs):
        cur_inputs = [inp.clone() for inp in inputs]
        cur_outputs = [None] * num_samples

        for i in tqdm(range(len(layers)),
                      desc=f"Bias collection epoch {epoch+1}/{num_epochs}"):
            handle = None
            if i in collectors:
                moe_block = get_moe_block(layers[i])
                if moe_block is not None:
                    handle = moe_block.register_forward_hook(collectors[i].hook_fn)

            for j in range(num_samples):
                cur_outputs[j] = layers[i](cur_inputs[j], **kwargs_list[j])[0]

            if handle is not None:
                handle.remove()

            cur_inputs, cur_outputs = cur_outputs, cur_inputs

        for lid in sorted(collectors.keys())[:3]:
            b = collectors[lid].bias
            print(f"  [Epoch {epoch+1}] Layer {lid} bias: "
                  f"min={b.min():.4f}, max={b.max():.4f}, "
                  f"#neg={int((b < 0).sum())}/{len(b)}")

    return {lid: c.bias for lid, c in collectors.items()}


# ---------------------------------------------------------------------------
# Weight magnitude score collection (data-free)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_weight_magnitude_scores(model, moe_layer_indices):
    """Per-expert L2 norm over all expert parameters (gate/up/down_proj)."""
    layers = model.model.layers
    scores_per_layer = {}

    for layer_idx in tqdm(moe_layer_indices, desc="Collecting weight magnitude scores"):
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        n_experts = len(moe_block.experts)
        expert_scores = torch.zeros(n_experts)

        for eid, expert in enumerate(moe_block.experts):
            all_params = torch.cat([p.data.float().flatten() for p in expert.parameters()])
            expert_scores[eid] = all_params.norm(p=2)

        scores_per_layer[layer_idx] = expert_scores

    return scores_per_layer


# ---------------------------------------------------------------------------
# Pure gradient magnitude score collection (router gate)
# ---------------------------------------------------------------------------

def collect_pure_gradient_scores(model, calib_batches, moe_layer_indices):
    """Mean |∂L/∂W_gate| per expert row from LM loss on calibration data."""
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)
    gate_params = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is not None:
            moe_block.gate.weight.requires_grad_(True)
            gate_params[layer_idx] = moe_block.gate.weight

    scores_per_layer = {lid: torch.zeros(gate_params[lid].shape[0])
                        for lid in gate_params}

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting pure gradient scores (router)"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for layer_idx, gate_w in gate_params.items():
            if gate_w.grad is None:
                continue
            importance = gate_w.grad.float().abs().mean(dim=1).detach().cpu()
            scores_per_layer[layer_idx] += importance

        model.zero_grad(set_to_none=True)

    n = len(calib_batches)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= max(n, 1)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return scores_per_layer


# ---------------------------------------------------------------------------
# Pure gradient magnitude score collection (expert params)
# ---------------------------------------------------------------------------

def collect_pure_expert_gradient_scores(model, calib_batches, moe_layer_indices):
    """Mean |∂L/∂W| over all expert parameters (gate_proj/up_proj/down_proj)."""
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)

    expert_params = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is None:
            continue
        expert_params[layer_idx] = []
        for eid, expert in enumerate(moe_block.experts):
            params = list(expert.parameters())
            for p in params:
                p.requires_grad_(True)
            expert_params[layer_idx].append((eid, params))

    scores_per_layer = {
        lid: torch.zeros(len(experts))
        for lid, experts in expert_params.items()
    }

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting pure gradient scores (experts)"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for layer_idx, experts in expert_params.items():
            for idx, (eid, params) in enumerate(experts):
                importance = 0.0
                n_elements = 0
                for p in params:
                    if p.grad is not None:
                        importance += p.grad.float().abs().sum().item()
                        n_elements += p.grad.numel()
                if n_elements > 0:
                    importance /= n_elements
                scores_per_layer[layer_idx][idx] += importance

        model.zero_grad(set_to_none=True)

    n = len(calib_batches)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= max(n, 1)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return scores_per_layer


# ---------------------------------------------------------------------------
# Pruning decision
# ---------------------------------------------------------------------------

def decide_experts_to_keep(scores_per_layer, num_experts_per_layer, preserve_n, method):
    """Returns {layer_idx: sorted list of expert indices to keep}."""
    keep_per_layer = {}

    if method == "layerwise_pruning":
        for layer_idx, scores in scores_per_layer.items():
            n_exp = num_experts_per_layer[layer_idx]
            n_keep = min(preserve_n, n_exp)
            if n_keep >= n_exp:
                keep_per_layer[layer_idx] = list(range(n_exp))
            elif n_keep <= 0:
                keep_per_layer[layer_idx] = []
            else:
                _, top_ids = torch.topk(scores, n_keep, largest=True)
                keep_per_layer[layer_idx] = sorted(top_ids.tolist())

    elif method == "global_pruning":
        all_scores = []
        layer_info = []
        for layer_idx in sorted(scores_per_layer.keys()):
            scores = scores_per_layer[layer_idx]
            n_exp = num_experts_per_layer[layer_idx]
            for eid in range(n_exp):
                all_scores.append(scores[eid].item())
                layer_info.append((layer_idx, eid))

        total_experts = len(all_scores)
        total_keep = round(preserve_n * len(scores_per_layer))
        total_keep = min(total_keep, total_experts)

        if total_keep >= total_experts:
            for layer_idx in scores_per_layer:
                keep_per_layer[layer_idx] = list(range(num_experts_per_layer[layer_idx]))
        elif total_keep <= 0:
            for layer_idx in scores_per_layer:
                keep_per_layer[layer_idx] = []
        else:
            all_scores_t = torch.tensor(all_scores)
            _, top_global = torch.topk(all_scores_t, total_keep, largest=True)
            top_global = set(top_global.tolist())

            for layer_idx in scores_per_layer:
                keep_per_layer[layer_idx] = []
            for gid in top_global:
                lid, eid = layer_info[gid]
                keep_per_layer[lid].append(eid)
            for lid in keep_per_layer:
                keep_per_layer[lid] = sorted(keep_per_layer[lid])
    else:
        raise ValueError(f"Unknown method: {method}")

    return keep_per_layer


# ---------------------------------------------------------------------------
# In-place pruning
# ---------------------------------------------------------------------------

def prune_model_inplace(model, keep_per_layer, moe_layer_indices):
    """Prune experts + gate in-place for each MoE layer. Update config accordingly."""
    layers = model.model.layers
    config = model.config
    num_layers = config.num_hidden_layers

    orig_num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[0]

    summary = []

    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        keep_ids = keep_per_layer.get(layer_idx, list(range(moe_block.gate.out_features)))
        orig = moe_block.gate.out_features
        drop_ids = sorted(set(range(orig)) - set(keep_ids))
        n_kept = len(keep_ids)

        summary.append({
            "layer": layer_idx,
            "original": orig,
            "kept": keep_ids,
            "dropped": drop_ids,
        })

        if not drop_ids:
            continue

        moe_block.experts = nn.ModuleList([moe_block.experts[i] for i in keep_ids])

        gate = moe_block.gate
        new_gate = nn.Linear(
            in_features=gate.in_features,
            out_features=n_kept,
            bias=gate.bias is not None,
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        new_gate.weight.data = gate.weight.data[keep_ids].clone()
        if gate.bias is not None:
            new_gate.bias.data = gate.bias.data[keep_ids].clone()
        moe_block.gate = new_gate

        moe_block.num_experts = n_kept
        moe_block.top_k = min(getattr(moe_block, "top_k", n_kept), n_kept)

        print(f"  Layer {layer_idx}: {orig} -> {n_kept} experts (dropped {len(drop_ids)})")

    num_experts_list = [None] * num_layers
    layer_experts_idx = [None] * num_layers

    for layer_idx in range(num_layers):
        if layer_idx in moe_layer_indices:
            if layer_idx in keep_per_layer:
                num_experts_list[layer_idx] = len(keep_per_layer[layer_idx])
                layer_experts_idx[layer_idx] = list(keep_per_layer[layer_idx])
            else:
                num_experts_list[layer_idx] = orig_num_experts
                layer_experts_idx[layer_idx] = list(range(orig_num_experts))

    config.num_experts = num_experts_list
    config.layer_experts_idx = layer_experts_idx

    valid_counts = [c for c in num_experts_list if isinstance(c, int) and c > 0]
    if valid_counts and hasattr(config, "num_experts_per_tok"):
        config.num_experts_per_tok = min(config.num_experts_per_tok, min(valid_counts))

    config.router_mask_kept = {str(s["layer"]): s["kept"] for s in summary}
    config.router_mask_dropped = {str(s["layer"]): s["dropped"] for s in summary}

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expert drop for Qwen3-MoE models")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preserve_n", type=int, default=64,
                        help="Number of experts to preserve per layer")
    parser.add_argument("--method", type=str, default="layerwise_pruning",
                        choices=["global_pruning", "layerwise_pruning",
                                 "bias_pruning",
                                 "weight_magnitude_pruning",
                                 "pure_gradient_pruning",
                                 "pure_expert_gradient_pruning",
                                 "random_pruning"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    # Calibration data sources (priority: dataset_name > calib_data)
    parser.add_argument("--calib_data", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="prompt")
    parser.add_argument("--shuffle_seed", type=int, default=None)
    # Bias-pruning knobs
    parser.add_argument("--bias_update_speed", type=float, default=4e-2)
    parser.add_argument("--bias_criterion", type=str, default="median",
                        choices=["mean", "median"])
    parser.add_argument("--bias_clip", type=float, default=10.0)
    parser.add_argument("--bias_epochs", type=int, default=2)
    parser.add_argument("--bias_threshold", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--mask_file", type=str, default=None,
                        help="Pre-computed expert_drop_summary.json; skip calibration.")
    args = parser.parse_args()

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

    data_free_methods = {"random_pruning", "weight_magnitude_pruning"}
    if args.mask_file:
        print(f"Mask file provided ({args.mask_file}); skipping calibration data loading.")
        calib_batches = []
    elif args.method in data_free_methods:
        print(f"Method {args.method} is data-free; skipping calibration data loading.")
        calib_batches = []
    elif args.dataset_name:
        print(f"Loading calibration data from HF dataset: {args.dataset_name} ...")
        calib_batches = load_calib_data_hf(
            tokenizer, args.dataset_name, args.dataset_config, args.dataset_split,
            args.n_samples, args.seq_len, args.text_column,
            shuffle_seed=args.shuffle_seed,
        )
    else:
        if not args.calib_data:
            raise ValueError("Must provide --dataset_name or --calib_data for this method.")
        print(f"Loading calibration data from {args.calib_data} ...")
        calib_batches = load_calib_data(tokenizer, args.calib_data, args.n_samples, args.seq_len)
    print(f"Loaded {len(calib_batches)} calibration samples (seq_len={args.seq_len})")

    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    # ---------------- Route to the right scoring path ----------------
    if args.mask_file:
        print(f"Loading pre-computed mask from {args.mask_file} ...")
        with open(args.mask_file, "r") as f:
            mask_data = json.load(f)
        keep_per_layer = {}
        for entry in mask_data["per_layer"]:
            keep_per_layer[int(entry["layer"])] = list(entry["kept"])
        print(f"Loaded mask for {len(keep_per_layer)} layers "
              f"(method={mask_data.get('method', 'unknown')}, "
              f"preserve_n={mask_data.get('preserve_n', 'unknown')})")

    elif args.method == "random_pruning":
        print(f"Random pruning (data-free), seed={args.random_seed}, preserve_n={args.preserve_n} ...")
        rng = random.Random(args.random_seed)
        keep_per_layer = {}
        for lid in sorted(moe_layer_indices):
            n_exp = num_experts_per_layer[lid]
            n_keep = min(args.preserve_n, n_exp)
            if n_keep >= n_exp:
                keep_per_layer[lid] = list(range(n_exp))
            elif n_keep <= 0:
                keep_per_layer[lid] = []
            else:
                keep_per_layer[lid] = sorted(rng.sample(range(n_exp), n_keep))

    elif args.method == "weight_magnitude_pruning":
        print("Collecting weight magnitude scores (data-free) ...")
        scores_per_layer = collect_weight_magnitude_scores(model, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} weight magnitude: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")
        print(f"Pruning with method=weight_magnitude_pruning (layerwise), preserve_n={args.preserve_n} ...")
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "pure_gradient_pruning":
        print("Collecting pure gradient magnitude scores (router gate weights) ...")
        scores_per_layer = collect_pure_gradient_scores(model, calib_batches, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} pure gate gradient: min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")
        print(f"Pruning with method=pure_gradient_pruning (layerwise), preserve_n={args.preserve_n} ...")
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "pure_expert_gradient_pruning":
        print("Collecting pure gradient magnitude scores (expert params) ...")
        scores_per_layer = collect_pure_expert_gradient_scores(model, calib_batches, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} pure expert gradient: min={s.min():.6f}, max={s.max():.6f}, mean={s.mean():.6f}")
        print(f"Pruning with method=pure_expert_gradient_pruning (layerwise), preserve_n={args.preserve_n} ...")
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "bias_pruning":
        print(f"Collecting bias scores (speed={args.bias_update_speed}, "
              f"criterion={args.bias_criterion}, clip={args.bias_clip}, "
              f"epochs={args.bias_epochs}) ...")
        bias_per_layer = collect_bias_scores(
            model, calib_batches, moe_layer_indices,
            args.bias_update_speed, args.bias_criterion, args.bias_clip,
            args.bias_epochs,
        )
        for lid in sorted(bias_per_layer.keys()):
            b = bias_per_layer[lid]
            print(f"  Layer {lid} final bias: min={b.min():.4f}, max={b.max():.4f}, "
                  f"#neg={int((b < 0).sum())}/{len(b)}")

        # Save bias states
        bias_states = {}
        for lid, bias in bias_per_layer.items():
            bias_states[f"model.layers.{lid}.mlp.gate"] = {
                "bias_values": bias.tolist(),
                "num_experts": len(bias),
            }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "moe_bias_states.json"), "w") as f:
            json.dump({
                "metadata": {
                    "method": "offline_bias_simulation",
                    "bias_update_speed": args.bias_update_speed,
                    "bias_criterion": args.bias_criterion,
                    "bias_clip": args.bias_clip,
                    "bias_epochs": args.bias_epochs,
                    "n_samples": args.n_samples,
                },
                "moe_bias_states": bias_states,
            }, f, indent=2)

        print(f"Pruning experts with bias < {args.bias_threshold} ...")
        keep_per_layer = {}
        for lid, bias in bias_per_layer.items():
            keep_ids = [i for i in range(len(bias)) if bias[i].item() >= args.bias_threshold]
            min_keep = num_experts_per_layer.get(lid, 4)
            min_keep = min(getattr(model.config, "num_experts_per_tok", 4), min_keep)
            if len(keep_ids) < min_keep:
                _, top_ids = torch.topk(bias, min_keep, largest=True)
                keep_ids = top_ids.tolist()
            keep_per_layer[lid] = sorted(keep_ids)

    else:
        # layerwise_pruning / global_pruning (routing-score based)
        print("Collecting routing scores ...")
        scores_per_layer = collect_routing_scores(model, calib_batches, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} scores: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")
        print(f"Pruning with method={args.method}, preserve_n={args.preserve_n} ...")
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, args.method
        )

    total_orig = sum(num_experts_per_layer.values())
    total_kept = sum(len(v) for v in keep_per_layer.values())
    print(f"Total experts: {total_orig} -> {total_kept} ({total_kept/total_orig*100:.1f}%)")

    print("Pruning model in-place ...")
    summary = prune_model_inplace(model, keep_per_layer, moe_layer_indices)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving pruned model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary_path = os.path.join(args.output_dir, "expert_drop_summary.json")
    summary_data = {
        "method": args.method,
        "preserve_n": args.preserve_n,
        "n_samples": args.n_samples,
        "total_orig_experts": total_orig,
        "total_kept_experts": total_kept,
        "per_layer": summary,
    }
    if args.method == "bias_pruning":
        summary_data.update({
            "bias_update_speed": args.bias_update_speed,
            "bias_criterion": args.bias_criterion,
            "bias_clip": args.bias_clip,
            "bias_epochs": args.bias_epochs,
            "bias_threshold": args.bias_threshold,
        })
    if args.method == "random_pruning":
        summary_data["random_seed"] = args.random_seed
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Done! Model saved to {args.output_dir}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
