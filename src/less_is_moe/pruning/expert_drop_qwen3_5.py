"""
Standalone expert-drop script for Qwen3.5-35B-A3B (and other Qwen3.5-MoE variants).

Supported methods (same 5 metrics as expert_drop_qwen3.py):
    - layerwise_pruning          (router score -> topk per layer)
    - global_pruning             (router score -> global topk across layers)
    - bias_pruning               (aux-free bias simulation -> keep bias>=threshold)
    - weight_magnitude_pruning   (L2 norm of expert weights, data-free)
    - pure_gradient_pruning      (mean |grad| of router gate weight rows)
    - pure_expert_gradient_pruning (mean |grad| of expert gate/up/down params)

PREREQ: transformers must support model_type=qwen3_5_moe (4.53.1 does NOT).
    pip install "git+https://github.com/huggingface/transformers.git"
Verify with:
    python -c "from transformers import AutoConfig; \\
               c=AutoConfig.from_pretrained('Qwen/Qwen3.5-35B-A3B', trust_remote_code=True); \\
               print(c.model_type, c.num_experts)"

Architecture notes (vs Qwen3-MoE):
    - Built on Qwen3-Next (hybrid attention: linear + full), so layer-by-layer
      replay may receive additional hybrid-state kwargs. We forward via **kwargs
      so routing-score / bias collectors transparently handle new fields.
    - MoE block still exposes .gate + .experts (ModuleList of MLP experts with
      gate_proj / up_proj / down_proj) — same as Qwen3-MoE.
    - Config still uses decoder_sparse_step / mlp_only_layers / num_experts.
    - No shared_expert (same as Qwen3-MoE).
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
# Qwen3.5-MoE arch compatibility helpers
# ---------------------------------------------------------------------------

def _get_text_config(cfg):
    """Qwen3.5-MoE (VL variant) nests the language-model config under
    ``cfg.text_config``. Pure-text checkpoints keep fields at the top level.
    Return whichever sub-config actually holds num_experts / num_hidden_layers.
    """
    tc = getattr(cfg, "text_config", None)
    if tc is not None and hasattr(tc, "num_experts"):
        return tc
    return cfg


def _get_decoder_layers(model):
    """Locate the MoE decoder layer stack across model wrapping styles:
        - pure Qwen3/3.5 text:      model.model.layers
        - VL/Conditional variants:   model.model.language_model.layers
                                     model.language_model.model.layers
                                     model.model.language_model.model.layers
    """
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("model", "model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise AttributeError(
        "Could not locate decoder layer stack. Tried: "
        + ", ".join("model." + ".".join(p) for p in candidates)
    )


def _load_qwen3_5_model(path, dtype):
    """Try AutoModelForCausalLM, fall back to AutoModel / AutoModelForImageTextToText
    for VL variants (Qwen3_5MoeForConditionalGeneration).
    """
    kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    errs = []
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except (ValueError, KeyError, AttributeError) as e:
        errs.append(f"AutoModelForCausalLM: {type(e).__name__}: {e}")
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    except Exception as e:
        errs.append(f"AutoModelForImageTextToText: {type(e).__name__}: {e}")
    try:
        from transformers import AutoModel
        return AutoModel.from_pretrained(path, **kwargs)
    except Exception as e:
        errs.append(f"AutoModel: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"Could not load Qwen3.5 model from {path}. Tried:\n  "
        + "\n  ".join(errs)
    )


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
# Identify Qwen3.5-MoE blocks
# ---------------------------------------------------------------------------

def get_moe_layer_info(model):
    """
    Return (moe_layer_indices, num_experts_per_layer) for Qwen3.5-MoE.
    Qwen3.5 uses decoder_sparse_step and mlp_only_layers (inherited from Qwen3).
    """
    config = _get_text_config(model.config)
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

    Qwen3/Qwen3.5 computes position_embeddings (cos, sin) before entering the decoder layers,
    so those must be captured and forwarded when replaying layer-by-layer.
    Returns (hidden_states_list, kwargs_list).
    """
    device = next(model.parameters()).device
    layers = _get_decoder_layers(model)
    inputs = []
    kwargs_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # Forward attribute access to the wrapped module (e.g. layer_type
            # which Qwen3.5-MoE's model.forward reads before calling the layer
            # to pick between linear_attention and full_attention mask).
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

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
        """Hook on Qwen3_5MoeTopKRouter. Its forward returns
            (router_logits_softmax, router_scores_topk, router_indices)
        where router_logits_softmax is already a full-distribution softmax over
        num_experts (shape (seq_len, num_experts)). Summing along batch gives
        per-expert routing mass.
        """
        probs = output[0] if isinstance(output, tuple) else output
        probs = probs.reshape(-1, probs.shape[-1]).float()
        batch_score = probs.sum(0)
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
    """Layer-by-layer replay with hooks on router (moe_block.gate) to collect
    per-expert softmax mass. Qwen3.5's SparseMoeBlock returns a single hidden
    tensor (no router_logits in the output tuple), so we hook on the inner
    router instead.
    """
    layers = _get_decoder_layers(model)
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
                n_exp = getattr(moe_block.gate, "num_experts",
                                getattr(moe_block.experts, "num_experts", None))
                collector = RouterScoreCollector(n_exp)
                handle = moe_block.gate.register_forward_hook(collector.hook_fn)

        for j in range(num_samples):
            # .contiguous() — Qwen3.5 MoE's grouped_mm kernel (torch 2.10)
            # requires 16-byte aligned data_ptrs; our captured hidden_states
            # is a view into embedding output that may not be aligned.
            _out = layer(inputs[j].contiguous(), **kwargs_list[j])
            outputs[j] = _out[0] if isinstance(_out, tuple) else _out

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
        """Hook on Qwen3_5MoeTopKRouter. The router already softmaxes inside
        forward, so we recompute pre-softmax logits from input + module.weight
        before adding bias (bias must be added to raw logits, not probs).
        """
        hidden = input[0]  # (N, hidden_dim)
        logits = F.linear(hidden.float(), module.weight.float())  # (N, num_experts)
        biased = logits + self.bias.to(logits.device)
        probs = F.softmax(biased, dim=-1, dtype=torch.float32)
        expert_usage = probs.sum(dim=0)

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
    layers = _get_decoder_layers(model)
    inputs, kwargs_list = _capture_layer0_inputs(model, calib_batches)
    num_samples = len(inputs)

    collectors = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is not None:
            n_exp = getattr(moe_block.gate, "num_experts",
                            getattr(moe_block.experts, "num_experts", None))
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
                    # Hook on router (gate), not SparseMoeBlock — see comment in
                    # collect_routing_scores.
                    handle = moe_block.gate.register_forward_hook(collectors[i].hook_fn)

            for j in range(num_samples):
                _out = layers[i](cur_inputs[j].contiguous(), **kwargs_list[j])
                cur_outputs[j] = _out[0] if isinstance(_out, tuple) else _out

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
    """Per-expert L2 norm over expert parameters.

    Qwen3.5-MoE stores expert weights as batched 3D Parameters on a single
    ``Qwen3_5MoeExperts`` module (no per-expert ModuleList):
        experts.gate_up_proj  (num_experts, 2*moe_inter, hidden)
        experts.down_proj     (num_experts, hidden,     moe_inter)
    so we slice along dim 0 to get each expert's weights.
    """
    layers = _get_decoder_layers(model)
    scores_per_layer = {}

    for layer_idx in tqdm(moe_layer_indices, desc="Collecting weight magnitude scores"):
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        experts = moe_block.experts
        n_experts = getattr(experts, "num_experts", None)
        if n_experts is None:
            # Legacy ModuleList fallback
            n_experts = len(experts)

        expert_scores = torch.zeros(n_experts)
        gu = getattr(experts, "gate_up_proj", None)
        dp = getattr(experts, "down_proj", None)

        if gu is not None and dp is not None:
            # Batched layout (Qwen3.5-MoE)
            for eid in range(n_experts):
                sq = gu[eid].float().pow(2).sum() + dp[eid].float().pow(2).sum()
                expert_scores[eid] = sq.sqrt()
        else:
            # Per-expert ModuleList layout (Qwen3-MoE and earlier)
            for eid in range(n_experts):
                expert = experts[eid]
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
        moe_block = get_moe_block(_get_decoder_layers(model)[layer_idx])
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
    """Mean |∂L/∂W| over each expert's parameters.

    Supports both layouts:
      - Qwen3.5-MoE batched: experts.gate_up_proj (N,2I,H) + experts.down_proj (N,H,I)
        → slice grad along dim 0 for per-expert score.
      - Qwen3-MoE / older: experts is ModuleList, each .gate_proj/.up_proj/.down_proj.
    """
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad_(False)

    # Record per-layer expert modules and which layout is used.
    #   batched_specs[lid] = (experts_module, n_experts)      (Qwen3.5 layout)
    #   list_specs[lid]    = [(eid, [params…]), …]            (legacy ModuleList)
    batched_specs = {}
    list_specs = {}

    layers = _get_decoder_layers(model)
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue
        experts = moe_block.experts
        gu = getattr(experts, "gate_up_proj", None)
        dp = getattr(experts, "down_proj", None)
        if gu is not None and dp is not None:
            gu.requires_grad_(True)
            dp.requires_grad_(True)
            batched_specs[layer_idx] = (experts, experts.num_experts)
        else:
            per_expert = []
            for eid in range(len(experts)):
                expert = experts[eid]
                params = list(expert.parameters())
                for p in params:
                    p.requires_grad_(True)
                per_expert.append((eid, params))
            list_specs[layer_idx] = per_expert

    scores_per_layer = {}
    for lid, (_, n) in batched_specs.items():
        scores_per_layer[lid] = torch.zeros(n)
    for lid, per_expert in list_specs.items():
        scores_per_layer[lid] = torch.zeros(len(per_expert))

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting pure gradient scores (experts)"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Batched layout: slice grads along expert dim
        for layer_idx, (experts, n) in batched_specs.items():
            gu_grad = experts.gate_up_proj.grad
            dp_grad = experts.down_proj.grad
            for eid in range(n):
                imp = 0.0
                cnt = 0
                if gu_grad is not None:
                    imp += gu_grad[eid].float().abs().sum().item()
                    cnt += gu_grad[eid].numel()
                if dp_grad is not None:
                    imp += dp_grad[eid].float().abs().sum().item()
                    cnt += dp_grad[eid].numel()
                if cnt > 0:
                    scores_per_layer[layer_idx][eid] += imp / cnt

        # Legacy ModuleList layout
        for layer_idx, per_expert in list_specs.items():
            for idx, (eid, params) in enumerate(per_expert):
                imp = 0.0
                cnt = 0
                for p in params:
                    if p.grad is not None:
                        imp += p.grad.float().abs().sum().item()
                        cnt += p.grad.numel()
                if cnt > 0:
                    scores_per_layer[layer_idx][idx] += imp / cnt

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
    """Prune experts + gate in-place for each MoE layer. Update config accordingly.

    For Qwen3.5-MoE (VL), config fields live under ``model.config.text_config``;
    for pure-text Qwen3/3.5 they live on ``model.config`` directly. We write to
    whichever one exposes num_experts so the saved checkpoint reloads correctly.
    """
    layers = _get_decoder_layers(model)
    config = _get_text_config(model.config)
    num_layers = config.num_hidden_layers

    orig_num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[0]

    summary = []

    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is None:
            continue

        # Determine original expert count: Qwen3.5 uses experts.num_experts
        # (batched), Qwen3/1.5 uses gate.out_features (nn.Linear) or len(experts).
        if hasattr(moe_block.experts, "num_experts"):
            orig = moe_block.experts.num_experts
        elif hasattr(moe_block.gate, "out_features"):
            orig = moe_block.gate.out_features
        else:
            orig = len(moe_block.experts)

        keep_ids = keep_per_layer.get(layer_idx, list(range(orig)))
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

        keep_idx_t = torch.tensor(keep_ids, dtype=torch.long)

        # Branch 1: Qwen3.5-MoE batched expert layout
        if hasattr(moe_block.experts, "gate_up_proj") and hasattr(moe_block.experts, "down_proj"):
            experts = moe_block.experts
            idx_gu = keep_idx_t.to(experts.gate_up_proj.device)
            idx_dp = keep_idx_t.to(experts.down_proj.device)
            experts.gate_up_proj = nn.Parameter(
                experts.gate_up_proj.data.index_select(0, idx_gu).clone()
            )
            experts.down_proj = nn.Parameter(
                experts.down_proj.data.index_select(0, idx_dp).clone()
            )
            experts.num_experts = n_kept

            # Router (Qwen3_5MoeTopKRouter): weight is (num_experts, hidden_dim).
            router = moe_block.gate
            idx_rw = keep_idx_t.to(router.weight.device)
            router.weight = nn.Parameter(
                router.weight.data.index_select(0, idx_rw).clone()
            )
            router.num_experts = n_kept
            router.top_k = min(getattr(router, "top_k", n_kept), n_kept)
        else:
            # Branch 2: legacy Qwen3-MoE / Qwen1.5-MoE ModuleList + nn.Linear gate
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

    # Qwen3.5-MoE's Qwen3_5MoeExperts.__init__ reads config.num_experts as an
    # int to size batched Parameters. If every MoE layer has the same kept
    # count (the common case with layerwise_pruning + fixed preserve_n),
    # collapse to int so save_pretrained → from_pretrained round-trips. Only
    # fall back to the list form when per-layer counts differ (global_pruning).
    valid_counts = [c for c in num_experts_list if isinstance(c, int) and c > 0]
    if valid_counts and all(c == valid_counts[0] for c in valid_counts):
        config.num_experts = valid_counts[0]
    else:
        config.num_experts = num_experts_list
    config.layer_experts_idx = layer_experts_idx

    if valid_counts and hasattr(config, "num_experts_per_tok"):
        config.num_experts_per_tok = min(config.num_experts_per_tok, min(valid_counts))

    config.router_mask_kept = {str(s["layer"]): s["kept"] for s in summary}
    config.router_mask_dropped = {str(s["layer"]): s["dropped"] for s in summary}

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expert drop for Qwen3.5-MoE models")
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
    model = _load_qwen3_5_model(args.model_name_or_path, torch_dtype)
    print(f"Loaded {type(model).__name__} "
          f"(text_config.num_experts={getattr(_get_text_config(model.config), 'num_experts', '?')}, "
          f"num_hidden_layers={getattr(_get_text_config(model.config), 'num_hidden_layers', '?')})")
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

        # Use top-`preserve_n` per layer (sorted by bias, descending) so the
        # resulting checkpoint has a uniform num_experts across all layers —
        # required by the Qwen3.5 vLLM plugin (Qwen3_5MoeExperts.__init__ reads
        # config.num_experts as a single int). The legacy "bias >= threshold"
        # path produced non-uniform per-layer counts which fails to load.
        print(f"Pruning experts: keep top-{args.preserve_n} per layer by bias score")
        print(f"  (legacy --bias_threshold={args.bias_threshold} ignored for uniformity)")
        keep_per_layer = {}
        for lid, bias in bias_per_layer.items():
            n_exp = num_experts_per_layer.get(lid, len(bias))
            n_keep = min(args.preserve_n, n_exp)
            if n_keep >= n_exp:
                keep_per_layer[lid] = list(range(n_exp))
            elif n_keep <= 0:
                keep_per_layer[lid] = []
            else:
                _, top_ids = torch.topk(bias, n_keep, largest=True)
                keep_per_layer[lid] = sorted(top_ids.tolist())

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

    # Transformers' Qwen3.5 batched-expert constructor accepts one scalar
    # expert count.  Refuse to save a checkpoint that cannot be loaded again.
    kept_counts = {len(keep_per_layer[lid]) for lid in moe_layer_indices}
    if len(kept_counts) != 1:
        raise ValueError(
            "Qwen3.5 structural expert removal currently requires the same "
            "number of surviving experts in every MoE layer. Choose a "
            "layerwise method/fixed preserve_n; the requested mask produced "
            f"counts {sorted(kept_counts)}."
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
