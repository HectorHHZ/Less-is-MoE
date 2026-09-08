"""
Standalone expert-drop script for OLMoE-1B-7B.

OLMoE differences from Qwen1.5-MoE:
  - No shared_expert / shared_expert_gate. Only routed experts exist.
  - All decoder layers are MoE (uniform `config.num_experts = 64`).
  - The decoder layer expects `position_embeddings = (cos, sin)` in kwargs
    (computed by the model's rotary_emb before the layer stack), so when we
    forward layers manually we must propagate every kwarg captured at layer 0.

Saving follows the _materialize_router_masks style:
  - Prune experts ModuleList in-place
  - Rebuild gate nn.Linear with kept rows only
  - Update config.num_experts to per-layer list + config.layer_experts_idx

Supported methods (mirrors expert_drop_qwen1.5_moe.py):
    layerwise_pruning              router-softmax mass per expert
    global_pruning                 router-softmax mass, picked globally
    bias_pruning                   simulated aux-free bias accumulation
    weight_magnitude_pruning       keep top-k L2 norm of expert params (data-free)
    low_magnitude_pruning          keep bottom-k L2 norm of expert params (data-free).
                                   On trained MoE the lowest-norm experts are often
                                   the ones whose weights moved furthest from
                                   initialization through frequent routing, so
                                   keeping low-norm tracks "specialized" experts.
    gradient_pruning               Taylor |W * dL/dW| on gate
    pure_gradient_pruning          mean |dL/dW| on gate
    expert_gradient_pruning        Taylor |W * dL/dW| on expert params
    pure_expert_gradient_pruning   mean |dL/dW| on expert params
    pure_expert_gradient_{up,down,gate,up_gate}_pruning
    densemixer_gradient_pruning    DenseMixer straight-through, gate gradient
    random_pruning                 reproducible random subset
"""

import argparse
import json
import os
import random
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def load_calib_data(tokenizer, calib_data_path, n_samples, seq_len):
    """Load JSON / JSONL calibration file and tokenize into batches.

    Supports both JSON array (``[{"text": ...}, ...]``) and JSONL
    (one JSON object per line). Short samples are kept as-is so small
    datasets like HumanEval still produce usable batches.
    """
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

    text_column may be comma-separated (e.g. "prompt,completion") to
    concatenate multiple columns per row.
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
# Benchmark-dataset presets (C-Eval / MATH / CMMLU)
# ---------------------------------------------------------------------------

def _format_ceval(row):
    q = row.get("question", "") or ""
    a = row.get("A", "") or ""
    b = row.get("B", "") or ""
    c = row.get("C", "") or ""
    d = row.get("D", "") or ""
    ans = row.get("answer", "") or ""
    exp = row.get("explanation", "") or ""
    if not q:
        return ""
    text = f"题目：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
    if ans:
        text += f"\n答案：{ans}"
    if exp:
        text += f"\n解析：{exp}"
    return text


def _format_cmmlu(row):
    q = row.get("Question") or row.get("question") or ""
    a = row.get("A", "") or ""
    b = row.get("B", "") or ""
    c = row.get("C", "") or ""
    d = row.get("D", "") or ""
    ans = row.get("Answer") or row.get("answer") or ""
    if not q:
        return ""
    text = f"题目：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
    if ans:
        text += f"\n答案：{ans}"
    return text


def _format_math(row):
    problem = row.get("problem", "") or ""
    solution = row.get("solution", "") or ""
    if not problem:
        return ""
    text = f"Problem: {problem}"
    if solution:
        text += f"\nSolution: {solution}"
    return text


_PRESET_REGISTRY = {
    "ceval": {
        "datasets": ["ceval/ceval-exam"],
        "default_split": "dev",
        "format_fn": _format_ceval,
    },
    "cmmlu": {
        "datasets": ["haonan-li/cmmlu"],
        "default_split": "dev",
        "format_fn": _format_cmmlu,
    },
    "math": {
        "datasets": ["EleutherAI/hendrycks_math", "lighteval/MATH",
                     "hendrycks/competition_math"],
        "default_split": "train",
        "format_fn": _format_math,
    },
}


def _iter_preset_rows(dataset_name, split):
    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [preset] get_dataset_config_names failed: {e}; trying default load")
        configs = []

    if not configs:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        for row in ds:
            yield None, row
        return

    for cfg in configs:
        try:
            ds = load_dataset(dataset_name, cfg, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  [preset] skip config {cfg}: {e}")
            continue
        for row in ds:
            yield cfg, row


def load_calib_data_preset(tokenizer, preset, n_samples, seq_len,
                           split=None, shuffle_seed=None):
    """Load calibration data from a benchmark preset: ceval / math / cmmlu."""
    preset = preset.lower()
    if preset not in _PRESET_REGISTRY:
        raise ValueError(f"Unknown preset '{preset}'. Supported: {list(_PRESET_REGISTRY)}")

    info = _PRESET_REGISTRY[preset]
    split = split or info["default_split"]
    format_fn = info["format_fn"]

    texts = []
    last_err = None
    for dataset_name in info["datasets"]:
        print(f"Loading preset '{preset}' from {dataset_name} (split={split}) ...")
        try:
            local_texts = []
            for _cfg, row in _iter_preset_rows(dataset_name, split):
                t = format_fn(row)
                if t:
                    local_texts.append(t)
            if local_texts:
                texts = local_texts
                print(f"  Extracted {len(texts)} text samples from {dataset_name}")
                break
        except Exception as e:
            last_err = e
            print(f"  [preset] {dataset_name} failed: {e}")

    if not texts:
        raise RuntimeError(
            f"Failed to load any data for preset '{preset}'. Last error: {last_err}"
        )

    if shuffle_seed is not None:
        print(f"  Shuffling preset samples with seed={shuffle_seed}")
        random.Random(shuffle_seed).shuffle(texts)

    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if ids.shape[1] == 0:
            continue
        batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches


# ---------------------------------------------------------------------------
# Identify MoE blocks (OLMoE: every decoder layer is MoE)
# ---------------------------------------------------------------------------

def get_moe_layer_info(model):
    config = model.config
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts  # uniform int (64) for OLMoE-1B-7B

    moe_layer_indices = list(range(num_layers))
    num_experts_per_layer = {i: num_experts for i in range(num_layers)}
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
# Layer-input capture (shared by all hook-based collectors)
# ---------------------------------------------------------------------------

def _capture_layer0_inputs(model, calib_batches):
    """
    Run calibration data through the embed + first-layer dispatch to capture
    the (hidden_states, kwargs) tuple that decoder layer 0 sees. We then forward
    every layer manually using these captured kwargs.

    OLMoE precomputes `position_embeddings = (cos, sin)` via rotary_emb before
    the decoder stack and passes it in kwargs, so capturing **kwargs is enough.
    """
    device = next(model.parameters()).device
    layers = model.model.layers

    inputs = []
    layer0_kwargs_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inputs.append(hidden_states)
            layer0_kwargs_list.append(kwargs)
            raise ValueError  # stop after layer 0

    layers[0] = Catcher(layers[0])
    for batch in calib_batches:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    return inputs, layer0_kwargs_list


# ---------------------------------------------------------------------------
# Routing-mass score collection (layerwise / global)
# ---------------------------------------------------------------------------

class RouterScoreCollector:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.scores = None
        self.nsamples = 0

    def hook_fn(self, module, input, output):
        router_logits = output[1]
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])
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
    layers = model.model.layers
    inputs, layer0_kwargs_list = _capture_layer0_inputs(model, calib_batches)

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
            outputs[j] = layer(inputs[j], **layer0_kwargs_list[j])[0]

        if handle is not None:
            handle.remove()
        if collector is not None:
            scores_per_layer[i] = collector.get_avg_scores()

        inputs, outputs = outputs, inputs

    return scores_per_layer


# ---------------------------------------------------------------------------
# Bias-based score collection (simulates aux-free bias accumulation)
# ---------------------------------------------------------------------------

class BiasCollector:
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
    layers = model.model.layers
    inputs, layer0_kwargs_list = _capture_layer0_inputs(model, calib_batches)
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
                cur_outputs[j] = layers[i](cur_inputs[j], **layer0_kwargs_list[j])[0]

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
# Weight magnitude (data-free)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_weight_magnitude_scores(model, moe_layer_indices):
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
# Gradient-based scoring (gate weights — Taylor and pure)
# ---------------------------------------------------------------------------

def _enable_gate_grads(model, moe_layer_indices):
    for param in model.parameters():
        param.requires_grad_(False)
    gate_params = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is not None:
            moe_block.gate.weight.requires_grad_(True)
            gate_params[layer_idx] = moe_block.gate.weight
    return gate_params


def collect_gradient_scores(model, calib_batches, moe_layer_indices):
    """Taylor importance |W * dL/dW| on gate weights, L2 norm per expert row."""
    device = next(model.parameters()).device
    gate_params = _enable_gate_grads(model, moe_layer_indices)
    scores_per_layer = {lid: torch.zeros(gate_params[lid].shape[0])
                        for lid in gate_params}

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting gradient scores"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for layer_idx, gate_w in gate_params.items():
            if gate_w.grad is None:
                continue
            taylor = (gate_w.data.float() * gate_w.grad.float()).abs()
            scores_per_layer[layer_idx] += taylor.norm(dim=1).detach().cpu()

        model.zero_grad(set_to_none=True)

    n = max(len(calib_batches), 1)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return scores_per_layer


def collect_pure_gradient_scores(model, calib_batches, moe_layer_indices):
    """Pure |dL/dW| on gate weights, mean per expert row."""
    device = next(model.parameters()).device
    gate_params = _enable_gate_grads(model, moe_layer_indices)
    scores_per_layer = {lid: torch.zeros(gate_params[lid].shape[0])
                        for lid in gate_params}

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting pure gradient scores (router)"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for layer_idx, gate_w in gate_params.items():
            if gate_w.grad is None:
                continue
            scores_per_layer[layer_idx] += gate_w.grad.float().abs().mean(dim=1).detach().cpu()

        model.zero_grad(set_to_none=True)

    n = max(len(calib_batches), 1)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return scores_per_layer


# ---------------------------------------------------------------------------
# Gradient-based scoring (expert params — Taylor and pure, full or by proj)
# ---------------------------------------------------------------------------

def _enable_expert_grads(model, moe_layer_indices, projs=None):
    """projs=None => all expert params; otherwise iterable of {gate_proj, up_proj, down_proj}."""
    for param in model.parameters():
        param.requires_grad_(False)
    expert_params = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(model.model.layers[layer_idx])
        if moe_block is None:
            continue
        expert_params[layer_idx] = []
        for eid, expert in enumerate(moe_block.experts):
            if projs is None:
                params = list(expert.parameters())
            else:
                params = []
                for proj_name in projs:
                    proj = getattr(expert, proj_name)
                    params.extend(list(proj.parameters()))
            for p in params:
                p.requires_grad_(True)
            expert_params[layer_idx].append((eid, params))
    return expert_params


def collect_expert_gradient_scores(model, calib_batches, moe_layer_indices):
    """Taylor sum |W*dL/dW| over each expert's full params."""
    device = next(model.parameters()).device
    expert_params = _enable_expert_grads(model, moe_layer_indices)
    scores_per_layer = {lid: torch.zeros(len(es)) for lid, es in expert_params.items()}

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting expert gradient scores"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for layer_idx, experts in expert_params.items():
            for idx, (eid, params) in enumerate(experts):
                importance = 0.0
                for p in params:
                    if p.grad is not None:
                        importance += (p.data.float() * p.grad.float()).abs().sum().item()
                scores_per_layer[layer_idx][idx] += importance

        model.zero_grad(set_to_none=True)

    n = max(len(calib_batches), 1)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return scores_per_layer


def collect_pure_expert_gradient_scores(model, calib_batches, moe_layer_indices,
                                        projs=None):
    """Pure mean |dL/dW| over each expert's params (full or chosen projs)."""
    device = next(model.parameters()).device
    expert_params = _enable_expert_grads(model, moe_layer_indices, projs=projs)
    scores_per_layer = {lid: torch.zeros(len(es)) for lid, es in expert_params.items()}

    desc = ("Collecting pure expert gradient scores"
            if projs is None
            else f"Collecting pure expert gradient scores ({'+'.join(sorted(projs))})")

    model.train()
    for batch in tqdm(calib_batches, desc=desc):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

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

    n = max(len(calib_batches), 1)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return scores_per_layer


# ---------------------------------------------------------------------------
# DenseMixer gradient (OLMoE has no shared_expert)
# ---------------------------------------------------------------------------

def _densemixer_olmoe_forward(self, hidden_states: torch.Tensor):
    """DenseMixer forward for OlmoeSparseMoeBlock.

    Forward value  = sparse top-k output (same as eager OLMoE).
    Backward grad  = dense (all experts via full softmax routing) output.
    """
    batch_size, seq_length, hidden_dim = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    flat_hidden = hidden_states.view(-1, hidden_dim)
    N_tokens = flat_hidden.size(0)

    router_logits = self.gate(flat_hidden).to(dtype=dtype)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    routing_weights_topk, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    routing_weights_topk = routing_weights_topk.to(dtype=dtype)
    if self.norm_topk_prob:
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
            dim=-1, keepdim=True
        )
        routing_weights_topk = routing_weights_topk.to(dtype=dtype)
    routing_weights = routing_weights.to(dtype=dtype)

    dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
    sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        expert_output = expert_layer(flat_hidden).to(dtype=dtype)

        activation_mask = (
            (selected_experts == expert_idx)
            .any(dim=1).float().unsqueeze(-1).to(dtype)
        )
        if expert_output.requires_grad:
            expert_output.register_hook(
                lambda grad, m=activation_mask: grad * m
            )

        weight_full = routing_weights[:, expert_idx].unsqueeze(-1)
        dense_outputs = dense_outputs + expert_output * weight_full

        matches = selected_experts == expert_idx
        if matches.any():
            token_indices, k_indices = torch.where(matches)
            w_topk = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)
            sparse_outputs[token_indices] = (
                sparse_outputs[token_indices] + expert_output[token_indices] * w_topk
            )

    final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
    final_output = final_flat.to(dtype=dtype).view(batch_size, seq_length, hidden_dim)
    return final_output, router_logits


def collect_densemixer_gradient_scores(model, calib_batches, moe_layer_indices):
    device = next(model.parameters()).device
    layers = model.model.layers

    orig_forwards = {}
    for layer_idx in moe_layer_indices:
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is not None:
            orig_forwards[layer_idx] = moe_block.forward
            moe_block.forward = types.MethodType(_densemixer_olmoe_forward, moe_block)

    gate_params = _enable_gate_grads(model, moe_layer_indices)
    scores_per_layer = {lid: torch.zeros(gate_params[lid].shape[0])
                        for lid in gate_params}

    model.train()
    for batch in tqdm(calib_batches, desc="Collecting DenseMixer gradient scores"):
        input_ids = batch.to(device)
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()

        for layer_idx, gate_w in gate_params.items():
            if gate_w.grad is None:
                continue
            scores_per_layer[layer_idx] += gate_w.grad.float().abs().mean(dim=1).detach().cpu()

        model.zero_grad(set_to_none=True)

    n = max(len(calib_batches), 1)
    for lid in scores_per_layer:
        scores_per_layer[lid] /= n

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for layer_idx, orig_fwd in orig_forwards.items():
        moe_block = get_moe_block(layers[layer_idx])
        if moe_block is not None:
            moe_block.forward = orig_fwd

    return scores_per_layer


# ---------------------------------------------------------------------------
# Pruning decision
# ---------------------------------------------------------------------------

def decide_experts_to_keep(scores_per_layer, num_experts_per_layer, preserve_n, method):
    """Returns dict: {layer_idx: sorted list of expert indices to keep}."""
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
    layers = model.model.layers
    config = model.config
    num_layers = config.num_hidden_layers
    orig_num_experts = (config.num_experts
                        if isinstance(config.num_experts, int)
                        else config.num_experts[0])

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

        # 1. Prune experts ModuleList
        moe_block.experts = nn.ModuleList([moe_block.experts[i] for i in keep_ids])

        # 2. Rebuild gate with kept rows only
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

        # 3. Module-level attrs
        moe_block.num_experts = n_kept
        moe_block.top_k = min(getattr(moe_block, "top_k", n_kept), n_kept)

        print(f"  Layer {layer_idx}: {orig} -> {n_kept} experts (dropped {len(drop_ids)})")

    # ---- Update config ----
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
    parser = argparse.ArgumentParser(description="Expert drop for OLMoE-1B-7B")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preserve_n", type=int, default=16,
                        help="Number of experts to preserve "
                             "(per-layer for layerwise, average for global)")
    parser.add_argument("--method", type=str, default="global_pruning",
                        choices=["global_pruning", "layerwise_pruning",
                                 "bias_pruning",
                                 "weight_magnitude_pruning",
                                 "low_magnitude_pruning",
                                 "gradient_pruning", "expert_gradient_pruning",
                                 "pure_gradient_pruning",
                                 "pure_expert_gradient_pruning",
                                 "pure_expert_gradient_up_pruning",
                                 "pure_expert_gradient_down_pruning",
                                 "pure_expert_gradient_gate_pruning",
                                 "pure_expert_gradient_up_gate_pruning",
                                 "densemixer_gradient_pruning",
                                 "random_pruning"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Path to local JSON/JSONL calibration file")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset (e.g. RoxanneWsyw/gsm)")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="prompt",
                        help="Column(s); comma-separated to concatenate")
    parser.add_argument("--calib_preset", type=str, default=None,
                        choices=["ceval", "math", "cmmlu"])
    parser.add_argument("--calib_preset_split", type=str, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    # Bias pruning
    parser.add_argument("--bias_update_speed", type=float, default=4e-2)
    parser.add_argument("--bias_criterion", type=str, default="median",
                        choices=["mean", "median"])
    parser.add_argument("--bias_clip", type=float, default=10.0)
    parser.add_argument("--bias_epochs", type=int, default=2)
    parser.add_argument("--bias_threshold", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--mask_file", type=str, default=None,
                        help="Pre-computed expert_drop_summary.json; "
                             "skips calibration and applies kept experts directly.")
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

    # ---- Calibration data ----
    data_free_methods = {"random_pruning", "weight_magnitude_pruning",
                         "low_magnitude_pruning"}
    if args.mask_file:
        print(f"Mask file provided ({args.mask_file}); skipping calibration data.")
        calib_batches = []
    elif args.method in data_free_methods:
        print(f"Method {args.method} is data-free; skipping calibration data.")
        calib_batches = []
    elif args.calib_preset:
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
            raise ValueError(
                "Provide one of --calib_preset / --dataset_name / --calib_data "
                f"for method {args.method!r}."
            )
        print(f"Loading calibration data from {args.calib_data} ...")
        calib_batches = load_calib_data(
            tokenizer, args.calib_data, args.n_samples, args.seq_len
        )
    if calib_batches:
        print(f"Loaded {len(calib_batches)} calibration samples (seq_len={args.seq_len})")

    # ---- MoE info ----
    moe_layer_indices, num_experts_per_layer = get_moe_layer_info(model)
    print(f"MoE layers: {len(moe_layer_indices)} layers, experts per layer: "
          f"{list(num_experts_per_layer.values())[:5]}...")

    # ---- Score / decide ----
    if args.mask_file:
        print(f"Loading pre-computed mask from {args.mask_file} ...")
        with open(args.mask_file, "r") as f:
            mask_data = json.load(f)
        keep_per_layer = {int(e["layer"]): list(e["kept"]) for e in mask_data["per_layer"]}
        print(f"Loaded mask for {len(keep_per_layer)} layers "
              f"(method={mask_data.get('method', 'unknown')}, "
              f"preserve_n={mask_data.get('preserve_n', 'unknown')})")

    elif args.method == "random_pruning":
        print(f"Random pruning, seed={args.random_seed}, preserve_n={args.preserve_n}")
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
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "low_magnitude_pruning":
        print("Collecting weight magnitude scores (data-free, low-magnitude keep) ...")
        scores_per_layer = collect_weight_magnitude_scores(model, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} weight magnitude: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")
        # Keep experts with the *smallest* L2 norm: in trained MoE these are
        # typically the most specialized / most-routed experts (their weights
        # moved furthest from init scale). Equivalent to layerwise topk on
        # the negated scores.
        scores_neg = {lid: -s for lid, s in scores_per_layer.items()}
        keep_per_layer = decide_experts_to_keep(
            scores_neg, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "gradient_pruning":
        print("Collecting Taylor gradient scores (gate weights) ...")
        scores_per_layer = collect_gradient_scores(model, calib_batches, moe_layer_indices)
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "expert_gradient_pruning":
        print("Collecting Taylor gradient scores (expert params) ...")
        scores_per_layer = collect_expert_gradient_scores(model, calib_batches, moe_layer_indices)
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "pure_gradient_pruning":
        print("Collecting pure gradient scores (gate weights) ...")
        scores_per_layer = collect_pure_gradient_scores(model, calib_batches, moe_layer_indices)
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "pure_expert_gradient_pruning":
        print("Collecting pure gradient scores (expert params) ...")
        scores_per_layer = collect_pure_expert_gradient_scores(
            model, calib_batches, moe_layer_indices, projs=None
        )
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method in {
        "pure_expert_gradient_up_pruning",
        "pure_expert_gradient_down_pruning",
        "pure_expert_gradient_gate_pruning",
        "pure_expert_gradient_up_gate_pruning",
    }:
        proj_map = {
            "pure_expert_gradient_up_pruning": ("up_proj",),
            "pure_expert_gradient_down_pruning": ("down_proj",),
            "pure_expert_gradient_gate_pruning": ("gate_proj",),
            "pure_expert_gradient_up_gate_pruning": ("up_proj", "gate_proj"),
        }
        projs = proj_map[args.method]
        print(f"Collecting pure expert gradient ({'+'.join(projs)}) ...")
        scores_per_layer = collect_pure_expert_gradient_scores(
            model, calib_batches, moe_layer_indices, projs=projs
        )
        keep_per_layer = decide_experts_to_keep(
            scores_per_layer, num_experts_per_layer, args.preserve_n, "layerwise_pruning"
        )

    elif args.method == "densemixer_gradient_pruning":
        print("Collecting DenseMixer gradient scores ...")
        scores_per_layer = collect_densemixer_gradient_scores(model, calib_batches, moe_layer_indices)
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

        for lid in sorted(bias_per_layer.keys())[:3]:
            b = bias_per_layer[lid]
            print(f"  Layer {lid} final bias: min={b.min():.4f}, max={b.max():.4f}, "
                  f"#neg={int((b < 0).sum())}/{len(b)}")

        os.makedirs(args.output_dir, exist_ok=True)
        bias_states = {f"model.layers.{lid}.mlp.gate":
                       {"bias_values": bias.tolist(), "num_experts": len(bias)}
                       for lid, bias in bias_per_layer.items()}
        bias_out_path = os.path.join(args.output_dir, "moe_bias_states.json")
        with open(bias_out_path, "w") as f:
            json.dump({"metadata": {"method": "offline_bias_simulation",
                                    "bias_update_speed": args.bias_update_speed,
                                    "bias_criterion": args.bias_criterion,
                                    "bias_clip": args.bias_clip,
                                    "bias_epochs": args.bias_epochs,
                                    "n_samples": args.n_samples},
                       "moe_bias_states": bias_states}, f, indent=2)
        print(f"Bias states saved to {bias_out_path}")

        keep_per_layer = {}
        if args.preserve_n and args.preserve_n > 0:
            # Ratio-controlled: keep the top-`preserve_n` experts by final bias
            # score (enables fixed-compression-ratio sweeps at 25/50/75%).
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
            # Legacy threshold mode (kept for backward compatibility).
            for lid, bias in bias_per_layer.items():
                keep_ids = [i for i in range(len(bias)) if bias[i].item() >= args.bias_threshold]
                min_keep = num_experts_per_layer.get(lid, 1)
                min_keep = min(getattr(model.config, "num_experts_per_tok", 1), min_keep)
                if len(keep_ids) < min_keep:
                    _, top_ids = torch.topk(bias, min_keep, largest=True)
                    keep_ids = top_ids.tolist()
                keep_per_layer[lid] = sorted(keep_ids)

    else:
        # global_pruning / layerwise_pruning (router softmax mass)
        print("Collecting routing scores ...")
        scores_per_layer = collect_routing_scores(model, calib_batches, moe_layer_indices)
        for lid in sorted(scores_per_layer.keys())[:3]:
            s = scores_per_layer[lid]
            print(f"  Layer {lid} scores: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")
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
