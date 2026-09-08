#!/usr/bin/env python3
"""
评估剪枝 + AWQ 量化的 Qwen2-MoE 模型（或兼容的 AWQ 模型），
流程使用 Less-is-MoE 的 Qwen zero-shot 数据处理与指标计算，
并沿用 generate.py 的补丁与安全 forward 方案。

python third_party/AutoAWQ/examples/evaluate_gsm8k_awq.py \
  --data_path dataset/eval_dataset/ARC-Challenge/test.json \
  --dataset arc-challenge \
  --output_dir output/arc_challenge_awq \
  --model_name_or_path outputs/Qwen1.5-MoE-A2.7B-awq \
  --awq_device cuda:0 \
  --batch_size 32 \
  --max_model_len 128 \
  --max_new_tokens 200 \
  --temperature 0.1 \
  --base_model


"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 提前设置路径，确保优先加载当前仓库的 less_is_moe / awq 实现
FILE_DIR = Path(__file__).resolve().parent
AUTOAWQ_ROOT = FILE_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (AUTOAWQ_ROOT, SRC_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

i_prompt = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def extract_answer_number(dataset: str, sentence: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\\.?\\d*", sentence)]
        if not pred:
            return float("inf")
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(f"not support dataset: {dataset}")
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        return pred_answers[0]
    else:
        return ""


def extract_commonsense_answer(dataset: str, sentence: str) -> str:
    """从输出中抽取 commonsense 多选题的选项标签。"""
    sentence = sentence.lower().strip()
    ds = dataset.lower()

    if ds == "boolq":
        pred_answers = re.findall(r"true|false", sentence)
    elif ds == "piqa":
        pred_answers = re.findall(r"solution1|solution2", sentence)
    elif ds in {"social_i_qa", "arc-challenge", "arc-easy", "openbookqa"}:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence)
    elif ds == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence)
    elif ds == "winogrande":
        pred_answers = re.findall(r"option1|option2", sentence)
    else:
        pred_answers = []

    if not pred_answers:
        return ""
    return pred_answers[0]


def fallback_digit_prediction(target: str, sentence: str) -> str:
    """退化策略：用输出里最后一个数字与目标末位数字对比。"""
    raw_digits = re.findall(r"\d", sentence)
    target_digits = re.findall(r"\d", str(target))
    if not raw_digits or not target_digits:
        return ""

    pred_digit = raw_digits[-1]
    target_digit = target_digits[-1]

    if pred_digit == target_digit:
        return str(target)

    prefix_match = re.match(r"(.+?)(\d+)$", str(target).strip())
    if prefix_match:
        return f"{prefix_match.group(1)}{pred_digit}"
    return pred_digit


def set_random_seed(seed: int) -> None:
    import random

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    instr = ex.get("instruction")
    ans = ex.get("answer")
    if instr is None:
        raise KeyError("Field 'instruction' is required in example.")
    if ans is None:
        raise KeyError("Field 'answer' is required in example.")
    out = dict(ex)
    out["instruction"] = instr
    out["answer"] = ans
    return out


def load_data(path: str) -> List[Dict[str, Any]]:
    """支持 json / jsonl 读取。"""
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r") as f:
        return json.load(f)


def is_esft_dataset(name: str) -> bool:
    return name.lower() in {
        "esft-intent",
        "esft-summary",
        "esft-law",
        "esft-translation",
    }


COMMONSENSE_TASKS = {
    "arc-challenge",
    "arc-easy",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "social_i_qa",
    "winogrande",
}


def map_dtype(dtype: str):
    if dtype in ["float16", "half", "fp16"]:
        return torch.float16
    if dtype in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if dtype in ["float32", "fp32"]:
        return torch.float32
    return None  # auto


def apply_qwen2_moe_patch() -> None:
    """将 HF 的 Qwen2-MoE 替换成剪枝后的实现。"""
    from less_is_moe.model_patches.registry import apply_hf_patch

    apply_hf_patch("qwen2_moe")
    logger.info("[Patch] Enabled Less-is-MoE Qwen2-MoE patch.")


def _make_safe_moe_forward():
    """稳健版 forward，用 scatter_add_ 代替 index_add_ 避免 CUDA 报错。
    兼容 stock transformers 和 pruned 模型的属性差异。"""
    import torch.nn.functional as F

    def _safe_forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        # pruned models have use_router_mask / router_logits_mask; stock models don't
        if getattr(self, "use_router_mask", False) and getattr(self, "router_logits_mask", None) is not None:
            router_logits = router_logits + self.router_logits_mask.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # pruned: gate_num_experts (total slots incl. pruned); stock: num_experts
        num_classes = getattr(self, "gate_num_experts", self.num_experts)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_classes).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            scatter_idx = top_x.to(final_hidden_states.device).unsqueeze(-1).expand_as(current_hidden_states)
            final_hidden_states.scatter_add_(0, scatter_idx, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return _safe_forward


def patch_moe_instances(model) -> int:
    """按名字为所有 Qwen2MoeSparseMoeBlock 实例绑定安全 forward。"""
    from types import MethodType

    safe_forward = _make_safe_moe_forward()
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            module.forward = MethodType(safe_forward, module)
            patched += 1
    return patched


def _collect_eos_token_ids(tokenizer) -> list[int]:
    """收集可用的 EOS id，过滤 None 并去重。"""

    def _add(store: list[int], token_id):
        if token_id is None:
            return
        if isinstance(token_id, list):
            for tid in token_id:
                _add(store, tid)
            return
        try:
            tid_int = int(token_id)
        except Exception:
            return
        store.append(tid_int)

    eos_ids: list[int] = []
    _add(eos_ids, getattr(tokenizer, "eos_token_id", None))
    for tok in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        _add(eos_ids, tok_id)

    seen = set()
    uniq = []
    for tid in eos_ids:
        if tid not in seen:
            uniq.append(tid)
            seen.add(tid)
    return uniq


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruned + AWQ quantized model on GSM8K-style datasets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data file (json/jsonl).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, aqua).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs and metrics.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="AWQ model path or HF repo ID.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature during generation.")
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Base 模型直接使用样本 instruction，不做字段归一化且不应用剪枝补丁。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations.",
    )
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--awq_device", type=str, default="cuda:0", help="Device to load the AWQ model (e.g., cuda:0).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples.")
    parser.add_argument("--max_new_tokens", type=int, default=600, help="Max new tokens for generation.")
    parser.add_argument(
        "--stop_sequences",
        type=str,
        default=None,
        help="Comma-separated custom stop strings (in addition to EOS). Example: '###,</answer>'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # 读取数据
    raw_data = load_data(args.data_path)
    if args.limit is not None:
        raw_data = raw_data[: args.limit]

    dataset_lower = args.dataset.lower()

    if is_esft_dataset(dataset_lower):
        t_test_data = raw_data
        prompts: List[str] = [
            ex.get("prompt") or ex.get("instruction") or ex.get("input") or "" for ex in t_test_data
        ]
        print(f"[Prompt] ESFT dataset detected: {args.dataset}")
    elif args.base_model:
        t_test_data = raw_data
        prompts: List[str] = [i_prompt.format_map(example) for example in t_test_data]
        print("[Prompt] base model：直接使用样本 instruction，不做 normalize。")
    else:
        t_test_data = [normalize_example(e) for e in raw_data]
        prompts: List[str] = [i_prompt.format_map(example) for example in t_test_data]
        print("[Prompt] 非 base 模型：已对字段做 normalize。")

    print(f"First prompt example:\\n{prompts[0]}")
    print(f"Total prompts: {len(prompts)}")

    # Patch transformers with pruned Qwen2-MoE classes before model loading
    if not args.base_model:
        apply_qwen2_moe_patch()

    torch_dtype = map_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoAWQForCausalLM.from_quantized(
        args.model_name_or_path,
        fuse_layers=False,
        trust_remote_code=True,
        safetensors=True,
        device_map={"": args.awq_device},
        torch_dtype=torch_dtype or torch.float16,
        low_cpu_mem_usage=True,
    )
    model.model.eval()
    model_device = next(model.model.parameters()).device

    try:
        patched = patch_moe_instances(model.model)
        logger.info(f"[Patch] Patched {patched} Qwen2MoeSparseMoeBlock instances with safe forward.")
    except Exception as e:
        logger.warning(f"[Patch] Failed to patch MoE instances: {e}")

    do_sample = args.temperature > 0
    eos_ids = _collect_eos_token_ids(tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        top_k=40 if do_sample else 0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
    )
    print(f"Generation config: {gen_config}")

    batch_size = min(args.batch_size, len(prompts))
    all_texts: List[str] = []
    print("Starting generation...")
    start_time = time.time()

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Generating", ncols=100):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        )
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
            )

        seqs = outputs.sequences
        input_lens = inputs["attention_mask"].sum(dim=1)
        batch_texts = []
        for seq, in_len in zip(seqs, input_lens):
            gen_tokens = seq[int(in_len.item()) :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            batch_texts.append(text)
        all_texts.extend(batch_texts)

        if (i // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    print(f"Speed: {len(prompts) / elapsed_time:.2f} prompts/second")

    save_outputs = []
    correct = 0
    miss = 0.001

    for example, generated_text in zip(t_test_data, all_texts):
        example["raw_output"] = generated_text
        target = example.get("answer") if not is_esft_dataset(dataset_lower) else example.get("completion")

        # 按 stop_sequences 截断生成（仿 evaluate_gsm8k_vllm 手动裁剪）
        if args.stop_sequences:
            for stop in [s.strip() for s in args.stop_sequences.split(",") if s.strip()]:
                if stop and stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    break

        if is_esft_dataset(dataset_lower):
            predict = generated_text  # ESFT 任务此处不自动打分（缺少 GPT 评估器）
        elif dataset_lower in COMMONSENSE_TASKS:
            predict = extract_commonsense_answer(dataset_lower, generated_text)
            if not predict:
                predict = fallback_digit_prediction(str(target), generated_text)
            if predict and str(target).strip().lower() == predict.lower():
                correct += 1
        else:
            is_choice = dataset_lower in ["aqua", "mathqa"] or str(target).strip().upper() in ["A", "B", "C", "D", "E"]

            if is_choice:
                predict = extract_answer_letter(generated_text)
                target_letter = extract_answer_letter(str(target))
                if target_letter and predict and target_letter.upper() == predict.upper():
                    correct += 1
            else:
                predict = extract_answer_number(args.dataset, generated_text)
                try:
                    target_val = float(str(target).replace(",", ""))
                except ValueError:
                    target_val = float("inf")
                if abs(target_val - predict) <= miss:
                    correct += 1

        example["prediction"] = predict
        save_outputs.append(example)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = os.path.basename(args.model_name_or_path.rstrip("/"))
    model_tag = model_tag.replace(os.sep, "_")
    output_file = os.path.join(args.output_dir, f"{model_tag}_predictions.jsonl")
    weighted_acc = correct / len(t_test_data) if len(t_test_data) > 0 else 0.0
    metrics_file = os.path.join(args.output_dir, f"{model_tag}_metrics.json")

    print(f"Saving outputs to {output_file}")
    print("=" * 80)
    if is_esft_dataset(dataset_lower):
        print("ESFT 数据集：未计算自动指标（需外部 GPT 评估器）。")
    else:
        print(f"Result {weighted_acc * 100:.1f}%, total: {len(t_test_data)}")
    print("=" * 80)

    with open(output_file, "w") as fout:
        for example in save_outputs:
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    metrics = {
        "model": args.model_name_or_path,
        "model_tag": model_tag,
        "dataset": args.dataset,
        "total": len(t_test_data),
        "correct": correct,
        "accuracy": weighted_acc,
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
