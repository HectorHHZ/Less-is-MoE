"""Optional Transformers fallback for environments where vLLM is unavailable.

The paper results use the dedicated vLLM protocol entry points. This module is
kept as a lightweight zero-shot compatibility path and is not the canonical
paper evaluator.
"""
import argparse
import json
import logging
import os
import re
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

from less_is_moe.model_patches.registry import apply_hf_patch, detect_model_family


i_prompt = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

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


def extract_answer_number(dataset: str, sentence: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
        if not pred:
            return float("inf")
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(f"not support dataset: {dataset}")
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        return pred_answers[0]
    return ""


def extract_commonsense_answer(dataset: str, sentence: str) -> str:
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


def map_dtype(dtype: str):
    if dtype in ["float16", "half", "fp16"]:
        return torch.float16
    if dtype in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if dtype in ["float32", "fp32"]:
        return torch.float32
    return None  # auto


def main(args):
    set_random_seed(args.seed)

    family = detect_model_family(
        args.model_name_or_path,
        validate_for_hf=True,
    )
    if family not in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe"}:
        raise ValueError(
            "The Hugging Face fallback is a zero-shot Qwen evaluator. "
            "Use vllm_olmoe_multishot for OLMoE; detected "
            f"{family!r}."
        )

    # Patch Transformers with pruned classes (skip for base models).
    if not args.base_model:
        apply_hf_patch(family)
        logger.info("[Patch] enabled Less-is-MoE HF patch for %s.", family)
    else:
        print("[Patch] base_model=True, skipping Less-is-MoE model patch.")

    dataset_lower = args.dataset.lower()

    # Load test data
    raw_data = load_data(args.data_path)
    if args.limit is not None:
        raw_data = raw_data[: args.limit]

    if is_esft_dataset(dataset_lower):
        t_test_data = raw_data
        prompts: List[str] = [
            ex.get("prompt") or ex.get("instruction") or ex.get("input") or ""
            for ex in t_test_data
        ]
        print(f"[Prompt] ESFT dataset detected: {args.dataset}")
    elif args.base_model:
        t_test_data = raw_data
        prompts = [i_prompt.format_map(example) for example in t_test_data]
        print("[Prompt] base model: using instruction directly, no normalize.")
    else:
        t_test_data = [normalize_example(e) for e in raw_data]
        prompts = [i_prompt.format_map(example) for example in t_test_data]
        print("[Prompt] pruned model: fields normalized.")

    print(f"First prompt example:\n{prompts[0]}")
    print(f"Total prompts: {len(prompts)}")

    # Load model
    torch_dtype = map_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    model_device = next(model.parameters()).device

    # Generation config
    do_sample = args.temperature > 0
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        top_k=40 if do_sample else 0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"Generation config: {gen_config}")

    # Parse stop sequences
    stop_seqs = None
    if args.stop_sequences:
        stop_seqs = [s for s in (seg.strip() for seg in args.stop_sequences.split(",")) if s]

    # Generate
    batch_size = min(args.batch_size, len(prompts))
    all_texts: List[str] = []
    print("Starting generation...")
    start_time = time.time()

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Generating", ncols=100):
        batch_prompts = prompts[i : i + batch_size]
        tok_kwargs = dict(return_tensors="pt", padding=True)
        if args.max_model_len is not None:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = args.max_model_len
        inputs = tokenizer(batch_prompts, **tok_kwargs)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, generation_config=gen_config, return_dict_in_generate=True
            )

        seqs = outputs.sequences
        # Decoder-only generation returns the complete padded input followed by
        # new tokens. With left padding, attention-mask lengths are shorter than
        # that shared padded width, so slicing by each mask sum leaks prompt
        # tokens into shorter examples.
        prompt_width = inputs["input_ids"].shape[1]
        batch_texts = []
        for seq in seqs:
            gen_tokens = seq[prompt_width:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            batch_texts.append(text)
        all_texts.extend(batch_texts)

        if (i // batch_size) % 5 == 0:
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    print(f"Speed: {len(prompts) / elapsed_time:.2f} prompts/second")

    # Score (mirrors the vLLM evaluators)
    save_outputs = []
    correct = 0
    miss = 0.001

    for example, generated_text in tqdm(
        zip(t_test_data, all_texts),
        total=len(t_test_data),
        desc="Evaluating",
    ):
        # Strip stop sequences
        if stop_seqs:
            for s in stop_seqs:
                if s and s in generated_text:
                    generated_text = generated_text.split(s)[0]
                    break
        generated_text = generated_text.rstrip()

        example["raw_output"] = generated_text
        target = (
            example.get("completion")
            if is_esft_dataset(dataset_lower)
            else example.get("answer")
        )

        if is_esft_dataset(dataset_lower):
            predict = generated_text
        elif dataset_lower in COMMONSENSE_TASKS:
            predict = extract_commonsense_answer(dataset_lower, generated_text)
            if not predict:
                predict = fallback_digit_prediction(str(target), generated_text)
            if predict and str(target).strip().lower() == predict.lower():
                correct += 1
        else:
            # Math / letter-choice tasks
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

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = os.path.basename(args.model_name_or_path.rstrip("/"))
    model_tag = model_tag.replace(os.sep, "_")
    output_file = os.path.join(args.output_dir, f"{model_tag}_predictions.jsonl")
    metrics_file = os.path.join(args.output_dir, f"{model_tag}_metrics.json")

    weighted_acc = correct / len(t_test_data) if len(t_test_data) > 0 else 0.0

    print(f"Saving outputs to {output_file}")
    print("=" * 80)
    if is_esft_dataset(dataset_lower):
        print("ESFT dataset: no auto metric (needs external GPT evaluator).")
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
    parser = argparse.ArgumentParser(
        description="Optional Transformers fallback evaluator (non-canonical)."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data (json/jsonl)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. gsm8k, arc-challenge)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model path or ID")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--base_model", action="store_true", help="Skip pruned patch, use stock HF model.")
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "half", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--max_model_len", type=int, default=None, help="Max input length (truncate). None = no truncation.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N examples")
    parser.add_argument(
        "--stop_sequences", type=str, default=None,
        help="Comma-separated stop strings. Example: '###,</answer>'",
    )

    args = parser.parse_args()
    main(args)
