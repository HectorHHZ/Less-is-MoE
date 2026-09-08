# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Run from the repository root with ``scripts/train/pruned.sh`` and one of
the editable templates under ``recipes/sft/pruned``. This entry point loads
an already-pruned checkpoint; additional online router pruning is opt-in.
"""

import json
import logging
import os
import gc
import sys
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datasets
import torch
import transformers
from less_is_moe.training.trainer import (
    RouterMaskCallback,
    CustomSFTTrainer as _BaseCustomSFTTrainer,
    _format_cuda_memory_stats,
)
from datasets import load_dataset
from transformers import set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from less_is_moe.training.configs import SFTConfig
from less_is_moe.training.utils import get_tokenizer
from less_is_moe.training.utils.callbacks import get_callbacks
from less_is_moe.training.utils.wandb_logging import init_wandb_training
from less_is_moe.model_patches.registry import (
    apply_hf_patch,
    detect_model_family,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


from trl.trainer.utils import (
    DataCollatorForChatML,
    pad,
    )

try:
    import yaml
except Exception:
    yaml = None

logger = logging.getLogger(__name__)

# torch.cuda.memory._record_memory_history()  # Disabled: causes OOM / hang at step 2 under multi-GPU


_MOE_BLOCK_NAMES = {"Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock"}


def _normalize_manual_mask(raw: Dict[Any, Any]) -> Optional[Dict[int, List[int]]]:
    """Convert raw dict/list values into {layer_idx: [expert_ids]}."""
    if raw is None:
        return None
    normalized: Dict[int, List[int]] = {}
    for key, value in raw.items():
        try:
            layer_idx = int(str(key).lstrip("L").strip())
        except Exception:
            continue
        if value is None:
            continue
        entries = value
        if isinstance(entries, str):
            entries = [p for p in entries.replace(" ", "").split(",") if p != ""]
        if isinstance(entries, (int, float)):
            entries = [int(entries)]
        if isinstance(entries, (list, tuple, set)):
            cleaned = sorted({int(v) for v in entries if str(v).strip() != ""})
            if cleaned:
                normalized[layer_idx] = cleaned
    return normalized or None


def _parse_manual_router_mask(mask_spec: Optional[str]) -> Optional[Dict[int, List[int]]]:
    """
    Parse manual mask spec. Accepts:
    - Path to JSON/YAML with {layer_idx: [expert_ids]} or list-of-lists (layer order).
    - Inline string like "0:1,3;2:0".
    """
    if mask_spec is None:
        return None
    spec = mask_spec.strip()
    if not spec:
        return None

    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            content = f.read()
        data = None
        if spec.endswith((".yml", ".yaml")) and yaml is not None:
            try:
                data = yaml.safe_load(content)
            except Exception:
                data = None
        if data is None:
            try:
                data = json.loads(content)
            except Exception:
                data = None
        if data is None and yaml is not None:
            try:
                data = yaml.safe_load(content)
            except Exception:
                data = None
        if isinstance(data, list):
            data = {idx: item for idx, item in enumerate(data)}
        if data is None or not isinstance(data, dict):
            raise ValueError(f"router_manual_mask file must contain a dict or list, got {type(data)}")
        normalized = _normalize_manual_mask(data)
        if normalized is None:
            raise ValueError(f"router_manual_mask file '{spec}' parsed but empty.")
        return normalized

    inline_map: Dict[int, List[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        layer_part, sep, expert_part = chunk.partition(":")
        if not sep:
            continue
        try:
            layer_idx = int(layer_part.lstrip("L").strip())
        except Exception:
            continue
        experts = [e for e in expert_part.split(",") if e.strip() != ""]
        inline_map[layer_idx] = [int(e.strip()) for e in experts]
    return _normalize_manual_mask(inline_map)


def _build_fixed_router_mask_fn(mask_map: Dict[int, List[int]]):
    """Create a mask_fn that returns tensors with -inf for disabled experts."""
    cleaned = {int(k): sorted(set(v)) for k, v in mask_map.items() if v}
    cached: Optional[Dict[int, torch.Tensor]] = None
    warned_layers = set()

    def _fn(model=None, **_):
        nonlocal cached
        if cached is not None:
            return cached
        if model is None:
            return None
        masks: Dict[int, torch.Tensor] = {}
        for module in getattr(model, "modules")():
            if module.__class__.__name__ not in _MOE_BLOCK_NAMES:
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None or layer_idx not in cleaned:
                continue
            num_exp = int(module.gate.out_features)
            target_ids = [e for e in cleaned[layer_idx] if 0 <= e < num_exp]
            dropped = set(cleaned[layer_idx]) - set(target_ids)
            if dropped and layer_idx not in warned_layers:
                logger.warning(
                    f"[RouterMask] Layer {layer_idx} has expert ids out of range (ignored): {sorted(dropped)}"
                )
                warned_layers.add(layer_idx)
            if not target_ids:
                continue
            mask = torch.zeros((num_exp,), device=module.gate.weight.device)
            mask[target_ids] = float("-inf")
            masks[layer_idx] = mask
        cached = masks or None
        return cached

    return _fn


def _apply_router_mask_to_model(model, mask: Dict[int, torch.Tensor]) -> None:
    """Apply a precomputed mask dict to all MoE blocks."""
    if mask is None:
        return
    unwrapped = getattr(model, "module", model)
    for module in unwrapped.modules():
        if not hasattr(module, "_set_router_logits_mask"):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        target = mask.get(layer_idx) if isinstance(mask, dict) else mask
        if target is None:
            continue
        module._set_router_logits_mask(target.to(module.gate.weight.device))


def _summarize_mask(mask: Optional[Dict[int, torch.Tensor]]) -> str:
    if mask is None:
        return "none"
    parts = []
    for layer_idx, tensor in sorted(mask.items()):
        if tensor is None:
            continue
        disabled = torch.isinf(tensor.detach().cpu()).nonzero(as_tuple=False).flatten().tolist()
        parts.append(f"L{layer_idx}:{disabled}")
    return "; ".join(parts) if parts else "none"


@dataclass
class FixedLengthDataCollatorForChatML(DataCollatorForChatML):
    """ChatML collator that pads every sample to max_length."""

    def _pad_to_length(self, seq, length, pad_value):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        if len(seq) >= length:
            return seq[:length]
        pad_len = length - len(seq)
        pad_chunk = [pad_value] * pad_len
        return pad_chunk + list(seq) if self.tokenizer.padding_side == "left" else list(seq) + pad_chunk

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                msg_input_ids = tokenized_message["input_ids"]
                msg_attention_mask = tokenized_message["attention_mask"]
            else:
                msg_input_ids = example["input_ids"]
                msg_attention_mask = example["attention_mask"]

            if isinstance(msg_input_ids, torch.Tensor):
                msg_input_ids = msg_input_ids.tolist()
            if isinstance(msg_attention_mask, torch.Tensor):
                msg_attention_mask = msg_attention_mask.tolist()

            # Truncate to max_length before padding.
            if len(msg_input_ids) > self.max_length:
                msg_input_ids = msg_input_ids[: self.max_length]
                msg_attention_mask = msg_attention_mask[: self.max_length]

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(msg_input_ids),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )
            prompt_ids = tokenized_prompt["input_ids"]
            prompt_attn = tokenized_prompt["attention_mask"]

            completion_start_idx = len(prompt_ids)
            label = [self.ignore_index] * len(msg_input_ids)
            label[completion_start_idx:] = msg_input_ids[completion_start_idx:]

            msg_input_ids = self._pad_to_length(msg_input_ids, self.max_length, self.tokenizer.pad_token_id)
            msg_attention_mask = self._pad_to_length(msg_attention_mask, self.max_length, 0)
            label = self._pad_to_length(label, self.max_length, self.ignore_index)

            input_ids.append(torch.tensor(msg_input_ids, dtype=torch.long))
            attention_mask.append(torch.tensor(msg_attention_mask, dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))
            prompts_input_ids.append(torch.tensor(prompt_ids, dtype=torch.long))
            prompt_attention_mask.append(torch.tensor(prompt_attn, dtype=torch.long))

        input_ids = pad(input_ids, padding_side=self.tokenizer.padding_side, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side=self.tokenizer.padding_side, padding_value=0)
        labels = pad(labels, padding_side=self.tokenizer.padding_side, padding_value=self.ignore_index)

        prompts_input_ids = pad(
            prompts_input_ids,
            padding_side=self.tokenizer.padding_side,
            padding_value=self.tokenizer.pad_token_id,
        )
        prompt_attention_mask = pad(
            prompt_attention_mask,
            padding_side=self.tokenizer.padding_side,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


class CustomSFTTrainer(_BaseCustomSFTTrainer):
    def _prepare_dataset(self, dataset, *args):
        # Some datasets come with a raw "messages" column; others are already tokenized.
        if "messages" in dataset.column_names:
            dataset = dataset.add_column("_messages", dataset["messages"])
            dataset = super()._prepare_dataset(dataset, *args)
            # Only rename back if the temp column exists (avoid crashes on tokenized datasets).
            if "_messages" in dataset.column_names:
                dataset = dataset.rename_column("_messages", "messages")
        else:
            dataset = super()._prepare_dataset(dataset, *args)
        return dataset


class MemLogCallback(TrainerCallback):
    """Log GPU memory right after training starts (model already on device)."""

    def __init__(self) -> None:
        self._printed = False

    def _is_main(self, args) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return False
        return getattr(args, "local_rank", -1) in (-1, 0)

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if self._printed:
            return
        try:
            if model is not None and getattr(model, "_student_device_logged", False):
                self._printed = True
                return
            if model is not None and getattr(model, "_mem_after_device_logged", False):
                self._printed = True
                return
            if self._is_main(args):
                print(f"[Student] Memory after device placement: {_format_cuda_memory_stats()}")
            self._printed = True
        except Exception:
            pass


class StepMemCallback(TrainerCallback):
    """Log memory on every training step end (main process only)."""

    def _is_main(self, args) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return False
        return getattr(args, "local_rank", -1) in (-1, 0)

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if not self._is_main(args):
            return
        try:
            step = getattr(state, "global_step", None)
            print(f"[MemDebug] step_end step={step if step is not None else 'N/A'}: {_format_cuda_memory_stats()}")
        except Exception:
            pass




def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    log_level = training_args.get_process_log_level()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Ensure both stdout and training.log receive all messages on every process.
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, "training.log")),
        ],
        force=True,  # override any handlers installed by accelerate/transformers
    )
    logger.setLevel(log_level)


    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Install the matching pruning-aware HF classes before AutoModel is loaded.
    # Detection comes from config metadata, so renamed/local checkpoints work too.
    model_name_lower = str(model_args.model_name_or_path).lower()
    model_family = detect_model_family(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        validate_for_hf=True,
    )
    is_qwen_family = model_family in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe"}
    apply_hf_patch(model_family)
    logger.info("[Patch] Enabled Less-is-MoE Hugging Face patch for %s.", model_family)

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets, from hugging face
    ################
    if script_args.dataset_name == "RoxanneWsyw/gsm":
        from datasets import DatasetDict
        dataset = DatasetDict({"train": load_dataset(script_args.dataset_name, data_files="train.jsonl", split="train")})
    elif script_args.dataset_name in ("hendrycks_math", "EleutherAI/hendrycks_math"):
        # Handled below in the conversion block; load is deferred there.
        dataset = None
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    print(f"script_args.dataset_name: {script_args.dataset_name}")

    if (script_args.dataset_name == "lmms-lab/Math10K") or (script_args.dataset_name == "HectorHe/math7k") or (script_args.dataset_name == "HectorHe/math14k"):
        # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        # dataset[script_args.dataset_train_split]
        # tune gsm8k scripts
        def convert_to_messages_format(example):
            messages = [
                {"content": example["instruction"], "role": "user"},
                {"content": f"<think>{example['output']}</think>\\boxed{{{example['answer']}}}", "role": "assistant"}
            ]
            return {"messages": messages}

        # Apply the conversion to both train and test splits
        dataset = dataset.map(convert_to_messages_format)
        # print(f"dataset length: {len(dataset[script_args.dataset_train_split])}")
        # print(f"first ten examples: {dataset[script_args.dataset_train_split][:10]}")
        # exit()
        # Optional: Sort by message length and filter if needed (similar to OpenR1-Math-220k)
        # train_dataset = dataset[script_args.dataset_train_split]
        # train_dataset = train_dataset.map(lambda x: {"message_length": len(str(x["messages"]))})
        # train_dataset = train_dataset.sort("message_length")
        # Uncomment if you want to limit the dataset size
        # train_dataset = train_dataset.select(range(45000)).shuffle(seed=42)
        # dataset[script_args.dataset_train_split] = train_dataset
    elif script_args.dataset_name == "openai/gsm8k":
        def convert_to_messages_format_gsm8k(example):
            # example["answer"] is a single string containing both reasoning and final answer
            # split by '####'
            if "####" in example["answer"]:
                think_part, answer_part = example["answer"].split("####", 1)
                think_part = think_part.strip()
                answer_part = answer_part.strip()
            else:
                # fallback: no explicit separator
                think_part = example["answer"].strip()
                answer_part = ""

            messages = [
                {
                    "role": "user",
                    "content": example["question"]
                },
                {
                    "role": "assistant",
                    "content": f"<think>{think_part}</think><answer>{answer_part}</answer>"
                }
            ]
            return {"messages": messages}

        dataset = dataset.map(convert_to_messages_format_gsm8k)

    elif script_args.dataset_name == "RoxanneWsyw/gsm":
        def convert_to_messages_format_roxanne_gsm(example):
            completion = example["completion"]
            if "####" in completion:
                think_part, answer_part = completion.split("####", 1)
                think_part = think_part.strip()
                answer_part = answer_part.strip()
            else:
                think_part = completion.strip()
                answer_part = ""

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": f"<think>{think_part}</think><answer>{answer_part}</answer>"},
            ]
            return {"messages": messages}

        dataset = dataset.map(convert_to_messages_format_roxanne_gsm)
        drop_cols = [c for c in ("prompt", "completion") if c in dataset["train"].column_names]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

    elif script_args.dataset_name == "fw407/Commonsense-15K":
        def convert_commonsense(example):
            return {
                "messages": [
                    {"role": "user", "content": example.get("instruction", "")},
                    {"role": "assistant", "content": example.get("output", "")},
                ]
            }

        dataset = dataset.map(convert_commonsense)

    elif script_args.dataset_name == "RoxanneWsyw/ESFT-translation" or script_args.dataset_name == "RoxanneWsyw/ESFT-summary" or script_args.dataset_name == "RoxanneWsyw/ESFT-law" or script_args.dataset_name == "RoxanneWsyw/ESFT-intent" or script_args.dataset_name == "RoxanneWsyw/MBPP":
        def convert_esft(example):
            return {
                "messages": [
                    {"role": "user", "content": example.get("prompt", "")},
                    {"role": "assistant", "content": example.get("completion", "")},
                ]
            }

        dataset = dataset.map(convert_esft)
        # Remove original prompt/completion columns to avoid TRL misclassifying the dataset as non-conversational
        drop_cols = [c for c in ("prompt", "completion") if c in dataset["train"].column_names]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

    elif script_args.dataset_name == "cais/mmlu":
        # MMLU: question (str), choices (list[str]), answer (int 0-3)
        _letters = ["A", "B", "C", "D"]

        def convert_mmlu(example):
            q = example.get("question", "")
            choices = example.get("choices", []) or []
            ans_idx = example.get("answer", 0)
            choice_block = "\n".join(
                f"{_letters[i]}. {c}" for i, c in enumerate(choices[:4])
            )
            ans_letter = _letters[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < 4 else "A"
            ans_text = choices[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < len(choices) else ""
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"The following is a multiple choice question. "
                            f"Answer with the letter and explanation.\n\n"
                            f"Question: {q}\n{choice_block}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"The answer is {ans_letter}. {ans_text}",
                    },
                ]
            }

        dataset = dataset.map(convert_mmlu)
        train_split = script_args.dataset_train_split  # "auxiliary_train" for cais/mmlu
        drop_cols = [c for c in ("question", "subject", "choices", "answer") if c in dataset[train_split].column_names]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

    elif script_args.dataset_name in ("hendrycks_math", "EleutherAI/hendrycks_math"):
        # Hendrycks MATH: problem / solution (solution contains \boxed{} answer).
        # Prefer local pre-downloaded JSONL; fall back to HF per-subject configs.
        local_train = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "dataset", "eval_dataset", "math", "train.jsonl",
        )
        if os.path.isfile(local_train):
            from datasets import DatasetDict
            dataset = DatasetDict({
                "train": load_dataset("json", data_files={"train": local_train}, split="train"),
            })
            print(f"[MATH] Loaded local JSONL: {local_train}")
        else:
            from datasets import DatasetDict, concatenate_datasets
            _math_cfgs = [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
            ]
            _parts = []
            for cfg in _math_cfgs:
                try:
                    _parts.append(load_dataset("EleutherAI/hendrycks_math", cfg, split="train"))
                except Exception as e:
                    print(f"[MATH] skip config {cfg}: {e}")
            dataset = DatasetDict({"train": concatenate_datasets(_parts)})
            print(f"[MATH] Loaded {len(_parts)} configs from HF, total {len(dataset['train'])} rows")

        def convert_hendrycks_math(example):
            return {
                "messages": [
                    {"role": "user", "content": example.get("problem", "")},
                    {"role": "assistant", "content": example.get("solution", "")},
                ]
            }

        dataset = dataset.map(convert_hendrycks_math)
        drop_cols = [c for c in ("problem", "level", "type", "solution", "subject", "text")
                     if c in dataset["train"].column_names]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        print(f"[MATH] {len(dataset['train'])} training examples")

    elif script_args.dataset_name == "cognitivecomputations/dolphin-r1":
        # todo
        print(f"dataset length: {len(dataset)}")

    elif script_args.dataset_name == "yentinglin/s1K-1.1-trl-format":
        print(f"dataset length: {len(dataset[script_args.dataset_train_split])}")

    elif script_args.dataset_name == "open-r1/OpenR1-Math-220k" or script_args.dataset_name == "open-r1/codeforces-cots":

        if script_args.dataset_name == "open-r1/codeforces-cots":
            def remove_prompt_in_keys(example):
                example.pop("prompt", None)
                return example
            dataset = dataset.map(remove_prompt_in_keys)

        train_dataset=dataset[script_args.dataset_train_split]
        print(f"dataset length: {len(train_dataset)}")
        train_dataset = train_dataset.map(lambda x: {"message_length": len(str(x["messages"]))})
        train_dataset = train_dataset.sort("message_length")
        train_dataset = train_dataset.select(range(38000)).shuffle(seed=42)
        dataset[script_args.dataset_train_split] = train_dataset
        print(f"dataset length: {len(dataset[script_args.dataset_train_split])}")
        # print(f"first ten examples: {train_dataset[:10]}")
        # exit()


    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[MemDebug] after tokenizer: {_format_cuda_memory_stats()}")

    ## fix the checkpoint loading issue.
    original_torch_load = torch.load
    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        # if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    data_collator = None
    if is_qwen_family:
        tokenizer.padding_side = 'left'
        print("padding size updated!!!!")
        data_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_length)
        # Keep conversational columns for the ChatML collator.
        training_args.remove_unused_columns = False

    # elif "qwen" in model_name_lower:
    #     tokenizer.padding_side = 'right'
    #     print("padding size updated!!!!")
    #     data_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_length)

    manual_mask_map = None
    manual_mask_fn = None
    try:
        manual_mask_map = _parse_manual_router_mask(getattr(training_args, "router_manual_mask", None))
    except Exception as exc:
        logger.error(f"[RouterMask] Failed to parse router_manual_mask: {exc}")
        raise
    if manual_mask_map:
        manual_mask_fn = _build_fixed_router_mask_fn(manual_mask_map)
        logger.info(f"[RouterMask] Loaded manual fixed mask spec: {manual_mask_map}")

    if model_family == "qwen3_5_moe" and (
        manual_mask_fn is not None
        or getattr(training_args, "router_prune_enable", False)
    ):
        raise ValueError(
            "Online/manual router pruning during Qwen3.5 SFT is not supported: "
            "its batched expert tensors cannot be materialized by the legacy "
            "SFT callback. Prune the checkpoint first, then run this entry "
            "point with router_prune_enable=false."
        )

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # warning: not supporting quantization for Qwen3 and GPT-oss due to version conflict (after transformers update)
    if model_name_lower not in ["openai/gpt-oss-20b", "qwen/qwen3-30b-a3b"]:
        quantization_config = get_quantization_config(model_args)
        # quantization_config = {} if quantization_config==None else quantization_config
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
    else:
        # quantization_config = get_quantization_config(model_args)
        # quantization_config = {} if quantization_config==None else quantization_config
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            # device_map=get_kbit_device_map() if quantization_config is not None else None,
            # quantization_config=quantization_config,
        )
    training_args.model_init_kwargs = model_kwargs

    #######################
    # Teacher model kwargs #
    #######################
    teacher_model_kwargs = None
    if training_args.teacher_model_name_or_path is not None:
        teacher_torch_dtype = training_args.teacher_torch_dtype
        if teacher_torch_dtype not in ["auto", None]:
            teacher_torch_dtype = getattr(torch, teacher_torch_dtype)

        teacher_model_kwargs = dict(
            revision=training_args.teacher_model_revision or model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=training_args.teacher_attn_implementation or model_args.attn_implementation,
            torch_dtype=teacher_torch_dtype,
            use_cache=True,
        )

    ############################
    # Initialize the SFT Trainer
    ############################

    callbacks = list(get_callbacks(training_args, model_args))
    callbacks.append(StepMemCallback())
    if model_family in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe"}:
        prune_enable = getattr(training_args, "router_prune_enable", False)
        # Log all prune-related knobs for debugging/visibility.
        print(
            "[RouterPrune] cfg",
            {
                "enable": prune_enable,
                "start_step": getattr(training_args, "router_prune_start_step", None),
                "interval": getattr(training_args, "router_prune_interval", None),
                "expert_per_layer": getattr(training_args, "router_prune_expert_per_layer", None),
                "min_keep": getattr(training_args, "router_prune_min_keep", None),
                "step_size": getattr(training_args, "router_prune_step_size", None),
                "score_tau": getattr(training_args, "router_prune_score_tau", None),
                "use_plan": getattr(training_args, "router_prune_use_plan", None),
                "entropy_slope_alpha": getattr(training_args, "entropy_slope_alpha", None),
                "entropy_slope_beta": getattr(training_args, "entropy_slope_beta", None),
            },
        )
        mask_fn = manual_mask_fn
        add_router_mask = prune_enable or (mask_fn is not None)
        if add_router_mask:
            prune_start_step = getattr(training_args, "router_prune_start_step", None)
            prune_enabled = prune_enable and prune_start_step is not None and prune_start_step >= 0
            if mask_fn is not None:
                logger.info(f"[RouterMask] Using fixed manual mask (applied every step).")
            callbacks.append(
                RouterMaskCallback(
                    mask_fn=mask_fn,
                    prune_start_step=prune_start_step if prune_enabled else None,
                    prune_interval=getattr(training_args, "router_prune_interval", 5),
                    prune_min_keep=getattr(training_args, "router_prune_min_keep", 1),
                    prune_step_size=getattr(training_args, "router_prune_step_size", 32),
                    prune_expert_per_layer=getattr(training_args, "router_prune_expert_per_layer", None),
                    use_prune_plan=getattr(training_args, "router_prune_use_plan", True),
                )
            )
        else:
            print("nonononono")

    if training_args.teacher_model_name_or_path is None:
        callbacks.append(MemLogCallback())

    TrainerCls = CustomSFTTrainer
    if is_qwen_family and (script_args.dataset_name in ["lmms-lab/Math10K", "HectorHe/math7k", "HectorHe/math14k", "openai/gsm8k", "RoxanneWsyw/gsm", "fw407/Commonsense-15K", "cais/mmlu",]):
        training_args.remove_unused_columns = False
        trainer_kwargs = dict(
            model=model_args.model_name_or_path,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            callbacks=callbacks,
            teacher_model_name_or_path=training_args.teacher_model_name_or_path,
            teacher_model_init_kwargs=teacher_model_kwargs,
            disable_teacher_dropout=training_args.disable_teacher_dropout,
            layer_entropy_l1_weight=training_args.layer_entropy_l1_weight,
            layer_entropy_l1_layers=getattr(training_args, "layer_entropy_l1_layers", None),
            last_entropy_weight=training_args.last_entropy_weight,
            attn_kl_weight=training_args.attn_kl_weight,
        )
    else:
        trainer_kwargs = dict(
            model=model_args.model_name_or_path,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            callbacks=callbacks,
        )
        if data_collator is not None:
            trainer_kwargs["data_collator"] = data_collator
        if TrainerCls is CustomSFTTrainer:
            trainer_kwargs.update(
                {
                    "teacher_model_name_or_path": training_args.teacher_model_name_or_path,
                    "teacher_model_init_kwargs": teacher_model_kwargs,
                    "disable_teacher_dropout": training_args.disable_teacher_dropout,
                    "layer_entropy_l1_weight": training_args.layer_entropy_l1_weight,
                    "layer_entropy_l1_layers": getattr(training_args, "layer_entropy_l1_layers", None),
                    "last_entropy_weight": training_args.last_entropy_weight,
                    "attn_kl_weight": training_args.attn_kl_weight,
                }
            )
    trainer = TrainerCls(**trainer_kwargs)
    logger.info("[Model] Training model loaded.")

    def _fix_layer_idx_for_qwen_moe(model):
        # Ensure each MoE block carries its layer index (HF 4.49 won't pass it).
        unwrapped = getattr(model, "module", model)
        if not hasattr(unwrapped, "model") or not hasattr(unwrapped.model, "layers"):
            return 0
        patched = 0
        for i, layer in enumerate(unwrapped.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp and mlp.__class__.__name__ in ("Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock"):
                mlp.layer_idx = i
                patched += 1
        return patched


    patched_blocks = _fix_layer_idx_for_qwen_moe(trainer.model)
    if patched_blocks > 0:
        logger.info(f"[LayerIdxPatch] Set layer_idx for {patched_blocks} MoE SparseMoeBlock modules.")

    if manual_mask_fn is not None:
        try:
            initial_mask = manual_mask_fn(model=trainer.model, step=0, args=training_args, state=trainer.state)
            if initial_mask:
                _apply_router_mask_to_model(trainer.model, initial_mask)
                logger.info(f"[RouterMask] Applied manual fixed mask before training: {_summarize_mask(initial_mask)}")
            else:
                logger.warning("[RouterMask] Manual mask spec was provided but resolved to empty mask.")
        except Exception as exc:
            logger.warning(f"[RouterMask] Failed to apply manual mask before first step: {exc}")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    logger.info(trainer.model)
    print(f"[MemDebug] before train: {_format_cuda_memory_stats()}")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 确保所有 rank 都到这一步
    trainer.accelerator.wait_for_everyone()

    logger.info("*** Save model ***")
    # NOTE: ZeRO-3 shards params across ranks; save_model internally runs an
    # all-gather collective that requires participation from EVERY rank.
    # Wrapping this call in `if is_world_process_zero()` causes rank 0 to enter
    # the gather while other ranks skip ahead → 30-min NCCL watchdog SIGABRT.
    # Trainer.save_model already gates the actual file write to rank 0.
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        logger.info(f"Model saved to {training_args.output_dir}")

    trainer.accelerator.wait_for_everyone()
    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
