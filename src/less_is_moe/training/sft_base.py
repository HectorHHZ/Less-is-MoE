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

Run from the repository root with ``scripts/train/base.sh`` and one of
the editable templates under ``recipes/sft/base``.
"""

import logging
import os
import sys
import functools
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from less_is_moe.training.configs import SFTConfig
from less_is_moe.training.utils import get_tokenizer
from less_is_moe.training.utils.callbacks import get_callbacks
from less_is_moe.training.utils.wandb_logging import init_wandb_training
from less_is_moe.model_patches.registry import detect_model_family
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
    DataCollatorForChatML
    )

logger = logging.getLogger(__name__)



class CustomSFTTrainer(SFTTrainer):
    def _prepare_dataset(self, dataset, *args):
        dataset = dataset.add_column("_messages", dataset["messages"])
        dataset = super()._prepare_dataset(dataset, *args)
        dataset = dataset.rename_column("_messages", "messages")
        return dataset

    def compute_loss(self, model, inputs, **kwargs):
        import inspect
        forward_fn = model.module.forward if hasattr(model, "module") else model.forward
        valid_keys = set(inspect.signature(forward_fn).parameters.keys())
        inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        return super().compute_loss(model, inputs, **kwargs)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    log_level = training_args.get_process_log_level()
    os.makedirs(training_args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, "training.log")),
        ],
        force=True,
    )
    logger.setLevel(log_level)


    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    model_family = detect_model_family(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        validate_for_hf=True,
    )
    logger.info("[Model] Detected family: %s.", model_family)

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
    # TODO: save the dataset to local and load from local for time saving
    ################
    if script_args.dataset_name == "RoxanneWsyw/gsm":
        from datasets import DatasetDict
        dataset = DatasetDict({"train": load_dataset(script_args.dataset_name, data_files="train.jsonl", split="train")})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    print(f"script_args.dataset_name: {script_args.dataset_name}")


    if (script_args.dataset_name == "lmms-lab/Math10K") or (script_args.dataset_name == "HectorHe/math7k") or (script_args.dataset_name == "HectorHe/math14k"):
        # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        # dataset[script_args.dataset_train_split]

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

    elif (script_args.dataset_name == "RoxanneWsyw/ESFT-translation") or (script_args.dataset_name == "RoxanneWsyw/ESFT-summary") or (script_args.dataset_name == "RoxanneWsyw/ESFT-law") or (script_args.dataset_name == "RoxanneWsyw/ESFT-intent") or (script_args.dataset_name == "RoxanneWsyw/MBPP"):
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

    elif script_args.dataset_name == "fw407/Commonsense-15K":
        def convert_commonsense(example):
                return {
                    "messages": [
                        {"role": "user", "content": example.get("instruction", "")},
                        {"role": "assistant", "content": example.get("output", "")},
                    ]
                }

        dataset = dataset.map(convert_commonsense)
        if "prompts" in dataset.column_names:
            dataset = dataset.remove_columns("prompts")

    elif script_args.dataset_name == "theblackcat102/evol-codealpaca-v1":

        def preprocess_function(example):
            # Combine instruction + input if input exists
            if example.get("input"):
                prompt = f"{example['instruction']}\nInput: {example['input']}"
            else:
                prompt = example['instruction']

            # Output is the target response - use 'completion' key instead of 'response'
            completion = example['output']

            return {
                "prompt": prompt,
                "completion": completion
            }
        dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)
        dataset[script_args.dataset_train_split] = dataset["train"]

        # Debug: Print first example to verify format
        print(f"Dataset length: {len(dataset[script_args.dataset_train_split])}")
        print(f"First example keys: {list(dataset[script_args.dataset_train_split][0].keys())}")
        print(f"First example: {dataset[script_args.dataset_train_split][0]}")
        # dataset = dataset.map(preprocess_function)
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

    ## fix the checkpoint loading issue.
    original_torch_load = torch.load
    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        # if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    if model_family in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe", "olmoe"}:
        tokenizer.padding_side = 'left'
        print("padding size updated!!!!")
        data_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_length)


    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # warning: not supporting quantization for Qwen3 and GPT-oss due to version conflict (after transformers update)
    if model_args.model_name_or_path not in ["openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B"]:
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

    ############################
    # Initialize the SFT Trainer
    ############################



    if model_family in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe", "olmoe"} and (script_args.dataset_name in ["lmms-lab/Math10K", "HectorHe/math7k", "HectorHe/math14k", "openai/gsm8k", "fw407/Commonsense-15K","RoxanneWsyw/ESFT-translation", "RoxanneWsyw/ESFT-summary", "RoxanneWsyw/ESFT-law", "RoxanneWsyw/ESFT-intent", "RoxanneWsyw/MBPP"]):
        training_args.remove_unused_columns = False
        trainer = CustomSFTTrainer(
            model=model_args.model_name_or_path,
            args=training_args,
            data_collator = data_collator,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
        )
    else:
        trainer = SFTTrainer(
            model=model_args.model_name_or_path,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            callbacks=get_callbacks(training_args, model_args),
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    logger.info(trainer.model)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

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
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
