# SuperGPQA evaluation

The existing `scripts/evaluate/zero_shot.sh` entry point accepts
`--dataset supergpqa`. It supports GPT-OSS and the stock/ragged model loaders
in the unified GPU environment.

## Protocol

Use the complete `SuperGPQA-all.jsonl` from
[`m-a-p/SuperGPQA`](https://huggingface.co/datasets/m-a-p/SuperGPQA), pinned to
revision `4430d4458112c7d4497fdcf94d7cc223313d6acf` (26,529 questions).
The question text and option order stay unchanged. The prompt follows the
[official zero-shot template](https://github.com/SuperGPQA/SuperGPQA/blob/main/config/prompt/zero-shot.yaml).

The [SuperGPQA paper, section 4.1](https://arxiv.org/html/2502.14739v3)
uses zero-shot evaluation for reasoning/chat models, temperature 0, and up to
32K generated tokens for reasoning models. Its five-shot protocol is for
pretrained base models. An *unpruned* GPT-OSS checkpoint remains a reasoning
model and uses the zero-shot protocol.

Qwen's [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
reports GPT-OSS-120B at **54.6%**. That table does not disclose the complete
GPT-OSS-specific evaluation configuration. The original SuperGPQA paper
predates GPT-OSS. Treat 54.6% as a published comparison, rather than claiming
an exact reproduction of an unavailable configuration.

This implementation records both sample accuracy and macro accuracy across
subfields, fields, and disciplines. It also records unparsable answers,
length truncation, token counts, raw generations, the final answer, and the
individual question UUID. GPT-OSS answers are extracted only from the Harmony
`final` channel. An answer mentioned only in unfinished reasoning is not scored
as correct. No question is silently dropped for exceeding the context budget.

## Full unpruned benchmark

Prepare a manifest without withholding any questions:

```bash
python -m less_is_moe.evaluation.supergpqa \
  --input /data/SuperGPQA-all.jsonl --output_dir /data/supergpqa-full \
  --calibration_size 0 --seed 42 --tokenizer_path /models/gpt-oss-120b-bf16 \
  --reasoning_effort high --prompt_date 2026-09-17

python -m less_is_moe.evaluation.vllm_zero_shot \
  --dataset supergpqa --data_path /data/supergpqa-full/evaluation.jsonl \
  --output_dir /outputs/supergpqa-base \
  --model_name_or_path /models/gpt-oss-120b-bf16 \
  --base_model --runtime_patch stock --dtype bf16 \
  --tensor_parallel_size 4 --enforce_eager --use_chat_template \
  --reasoning_effort high --prompt_date 2026-09-17 \
  --seed 42 --temperature 0 --max_tokens 32768 --max_model_len 49152 \
  --batch_size 256 --max_num_batched_tokens 8192 --gpu_memory_utilization 0.9
```

Use `--limit 16` and a separate output directory for an initial smoke test.
Omit `--limit` for the complete dataset. Append `--resume` to continue a run;
the evaluator rejects changed settings or a different data/prompt hash.
Results are saved after each completed question. Continuous batching keeps the
scheduler fed while other requests finish long reasoning traces. Resume skips
already saved UUIDs without requiring completion order to match dataset order.

The runner uses the tokenizer's chat template, with a fixed date because
GPT-OSS's default template inserts the current date. Sampling and the entire
rendered input token sequence are recorded or hashed. No tools are provided.
The maximum token count includes reasoning and the final answer.

## Later pruning comparison

Freeze a 128-question calibration split before inspecting individual results:

```bash
python -m less_is_moe.evaluation.supergpqa \
  --input /data/SuperGPQA-all.jsonl --output_dir /data/supergpqa-calib128 \
  --calibration_size 128 --seed 42 --tokenizer_path /models/gpt-oss-120b-bf16 \
  --reasoning_effort high --prompt_date 2026-09-17
```

The selection is deterministic and balanced across the 13 broad disciplines.
Normalized duplicate questions stay out of the evaluation set if any copy is
used for calibration. The manifest records calibration/evaluation UUIDs and
any excluded duplicate rows. Calibration text contains the same user prompt
and a gold answer, formatted with the same model chat template.

For the IntDim-E/L/G 50% experiment, use this exact calibration file and its
remaining evaluation questions for every variant. Filter the saved full-base
predictions to those same evaluation UUIDs before comparing scores; do not
compare a full-dataset base score against a smaller pruned-model evaluation.
Keep tokenizer, precision, reasoning effort, generation budget, seed, prompt,
TP, and scoring fixed. Use stock loading for uniform IntDim-E checkpoints and
`--runtime_patch ragged` for compact IntDim-L/G checkpoints.

GPT-OSS's released expert weights are MXFP4. The structural-pruning workflow
uses BF16 weights dequantized from that checkpoint. Use the same unpruned BF16
export for the primary pruning baseline, and label it explicitly as dequantized
BF16. Native MXFP4 inference is a separate precision setting.
