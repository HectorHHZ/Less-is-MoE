# GPT-OSS-120B IntDim 50% on held-out GPQA-main

This record compares the BF16 GPT-OSS-120B baseline with IntDim-E, IntDim-L,
and IntDim-G checkpoints pruned by 50%. All three pruning runs use the same 64
GPQA-main calibration examples. Evaluation uses the remaining 384 disjoint
questions.

The reported metric is **mean pass@1 over eight independent samples per
question** (`avg@8`). Each completion is scored independently; there is no
majority vote, best-of-eight selection, or pass@8 calculation.

## Results

| Checkpoint | Loader | Accuracy | Correct | Delta from base | Unparsed | Completion tokens |
|---|---|---:|---:|---:|---:|---:|
| BF16 baseline | stock | **76.99%** | 2,365 / 3,072 | — | 3 | 28,039,222 |
| IntDim-E 50% | stock | **25.36%** | 779 / 3,072 | -51.63 pp | 188 | 1,586,117 |
| IntDim-L 50% | ragged | **62.99%** | 1,935 / 3,072 | -14.00 pp | 12 | 15,567,420 |
| IntDim-G 50% | ragged | **58.46%** | 1,796 / 3,072 | -18.52 pp | 44 | 40,743,679 |

Every run completed all 3,072 requested generations with zero length
truncations. Unparsed answers count as incorrect. IntDim-L retains the most
accuracy in this experiment. IntDim-E is near four-choice chance accuracy even
when unparsed outputs are excluded (27.01%), so answer parsing alone does not
explain its degradation.

## Models and calibration

The baseline is `openai/gpt-oss-120b` dequantized from the released MXFP4
weights to BF16. Pruning used the full BF16 weights and removed exactly 50% of
routed-expert intermediate units.

| Method | Checkpoint | Revision | Drop-plan SHA-256 |
|---|---|---|---|
| IntDim-E | [`jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-e-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-e-50) | `969e733dff70aee5e94d1cb122ff4d3becb56b10` | `646c93afcafab8994bd645cc47720a9bf62b781a0ef26ed24cff96c57ddcabaf` |
| IntDim-L | [`jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-l-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-l-50) | `c180424051bcc3a1a767579747cbaabef33b0d79` | `6c87679f5245637da6aafaed9ede904c0363be44c456b84610b3f32d06a612d7` |
| IntDim-G | [`jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-g-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-gpqa-main-64-intdim-g-50) | `06bd011c40b8673efb40c17f3175927787274be6` | `65d3b4f30f0f349815457f2de73f4830eebf7ff72c46a3e74597bbe2df21249f` |

The exact calibration and held-out split is stored in the access-controlled
Hugging Face dataset
[`jayzou3773/less-is-moe-gpqa-main-calibration-64`](https://huggingface.co/datasets/jayzou3773/less-is-moe-gpqa-main-calibration-64),
revision `7134dfef5af4605eae0706c30efa9226f49aed96`. It contains 64 calibration
rows and 384 held-out rows selected from `Idavidrein/gpqa` config `gpqa_main`,
revision `633f5ee89ab8ad4522a9f850766b73f62147ffdd`. Calibration selection uses
`numpy.random.default_rng(1234).permutation(448)` and takes the first 64 rows.
No calibration input was truncated; GPT-OSS sees 28,180 tokens in total, with
150–1,511 tokens per example.

Importance is the released mean absolute gradient over the next-token loss.
The run uses BF16 weights and gradients, FP32 reduction and accumulation, and
no optimizer step. The score tensor SHA-256 is
`796f7a9c227bde4f563ded6b8892d3d08a40291cbcb92f165272d4f9c77d3a3d`.

## Evaluation protocol

- Dataset: 384 held-out GPQA-main questions after removing calibration rows
- Samples: eight independent completions per question, 3,072 per model
- Options: deterministic per-record shuffle, option seed 42
- Prompt profile: `qwen35-mcq`; GPT-OSS Harmony chat template; no tools
- Reasoning effort: `high`; fixed prompt date `2026-09-17`
- Temperature: 1.0; top-p: 1.0; top-k: -1; min-p: 0
- Presence penalty: 0; repetition penalty: 1.0
- Maximum generated tokens and model length: 131,072
- TP: 4; maximum sequences: 128; maximum batched tokens: 8,192
- Precision: BF16; eager execution; Triton attention and MoE kernels
- Scoring: last Harmony `final` channel only, using the same frozen parser for
  every model without consulting the gold answer

The evaluation JSONL SHA-256 is
`c7475f3382e65a23a5f3c1da97947e1ac9c110686efcc23df16c44c9b65c44cb`;
the split-manifest SHA-256 is
`ff0bf0307c875869ea9b90feab29413a9b6a7777ed844c5add4626b93d7b941a`.

The [artifact directory](artifacts/gpt-oss-120b-gpqa-main-calib64/README.md)
contains all 12,288 question-safe per-sample predictions, complete metrics,
run settings, grading audits, and frozen manifests. Because GPQA asks users not
to reveal its examples, committed predictions omit question text, options,
answer text, raw generations, and final-answer prose. They retain every field
needed to reproduce the headline scores.

The prompt and sampling settings follow the public GPT-OSS settings used with
Qwen3.5's published multiple-choice format. The complete internal benchmark
harness behind the Qwen3.5 report is unavailable, so these scores are not an
exact reproduction of that report.

## Environment

- 4 × NVIDIA B200 per run
- Docker image: `less-is-moe:reasoning-matrix-20260918-v5`
- Image ID: `sha256:009aafb479cf5275745c92d116198877d67c75cdda9a5a60320c5b8d7a98ebcb`
- Python 3.12.14
- PyTorch 2.13.0+cu130; CUDA 13.0
- Transformers 5.17.0
- vLLM 0.29.0
- tokenizers 0.23.1

## Reproduction command

Set `RUNTIME=stock` for the baseline and IntDim-E, and `RUNTIME=ragged` for
IntDim-L/G.

```bash
MODEL=/models/gpt-oss-checkpoint
OUTPUT=/outputs/gpt-oss-gpqa-main
RUNTIME=ragged

scripts/evaluate/zero_shot.sh \
  --dataset gpqa_main \
  --data_path /data/gpqa-main-64/test.jsonl \
  --output_dir "$OUTPUT" \
  --model_name_or_path "$MODEL" \
  --runtime_patch "$RUNTIME" \
  --dtype bf16 \
  --tensor_parallel_size 4 \
  --enforce_eager \
  --use_chat_template \
  --reasoning_effort high \
  --prompt_date 2026-09-17 \
  --seed 42 \
  --mcq_profile qwen35-mcq \
  --n_samples_per_problem 8 \
  --temperature 1 \
  --top_p 1 \
  --top_k -1 \
  --min_p 0 \
  --presence_penalty 0 \
  --repetition_penalty 1 \
  --max_tokens 131072 \
  --max_model_len 131072 \
  --batch_size 128 \
  --max_num_batched_tokens 8192 \
  --gpu_memory_utilization 0.9 \
  --resume
```
