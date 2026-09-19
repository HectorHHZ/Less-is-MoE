# GPT-OSS-120B IntDim 50% on GPQA-Diamond

This record compares the BF16 GPT-OSS-120B baseline with the public IntDim-E,
IntDim-L, and IntDim-G checkpoints produced from the same 128-sample S1K
calibration set. It uses the frozen GPQA-Diamond evaluator introduced in
[#34](https://github.com/HectorHHZ/Less-is-MoE/pull/34).

The reported metric is **mean pass@1 over eight independent samples per
question** (`avg@8`). Each completion is scored independently. There is no
majority vote, best-of-eight selection, or pass@8 calculation.

## Results

| Checkpoint | Loader | Accuracy | Correct | Delta from base | Unparsed | Completion tokens |
|---|---|---:|---:|---:|---:|---:|
| BF16 baseline | stock | **79.23%** | 1,255 / 1,584 | — | 0 | 17,951,623 |
| IntDim-E 50% | stock | **20.96%** | 332 / 1,584 | -58.27 pp | 85 | 997,172 |
| IntDim-L 50% | ragged | **54.67%** | 866 / 1,584 | -24.56 pp | 0 | 16,065,490 |
| IntDim-G 50% | ragged | **64.33%** | 1,019 / 1,584 | -14.90 pp | 4 | 17,701,545 |

All four runs completed all 1,584 requested generations without length
truncation. Unparsed final answers count as incorrect. IntDim-G retains the
most accuracy of the three 50% pruning methods. IntDim-E also changes the
generation-length distribution substantially: it averages 630 completion
tokens, compared with 11,333 for the baseline, 10,142 for IntDim-L, and 11,175
for IntDim-G. This observation describes the run; it does not by itself
identify the cause of the IntDim-E degradation.

## Models and calibration

The baseline is `openai/gpt-oss-120b` dequantized from the released MXFP4
weights to BF16. Pruning used the full BF16 weights and removed exactly 50% of
routed-expert intermediate units.

| Method | Public checkpoint | Revision | Drop-plan SHA-256 |
|---|---|---|---|
| IntDim-E | [`jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-e-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-e-50) | `cc6d97d3b085ecfe414df635bc60be05a4afff3a` | `9bb0be179d2dc77b8a89eec99cc510379854faef49bf6ce6e067f82d0c9a5837` |
| IntDim-L | [`jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-l-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-l-50) | `675bb874b8c9ee277eda04cddea4bd82d88911cd` | `920eef85f081035db9f357d109b5245f177ce104a6d9d8f1aa9c4db76c6934b4` |
| IntDim-G | [`jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-g-50`](https://huggingface.co/jayzou3773/less-is-moe-gpt-oss-120b-s1-128-seq8192-intdim-g-50) | `0ebeced308f7ddcfd3577c047127b4d7ecc4009e` | `43625dc6ab122f6f29a2e76ffd51698b934d47ba649ed61e93e72e2664688ee1` |

The calibration inputs are public at
[`jayzou3773/less-is-moe-s1-calibration-128-seq8192`](https://huggingface.co/datasets/jayzou3773/less-is-moe-s1-calibration-128-seq8192),
revision `678b4e666183e16ec00376960df03b6381632ed1`. The source is
`yentinglin/s1K-1.1-trl-format`, revision
`58a01564d278477da20ead1bcf1cde8e31f36251`; the selection uses split `train`,
shuffle seed 1234, and the first 128 nonempty samples. GPT-OSS tokenization
uses prefix truncation at 8,192 tokens without padding: 915,210 input tokens in
total and 79 truncated samples.

Importance is the released mean absolute gradient over the next-token loss.
The run uses BF16 weights and gradients, FP32 reduction and accumulation, and
no optimizer step. The score tensor SHA-256 is
`5797d270d95bc34699a5a7bf57e406e4c111df87f0f35395af723ed61b124bdc`.

## GPQA-Diamond protocol

- Dataset: `Idavidrein/gpqa`, revision
  `633f5ee89ab8ad4522a9f850766b73f62147ffdd`
- Questions: all 198 Diamond questions; deterministic option shuffle, seed 42
- Samples: eight independent completions per question, 1,584 total
- Prompt profile: `qwen35-mcq`; GPT-OSS Harmony chat template; no tools
- Reasoning effort: `high`; fixed prompt date `2026-09-17`
- Temperature: 1.0; top-p: 1.0; top-k: -1; min-p: 0
- Presence penalty: 0; repetition penalty: 1.0
- Maximum generated tokens: 131,072
- Maximum model length: 131,072
- Per-request budget: `min(131072, 131072 - prompt_tokens)`
- TP: 4; maximum sequences: 128; maximum batched tokens: 8,192
- Precision: BF16; eager execution; Triton attention and MoE kernels
- Scoring: last Harmony `final` channel only; the frozen final-answer parser is
  applied uniformly without consulting the gold answer

The evaluation JSONL SHA-256 is
`9312a13f2c88c80082892fa4c60d761b5e6a04b943d36976f8904ac72e93f542`;
the split-manifest SHA-256 is
`853ce946fbd9cd4309ce514ed69cbc571437df3d6bf30430f61d847a1350ff07`.
The machine-readable record includes hashes for every raw prediction file and
the frozen grading source.

The protocol follows the public GPT-OSS settings used with Qwen3.5's published
multiple-choice format. The complete internal benchmark harness behind the
Qwen3.5 report is not public, so these results should not be described as an
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

Prepare the frozen GPQA-Diamond manifest as documented in
[`docs/supergpqa.md`](../supergpqa.md), then run the same command for each
checkpoint. Set `RUNTIME=stock` for the baseline and IntDim-E; set
`RUNTIME=ragged` for IntDim-L/G.

```bash
MODEL=/models/gpt-oss-checkpoint
OUTPUT=/outputs/gpt-oss-gpqa-diamond
RUNTIME=ragged

scripts/evaluate/zero_shot.sh \
  --dataset gpqa_diamond \
  --data_path /data/gpqa-diamond-full/evaluation.jsonl \
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

The evaluator rejects resume attempts when the saved protocol, model config,
prompt tokens, data, or split hashes do not match.
