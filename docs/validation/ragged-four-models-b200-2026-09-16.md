# Four full pretrained models: IntDim-L/G on B200

Issue [#29](https://github.com/HectorHHZ/Less-is-MoE/issues/29),
PR [#30](https://github.com/HectorHHZ/Less-is-MoE/pull/30).
[Machine-readable results](ragged-four-models-b200-2026-09-16.json).
[Full-model FP32 controls and vLLM divergence analysis](ragged-precision-b200-2026-09-16.md).

## Result

All **eight full-weight cases** completed calibration-based 50% routed-neuron
pruning, physical compaction, checkpoint save, exact HF reload and adapted
vLLM GPU generation. The Qwen3 checkpoint is **Qwen/Qwen3-30B-A3B**, not Coder.
Qwen3.5 uses its complete **language model** (40 layers); vision and MTP are
excluded by the stock causal-LM loader. The other three checkpoints are their
complete original causal language models.

| Full pretrained checkpoint | Scope | Total parameters: original → saved | Width range / distinct widths | Saved checkpoint vLLM inference | Tokens matching stock zero-mask |
| --- | --- | --- | --- | --- | --- |
| Qwen1.5-MoE-A2.7B | L | 14.3158 → 8.0872B | 0–1408 / 623 | Passed | 134/256 |
| Qwen1.5-MoE-A2.7B | G | 14.3158 → 8.0872B | 0–1408 / 580 | Passed | 256/256 |
| OLMoE-1B-7B-0924 | L | 6.9192 → 3.6979B | 0–1024 / 420 | Passed | 148/256 |
| OLMoE-1B-7B-0924 | G | 6.9192 → 3.6979B | 0–1024 / 408 | Passed | 256/256 |
| Qwen3-30B-A3B | L | 30.5321 → 16.0366B | 0–768 / 584 | Passed | 30/256 |
| Qwen3-30B-A3B | G | 30.5321 → 16.0366B | 0–768 / 575 | Passed | 254/256 |
| Qwen3.5-35B-A3B | L | 34.6606 → 18.5545B | 0–512 / 256 | Passed | 256/256 |
| Qwen3.5-35B-A3B | G | 34.6606 → 18.5545B | 0–512 / 155 | Passed | 175/256 |

The 50% budget applies to routed expert intermediate neurons, not all model
parameters. Attention, embeddings, router and shared-expert weights remain.
The layer/expert/hidden dimensions were checked against the original config;
parameter removal equals `3 * hidden_size * dropped_neurons` in every case.
Original expert/router IDs, including zero-width experts, are preserved.
The JSON records the verified download revisions for the three downloaded
models; the existing Qwen3.5 cache has no recorded upstream revision, so its
source is identified by config hash, shard manifest and B200 path.
Token matches count equal tokens at the same positions across two 128-token
outputs; each per-prompt record also includes its common-prefix length and
first differing generated-token position (one-based). These counts are not an accuracy metric or an equivalence claim.

## Long-generation divergence

First mismatch positions count only generated tokens, starting at 1. `None`
means all 128 generated tokens match for that prompt. After a divergence, the
models receive different generated histories, so subsequent token matches
are descriptive and are not a same-context numerical comparison.

| Model | Scope | Code prompt matches | First mismatch | Explanation prompt matches | First mismatch |
| --- | --- | --- | --- | --- | --- |
| qwen15 | L | 128/128 | None | 6/128 | 6 |
| qwen15 | G | 128/128 | None | 128/128 | None |
| olmoe | L | 42/128 | 36 | 106/128 | 20 |
| olmoe | G | 128/128 | None | 128/128 | None |
| qwen3 | L | 11/128 | 10 | 19/128 | 19 |
| qwen3 | G | 126/128 | 21 | 128/128 | None |
| qwen35 | L | 128/128 | None | 128/128 | None |
| qwen35 | G | 128/128 | None | 47/128 | 43 |

## Identical settings

- Full source weights, GPU BF16; no CPU model execution or offload.
- IntDim-L and IntDim-G; drop ratio 0.5; seed 7.
- The same four calibration texts from `docker/ragged_qwen_smoke.py`, maximum
  64 tokens each. Actual lengths: OLMoE `[22,20,26,20]` (88 total), other models
  `[21,19,23,19]` (82 total). Scores are computed on the full original model
  and reused for L/G. Confirmation runs also reuse those scores.
- Two held-out prompts: Fibonacci completion and a sky-color explanation.
  This is pipeline validation, not representative calibration or quality evaluation.
- Each saved checkpoint is loaded in a fresh vLLM process. BF16, TP=PP=DP=1,
  eager, Triton attention, max context 256, max sequences 2, batched tokens 256,
  GPU memory utilization 0.6, temperature 0, 128 new tokens, `ignore_eos=True`.
- Baseline: stock vLLM with zero-masked full-shape weights. Compact: the opt-in
  `less_is_moe_ragged` backend. HF preparation and vLLM initialization are
  sequential on each GPU; different model jobs use separate B200s.

## Numerical checks

All compact tensors reload bitwise, and the reloaded HF logits exactly match
the in-memory compact model. FP32 expert checks (TF32 off, GPU) sample three
experts in every layer and must pass `rtol=1e-4, atol=1e-5`.
Full-model BF16 zero-mask/compact comparisons are reported separately:

| Model | Scope | FP32 expert checks | FP32 max absolute error | BF16 logit max absolute error | Maximum KL | BF16 sanity gate |
| --- | --- | --- | --- | --- | --- | --- |
| qwen15 | L | 72 | 3.57628e-07 | 0.34375 | 0.00239997 | Passed |
| qwen15 | G | 72 | 9.53674e-07 | 0.367188 | 0.00188942 | Passed |
| olmoe | L | 48 | 5.96046e-07 | 0.125 | 0.000463136 | Passed |
| olmoe | G | 48 | 4.76837e-07 | 0.25 | 0.000573408 | Passed |
| qwen3 | L | 144 | 1.90735e-06 | 1.0625 | 0.0111775 | **Failed** |
| qwen3 | G | 144 | 1.43051e-06 | 0.75 | 0.0167838 | **Failed** |
| qwen35 | L | 120 | 6.70552e-08 | 0.580078 | 0.000847458 | Passed |
| qwen35 | G | 120 | 5.21541e-08 | 0.359375 | 0.00109198 | Passed |

The sanity gate is cosine similarity >= 0.995 and reference-to-compact KL <=
0.01 on both prompts. **Qwen3 IntDim-L and IntDim-G fail this gate**. The initial IntDim-L run stopped at that failure.
The failure log is retained. The standalone preparation command still fails
by default; the matrix explicitly opts into `--allow-logit-drift` to record
numerical failures while independently checking save/reload/inference.
FP32 correctness and exact reload checks are never bypassed. BF16 matrix-shape
changes alter rounding and later routing can amplify it. Successful inference
does not establish BF16 equivalence, model quality, or a speedup.

## Environment and regressions

- B200 host GPUs 5, 6 and 7, one GPU per process, SM100.
- Existing image `less-is-moe:x9zou-intdim-gpu`, ID
  `sha256:959d2ed008772283e20a0d7302eb9663be56c1dea51122e2451ce3ff1f1d3cb1`.
  PR code is mounted and installed editable with `--no-deps`; no third-party
  versions changed and no new Docker release was published.
- Python 3.12.14, Torch 2.13.0+cu130, Transformers 5.17.0, vLLM 0.29.0,
  tokenizers 0.23.1, CUDA runtime 13.0.
- All 211 locked package versions and `pip check` passed, plus GPU BF16 matmul
  and compiled vLLM RMSnorm checks.
- **33 GPU unit tests passed** across the four families: calibrated L/G,
  exact weights, non-expert/shared-expert preservation, CLI save/reload,
  invalid plans, zero-width experts and entirely zero routed layers.
- **Eight random-fixture L/G cases** also generated with both stock and
  adapted vLLM. These fixtures are supplementary; full-weight evidence is above.
- Initial Qwen2 adapter startup exposed a missing upstream eager-mode flag;
  `_init_model` now initializes it. The Qwen2 and Qwen3.5 adapter reruns pass.
- Qwen3.5 HF calibration uses reference PyTorch GPU implementations of its
  convolution and Gated DeltaNet operators in this pinned image. vLLM uses
  its native GPU implementations. This is not CPU fallback.

The general format and shared kernels cover these four audited SiLU families.
Quantization, multi-GPU, CUDA Graphs, LoRA and speculative decoding remain
unsupported. GPT-OSS and Gemma are not included in this backend expansion.

## Reproduce / retained checkpoints

See [the common commands](../RAGGED_EXPERTS.md#reproduce-validation).
`docker/ragged_model_matrix.py` runs both scopes for one full model using the
same `docker/ragged_qwen_smoke.py` implementation. GPU unit tests are in
`tests/test_ragged_intdim.py`, parameterized by model family.

All eight compact checkpoints and raw logs are retained on B200:

```text
/raid/x9zou/less-is-moe-four.YHONHJ/results/
  qwen15/{layer,global}/compact/
  olmoe/{layer,global}/compact/
  qwen3/{layer,global}/compact/
  qwen35/{layer,global}/compact/
```

Earlier 16-token reports are retained as `history16-<case>.tgz` on B200.
Temporary full-size zero-mask baselines were removed after their comparisons;
the generated outputs and numerical evidence remain. Per-case `matrix.json`
contains the source config, shard sizes, launch-time source hashes and all
results. All four launch snapshots match the reviewed runtime/harness files.
The aggregate JSON also records final source hashes. All eight resulting
artifacts were independently checked for dimensions, exact parameter/neuron
budgets and completed 128-token generation.
