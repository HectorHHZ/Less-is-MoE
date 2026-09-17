# Expanded full-weight B200 validation and precision diagnosis

[Raw configurations, hashes, widths, output tokens and precision metrics](ragged-expanded-b200-2026-09-17.json).
This extends the [original four-model experiment](ragged-four-models-b200-2026-09-16.md)
and [first precision diagnosis](ragged-precision-b200-2026-09-16.md) in PR #30.
Across both reports, seven distinct full pretrained models have completed L/G.

## Full-model inference

All four requested models use IntDim-L and IntDim-G at **50% of routed intermediate
neurons**, with the same cached calibration scores and selected neurons for stock
zero-mask and physically compact representations. Both compact checkpoints are
saved, every HF weight reloads exactly, and HF reload logits are bitwise equal.
Each vLLM path generates **128 new tokens on each of two prompts**.

| Full pretrained model | L token matches | G token matches | vLLM PP | Save/reload/inference |
| --- | --- | --- | --- | --- |
| GPT-OSS-120B | 110/256 | 129/256 | 3 | L/G pass |
| Qwen3.5-122B-A10B | 134/256 | 207/256 | 3 | L/G pass |
| Qwen3.5-35B-A3B | 256/256 | 175/256 | 1 | L/G pass |
| Gemma-4-26B-A4B | 130/256 | 23/256 | 1 | L/G pass |

Matches are positional comparisons of two free-running completions, not an
accuracy score. After the first differing token, the contexts also differ.
Inference success does not establish numerical equivalence.

## Supplemental precision comparison

Final acceptance is physical removal of the zero-masked neurons, checkpoint
reload, and successful adapted-vLLM inference. Token equality and full-model
FP32 diagnosis are not acceptance gates. Completed controls remain below;
GPT-OSS full-model FP32 controls were not run after scope was narrowed.

For the completed Qwen3.5-35B, Qwen3.5-122B and Gemma L/G controls, run the complete zero-masked source and saved compact
model sequentially on GPU. Evaluate BF16, then promote those same weights to
FP32 with TF32 disabled. Check each initial prompt and the common prefix just
before each first vLLM token divergence. Compare last-token logits and unordered
expert-selection sets for every input token at every layer. These are HF full
forwards without a KV cache, not FP32 vLLM decoding.

| Model | Scope | Context pairs | BF16 max logit error | FP32 max logit error | Changed token-layer route sets: BF16 → FP32 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B | L | 4 | 0.4375 | 2.59876e-05 | 1048 → 0 |
| Qwen3.5-122B-A10B | G | 3 | 0.375 | 3.24249e-05 | 680 → 0 |
| Qwen3.5-35B-A3B | L | 2 | 0.580078 | 1.33514e-05 | 136 → 0 |
| Qwen3.5-35B-A3B | G | 3 | 0.523438 | 1.85966e-05 | 410 → 0 |
| Gemma-4-26B-A4B | L | 3 | 5.59375 | 3.8147e-05 | 108 → 0 |
| Gemma-4-26B-A4B | G | 4 | 5.57812 | 3.91006e-05 | 306 → 0 |

Across **19 context pairs**, FP32 has the same next-token argmax and
zero changed expert-selection sets in every pair. BF16 has **2,688**
changed token-layer route sets; FP32 maximum logit error is **3.91006e-05**.
These counts include overlapping prefixes and are not independent quality samples.

This strongly supports finite precision and changed arithmetic order as the main
cause on these measured contexts. Compact matrix shapes change reductions;
small logit changes can alter top-k routing and then accumulate through layers.
Different next tokens subsequently cause different autoregressive histories.
Stock vLLM and the custom backend also differ in activation fusion and when
routing weights and down-projection accumulators are rounded. The SiLU boundary
audit is in the [earlier diagnosis](ragged-precision-b200-2026-09-16.md).
GPT-OSS adds clipped SwiGLU and bias arithmetic; Gemma adds GELU-tanh and its
native scaled routing. GPU kernel tests cover these separate semantics.

**The HF FP32 control does not isolate or prove every custom vLLM operation.**
BF16 serving still differs, and the tested prompt set is small. The new report
retains these failed BF16 sanity gates (cosine >= 0.995 and KL <= 0.01):

- Gemma-4-26B-A4B L: cosine minimum 0.9719316, KL maximum 0.01145153

Sampled GPT-OSS FP32 expert diagnostics:

- GPT-OSS L: 52/108 sampled expert comparisons exceed `rtol=1e-4, atol=1e-5`; maximum absolute error 0.005859375. Exact retained-weight/bias and removed-zero-column checks pass for all 4,608 experts.
- GPT-OSS G: 47/108 sampled expert comparisons exceed `rtol=1e-4, atol=1e-5`; maximum absolute error 0.001953125. Exact retained-weight/bias and removed-zero-column checks pass for all 4,608 experts.

The earlier Qwen3 L/G KL failures also remain in their original report. Numerical tolerances are unchanged. Under the final acceptance scope,
`--allow-logit-drift` records numerical failures while exact retained weights,
removed zero columns and checkpoint reload are mandatory. The matrix explicitly records gate failures while independently
checking physical export, exact reload and successful inference.

## What is loaded and pruned

- GPT-OSS uses official revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
  Published experts are MXFP4. Both paths explicitly dequantize the **same source
  to BF16** before calibration. Sampled FP32 expert checks promote those same
  dequantized weights; full-model GPT-OSS FP32 controls were not run.
  This experiment cannot recover or measure information lost to source MXFP4
  quantization, and does not compare native MXFP4 serving with BF16 serving.
- Gemma is `google/gemma-4-26B-A4B`, revision
  `24548b62aa021d562695c04aaf7758a1ea47990b`, the base model. Explicit prefix mapping
  loads the complete language tower; missing language weights cause failure.
- Qwen3.5 and Gemma use all original language layers/experts/hidden dimensions
  before pruning. Vision/audio/MTP weights are excluded. Shared experts, dense
  MLP branches, routers, norms and attention are retained.
- GPT-OSS gate/up biases are physically reduced with their neurons; down biases
  remain, including for zero-width experts. Version 2 metadata records the
  activation/bias semantics. Existing SiLU checkpoints retain version 1.

| Model | Scope | Parameters: original → compact | Retained width range | Distinct widths |
| --- | --- | --- | --- | --- |
| GPT-OSS-120B | L | 116,829,156,672 → 59,484,992,832 | 0–2880 | 507 |
| GPT-OSS-120B | G | 116,829,156,672 → 59,484,992,832 | 0–2880 | 312 |
| Qwen3.5-122B-A10B | L | 122,111,526,912 → 64,129,468,416 | 0–1024 | 888 |
| Qwen3.5-122B-A10B | G | 122,111,526,912 → 64,129,468,416 | 0–1024 | 884 |
| Qwen3.5-35B-A3B | L | 34,660,610,688 → 18,554,483,328 | 0–512 | 256 |
| Qwen3.5-35B-A3B | G | 34,660,610,688 → 18,554,483,328 | 0–512 | 155 |
| Gemma-4-26B-A4B | L | 25,233,141,760 → 13,814,149,120 | 0–704 | 364 |
| Gemma-4-26B-A4B | G | 25,233,141,760 → 13,814,149,120 | 0–704 | 378 |

A separate read of every saved safetensors header confirms that on-disk
parameter counts equal the compact counts, and that every expert tensor has
the flat shape implied by its retained widths. These files contain compact
weights, not original-size tensors with zeros or padded expert widths.
Gemma also stores 30 required BF16 `layer_scalar` buffers (one per layer);
the header audit counts these separately from parameters and includes their
60 bytes in total tensor storage.

The exact parameter decrease equals removed neurons times `3*hidden_size`, plus
2 gate/up bias parameters per removed GPT-OSS neuron. Fifty percent of routed
neurons is not fifty percent of total model parameters.

## Environment and test settings

- Existing image `less-is-moe:x9zou-intdim-gpu`, ID
  `sha256:959d2ed008772283e20a0d7302eb9663be56c1dea51122e2451ce3ff1f1d3cb1`.
  Python 3.12.14, Torch 2.13.0+cu130, Transformers 5.17.0, vLLM 0.29.0,
  tokenizers 0.23.1, CUDA 13.0. Third-party pins are unchanged; PR source is
  installed with `uv pip install --no-deps --no-build-isolation -e .`.
- B200 SM100. GPT-OSS and Qwen3.5-122B use GPUs 5/6/7, GPU-only HF balanced
  placement and vLLM PP=3. Smaller full runs use one GPU. Within each comparison,
  baseline and compact have identical parallelism. TP=DP=1 throughout.
- BF16 serving; eager mode, Triton attention; seed 7, temperature 0, ignore EOS,
  max context 256, max sequences 2, max batched tokens 256, GPU utilization 0.6.
- Same four calibration texts, max 64 tokens each; same two held-out prompts.
  Text/token IDs and model-specific lengths are recorded in JSON.
- **101 GPU regressions pass**, including 79 ragged tests and 22 existing runtime
  tests in FP32/BF16. Wrong layouts, dead experts, zero-width GPT bias, all-zero
  layers, unaligned widths, checkpoint preservation and activation variants are
  covered. GPT L/G PP=2 random fixtures and four-layer hybrid Qwen L/G PP=2
  fixtures also complete stock and compact inference. Random fixtures are
  separate from the full-weight evidence above.

## Issues encountered and resolved

1. Full Gemma requires explicit causal-LM selection and a multimodal-prefix map.
   Early attempts were rejected/aborted; their invalid artifacts were not used.
   A strict loading-key guard and regression now prevent missing text weights.
2. The original 122B layout probe sampled eight nonzero but negligible units in
   layer 12, all producing zero measured output change. Disabling TF32 did not
   fix it. Probe candidates now rank gate/up/down magnitudes, with the same
   verification tolerance and unique-layout requirement. The real layer's
   correct layout passes and the wrong layout fails. This changes probe sample
   selection only, not calibration scores or pruning decisions. A sparse-signal
   regression and the existing runtime tests pass.
3. Stock vLLM 0.29.0 cannot initialize a hybrid PP stage with no linear-attention
   layer. The two-layer PP fixture exposed this; four layers give each stage
   both layer types. The complete 48-layer 122B PP=3 partition has linear
   attention in every stage; the custom adapter rejects invalid partitions.
4. The first full 122B FP32 promotion exhausted memory with the default allocator
   (154.62 GiB allocated plus 21.64 GiB reserved but unused on one GPU).
   The precision controls use `PYTORCH_ALLOC_CONF=expandable_segments:True`
   on the same three GPUs. The failure log is retained; arithmetic precision
   and contexts are unchanged.
5. The first full GPT-OSS attempt exceeded the sampled FP32 tolerance at one
   element (absolute error 1.78814e-5). After acceptance was narrowed to physical
   compaction and inference, every expert received exact retained-weight/bias
   and removed-zero-column checks. Sampled FP32 errors remain recorded under
   `--allow-logit-drift`, rather than blocking a structurally exact export.
6. Balanced HF reload may place a tensor on a different GPU. The exact weight
   check transfers it to the original tensor's GPU before `torch.equal`;
   it does not change values or loosen equality.

## Artifacts and provenance

Logs, calibration scores, diagnostic tensors, and JSON are on B200 under
`/raid/x9zou/less-is-moe-expanded.ZIPfHV/results`. Gemma compact checkpoints are
there too. Qwen3.5-35B evidence/checkpoints remain under
`/raid/x9zou/less-is-moe-four.YHONHJ/results/qwen35`.

**The four 120B-class compact checkpoints are RAM-backed** at
`/dev/shm/x9zou-less-is-moe-expanded-ZIPfHV/{gptoss,qwen35122}/{layer,global}/compact`,
linked from their result directories, because RAID did not have sufficient
free disk space. They survive container exit but are lost on host reboot.
Compact config/index copies and experimental evidence are retained on RAID.
Move the checkpoints to a sufficiently large persistent volume for long-term
retention. Temporary zero-mask baselines are removed after comparison.

Each matrix records its actual launch source hashes. Qwen3.5-35B generation
reuses the completed original full run; its L FP32 control is newly added.
Both L/G compact checkpoints were also served again with the final expanded
backend on an idle fourth B200 (host index 4), exactly reproducing all four
previous 128-token completions. The rerun output and source hashes are included.
Gemma full runs preceded the probe-selection fix. Final source hashes and
additional regression results are recorded separately; no claim is made that
historical experiments ran a later commit. Qwen3.5 local source caches have no
recorded upstream revision; their source config hashes, complete shard manifests
and host paths identify the inputs.

Reproduction commands are in [the backend documentation](../RAGGED_EXPERTS.md).
Run `docker.ragged_precision_diagnostic` for both scopes after the shared matrix,
with `PYTORCH_ALLOC_CONF=expandable_segments:True` for the completed 122B FP32
controls. No GPT-OSS full-model FP32 control is included.
This is full-weight feasibility and numerical diagnosis, not a quality,
perplexity, throughput or production deployment evaluation.
