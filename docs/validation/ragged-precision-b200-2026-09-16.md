# BF16 divergence diagnosis on full pretrained models

[Raw metrics, routing counts and vLLM token scores](ragged-precision-b200-2026-09-16.json).
[Eight-case 128-token experiment](ragged-four-models-b200-2026-09-16.md).

## Conclusion

The controls strongly support **finite-precision execution and different rounding
boundaries as the main cause** of the observed differences. This is not a
claim of bitwise BF16 equivalence, and it does not isolate every source of
error inside vLLM. No runtime kernel or original result was changed to obtain
these controls.

## Full-model FP32 control

For all five cases with differing vLLM completions, load the complete original
model, apply the same cached 50% L/G plan, and compare it with the saved compact
checkpoint. First evaluate both in BF16; then promote the **same BF16 weights**
to FP32 and disable TF32 for matmul and convolution. All model execution and
metric reductions run on B200. Original BF16 weights are not recovered to a
higher-precision pretrained source. Source and compact models load sequentially.

Each case checks both original prompts plus their common generated prefix
immediately before each first vLLM divergence. These are full HF forwards with
no KV cache, not a reproduction of vLLM's decode execution. Expert choices are
compared as sets for every token at every layer, ignoring top-k order.

| Model | Scope | Same-context checks | BF16 maximum logit error | FP32 maximum logit error | Changed token-layer route sets: BF16 → FP32 |
| --- | --- | --- | --- | --- | --- |
| qwen15 | L | 3 | 0.34375 | 2.19345e-05 | 41 → 0 |
| olmoe | L | 4 | 0.390625 | 1.4782e-05 | 62 → 0 |
| qwen3 | L | 4 | 1.09473 | 3.71933e-05 | 415 → 0 |
| qwen3 | G | 3 | 1.17969 | 2.76566e-05 | 427 → 0 |
| qwen35 | G | 3 | 0.523438 | 1.85966e-05 | 410 → 0 |

Across all **17 context pairs**, FP32 chooses the same next token and has zero
changed expert-selection sets. BF16 has 1,355 changed token-layer route sets;
its logit error drops by roughly four to five orders of magnitude in FP32.
The counts include overlapping prompt prefixes and are diagnostic counts,
not independent examples or an accuracy estimate.

This supports the following causal chain: compaction changes matrix shapes,
reduction order and rounding; perturbations can change top-k expert selection;
subsequent layers amplify differences; a changed next token gives the two
autoregressive runs different later inputs. The zero-mask and compact models
implement the same intended pruned function, but not the same finite-precision
execution trace. FP32 results support structural correctness on these contexts;
they do not prove every future input is equivalent.

## Concrete vLLM first-divergence evidence

Repeat Qwen3 IntDim-L with the original BF16 settings, retaining the top five
log probabilities at every generation step. Both engines independently
reproduce their previous 128-token outputs exactly for both prompts. This
rules out run-to-run variation for these repetitions.

- Code prompt, generated token **10**: stock selects ID `11`, custom selects
  `510`. Stock scores these tokens `-0.543246` and `-1.043246`; custom scores
  them `-1.826055` and `-0.326055`. The rankings reverse. The stock gap is 0.5,
  so this is not simply a tie at the final output: accumulated internal drift
  is material by this point.
- Explanation prompt, token **19**: stock gives IDs `374` and `61722` exactly
  equal reported scores (`-0.856256`) and selects `374`; custom scores them
  `-0.929842` and `-0.804842` and selects `61722`. This is an example of a tie
  or near tie being resolved differently.

At each first divergence both engines still see the same token history.
Afterward, positional token matches cannot measure same-context logit error.
Temperature zero makes each engine choose greedily; it cannot force two
numerically different engines to choose the same token.

## Why the two BF16 paths differ

The audited vLLM 0.29.0 code and PR implementation have different rounding
boundaries in addition to the changed matrix dimensions:

1. `RaggedMLP.forward` converts top-k routing probabilities to BF16 before the
   expert kernel. Stock routing can retain FP32 weights.
2. `_gate_up` explicitly rounds `SiLU(gate)` to BF16 before multiplying by
   `up`, matching HF eager. Stock vLLM calls its fused `silu_and_mul` operator.
3. `_down` rounds the down-projection accumulator to BF16 before applying the
   routing weight. Stock Triton `fused_moe_kernel` multiplies its FP32
   accumulator by that weight **before** the final output-dtype conversion
   (`fused_moe.py`, installed 0.29.0 source, lines 593–603).
4. The compact reduction length and tiling differ from the zero-padded
   rectangular computation; floating-point addition is order-dependent.

Thus “both use BF16” does not mean “both round at the same places.” The current
custom backend follows HF reference boundaries, not bitwise stock-vLLM
boundaries. These are precision-related implementation differences; the FP32
HF control does not, by itself, certify the custom Triton kernel on all inputs.
GPU kernel regressions and full inference checks provide separate evidence.
PyTorch also documents that mathematically identical computations can differ
with batching, backend and reduction order in its
[numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

## Interpretation and next steps

- Keep the original Qwen3 BF16 KL-gate failures visible; do not loosen the gate
  or label all eight cases numerically equivalent.
- Align the custom kernel's routing-weight dtype, activation fusion and
  down-projection rounding with stock vLLM before claiming closer BF16 parity.
  Changing these boundaries is a separate runtime change requiring reruns.
- For diagnosis, use identical contexts, logits, top-k margins and route sets.
  Free-running 128-token equality is useful but insufficient on its own.
- FP32 here is a diagnostic control, not a proposed serving configuration or
  a quality/performance benchmark. Only four short calibration texts and two
  held-out prompts were used.

## Reproduce

```bash
python -m docker.ragged_precision_diagnostic \
  --model /work/models/Qwen3-30B-A3B --results /work/results/qwen3 \
  --scope layer --masked-dir /dev/shm/diagnostic-qwen3-layer

VLLM_PLUGINS=less_is_moe_ragged python -m docker.ragged_vllm_diagnostic \
  --checkpoint /dev/shm/diagnostic-qwen3-layer \
  --results /work/results/qwen3/layer --kind masked
VLLM_PLUGINS=less_is_moe_ragged python -m docker.ragged_vllm_diagnostic \
  --checkpoint /work/results/qwen3/layer/compact \
  --results /work/results/qwen3/layer --kind compact
```

Run sequentially on an idle B200 in the pinned Docker. For the other cases,
substitute their full source, result directory and scope. Raw logits and route
sets are retained as `precision-tensors.pt` beside each compact checkpoint on
B200; summarized metrics and script hashes are in the committed JSON.
