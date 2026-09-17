# Non-uniform MoE experts

IntDim-L/G selects neurons with the existing calibration-gradient algorithm.
The new `ragged` export mode physically removes those neurons even when each
expert keeps a different count. It requires the explicit HF loader or opt-in
vLLM plugin below. Uniform IntDim-E `structural` and stock zero-mask checkpoints
retain their existing behavior.

## Supported implementation

- Qwen1.5-MoE (`qwen2_moe`), OLMoE, Qwen3-MoE and the Qwen3.5-MoE text tower:
  SiLU, bias-free routed experts, one MoE block per decoder layer.
  Other audited variants are GPT-OSS (biased, clipped SwiGLU) and Gemma4
  (GELU-tanh and per-expert routing scales). They share the packed GPU GEMMs
  with activation/bias specializations. Unsupported families fail before scoring.
- Shared experts, gates and each family's routing normalization remain intact.
  Qwen3.5 retains every language layer, including Gated DeltaNet attention;
  vision and MTP weights are not part of the exported causal language model.
  Gemma4 likewise exports its complete language tower and retains its dense MLP
  branch, norms and routing scales. Its multimodal checkpoint prefix is mapped
  explicitly; missing language-model weights fail loading.
- GPU inference only. vLLM **0.29.0**, BF16, TP=DP=1, eager mode. Pipeline
  parallelism partitions whole layers across GPUs for 120B-class models; expert
  tensor parallelism remains unsupported. HF uses balanced GPU placement when
  multiple GPUs are visible, and rejects CPU/disk offload.
- Unequal widths, widths not divisible by kernel tile size, zero-width experts,
  and all-zero expert layers preserve their original router IDs and weights.
- Quantization, LoRA, speculative decoding, CUDA Graph capture, sequence/expert
  parallelism and EPLB are not supported by this first backend.
- This is an inference feasibility implementation, not a throughput claim.

## Format

`config.json` uses the corresponding registered architecture:

| Original model type | Compact architecture |
| --- | --- |
| `qwen2_moe` | `RaggedQwen2MoeForCausalLM` |
| `olmoe` | `RaggedOlmoeForCausalLM` |
| `qwen3_moe` | `RaggedQwen3MoeForCausalLM` |
| `qwen3_5_moe_text` | `RaggedQwen3_5MoeForCausalLM` |
| `gpt_oss` | `RaggedGptOssForCausalLM` |
| `gemma4_text` | `RaggedGemma4ForCausalLM` |

The original four SiLU families retain version 1 metadata:

```json
{
  "less_is_moe": {
    "format_version": 1,
    "weight_layout": "flat_gate_up_down_v1",
    "expert_intermediate_sizes": {"0": [256, 512, 768], "1": [384, 640, 512]}
  }
}
```

Layer IDs and original expert IDs determine the order. The example has two
layers and three experts; a real config must describe every layer/expert.
`moe_intermediate_size` (OLMoE/GPT-OSS: `intermediate_size`) remains the original scalar width. The custom loader
uses `expert_intermediate_sizes` as authoritative for compact expert tensors.

For each layer, two flat safetensors parameters retain the names
`model.layers.L.mlp.experts.gate_up_proj` and `...down_proj`:

- Concatenate each expert's contiguous `[2*I_e, H]` gate/up matrix (gate first).
- Concatenate each expert's contiguous `[H, I_e]` down matrix.

Gemma4 uses `model.layers.L.experts` as the expert prefix.

Prefix sums of widths determine tensor offsets. No per-neuron indices are
needed during inference: all three projections were compacted consistently.
The pruning summary retains the original dropped indices for auditing.
GPT-OSS and Gemma4 use `format_version: 2`, `weight_layout:
flat_gate_up_down_v2`, and explicit `expert_options` (`activation` and `bias`).
For GPT-OSS, gate/up biases are also packed in gate-then-up order; down biases
remain `[E,H]`. A zero-width GPT-OSS expert still contributes its routed down
bias. Its clipped SwiGLU retains alpha 1.702, limit 7 and the `up + 1` offset.

The official GPT-OSS checkpoint is MXFP4. The source loader explicitly
**dequantizes it to BF16 for both zero-mask and compact paths**. Native MXFP4
serving is not compared with BF16 serving. Normalization modules retain the
upstream loader's precision policy. This backend does not serve quantized
compact weights or recover precision lost in the published source weights.

No Tensor Parallel shard sizes or GPU launch settings are embedded in this
portable format. Unsupported versions, layers and widths fail validation.

## Prune and load

Use the existing unified GPU Docker and install this revision of the package.
All third-party versions remain pinned. The model must fit entirely on GPU.

```bash
python -m less_is_moe.intdim.prune \
  --model_name_or_path /models/Qwen3-30B-A3B \
  --output_dir /outputs/qwen-layer \
  --mode ragged --prune_mode layer --drop_ratio 0.5 \
  --calib_data /data/calibration.jsonl --n_samples 128 --seq_len 256 --dtype bf16
```

Use `--prune_mode global` for IntDim-G. HF GPU reference loading:

```python
from less_is_moe.intdim.ragged import load_checkpoint
model = load_checkpoint("/outputs/qwen-layer")
```

vLLM must load the new plugin in every worker. The Docker default disables all
plugins; explicitly enable only this one for compact checkpoints:

```bash
VLLM_PLUGINS=less_is_moe_ragged vllm serve /outputs/qwen-layer \
  --dtype bfloat16 --tensor-parallel-size 1 --enforce-eager
```

This registers separate architectures; it does not replace the stock model classes
or enable the historical model patches. Model files do not execute remote code.
The implementation lives in this repository's plugin and Triton modules;
the installed vLLM package and its pinned version are unchanged.

## GPU kernel

The backend reuses vLLM's GPU token-to-expert alignment. The first Triton GEMM
reads each expert's compact gate/up matrix and applies the family's activation
and optional bias. The second GEMM
uses that expert's actual intermediate width as its reduction length. Width
bounds skip unused column tiles. Routing and output reduction stay on GPU.
Weights contain no padding. Intermediate activation scratch currently uses the
largest retained width, and the launch grid reserves column tiles for that
width; unused tiles return without matrix arithmetic. This is not yet an
optimal ragged tile scheduler. The HF reference intentionally uses per-expert
GPU operations and is a correctness oracle, not the accelerated backend.

## Reproduce validation

Completed B200 evidence: [expanded GPT-OSS, Qwen3.5 and Gemma coverage](validation/ragged-expanded-b200-2026-09-17.md),
[original four full models and eight L/G cases](validation/ragged-four-models-b200-2026-09-16.md),
and [the earlier BF16/FP32 diagnosis](validation/ragged-precision-b200-2026-09-16.md).

Acceptance is physical removal of the same neurons selected for zero-masking,
saved compact weights/config, exact HF reload and successful adapted-vLLM
inference. Generation equality is recorded for diagnosis, not required for
this feasibility check. No end-to-end throughput improvement is claimed.

One shared full-model harness runs both IntDim-L and IntDim-G at 50%, saves
both compact checkpoints, verifies exact HF reload, and generates with stock
zero-mask vLLM and adapted compact vLLM in separate processes:

```bash
python -m docker.ragged_model_matrix --case qwen15 --model /models/Qwen1.5-MoE-A2.7B --output /results/qwen15
python -m docker.ragged_model_matrix --case olmoe --model /models/OLMoE-1B-7B-0924 --output /results/olmoe
python -m docker.ragged_model_matrix --case qwen3 --model /models/Qwen3-30B-A3B --output /results/qwen3
python -m docker.ragged_model_matrix --case qwen35 --model /models/Qwen3.5-35B-A3B --output /results/qwen35
python -m docker.ragged_model_matrix --case gemma4 --model /models/Gemma-4-26B-A4B --output /results/gemma4
# Three visible GPUs for full 120B-class BF16 calibration and PP=3 serving:
python -m docker.ragged_model_matrix --case qwen35122 --model /models/Qwen3.5-122B-A10B --output /results/qwen35122 --pipeline-parallel-size 3
python -m docker.ragged_model_matrix --case gptoss --model /models/gpt-oss-120b --output /results/gptoss --pipeline-parallel-size 3
```

Run sequentially per selected GPU group. Both baseline and compact runs use
the same pipeline-parallel size. Ensure enough GPU memory for calibration
weights plus gradients, and enough checkpoint storage for the selected models. `--scratch` defaults to `/dev/shm` for temporary
zero-mask baselines; use a disk directory with enough space if needed. Compact
checkpoints remain under `<output>/{layer,global}/compact`. The harness uses
the same four short calibration texts, two held-out prompts and 128 generated
tokens per prompt for each model. Tokenization differs by model. It verifies
the original layer/expert/hidden dimensions and the exact 50% neuron budget.
Use `--max-tokens` to change generation length. Each matrix records token
match counts, common-prefix lengths and the first differing generated token.
This is a full-weight pipeline smoke test, not a quality or speed benchmark.

From the repository root, inside the pinned image with a visible B200:

```bash
uv pip install --no-deps --no-build-isolation -e .
python -m pytest -q tests/test_ragged_intdim.py
export VLLM_PLUGINS=less_is_moe_ragged
python -m docker.ragged_qwen_smoke prepare --model tiny --scope layer --output /results/tiny-layer
python -m docker.ragged_qwen_smoke generate --output /results/tiny-layer --checkpoint masked
python -m docker.ragged_qwen_smoke generate --output /results/tiny-layer --checkpoint compact
```

Repeat with `--scope global` and a separate output directory. `tiny` is an
explicit random, two-layer architecture fixture; it is not pretrained evidence.
For a complete model, replace `tiny` with its local checkpoint path. The harness
uses the entire checkpoint, four short calibration examples and two held-out
prompts; there is no layer, expert or hidden-size reduction before pruning.
These examples establish pipeline feasibility, not model quality.

Run HF preparation and vLLM initialization **serially** on a dedicated GPU.
vLLM profiles available memory at startup; another process allocating model
weights during that interval can invalidate its KV-cache memory estimate.

The harness stores zero-mask and compact checkpoints, exact HF reload checks,
logit errors, widths, parameter counts, memory, environment, and vLLM output
tokens. Prepare and generate run in separate processes. Compare stock vLLM
zero-mask with custom vLLM compact under identical settings before attributing
any future measured speedup to pruning.

### Numerical checks

The compact weights reload bitwise. Removing zero columns can change BF16 GEMM
rounding; later top-k routers can amplify small differences, so full-model
BF16 logits are not expected to be bitwise identical. The current harness exactly checks retained weights/biases and removed zero
down columns for every expert. It also compares three experts in every layer
against their zero-masked counterpart in FP32 on GPU
(`rtol=1e-4`, `atol=1e-5`, TF32 disabled). It separately records BF16 logit
maximum error, RMSE, cosine similarity, KL divergence and first-token agreement.
For full checkpoints, cosine >= 0.995 and reference-to-compact KL <= 0.01 are
numerical sanity gates on the two smoke prompts, not an accuracy benchmark or
a claim that every output token will match. The standalone harness fails when
these gates fail by default. The full-model matrix explicitly passes
`--allow-logit-drift`: it preserves the failed numerical check in the report
and independently completes checkpoint/reload/inference validation. FP32 expert tolerance failures are also recorded when this flag is enabled.
Exact retained-weight/removed-column checks and exact reload remain mandatory. An inference pass must not be
reported as a passed numerical-equivalence check. Tiny fixtures retain a direct
elementwise BF16 comparison. The first full-model elementwise tolerance check
failed; that difference is retained in the validation report rather than
being described as exact equivalence.
