# Non-uniform Qwen3-MoE experts

IntDim-L/G selects neurons with the existing calibration-gradient algorithm.
The new `ragged` export mode physically removes those neurons even when each
expert keeps a different count. It requires the explicit HF loader or opt-in
vLLM plugin below. Uniform IntDim-E `structural` and stock zero-mask checkpoints
retain their existing behavior.

## Supported first implementation

- Qwen3-MoE, including Qwen3-Coder-30B-A3B; SiLU, bias-free routed experts,
  one MoE block per decoder layer. Other families are rejected before scoring.
- GPU inference only. vLLM **0.29.0**, BF16, one GPU (TP=PP=DP=1), eager mode.
- Unequal widths, widths not divisible by kernel tile size, zero-width experts,
  and all-zero expert layers preserve their original router IDs and weights.
- Quantization, LoRA, speculative decoding, CUDA Graph capture, sequence/expert
  parallelism and EPLB are not supported by this first backend.
- This is an inference feasibility implementation, not a throughput claim.

## Format

`config.json` has `architectures: ["RaggedQwen3MoeForCausalLM"]` and:

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
`moe_intermediate_size` remains the original scalar width. The custom loader
uses `expert_intermediate_sizes` as authoritative for compact expert tensors.

For each layer, two flat safetensors parameters retain the names
`model.layers.L.mlp.experts.gate_up_proj` and `...down_proj`:

- Concatenate each expert's contiguous `[2*I_e, H]` gate/up matrix (gate first).
- Concatenate each expert's contiguous `[H, I_e]` down matrix.

Prefix sums of widths determine tensor offsets. No per-neuron indices are
needed during inference: all three projections were compacted consistently.
The pruning summary retains the original dropped indices for auditing.
No Tensor Parallel shard sizes or GPU launch settings are embedded in this
portable format. Unsupported versions, layers and widths fail validation.

## Prune and load

Use the existing unified GPU Docker and install this revision of the package.
All third-party versions remain pinned. The model must fit entirely on GPU.

```bash
python -m less_is_moe.intdim.prune \
  --model_name_or_path /models/Qwen3-Coder-30B-A3B-Instruct \
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

This registers a new architecture; it does not replace the stock Qwen classes
or enable the historical model patches. Model files do not execute remote code.

## GPU kernel

The backend reuses vLLM's GPU token-to-expert alignment. The first Triton GEMM
reads each expert's compact gate/up matrix and applies SiLU. The second GEMM
uses that expert's actual intermediate width as its reduction length. Width
bounds skip unused column tiles. Routing and output reduction stay on GPU.
Weights contain no padding. Intermediate activation scratch currently uses the
largest retained width, and the launch grid reserves column tiles for that
width; unused tiles return without matrix arithmetic. This is not yet an
optimal ragged tile scheduler. The HF reference intentionally uses per-expert
GPU operations and is a correctness oracle, not the accelerated backend.

## Reproduce validation

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
BF16 logits are not expected to be bitwise identical. The harness checks three
experts in every layer against their zero-masked counterpart in FP32 on GPU
(`rtol=1e-4`, `atol=1e-5`, TF32 disabled). It separately records BF16 logit
maximum error, RMSE, cosine similarity, KL divergence and first-token agreement.
For full checkpoints, cosine >= 0.995 and reference-to-compact KL <= 0.01 are
numerical sanity gates on the two smoke prompts, not an accuracy benchmark or
a claim that every output token will match. Tiny fixtures retain a direct
elementwise BF16 comparison. The first full-model elementwise tolerance check
failed; that difference is retained in the validation report rather than
being described as exact equivalence.
