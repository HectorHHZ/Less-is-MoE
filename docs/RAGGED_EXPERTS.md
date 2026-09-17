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

Build the existing root Dockerfile from this revision. It installs the pruning
implementation and vLLM plugin into the image with the same locked dependencies;
no source mount or package installation is needed when running the container.

```bash
docker build --platform linux/amd64 \
  --build-arg REVISION="$(git rev-parse HEAD)" --build-arg VERSION=dev \
  -t less-is-moe:dev-unified .
```

Mount models, calibration data and outputs as described in
[Docker usage](DOCKER.md#mount-data-and-update-dependencies). Inside that GPU
container, prune with the command below. The full language model must fit on
the visible GPUs.

```bash
python -m less_is_moe.intdim.prune \
  --model_name_or_path /models/Qwen3-30B-A3B \
  --output_dir /outputs/qwen-layer \
  --mode ragged --prune_mode layer --drop_ratio 0.5 \
  --calib_data /data/calibration.jsonl --n_samples 128 --seq_len 256 --dtype bf16
```

Use `--prune_mode global` for IntDim-G. Both modes directly compact the same
calibration-selected neuron plan used by zero-masking. The separate
`--from_zeroed_model` option remains limited to uniform IntDim-E structural
conversion.

HF GPU reference loading:

```python
from less_is_moe.intdim.ragged import load_checkpoint
model = load_checkpoint("/outputs/qwen-layer")
```

vLLM must load the new plugin in every worker. The Docker default disables all
plugins; explicitly enable only this one for compact checkpoints:

```bash
docker run --rm --gpus 'device=0' --shm-size=8g -p 8000:8000 \
  -e VLLM_PLUGINS=less_is_moe_ragged \
  --mount type=bind,src=/absolute/path/outputs,dst=/outputs,readonly \
  less-is-moe:dev-unified \
  vllm serve /outputs/qwen-layer \
  --dtype bfloat16 --tensor-parallel-size 1 --enforce-eager
```

For a large model, expose multiple allocated GPUs and add
`--pipeline-parallel-size N`, where `N` is the number of visible GPUs. Qwen3.5
requires a linear-attention layer in each pipeline stage with vLLM 0.29.0.

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

Removing zero columns changes matrix reduction order, so BF16 outputs can
vary from a zero-mask baseline. Exact token equality and throughput improvement
are not guaranteed by this backend.
