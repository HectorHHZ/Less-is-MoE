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
- GPU inference only. vLLM **0.29.0**, BF16, eager mode. Tensor parallelism
  splits each expert's retained neurons across GPUs. Pipeline parallelism
  partitions whole layers. Data parallelism replicates the compact model across
  serving engines, with optional TP inside each replica. HF uses balanced GPU placement when
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

For two-way tensor parallelism, expose two allocated GPUs (for example
`--gpus '"device=0,1"'`) and use `--tensor-parallel-size 2`. Add
`--max-num-seqs 2` to allow two concurrent sequences, then submit two prompts
in one completions request or send two concurrent requests. This scheduler
limit alone does not create a batch. The same compact checkpoint works at
TP=1/2/4/8 without re-pruning or changing its config, provided the model
satisfies the native divisibility rules below. For TP=4 or TP=8, expose four
or eight allocated GPUs and set `--tensor-parallel-size 4` or `8`.

For pipeline parallelism, use `--pipeline-parallel-size N` and expose
`TP * N` allocated GPUs. Qwen3.5 requires a linear-attention layer in each
pipeline stage with vLLM 0.29.0. Combined TP and PP requires separate validation.

### Tensor parallelism and larger batches

Each rank `r` in a TP group of size `P` loads the retained-neuron interval
`[floor(I_e*r/P), floor(I_e*(r+1)/P))` of expert `e`. Gate/up rows and down
columns use the same interval. Odd widths, widths smaller than `P` and empty
experts require no padding. Original expert IDs and routing remain unchanged.
The shared ragged kernels compute local contributions; a TP all-reduce sums
them before shared-expert addition or downstream normalization. GPT-OSS gate/up
biases follow the neuron slices; rank zero alone contributes the down bias,
including for an empty expert. Upstream vLLM handles attention, embedding,
dense and shared-expert tensor parallelism.
GPT-OSS attention sinks are also sliced by head in the compact weight loader,
following its upstream loader.

There is no hardcoded TP=2 or batch-size=2 limit. Larger TP must still satisfy
the original model's attention/head and dense/shared projection divisibility
rules. Narrower expert shards can underutilize GPU tiles, and the routed branch
adds one all-reduce per MoE layer. With the pinned vLLM attention implementations,
OLMoE and GPT-OSS specifically require TP to divide `num_key_value_heads`;
replicating KV heads across a larger TP group is rejected early.
Loading currently slices global checkpoint
tensors on each rank, so startup I/O and host-memory pressure also need checking
at larger TP. Different BF16 reduction orders can change logits and tokens.

For larger batches, increase `--max-num-seqs` within the GPU memory budget.
KV-cache usage grows with concurrent sequence lengths, and temporary expert
buffers grow with scheduled tokens and top-k (including the largest local
expert width). Prefill may schedule many more tokens than the request count.
The scheduler's token budget and available memory remain constraints; higher
TP or batch size is not a guarantee of higher throughput.

### Data parallel serving

Use vLLM's native API server and internal load balancing. For example, two
replicas, each split across two GPUs, require **four allocated GPUs**:

```bash
docker run --rm --gpus '"device=0,1,2,3"' --shm-size=8g -p 8000:8000 \
  -e VLLM_PLUGINS=less_is_moe_ragged \
  --mount type=bind,src=/absolute/path/outputs,dst=/outputs,readonly \
  less-is-moe:dev-unified \
  vllm serve /outputs/qwen-layer \
  --dtype bfloat16 --enforce-eager \
  --tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 4
```

Each DP replica has its own KV cache and a complete set of routed experts;
only the TP ranks **within that replica** partition each expert and sum its
output. The compact backend does not use stock FusedMoE's DP×TP expert
partitioning or all-to-all dispatch. The native vLLM coordinator still manages
MoE forward waves, including idle replicas. Routing within a request remains
unchanged, and no checkpoint/config conversion is needed.

Submit concurrent requests to the same endpoint; vLLM distributes them across
replicas. `--max-num-seqs` is a per-replica scheduling ceiling. DP=2/TP=1 uses
two GPUs; DP=2/TP=2 uses four; DP=2/TP=4 uses eight. DP replicates memory rather
than making one model fit in less total memory. These counts assume PP=1.

Use `vllm serve` or the native `AsyncLLM` engine for this mode. In vLLM 0.29.0,
single-process synchronous `LLM(data_parallel_size=2)` is not supported with
the default executor. See [vLLM's DP serving guide](https://docs.vllm.ai/en/v0.29.0/serving/data_parallel_deployment/).
The delivery validation covers single-node multiprocessing with PP=1;
multi-node/Ray, DP combined with PP, and overlapping microbatches require
separate validation. EP, EPLB and sequence parallelism remain unsupported.

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
