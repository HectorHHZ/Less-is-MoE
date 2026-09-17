# Unified GPU runtime and IntDim-E validation

The unified image builds on the generic `intdim` implementation introduced in
PR #24 and the unified pruning pipeline from PR #26. Its GPU checks establish two things: the new IntDim-E path preserves
the released algorithm, and structurally pruned checkpoints can use stock HF
and vLLM implementations in this fixed environment.

## Fixed stack

| Component | Version |
| --- | --- |
| Platform | Linux x86-64 / Ubuntu 24.04 |
| CUDA toolkit | 13.0.3, base image pinned by digest |
| Python | 3.12.14 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.17.0 |
| vLLM | 0.29.0 |
| Tokenizers | 0.23.1 |
| uv | 0.12.5, image pinned by digest |

`environments/unified/requirements.in` specifies the core stack. Its generated
`requirements.txt` locks all transitive dependencies, the project's base and
test requirements, and installer/build tools. Installation checks hashes and
uses wheels only, without third-party metadata overrides. Only the local
project is installed with `--no-deps --no-build-isolation` after its dependencies
and build tools have been installed from the lock. `pip check` validates the
result. The original `setup.sh` reproduction profiles are unchanged.

The image records Python packages in `/opt/less-is-moe-environment.txt` and
OS packages in `/opt/less-is-moe-system-packages.txt`. Apt repositories still
supply OS utility updates; rebuilding is not promised to be byte-identical.
Use the published image **digest** to reproduce exactly the same filesystem.

## GPU validation scope

All model execution in this validation suite runs on GPU, using small random
models rather than downloaded pretrained weights. Equivalence tests have one
entry point: **`tests/test_intdim_equivalence.py`**.

### One common matrix for seven configurations

`test_uniform_intdim_e` runs the same workflow for each row in FP32 and BF16
(14 cases). All configurations use two layers, hidden size 128, four experts,
top-2 routing, two fixed six-token calibration sequences, and 50% removal per
expert. Logits are checked on both calibration sequences and a held-out
four-token sequence.

| Fixture | Expert width before -> after | Independent reference |
| --- | --- | --- |
| Qwen2-MoE / Qwen1.5-MoE | 256 -> 128 | Released Qwen1.5 per-expert functions |
| Qwen3-MoE | 256 -> 128 | Released Qwen3 per-expert functions |
| OLMoE | 256 -> 128 | Released OLMoE per-expert functions |
| Qwen3.5-35B-A3B | 512 -> 256 | Released Qwen3.5 functions |
| Qwen3.5-122B-A10B | 1024 -> 512 | Released Qwen3.5 functions |
| GPT-OSS-120B | 2880 -> 1440 | Explicit GPT-OSS port of the released method |
| Gemma-4-26B-A4B | 704 -> 352 | Explicit Gemma 4 port of the released method |

Every row checks:

1. Independent reference scoring and bottom-k selection. Autodetect APIs are
   made to raise while the reference runs, before automatic scores/IDs exist.
2. Reference zero-masking, with unchanged config/shapes and an audit of every
   masked and untouched tensor.
3. The actual `intdim.prune` scoring and selection APIs, followed by its public
   pipeline for E/L/G zero-masking and E structural pruning. All three mask
   plans, tensors and logits match independent legacy selection/manual masks.
   Kept IDs and weights/biases are exact; `--from_zeroed_model` produces the
   same structural tensors as direct pruning.
4. Full-model masked/structural logits. For the five cases with released
   structural functions, their weights and logits must also match the generic
   structural path exactly.
5. Both zero-mask and structural stock HF checkpoints: save, verify, reload,
   exact tensor/logit recovery, and discovery of the reduced width.
6. With `INTDIM_TEST_VLLM=1`: stock vLLM generation from original, zero-mask and
   structural BF16 checkpoints. Masked and structural token IDs must match.
   This is **21 generations**: three checkpoints for each of seven cases.

The Qwen2/Qwen3/OLMoE reference functions require per-expert Linear modules.
The common comparison explicitly reconstructs that layout from the same stock
weights and uses it in both scoring branches. Both resulting checkpoints are
then independently repacked into native fused storage for stock HF/vLLM;
repacking must preserve outputs within the logit tolerance. No autodetect
helper is used to construct the reference or repack its checkpoint. Separate
native-layout comparisons described below retain coverage of the automatic
fused path and its GEMM rounding differences.

The two model-specific ports are `pruning/neuron_drop_gpt_oss.py` and
`pruning/neuron_drop_gemma4.py`. They implement the released mean-absolute-gradient
reduction using explicit model paths/tensor axes and reuse the unchanged
released bottom-k selector. They are newly implemented references, not
historically released implementations for those two models. GPT-OSS's
interleaved gate/up weights and biases are zeroed together with down rows;
down/output bias is preserved. Gemma 4 zeroes concatenated gate/up rows and
down columns, preserving the dense MLP and router. These callable APIs support
stock floating-point text models, not packed MXFP4 or multimodal wrappers:

```python
from less_is_moe.pruning import neuron_drop_gpt_oss as reference
# For Gemma4ForCausalLM, import neuron_drop_gemma4 instead.
layers = list(range(model.config.num_hidden_layers))
scores = reference.collect_neuron_gradient_scores(model, batches, layers)
dropped = reference.decide_neurons_to_drop(scores, drop_ratio=0.5)
reference.zero_dropped_neurons(model, dropped, layers)
model.save_pretrained(zero_mask_directory)
```

The common matrix requires bitwise equal scores for released same-layout
references. GPT-OSS/Gemma 4 scores allow FP32 reduction rounding with
`rtol=1e-6, atol=0`; **kept IDs and tensors must still be exact for every case**.
Masked/structural and repacked logits allow FP32 `rtol=1e-5, atol=1e-5` and BF16
`rtol=0.02, atol=0.002`. Saved/reloaded tensors and logits must be bitwise equal.
GPT-OSS tests use nonzero expert biases. Gemma 4 disables per-layer input
embeddings, and all fixtures use text inputs.

### Additional regression coverage

The same equivalence file retains the original **14 small-layout comparisons**:
four released families in the same layout and three old Linear vs native fused
comparisons, each in FP32/BF16. Same-layout comparisons are bitwise equal.
Native-layout score tolerances are FP32 `rtol=1e-5, atol=1e-8` and BF16
`rtol=0.02, atol=1e-8`; logits use FP32 `rtol=1e-5, atol=1e-6` and BF16
`rtol=0.02, atol=0.002`. Selected IDs and kept weights remain exact.

`test_uniform_prune_cli` adds **14 GPU CLI cases** in the same equivalence
file: seven configurations in FP32/BF16. Each executes actual tokenizer/file
calibration, stock model loading, direct structural pruning, zero-masking,
mask-to-structural conversion, save and verification. All loaded parameters
must be on CUDA; expert execution is explicitly set to stock eager, matching
the other HF tests. Reloaded direct/converted structural weights are exact,
and their logits match the masked checkpoint.

`tests/test_intdim_runtime.py` keeps **22 structural/probe/scoring regressions**
separate from method equivalence: six families in FP32/BF16, probe-failure
preservation, dead-expert/unit discovery, empty calibration rejection and stale
gradient isolation. This preserves tiny/square layout cases, random unsorted
masks and exception safety. The two files together contain **64 GPU cases**. The previous separate
new-model and zero-mask equivalence files are consolidated into the common
matrix; their coverage is retained or strengthened.

`docker/intdim_vllm_smoke.py --family all` additionally keeps the original
14 original/random-half-width generation checks. GPU release gates run both
this smoke and the common gradient-selected matrix.

HF comparisons use stock eager experts on GPU. vLLM uses upstream Triton
attention and MoE kernels, BF16, TP=1, and `enforce_eager=True`, with context
limit 128, one request and two generated tokens. Project plugins and remote
model code are disabled. Other kernel backends, native MXFP4, multi-GPU/TP,
full-model accuracy, benchmarks, SFT and AWQ/GPTQ are outside this matrix.
These are **architecture/width tests, not full pretrained model runs**.

## Build and run

Use Linux x86-64 with Docker/BuildKit, a compatible NVIDIA GPU/driver and
NVIDIA Container Toolkit. Consult [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
for the driver/GPU combination. The build checks dependencies/imports without
executing a model; runtime validation requires a GPU. Reserve at least
**60 GiB free disk** and do not prune shared Docker caches to make space.

```bash
docker build --platform linux/amd64 \
  --build-arg REVISION="$(git rev-parse HEAD)" --build-arg VERSION=dev \
  -t less-is-moe:dev-unified .

# Locked dependencies and compiled CUDA extensions.
docker run --rm --gpus 'device=0' --network none \
  less-is-moe:dev-unified python docker/smoke_test.py --gpu

# All equivalence cases, including seven-model BF16 vLLM parity.
docker run --rm --gpus 'device=0' --network none --shm-size=2g -e OMP_NUM_THREADS=2 \
  -e INTDIM_TEST_VLLM=1 less-is-moe:dev-unified \
  python -m pytest tests/test_intdim_equivalence.py -q -s

# Structural/probe regressions and additional random-mask vLLM smoke.
docker run --rm --gpus 'device=0' --network none -e OMP_NUM_THREADS=2 \
  less-is-moe:dev-unified python -m pytest tests/test_intdim_runtime.py -q
docker run --rm --gpus 'device=0' --network none --shm-size=2g -e OMP_NUM_THREADS=2 \
  less-is-moe:dev-unified python docker/intdim_vllm_smoke.py --family all
```

Inside the GPU image, use `-k uniform` to select only the seven-model matrix,
or `-k 'gpt_oss or gemma4'` to select model-specific cases. Without
`INTDIM_TEST_VLLM=1`, the equivalence file runs its GPU HF checks only.

Expose only your allocated GPU. For Docker hosts configured with
[NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html),
replace `--gpus 'device=0'` with `--device nvidia.com/gpu=0`. Docker Desktop on
macOS cannot supply NVIDIA GPU execution. vLLM has a ten-minute timeout per
checkpoint and downloads no model weights.

## Stock models and legacy patches

The image sets `LESS_IS_MOE_RUNTIME_PATCH=stock` and `VLLM_PLUGINS=""`.
Evaluation entry points honor stock mode, and the legacy plugin is a no-op in
stock mode even when explicitly discovered. Outside this image, use
`--runtime_patch stock` for uniform-width IntDim-E checkpoints.

Legacy patch files remain available for expert-drop, router-mask and historical
reproduction workflows. Their original environment defaults are preserved;
this change retires them from the **unified IntDim-E path**, not from every
baseline. It does not delete or silently upgrade old reproduction environments.

Use the unified CLI from PR #26:

```bash
python -m less_is_moe.intdim.prune \
  --model_name_or_path /models/MODEL --output_dir /outputs/structural \
  --mode structural --drop_ratio 0.5 --dtype bf16 --calib_data /data/calib.jsonl
```

`--mode mask --prune_mode expert|layer|global` supports E/L/G zero-masking.

For Qwen1.5-MoE, OLMoE, Qwen3-MoE, Qwen3.5-MoE, GPT-OSS and Gemma4 text models,
`--mode ragged --prune_mode layer|global` adds a separate compact
non-uniform format. Its vLLM backend requires
`VLLM_PLUGINS=less_is_moe_ragged`, BF16, TP=DP=1 and eager execution. Pipeline
parallelism can partition large models across GPUs. GPT-OSS's published MXFP4
weights are dequantized to BF16 before pruning and before both comparisons. See
[ragged expert support](RAGGED_EXPERTS.md); stock runtime defaults stay the same.
`--mode structural --from_zeroed_model` compacts a uniformly zero-masked
checkpoint without calibration. Empty calibration for scoring is rejected.

`intdim.scoring.collect_scores` and `select_expert_units` are compatibility
helpers that delegate to `intdim.prune`; no separate scoring algorithm is
maintained. The existing Python workflow remains valid:

```python
from less_is_moe.intdim import discover, verify_checkpoint
from less_is_moe.intdim.scoring import collect_scores, select_expert_units

# model is a stock HF model on CUDA; batches are token-ID tensors.
handles = discover(model)
scores = collect_scores(model, batches, handles)
keep = select_expert_units(scores, drop_ratio=0.5)
for handle in handles:
    handle.apply_units(keep[handle.layer_index])
widths = {handle.intermediate_size for handle in handles}
assert len(widths) == 1  # one scalar width for this IntDim-E checkpoint
handles[0].intermediate_size_key.set(model.config, widths.pop())
model.save_pretrained(output_directory)
assert verify_checkpoint(output_directory).ok
```

The unified calibration and CLI implementation is included from PR #26.
Historical launcher retirement and full pretrained-checkpoint experiments
remain tracked by #25. See [the upstream review](PR24_PR26_REVIEW.md) for the
original-revision failures and integration fixes; these GPU results apply to
#27's integrated code, not unmodified #24/#26.

## Mount data and update dependencies

Mount weights/data read-only and keep outputs and caches outside the container:

```bash
docker run --rm -it --gpus 'device=0' --shm-size=8g \
  --mount type=bind,src=/absolute/path/models,dst=/models,readonly \
  --mount type=bind,src=/absolute/path/data,dst=/data,readonly \
  --mount type=bind,src=/absolute/path/outputs,dst=/outputs \
  --mount type=bind,src=/absolute/path/hf-cache,dst=/cache/huggingface \
  less-is-moe:dev-unified bash
```

The image runs as root by default. On shared storage use
`--user "$(id -u):$(id -g)" -e HOME=/tmp` with writable output/cache mounts.
For gated downloads, set `HF_TOKEN` on the host and pass `--env HF_TOKEN`;
never pass tokens as build arguments. The context allowlist excludes credentials
and common model artifacts. SFT/TRL/DeepSpeed and AWQ/GPTQ extras are omitted.

Regenerate the lock with uv 0.12.5 using `./environments/unified/lock.sh`;
pass `--upgrade` only in an explicit environment-upgrade PR. Resolution targets
Python 3.12.14, Linux x86-64/glibc 2.39 and cu130, even from macOS. Rerun the
GPU checks after changing dependencies; model-adapter PRs should consume the
released digest without changing the lock.

## Publish

The **Unified Docker runtime** workflow checks syntax on pull requests. Builds
and GPU gates run on manual dispatch or stable `vMAJOR.MINOR.PATCH` tags:

1. Set `DOCKER_RUNNER` to a dedicated/ephemeral Linux x64 NVIDIA GPU Docker
   runner label with at least 60 GiB free. Set `DOCKER_GPU_DEVICE` to its allocated
   GPU index (default `0`), and `DOCKER_GPU_ACCESS=cdi` for CDI hosts (default
   `gpus`). The workflow does not register a runner or change a shared server.
2. After merging, manually dispatch from the release commit. This builds and
   runs both GPU test groups without publishing; inspect the job summary.
3. Push a stable tag matching `pyproject.toml`. The first planned `v0.1.0`
   publishes `ghcr.io/hectorhhz/less-is-moe:0.1.0-unified`. Only after every GPU
   gate passes does the workflow push that tested image and promote
   `latest-unified`. It records the registry digest in the summary and uses
   `GITHUB_TOKEN` with `packages: write`.
4. Published version tags are not overwritten. Inspect partial publications;
   promote a verified existing digest manually or publish a new version. Limit
   other registry writers to preserve this policy outside the workflow.
5. Make a newly created GHCR package public and verify an anonymous pull.
   Existing packages must grant the repository Actions write access.

Until publication succeeds, these GHCR tags are planned names. Pin the actual
published digest for experiments; `latest-unified` is a convenience alias.
