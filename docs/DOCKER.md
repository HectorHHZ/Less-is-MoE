# Unified GPU runtime and IntDim-E validation

The unified image builds on the generic `intdim` implementation introduced in
PR #24. Its GPU checks establish two things: the new IntDim-E path preserves
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

All model execution in this validation suite runs on a GPU. Tiny randomly
initialized models make the tests independent of private weights and datasets.

| Check | Coverage | Acceptance |
| --- | --- | --- |
| Released algorithm vs generic handles | Qwen2-MoE, Qwen3-MoE, OLMoE, Qwen3.5-MoE; FP32 and BF16 | Same-layout scores, selected indices, all weights and logits are bitwise equal |
| Old Linear layout vs native fused layout | Qwen2-MoE, Qwen3-MoE, OLMoE; FP32 and BF16 | Selected indices and canonical weights are bitwise equal; scores/logits allow documented floating-point rounding |
| HF structural roundtrip | All six families, including GPT-OSS and Gemma 4; FP32 and BF16 | Unsorted per-expert selections equal zero-masking; save, stock verify, reload and forward; exact saved tensor recovery |
| Probe failure | GPT-OSS, FP32 and BF16 | A failed discovery probe leaves the live weights unchanged |
| vLLM generation | Original and half-width checkpoints for seven fixture configurations | Stock upstream model classes generate two valid tokens on GPU |

The vLLM fixtures cover Qwen2-MoE, Qwen3-MoE, OLMoE, both Qwen3.5 target
widths (512 -> 256 and 1024 -> 512), GPT-OSS (2880 -> 1440), and Gemma 4
(704 -> 352). Layers, expert counts and hidden sizes are reduced; these are
**architecture/width tests, not runs of the full pretrained 120B/122B models**.
GPT-OSS uses BF16 expert tensors, not native packed MXFP4.

The old Qwen2/Qwen3/OLMoE functions require per-expert Linear modules. Tests
reconstruct that storage from identical native HF weights and call the actual,
unchanged released functions. They compare against both the generic Linear
path and the native fused path within the same runtime, isolating the pruning
algorithm from framework-version changes. Fused and separate GEMMs can round
differently: native-layout score tolerances are FP32 `rtol=1e-5, atol=1e-8`
and BF16 `rtol=0.02, atol=1e-8`; logits use FP32 `rtol=1e-5, atol=1e-6`
and BF16 `rtol=0.02, atol=0.002`. **Indices and kept weights must still match
exactly**, so a ranking change fails the test.

HF reference comparisons use stock eager experts on GPU; tiny fixture widths
are not aligned for grouped GEMMs. The vLLM test uses upstream Triton attention
and MoE kernels, BF16, TP=1, and `enforce_eager=True` (CUDA execution without
CUDA graphs). It disables project plugins and remote model code. Other kernel
backends, multi-GPU/TP, full-model accuracy, benchmark scores, SFT, and AWQ/GPTQ
are outside this test matrix. FlashInfer auto-selection may compile kernels on
first use; the development toolkit and compiler remain in the image.

## Build and run the two test groups

Use Linux x86-64 with Docker/BuildKit, a compatible NVIDIA GPU/driver and
NVIDIA Container Toolkit. Consult [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
for the driver/GPU combination. The build checks dependencies and imports
without executing a model; runtime validation requires a GPU.

Reserve at least **60 GiB free disk** as a conservative build guard and
measure usage on your builder. Do not prune shared Docker caches to make space.

```bash
docker build --platform linux/amd64 \
  --build-arg REVISION="$(git rev-parse HEAD)" --build-arg VERSION=dev \
  -t less-is-moe:dev-unified .

# Dependency versions, BF16 CUDA GEMM, and vLLM's compiled RMSNorm extension.
docker run --rm --gpus 'device=0' --network none \
  less-is-moe:dev-unified python docker/smoke_test.py --gpu

# Test 1: released IntDim-E vs generic and native fused paths.
docker run --rm --gpus 'device=0' --network none -e OMP_NUM_THREADS=2 \
  less-is-moe:dev-unified python -m pytest tests/test_intdim_equivalence.py -q

# Test 2: HF roundtrips and stock vLLM generation.
docker run --rm --gpus 'device=0' --network none -e OMP_NUM_THREADS=2 \
  less-is-moe:dev-unified python -m pytest tests/test_intdim_runtime.py -q
docker run --rm --gpus 'device=0' --network none --shm-size=2g -e OMP_NUM_THREADS=2 \
  less-is-moe:dev-unified python docker/intdim_vllm_smoke.py --family all
```

Expose only the GPU allocated to your job. On a Docker host configured for
[NVIDIA CDI](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html),
replace `--gpus 'device=0'` with `--device nvidia.com/gpu=0`. Docker Desktop on
macOS cannot supply NVIDIA GPU execution. The vLLM checks have a ten-minute
timeout per checkpoint and download no model weights.

## Stock models and legacy patches

The image sets `LESS_IS_MOE_RUNTIME_PATCH=stock` and `VLLM_PLUGINS=""`.
Evaluation entry points honor stock mode, and the legacy plugin is a no-op in
stock mode even when explicitly discovered. Outside this image, use
`--runtime_patch stock` for uniform-width IntDim-E checkpoints.

Legacy patch files remain available for expert-drop, router-mask and historical
reproduction workflows. Their original environment defaults are preserved;
this change retires them from the **unified IntDim-E path**, not from every
baseline. It does not delete or silently upgrade old reproduction environments.

`intdim.scoring.collect_scores` preserves the released mean-absolute-gradient
criterion, and `select_expert_units` preserves its bottom-k selection. A Python
workflow is:

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

The complete calibration/CLI migration remains tracked by #25. This PR adds
the scoring API needed to verify equivalence, not a replacement for every
historical launcher or calibration option.

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
