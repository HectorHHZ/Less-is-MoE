# Review of PR #24 and PR #26

Reviewed on 2026-09-16 against #24 `8b7ed013a0e7e8ba9a457d8fcaba4eef2ef776eb`
and #26 `8cfe21417727348fc2d4e243545e552b354a27a4`. The findings below refer to
those upstream revisions, before the integration fixes in #27.

## Findings

### P1: BF16 discovery rejects supported models (#24, inherited by #26)

The probe uses BF16 forwards and `atol=rtol=0.03`. Individual neurons in a
randomly initialized expert contribute less than this threshold, so zeroing
them is treated as having no effect and every layout hypothesis is rejected.
#26's search over eight live units does not fix the precision issue.

On a B200, all seven FP32 fixture configurations were discovered successfully;
all seven BF16 configurations failed with `layout probe found 0 consistent
layouts`. This blocks #26's default `--dtype bf16` before scoring begins.

Location: [#26 discover.py, lines 474-475 and 515-522](https://github.com/HectorHHZ/Less-is-MoE/blob/8cfe21417727348fc2d4e243545e552b354a27a4/src/less_is_moe/intdim/discover.py#L474).

Fix in #27: probe one isolated FP32 expert on its original GPU, preserving
#26's live-expert/unit search. The full model's dtype and weights are unchanged.
The unified BF16 matrix and dead-expert tests cover this combination.

### P1: Unsorted keep indices mismatch gate/up and down (#24, inherited by #26)

`gate_up_indices()` sorts its indices, but `select_units()` gathers down
weights in the original `keep` order. Passing descending or random indices
therefore pairs one neuron's activation with another neuron's down weight.
Saving and stock-loading still succeed because tensor shapes remain valid.

Reproduced with a Qwen3.5 GPU fixture: descending `keep` yielded ascending
gate/up weights and descending down weights. #26's current selector happens
to produce ascending indices; the public handle API does not require them.

Location: [#24 discover.py, lines 208-212](https://github.com/HectorHHZ/Less-is-MoE/blob/8b7ed013a0e7e8ba9a457d8fcaba4eef2ef776eb/src/less_is_moe/intdim/discover.py#L208).

Fix already in #27: validate and sort the keep set before gathering any
projection. Random unsorted masks are checked against zero-masked expert
outputs in every family and dtype.

### P2: A failed probe leaves the caller's weights zeroed (#24 and #26)

The probe zeroes live gate/up weights and calls forward before restoring them.
An exception from that forward bypasses restoration. Catching the discovery
failure and retrying therefore operates on a changed model.

Injecting a failure into the second probe forward leaves `gate_up_proj`
modified in both original revisions.

Location: [#26 discover.py, lines 502-508](https://github.com/HectorHHZ/Less-is-MoE/blob/8cfe21417727348fc2d4e243545e552b354a27a4/src/less_is_moe/intdim/discover.py#L502).

Fix in #27: all mutations are confined to the isolated probe copy. GPU
exception-injection tests assert that all live model tensors stay identical.

### P2: Empty calibration silently prunes using all-zero scores (#26)

An empty JSON/JSONL file, or a file with no recognized text fields, produces
zero batches. The collector accepts them, divides its zero scores by
`max(0, 1)`, and the selector arbitrarily chooses tied neurons. The pipeline
reports success and can save a structurally valid but uncalibrated checkpoint.

On the GPU Qwen3.5 fixture, `prune(model, args, [])` returned success, reduced
width 512 to 256, and removed 2,048 neurons without one backward pass.

Location: [#26 prune.py, lines 439-441](https://github.com/HectorHHZ/Less-is-MoE/blob/8cfe21417727348fc2d4e243545e552b354a27a4/src/less_is_moe/intdim/prune.py#L439).

Fix in #27: reject empty calibration before discovery or mutation, both at
the pipeline and collector entry points. `--from_zeroed_model` remains valid
without calibration. The collector also clears stale gradients before scoring
and cleans up gradients/training state on exceptions.

## Validation boundary

The original-revision reproductions used the fixed #27 Docker dependencies:
Python 3.12.14, torch 2.13.0+cu130, Transformers 5.17.0, vLLM 0.29.0,
CUDA toolkit 13.0.3; NVIDIA B200, driver 595.71.05. All model execution was on
GPU. The architecture fixtures are reduced random models, not full pretrained
checkpoints. No claim is made about benchmark accuracy or original MXFP4 weights.

#27 now imports #26's actual `intdim.prune` implementation. Its former
`intdim.scoring` API delegates to that implementation. The independent legacy
references never call autodetect or the new pruning API. The GPU matrix checks
E/L/G zero-masking, E structural pruning, mask-to-structural conversion, real
CLI loading/calibration/saving, stock HF reloads, and BF16 vLLM generation.
The raw upstream revisions and the integrated, corrected revision are reported
separately; a passing integrated run does not mean the original PRs passed.
