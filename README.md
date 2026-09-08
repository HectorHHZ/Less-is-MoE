# Less is MoE: Trimming Experts in Domain-Specialist Language Models

**Official implementation of Fisher-MoE — structured expert pruning and compression for Mixture-of-Experts (MoE) large language models.**

[![arXiv](https://img.shields.io/badge/arXiv-2606.05538-b31b1b.svg)](https://arxiv.org/abs/2606.05538)
[![Venue](https://img.shields.io/badge/EMNLP%202026-Main%20Conference%20%7C%20Oral-4c1.svg)](https://arxiv.org/abs/2606.05538)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Less is MoE: Trimming Experts in Domain-Specialist Language Models**
> Haoze He\*, Xinkai Zou\*, Xuan Jiang, Xingyuan Ding, Ao Qu, Juncheng Billy Li, Heather Miller
> **EMNLP 2026, Main Conference (Oral)** · [arXiv:2606.05538](https://arxiv.org/abs/2606.05538)
> \*Equal contribution

## Abstract

Mixture-of-Experts (MoE) models achieve strong performance through conditional
computation, but their large parameter footprint poses deployment challenges.
Prior MoE compression approaches catastrophically fail when evaluated on
general-purpose benchmarks beyond commonsense reasoning. We trace this failure
to the granularity of compression: important capabilities are distributed across
experts but concentrated in FFN sparse intermediate dimensions. To identify
these dimensions, we use Fisher importance which outperforms activation-,
router-score-, and magnitude-based alternatives, and identifies tiny sets of
task-critical dimensions: in Qwen1.5-MoE, removing as few as 12 of 1.35M routed-FFN
intermediate dimensions collapses GSM8K accuracy while largely preserving
factual-knowledge performance. Building on this, we propose Fisher-MoE, which
operates within FFN to remove intermediate dimensions ranked by Fisher
importance. At the same 50% MoE compression ratio, Fisher-MoE preserves model
capability, while reducing weight memory by ~45% and improving inference
throughput by 21%. These findings suggest intermediate dimension granularity is
an effective unit for both compression and ranking where capability concentrates
in MoE models.

## TL;DR

- **Problem.** Existing **MoE compression** and **expert pruning** methods look
  fine on commonsense reasoning and collapse on general-purpose benchmarks.
- **Diagnosis.** The failure is one of *granularity*. Capability is spread
  across experts but concentrated in a tiny number of **FFN intermediate
  dimensions** — in Qwen1.5-MoE, deleting 12 out of 1.35M routed-FFN dimensions
  is enough to destroy GSM8K.
- **Method.** **Fisher-MoE** performs **intra-expert structured pruning**: it
  ranks and removes FFN intermediate dimensions (neurons) by Fisher importance,
  instead of dropping or merging whole experts.
- **Result.** At a 50% MoE compression ratio: capability preserved, **~45% less
  weight memory**, **+21% inference throughput**.

Where this sits in the literature: most work on compressing sparse
Mixture-of-Experts LLMs operates at **whole-expert granularity** — expert
pruning, expert dropping, expert merging, expert skipping, or low-rank expert
decomposition. Fisher-MoE instead prunes *inside* the expert, at
**neuron / FFN intermediate-dimension granularity**, and shows that this is the
granularity at which task capability actually concentrates. The repository
supports both **mask-based (zeroed) pruning** and **structural (physically
removed) pruning**, plus post-pruning SFT, AWQ/GPTQ quantization, and evaluation.

**Keywords:** mixture-of-experts, MoE compression, expert pruning, expert
trimming, expert dropping, structured pruning, neuron pruning, intra-expert
pruning, Fisher information, LLM compression, efficient inference, Qwen-MoE,
Qwen3-MoE, OLMoE, sparse upcycled models.

## What is included

This repository contains the release implementation for pruning, loading,
fine-tuning, quantizing, and evaluating pruned mixture-of-experts language
models. It consolidates the experiment scripts and runtime model patches into
an installable `src/` package, with thin launchers under `scripts/`.

| Area | Supported model families | Entry points |
| --- | --- | --- |
| Mask-based neuron pruning | Qwen1.5-MoE, Qwen3-MoE, Qwen3.5-MoE, OLMoE | `scripts/prune/neuron_drop_*.sh` |
| Structural neuron pruning | Qwen1.5-MoE, Qwen3-MoE, Qwen3.5-MoE, OLMoE | `scripts/prune/neuron_structure_drop_*.sh` |
| Pruned checkpoint loading | The same four families, for Hugging Face and vLLM | `less_is_moe.model_patches` |
| Evaluation | Qwen strict zero-shot; OLMoE multi-shot | `scripts/evaluate/` |
| SFT | Base checkpoints and already-pruned checkpoints only | `scripts/train/` |
| Quantization | AWQ for legacy/Qwen3; GPTQ for Qwen1.5/Qwen2-MoE | `scripts/quantize/` |

Important protocol details:

- The released pruning code ranks parameters with
  `mean(abs(gradient))`. It does **not** compute squared-gradient Fisher
  information. The code preserves the criterion actually used for the released
  experiments. Please report this implementation detail when comparing against
  the paper's Fisher importance formulation.
- Qwen1.5-MoE, Qwen3-MoE, and Qwen3.5-MoE use the strict zero-shot evaluator.
  The multi-shot evaluator is only for OLMoE.
- The two supported SFT paths are ordinary base-model SFT and SFT of an
  already-pruned checkpoint. Online pruning during SFT is disabled in the
  release recipes.
- Qwen3.5 AWQ/GPTQ is not supported. Do not use the legacy quantization scripts
  for a Qwen3.5 checkpoint.

See [the script reference](docs/SCRIPT_REFERENCE.md) for the per-model pruning
behavior and a comparison of the five original evaluation scripts.

## Installation

Python 3.11 and a CUDA-capable Linux host are recommended. Select the profile
that matches the model family; the environments are intentionally separate.

```bash
git clone https://github.com/HectorHHZ/Less-is-MoE.git
cd Less-is-MoE

./setup.sh legacy   # Qwen1.5-MoE and OLMoE
# ./setup.sh qwen3  # Qwen3-MoE
# ./setup.sh qwen35 # Qwen3.5-MoE

source .venv-legacy/bin/activate
```

For optional quantization dependencies:

```bash
./setup.sh legacy --with-awq
./setup.sh legacy --with-gptq
```

Each profile creates its own virtual environment, so installing Qwen3.5 does
not overwrite the Transformers/vLLM versions needed by the earlier families.
The legacy and Qwen3.5 profiles intentionally restore the experiment's
Transformers version after installing vLLM, despite incompatible vLLM package
metadata; keep these environments isolated and do not run a broad dependency
upgrade in them.
See [the environment matrix](docs/ENVIRONMENTS.md) for exact core versions and
build notes.

## Pruning

Every shell launcher forwards its arguments to a package module. The following
example performs structural Qwen3-MoE pruning; replace the model, data, and
output values with paths available on your system.

```bash
scripts/prune/neuron_structure_drop_qwen3.sh \
  --model_name_or_path MODEL_ID_OR_LOCAL_PATH \
  --output_dir outputs/qwen3-structural-p50 \
  --drop_ratio 0.5 \
  --calib_data data/calibration.jsonl \
  --n_samples 128 \
  --seq_len 2048
```

Use `neuron_drop_*` to produce a masked/zeroed checkpoint and
`neuron_structure_drop_*` to remove the selected structures physically. Run a
launcher with `--help` for model-specific options. Calibration datasets and
model weights are not distributed in this repository.

## Evaluation

Use strict zero-shot for every Qwen family:

```bash
scripts/evaluate/zero_shot.sh \
  --model_name_or_path MODEL_ID_OR_LOCAL_PATH \
  --data_path data/gsm8k_test.jsonl \
  --dataset gsm8k \
  --output_dir results/qwen-gsm8k \
  --temperature 0 \
  --n_samples_per_problem 1
```

Use the dedicated multi-shot path only for OLMoE:

```bash
scripts/evaluate/olmoe_multishot.sh \
  --model_name_or_path MODEL_ID_OR_LOCAL_PATH \
  --data_path data/gsm8k_test.jsonl \
  --dataset gsm8k \
  --output_dir results/olmoe-gsm8k
```

Pass `--base_model` for an unpruned checkpoint. Without it, the evaluator
detects the model family from checkpoint configuration and installs the pruned
runtime patch. `scripts/evaluate/hf_optional.sh` is a slower, Qwen-only Hugging
Face fallback; it rejects OLMoE so the paper's protocol cannot be selected by
mistake.

The ESFT summary/law/translation judges additionally require
`pip install -e '.[judge]'` and an `OPENAI_API_KEY`; ordinary benchmark
evaluation does not install or require that optional client.

## Fine-tuning

The YAML files under `recipes/sft/` are safe, editable templates. Values named
`REPLACE_WITH_*` must be changed before a run; no unpublished checkpoint path is
assumed.

```bash
# Base-model SFT
scripts/train/base.sh --config recipes/sft/base/qwen3_moe.yaml

# SFT of a checkpoint that has already been pruned
scripts/train/pruned.sh --config recipes/sft/pruned/qwen3_moe.yaml
```

The default launcher uses `recipes/accelerate/zero2.yaml`. Select another
Accelerate configuration without editing a script:

```bash
ACCELERATE_CONFIG=recipes/accelerate/zero3.yaml \
  scripts/train/pruned.sh --config recipes/sft/pruned/olmoe.yaml
```

## Quantization

AWQ uses the compatibility fork vendored at `third_party/AutoAWQ`. Install it
with `--with-awq`, then run, for example:

```bash
scripts/quantize/awq_pruned.sh \
  --model_path PRUNED_MODEL_PATH \
  --output_path outputs/model-awq \
  --model_type qwen2_moe \
  --q_group_size 32
```

GPTQ is exposed for pruned Qwen1.5/Qwen2-MoE checkpoints:

```bash
scripts/quantize/gptq_pruned.sh \
  --model-path PRUNED_MODEL_PATH \
  --output-path outputs/model-gptq
```

See [quantization notes](docs/QUANTIZATION.md) for the support boundary.

## Repository layout

```text
src/less_is_moe/
  pruning/          Pruning implementations
  model_patches/    Hugging Face and vLLM compatibility patches
  evaluation/       Zero-shot, OLMoE multi-shot, and optional HF evaluators
  training/         Base and already-pruned SFT
  quantization/     AWQ/GPTQ helpers
scripts/            Thin command-line launchers
recipes/            Accelerate configs and editable SFT templates
environments/       Mutually compatible dependency profiles
third_party/AutoAWQ Vendored compatibility fork (MIT licensed)
```

For a start-to-finish checklist, read
[Reproducibility](docs/REPRODUCIBILITY.md). The
[migration map](docs/MIGRATION.md) records exactly how the experiment files
were consolidated. Please review
[Security](SECURITY.md) before sharing logs or configuration files.

## Related work from the authors

- **Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning**
  (COLM 2026) — auxiliary-loss-free MoE supervised fine-tuning with gated
  condenser experts. [arXiv:2604.23036](https://arxiv.org/abs/2604.23036) ·
  [ExpertCondenser code](https://github.com/HectorHHZ/ExpertCondenser)
- **SMT: Fine-Tuning Large Language Models with Sparse Matrices** (ICLR 2025).
  [code](https://github.com/HectorHHZ/Sparse_Matrix_Tuning)

## Citation

Accepted at **EMNLP 2026, Main Conference (Oral)**. Would appreciate your citation :). *Preserving Long-Tailed Expert Information in
Mixture-of-Experts Tuning* (COLM 2026) is the analysis that Less is MoE is built
on. It systematically pruned experts and found that although a small number of
super experts dominate activation, discarding the rarely activated **long-tailed
experts** still causes notable degradation: those experts encode non-trivial
knowledge that downstream tasks depend on. Knowledge in a sparse MoE is
therefore *distributed across the long tail of experts*, which means
**whole-expert pruning cannot be lossless no matter how the experts are ranked**. The failure is inherent to the granularity, not to the scoring criterion.
Less is MoE follows directly from that result: rather than removing experts, it
moves compression *inside* the expert and prunes FFN intermediate dimensions,
the granularity at which task capability actually concentrates.

```bibtex
@article{he2026lessismoe,
  title   = {Less is {MoE}: Trimming Experts in Domain-Specialist Language Models},
  author  = {He, Haoze and Zou, Xinkai and Jiang, Xuan and Ding, Xingyuan and
             Qu, Ao and Li, Juncheng Billy and Miller, Heather},
  journal = {arXiv preprint arXiv:2606.05538},
  year    = {2026}
}

@article{he2026expertcondenser,
  title   = {Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning},
  author  = {He, Haoze and Ding, Xingyuan and Jiang, Xuan and Zou, Xinkai and
             Cheng, Alex and Zhao, Yibo and Li, Juncheng Billy and Miller, Heather},
  journal = {arXiv preprint arXiv:2604.23036},
  year    = {2026}
}
```

<!-- Replace the Less-is-MoE entry above once the EMNLP 2026 proceedings are on
     the ACL Anthology; fill in the real booktitle, pages, and anthology URL:

@inproceedings{he2026lessismoe,
  title     = {Less is {MoE}: Trimming Experts in Domain-Specialist Language Models},
  author    = {He, Haoze and Zou, Xinkai and Jiang, Xuan and Ding, Xingyuan and
               Qu, Ao and Li, Juncheng Billy and Miller, Heather},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  pages     = {TBD},
  url       = {TBD}
}
-->

## License

Less-is-MoE is released under Apache-2.0. Vendored or adapted third-party code
retains its own license; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).
