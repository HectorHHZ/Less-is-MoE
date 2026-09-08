# Reproducibility checklist

## 1. Record immutable inputs

Record the base checkpoint revision, calibration dataset revision, calibration
sample order, number of samples, sequence length, random seed, dtype, and drop
ratio. Local datasets and model weights are intentionally not committed.

## 2. Select one environment

Use `legacy` for Qwen1.5-MoE/OLMoE, `qwen3` for Qwen3-MoE, or `qwen35` for
Qwen3.5-MoE. Do not upgrade one profile in place to run another family; create
the second virtual environment alongside it.

## 3. Prune

Choose a model-specific launcher in `scripts/prune/`. The `neuron_drop_*`
scripts retain tensor shapes and zero/mask selected units. The
`neuron_structure_drop_*` scripts create physically smaller structures and
write configuration metadata needed by the runtime patches.

The importance statistic in this release is the mean absolute gradient:

```text
score = mean(abs(gradient))
```

It is not squared-gradient Fisher information. Report that implementation detail
when comparing checkpoints produced by this code.

## 4. Optionally fine-tune

Use only one of the two release paths:

- `scripts/train/base.sh` for an unpruned base checkpoint.
- `scripts/train/pruned.sh` for a checkpoint already produced by a pruning
  script.

Copy an editable recipe from `recipes/sft/base/` or `recipes/sft/pruned/`, fill
in each `REPLACE_WITH_*` value, and keep the completed recipe next to the run
metadata. Pruned templates set `router_prune_enable: false`; they do not perform
a second pruning operation during SFT.

## 5. Evaluate with the matching protocol

- Qwen1.5-MoE, Qwen3-MoE, and Qwen3.5-MoE: strict zero-shot through
  `scripts/evaluate/zero_shot.sh`.
- OLMoE: the dedicated multi-shot evaluator through
  `scripts/evaluate/olmoe_multishot.sh`.

For deterministic zero-shot generation, keep `--temperature 0` and
`--n_samples_per_problem 1`. Record any prompt-mode, chat-template, thinking,
stop-sequence, maximum-length, or tensor-parallel override.

## 6. Preserve outputs

Archive the completed recipe, command line, git commit, model configuration,
metrics JSON, and generated-answer file. Never archive access tokens in a
recipe, shell transcript, scheduler script, or public experiment log.
