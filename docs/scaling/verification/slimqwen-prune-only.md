# SlimQwen prune-only baseline

Answers [#4](https://github.com/HectorHHZ/Less-is-MoE/issues/4). Source: *SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training*, [arXiv:2605.08738v2](https://arxiv.org/abs/2605.08738).

## Answer

**Yes, the pruning stage can run without distillation.** SlimQwen compresses the model first and trains afterwards. Compression needs only forward passes over calibration data and closed-form keep/merge rules, so it is separable from the continued pretraining that follows.

Three caveats shape the baseline:

- **No official code.** It must be implemented.
- **No zero-training results.** Every compressed model in the paper is evaluated after continued pretraining.
- **The final model's expert importance metric is not stated.** The configuration below is our choice and needs sign-off.

## What SlimQwen does

SlimQwen compresses Qwen3-Next-80B-A3B into 23B-A2B in three stages, then trains with a combined distillation and language-modeling loss (progressive schedule: 40B + 360B tokens).

| Stage | Rule | Effect |
| --- | --- | --- |
| Depth | Drop the last N layers | 48 → 36 layers |
| Width | Keep the hidden dimensions with the largest mean `abs(RMSNorm(X))` activation | d_model 2048 → 1536, in every module |
| Experts | Per layer, keep the top ⌊Ñ/2⌋ experts by importance intact, select Ñ/2 merge bases, and merge the remaining experts into their most similar base by an importance-weighted average | 512 → 256 experts, top-k 10 → 8, shared expert kept |

Expert intermediate size is unchanged (512). Importance uses 1024 calibration samples from pretraining data; the sequence length is not stated.

Expert importance metrics, with A(x) the top-k experts, z_i(x) the router logit, and E_i(x) the expert output:

- **Frequency:** I_i = E_x[𝟙(i ∈ A(x))]
- **Soft-logits:** I_i = E_x[𝟙(i ∈ A(x)) · z_i(x) / Σ_{j∈A(x)} z_j(x)]
- **REAP:** I_i = (1/|X_i|) Σ_{x∈X_i} z_i(x) · ‖E_i(x)‖₂

## Evidence for choosing a configuration

Table 2 of the paper compares expert compression methods, all measured **after 400B tokens of continued pretraining**. Averages over its eight benchmarks, computed here:

| Method | Metric | Grouping | Partial preservation | Average |
| --- | --- | --- | --- | ---: |
| Merge | REAP | Expert vector | Yes | **64.10** |
| Prune | Soft-logits | — | — | 63.75 |
| Prune | REAP | — | — | 63.66 |
| Merge | REAP | Router logits | Yes | 63.39 |
| Merge | Soft-logits | Router weights | Yes | 63.36 |
| Merge | Soft-logits | Router logits | Yes | 63.23 |
| Merge | Frequency | Router logits | Yes | 63.03 |
| Merge | Soft-logits | Router weights | No | 63.00 |
| Merge | Soft-logits | Expert vector | Yes | 62.84 |

The whole spread is 1.26 points, and the paper states that "no single one-shot pruning or merging method establishes consistent superiority across all downstream tasks." These results cannot tell us which method is best *without* training.

## Proposed definition

**SlimQwen (prune-only)** is defined as:

1. **Expert stage only, uniform per layer: E → E/2.** This removes exactly 50% of routed-expert FFN parameters, the same budget as the Fisher-MoE rows. Depth and width pruning are excluded: they remove attention, GatedDeltaNet, embedding, and norm parameters, which breaks the matched routed-FFN budget.
2. **Merge with partial preservation, REAP importance, expert-vector grouping.** This is the procedure SlimQwen uses for its final model, with the configuration that has the best Table 2 average.
3. **Top-k and shared experts unchanged**, as in the Fisher-MoE rows. SlimQwen's 10 → 8 top-k change belongs to its target architecture.
4. **The same 128 calibration samples as Fisher-MoE**, instead of 1024.
5. **No training of any kind.**

State these deviations from the published SlimQwen in every table that includes the row: no distillation or continued pretraining, expert stage only, 128 calibration samples, and an importance metric chosen by us.

If budget allows, a **pure REAP drop** row (Table 2's "Prune / REAP") costs little extra, because it reuses the same importance scores, and it separates the effect of merging from that of the metric.

## Implementation notes for #14

- **No usable SlimQwen code.** The only repository with the name, `HowardZorn/Megatron-Bridge-SlimQwen`, mirrors `NVIDIA-NeMo/Megatron-Bridge`: its head commit `8c80963c10` exists upstream, and it contains no SlimQwen-specific files.
- **REAP importance can be reused.** Both implementations are Apache-2.0:
  - `CerebrasResearch/reap` supports Qwen3-MoE, Llama-4, Mixtral, DeepSeek-V2, ERNIE-4.5, and GLM-4-MoE — none of the new backbones.
  - The `vllm-project/llm-compressor` REAP modifier converts fused experts through mappings that cover `qwen3_5_moe`, `gemma4`, and `gpt_oss`. It prunes only; the SlimQwen merge step must be written.
- **Gemma-4 router.** `llm-compressor` shrinks a router's `weight`, `bias`, and `e_score_correction_bias`. Gemma-4 keeps its router weights in `router.proj` and adds a per-expert `per_expert_scale`, so both need explicit handling.
- **gpt-oss.** Removing or merging experts must slice the MXFP4 `*_blocks` and `*_scales` tensors and both bias tensors, or run on dequantized BF16 weights according to the MXFP4 decision in the roadmap.
- **Details to confirm from the paper's Algorithm 1:** the exact expert-vector similarity and how merge bases are selected.
