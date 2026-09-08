# Script migration map

This map records how the original experiment working tree was reduced to the
public release interface. New shell files are thin launchers; implementations
live in the installable `less_is_moe` package.

## Pruning

| Original script | Package module | Launcher |
| --- | --- | --- |
| `scripts/neuron_structure_drop_qwen3.py` | `less_is_moe.pruning.neuron_structure_drop_qwen3` | `scripts/prune/neuron_structure_drop_qwen3.sh` |
| `scripts/neuron_structure_drop_qwen3_5.py` | `less_is_moe.pruning.neuron_structure_drop_qwen3_5` | `scripts/prune/neuron_structure_drop_qwen3_5.sh` |
| `scripts/neuron_structure_drop_qwen1.5_moe.py` | `less_is_moe.pruning.neuron_structure_drop_qwen15_moe` | `scripts/prune/neuron_structure_drop_qwen15_moe.sh` |
| `scripts/neuron_structure_drop_olmoe.py` | `less_is_moe.pruning.neuron_structure_drop_olmoe` | `scripts/prune/neuron_structure_drop_olmoe.sh` |
| `scripts/neuron_drop_qwen3.py` | `less_is_moe.pruning.neuron_drop_qwen3` | `scripts/prune/neuron_drop_qwen3.sh` |
| `scripts/neuron_drop_qwen3_5.py` | `less_is_moe.pruning.neuron_drop_qwen3_5` | `scripts/prune/neuron_drop_qwen3_5.sh` |
| `scripts/neuron_drop_qwen1.5_moe.py` | `less_is_moe.pruning.neuron_drop_qwen15_moe` | `scripts/prune/neuron_drop_qwen15_moe.sh` |
| `scripts/neuron_drop_olmoe.py` | `less_is_moe.pruning.neuron_drop_olmoe` | `scripts/prune/neuron_drop_olmoe.sh` |

Dots were removed from Python module names (`qwen1.5` became `qwen15`) so the
modules can be imported normally.

## Evaluation

| Original script | Release disposition |
| --- | --- |
| `src/open_r1/evaluate_qwen1.5_moe_vllm.py` | Consolidated into `less_is_moe.evaluation.vllm_zero_shot` and `scripts/evaluate/zero_shot.sh` |
| `src/open_r1/evaluate_gsm8k_vllm.py` | Preserved as the OLMoE-only multi-shot protocol in `less_is_moe.evaluation.vllm_olmoe_multishot` and `scripts/evaluate/olmoe_multishot.sh` |
| `src/open_r1/evaluate_gsm8k_vllm_old.py` | Not migrated; superseded experiment snapshot |
| `src/open_r1/evaluate_gsm8k_vllm_new.py` | Not migrated; superseded experiment snapshot |
| `src/open_r1/evaluate_gsm8k_hf.py` | Kept only as optional fallback in `less_is_moe.evaluation.hf` and `scripts/evaluate/hf_optional.sh` |

The canonical rule is protocol-based, not filename-based: every Qwen family is
strict zero-shot, while OLMoE uses the multi-shot evaluator.

## Quantization

| Original script | Package module / launcher |
| --- | --- |
| `src/open_r1/quantize_base_awq.py` | `less_is_moe.quantization.awq_base` / `scripts/quantize/awq_base.sh` |
| `src/open_r1/quantize_pruned_awq.py` | `less_is_moe.quantization.awq_pruned` / `scripts/quantize/awq_pruned.sh` |
| `src/open_r1/quantize_pruned_gptq.py` | `less_is_moe.quantization.gptq_pruned` / `scripts/quantize/gptq_pruned.sh` |
| `src/open_r1/evaluate_awq_vllm.py` | `scripts/quantize/evaluate_awq.sh`, which delegates to the canonical zero-shot evaluator with `--quantization awq` |

The quantized evaluator is not a separate prompt/scoring implementation. That
prevents the AWQ results from silently diverging from the Qwen zero-shot
protocol.

## SFT

| Original script | Package module / launcher |
| --- | --- |
| `src/open_r1/sft_base.py` | `less_is_moe.training.sft_base` / `scripts/train/base.sh` |
| `src/open_r1/sft_baseline.py` | Renamed for clarity to `less_is_moe.training.sft_pruned` / `scripts/train/pruned.sh` |

Other online-pruning, ablation, and auxiliary SFT variants are not part of the
release interface. The pruned SFT entry point expects an already-pruned model.
