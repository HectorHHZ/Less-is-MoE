# Script reference

This page records what the experiment scripts do and why only three evaluation
entry points remain in the release. For the old-to-new path mapping, see
[Migration](MIGRATION.md).

## Neuron pruning

All eight neuron-pruning entry points estimate one score per routed-expert FFN
neuron from calibration-language-model loss:

```text
score[j] = mean(abs(gradient slices for gate[j], up[j], and down[:, j]))
```

Despite the historical `pure_gradient` naming in helper functions, this is
mean absolute gradient, not squared-gradient Fisher information. Shared experts
are left unchanged.

The four `neuron_drop_*` modules keep every tensor shape unchanged and zero the
selected `gate_proj`/`up_proj` rows and matching `down_proj` columns. Each has
three selection scopes:

- `expert`: rank independently inside each expert. Every expert loses the same
  number of neurons, so the result can later be structurally compacted.
- `layer`: rank across all routed experts in one layer. Per-expert counts may
  differ, so this is a zero-mask-only result.
- `global`: rank across all routed experts and layers. Both per-expert and
  per-layer counts may differ, so this is also zero-mask-only.

The four `neuron_structure_drop_*` modules physically replace expert
projections with smaller linear layers. They can either score a base checkpoint
directly with `--drop_ratio`, or compact a compatible masked checkpoint with
`--from_zeroed_model`. Structural compaction requires one uniform surviving
width; the scripts reject layer/global masks instead of writing an unloadable
checkpoint.

The newer unified command `python -m less_is_moe.intdim.prune --mode ragged`
can compact non-uniform Qwen3-MoE layer/global plans using a dedicated checkpoint
format and vLLM plugin. See [ragged expert support](RAGGED_EXPERTS.md) for its
GPU-only scope and loader requirements; this does not change the legacy
per-family scripts described above.

| Family | Mask module | Structural module | Family-specific behavior |
| --- | --- | --- | --- |
| Qwen1.5/Qwen2-MoE | `neuron_drop_qwen15_moe` | `neuron_structure_drop_qwen15_moe` | Routed experts only; the shared expert is preserved. |
| Qwen3-MoE | `neuron_drop_qwen3` | `neuron_structure_drop_qwen3` | ModuleList routed experts; no shared expert in the target model. |
| Qwen3.5-MoE | `neuron_drop_qwen3_5` | `neuron_structure_drop_qwen3_5` | Handles the Transformers 5 batched-expert layout and legacy ModuleList layout; use the `qwen35` environment. |
| OLMoE | `neuron_drop_olmoe` | `neuron_structure_drop_olmoe` | Routed experts only; no shared expert in the target model. |

Every implementation accepts a local JSON/JSONL calibration file or a Hugging
Face dataset. Qwen1.5 and OLMoE also retain the `ceval`, `math`, and `cmmlu`
calibration presets. Output checkpoints include a JSON pruning summary.

## Evaluation implementations

| Release module | Intended models | Prompt protocol | Notes |
| --- | --- | --- | --- |
| `evaluation.vllm_zero_shot` | Qwen1.5-MoE, Qwen3-MoE, Qwen3.5-MoE | Strict zero-shot | GSM8K, MBPP, MATH, MultiArith, and the other supported tasks receive no demonstrations. The entry point rejects OLMoE. |
| `evaluation.vllm_olmoe_multishot` | OLMoE only | Paper multi-shot | GSM8K uses 8-shot CoT, MATH 4-shot CoT, MBPP 3-shot, and MultiArith 4-shot; task-specific zero-shot formatting remains for benchmarks without a released demonstration set. The entry point rejects Qwen. |
| `evaluation.hf` | Qwen families only | Optional zero-shot fallback | Slower Transformers generation path for debugging; it is not the canonical paper evaluator and rejects OLMoE. |

The old files differed as follows:

- `evaluate_qwen1.5_moe_vllm.py` was already the strict zero-shot branch. It is
  now the family-checked `vllm_zero_shot` module.
- `evaluate_gsm8k_vllm.py` accumulated the paper demonstrations, broader
  benchmark scoring, chat/thinking controls, and several model patches. Its
  prompt/scoring behavior is now isolated as the OLMoE-only multi-shot module.
- `evaluate_gsm8k_vllm_old.py` was an earlier general evaluator, before the
  paper multi-shot prompt additions. It is a superseded experiment snapshot.
- `evaluate_gsm8k_vllm_new.py` was not a newer general evaluator: it was a
  narrow diagnostic for an auxiliary-loss-free router-bias experiment, with a
  simple Alpaca prompt and ad-hoc vLLM loader patching. It is unrelated to the
  released neuron-pruning path.
- `evaluate_gsm8k_hf.py` was the non-vLLM fallback. The retained version fixes
  batched decoding and is explicitly noncanonical.

The two superseded snapshots are intentionally not shipped. Keeping one
zero-shot implementation and one OLMoE multi-shot implementation prevents a
filename choice from silently changing the reported protocol.
