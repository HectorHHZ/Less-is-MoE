# Script reference

This page records what the experiment scripts do and why only three evaluation
entry points remain in the release. For the old-to-new path mapping, see
[Migration](MIGRATION.md).

## Neuron pruning

One command prunes every supported family:

```bash
python -m less_is_moe.intdim.prune --mode mask|structural --prune_mode expert|layer|global
```

It estimates one score per routed-expert FFN neuron from the calibration
language-model loss:

```text
score[j] = mean(abs(gradient slices for gate[j], up[j], and down[:, j]))
```

Despite the historical `pure_gradient` naming in the retired modules, this is
mean absolute gradient, not squared-gradient Fisher information. Shared experts
are left unchanged.

`--mode mask` keeps every tensor shape unchanged and zeroes the selected
`gate_proj`/`up_proj` rows and matching `down_proj` columns. Three selection
scopes are available:

- `expert`: rank independently inside each expert. Every expert loses the same
  number of neurons, so the result can later be structurally compacted.
- `layer`: rank across all routed experts in one layer. Per-expert counts may
  differ, so this is a zero-mask-only result.
- `global`: rank across all routed experts and layers. Both per-expert and
  per-layer counts may differ, so this is also zero-mask-only.

`--mode structural` physically replaces expert projections with smaller ones.
It can either score a base checkpoint directly with `--drop_ratio`, or compact
a compatible masked checkpoint with `--from_zeroed_model`. Structural
compaction requires one uniform surviving width; layer and global masks are
rejected instead of writing an unloadable checkpoint. The saved checkpoint is
verified with the stock Transformers loader unless `--skip_verify` is passed.

Family-specific behavior that the shared implementation preserves:

| Family | Behavior |
| --- | --- |
| Qwen1.5/Qwen2-MoE | Routed experts only; the shared expert is preserved. |
| Qwen3-MoE | No shared expert in the target model. |
| Qwen3.5-MoE | Handles the Transformers 5 batched-expert layout and the legacy ModuleList layout. |
| OLMoE | Routed experts only; no shared expert in the target model. |
| gpt-oss | Interleaved gate/up rows with biases; the bias entries of dropped units are zeroed with their rows. |
| Gemma-4 | GeGLU experts; the parallel dense MLP is not pruned. |

`--unwrap_message_content` (default on) reproduces the Qwen3/Qwen3.5 handling
of chat-message fields in Hugging Face datasets;
`--no-unwrap_message_content` reproduces the Qwen1.5-MoE and OLMoE handling.
The deprecated launchers under `scripts/prune/` pass the value their family
used.

The ten retired per-family modules now live in `tests/legacy_reference/`. They
are not installed and have no launchers; the equivalence suite runs them as the
oracle for the shared implementation. See [Migration](MIGRATION.md).

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
