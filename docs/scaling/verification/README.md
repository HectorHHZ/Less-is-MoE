# Scaling verification

Answers to the verification questions that gate the [Less is MoE at Scale roadmap](https://github.com/HectorHHZ/Less-is-MoE/issues/2). Resolve these before any budget is fixed or baseline is implemented.

| Question | Issue | Answer |
| --- | --- | --- |
| Do the model specs hold, and what is the pruning budget? | [#3](https://github.com/HectorHHZ/Less-is-MoE/issues/3) | [Model configs and budget math](model-configs.md) |
| Can SlimQwen's pruning stage run without distillation? | [#4](https://github.com/HectorHHZ/Less-is-MoE/issues/4) | [SlimQwen prune-only baseline](slimqwen-prune-only.md) |

## Key results

- **Specs.** All five backbones match the plan, except that Gemma-4's "shared expert" is an ungated parallel dense MLP. Qwen3.5-9B is dense and evaluation-only.
- **Budget.** p = 50% removes half of every routed expert's intermediate dimensions, giving `p_model` of 45.3%–49.1% on a language-model basis.
- **Accounting.** Report `p_model` on the language model, excluding the vision tower and MTP module. The paper's 48.4% for Qwen3.5-35B-A3B mixes bases; the consistent value is 46.5%.
- **SlimQwen.** Its pruning stage is training-free and specified by closed-form formulas, so it can run without distillation. No official code exists, so it must be implemented.

## Reproduce

`inspect_checkpoints.py` needs only the Python standard library and reads checkpoint headers, not weights:

```bash
python docs/scaling/verification/inspect_checkpoints.py
```
