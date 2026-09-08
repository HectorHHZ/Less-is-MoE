# Quantization support

Quantization is optional and has a narrower support matrix than pruning.

| Workflow | Base model | Structurally pruned model | Qwen3.5 |
| --- | --- | --- | --- |
| AWQ | Legacy/Qwen3 adapters exposed by AutoAWQ | Qwen2-MoE and Qwen3-MoE adapters in the bundled fork | Not supported |
| GPTQ | Use `--base-model` with the Qwen2 helper if appropriate | Qwen2-MoE (Qwen1.5-MoE) | Not supported |

The repository vendors a compatibility fork of AutoAWQ under
`third_party/AutoAWQ`. It remains MIT licensed. Install it explicitly:

```bash
./setup.sh legacy --with-awq
```

The helper restores pruning-specific configuration fields after AWQ writes the
checkpoint. Keep the source checkpoint until you have verified both its
configuration and generated outputs.

The pruned AWQ launcher defaults to 4-bit weights and a group size of 32. It
checks routed-expert projection dimensions before quantization and fails with a
clear error if a custom pruning ratio produces a width that the bundled kernels
cannot pack.

GPTQModel is optional as well:

```bash
./setup.sh legacy --with-gptq
```

GPTQ is restricted to `legacy`; `setup.sh qwen3 --with-gptq` is rejected. The
release GPTQ adapter targets Qwen1.5/Qwen2-MoE structurally pruned checkpoints.

Neither route implements Qwen3.5's hybrid architecture. The setup script rejects
those extras under the Qwen3.5 profile to prevent a superficially successful but
unloadable artifact.
