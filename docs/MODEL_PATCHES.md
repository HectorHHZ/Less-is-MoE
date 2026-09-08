# Runtime model patches

Structurally pruned MoE checkpoints contain layer-wise expert dimensions that
the stock Hugging Face and vLLM implementations do not understand. The modules
under `src/less_is_moe/model_patches/` provide those compatibility changes.

## Hugging Face

The public registry API is:

```python
from less_is_moe.model_patches import apply_hf_patch, detect_model_family

family = detect_model_family("MODEL_ID_OR_LOCAL_PATH")
apply_hf_patch(family)
```

Detection reads `model_type`, `architectures`, and (where applicable) nested
text configuration. It does not infer a family from a directory name. Apply
the patch before constructing the model.

Supported canonical names are `qwen2_moe` (Qwen1.5-MoE), `qwen3_moe`,
`qwen3_5_moe`, and `olmoe`.

Qwen3.5's Hugging Face implementation stores experts in batched tensors and
therefore requires one scalar `num_experts` shared by all layers. The registry
fails early for list-valued/uneven expert counts instead of constructing a
checkpoint incorrectly. The released Qwen3.5 neuron mask and structural-neuron
workflows keep a uniform expert count and are supported. Online/manual expert
mask materialization during Qwen3.5 SFT is not supported; prune first and then
run `scripts/train/pruned.sh`.

## vLLM

Evaluation entry points install the corresponding vLLM patch before creating
an engine. Qwen3.5 additionally needs the process-safe vLLM general plugin,
because vLLM 0.19.1 starts an EngineCore subprocess. Installing this repository
with `pip install -e .` registers:

```text
less_is_moe.model_patches.vllm_plugin:register
```

The plugin is import-light and becomes a no-op on older vLLM versions that do
not expose Qwen3.5. It therefore remains safe to leave the package installed in
the legacy and Qwen3 environments.

Use `--base_model` in evaluation commands when loading an ordinary, unpruned
checkpoint. Applying a structural-pruning patch to a base checkpoint is neither
needed nor part of the reported protocol.
