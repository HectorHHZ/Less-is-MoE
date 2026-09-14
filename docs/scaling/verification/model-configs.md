# Model configs and budget math

Answers [#3](https://github.com/HectorHHZ/Less-is-MoE/issues/3). Every number on this page comes from each checkpoint's `config.json` and safetensors headers, produced by [`inspect_checkpoints.py`](inspect_checkpoints.py). No weights are downloaded.

```bash
python docs/scaling/verification/inspect_checkpoints.py
```

## Summary

- The scaling plan's model table is correct, with one correction: Gemma-4's "shared expert" is an ungated dense MLP added in parallel to the MoE block.
- Qwen3.5-9B is dense. It is an evaluation-only reference and is never pruned.
- Every FFN is a gate/up/down triple, so intermediate-dimension pruning applies to all four MoE backbones.
- At p = 50%, the whole-model reduction `p_model` is 45.3%–49.1%.
- The paper's `p_model` for Qwen3.5-35B-A3B (48.4%) compares the full checkpoint with a text-only compressed model. On a consistent basis it is 46.5%. See [Accounting convention](#accounting-convention).

## Checked specifications

Revisions: gpt-oss-120b `b5c939de8f75`, Qwen3.5-122B-A10B `dc4d348443bc`, Qwen3.5-35B-A3B `59d61f3ce65a`, gemma-4-26B-A4B `24548b62aa02`, Qwen3.5-9B `c20223623576`.

| Model | Layers | Routed experts | Top-k | Expert intermediate | Hidden | Shared / dense FFN | Plan | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| gpt-oss-120b | 36 | 128 | 4 | 2880 | 2880 | none | 36 / 128 / 4 / 0 | ✅ matches |
| Qwen3.5-122B-A10B | 48 | 256 | 8 | 1024 | 3072 | 1 gated shared expert (1024) | 48 / 256 / 8 / 1 | ✅ matches |
| Qwen3.5-35B-A3B | 40 | 256 | 8 | 512 | 2048 | 1 gated shared expert (512) | 40 / 256 / 8 / 1 | ✅ matches |
| gemma-4-26B-A4B | 30 | 128 | 8 | 704 | 2816 | ungated parallel dense MLP (2112) | 30 / 128 / 8 / 1 | ⚠️ not a shared expert |
| Qwen3.5-9B | 32 | — | — | — | 4096 | dense FFN (12288) | dense | ✅ dense |

## Architecture findings

**FFN structure.** Checked against the Hugging Face Transformers modeling code.

| Family | Expert weights on disk | Gate/up layout | Activation | Biases |
| --- | --- | --- | --- | --- |
| Qwen3.5 | fused `gate_up_proj` (E, 2I, H), `down_proj` (E, H, I) | concatenated: gate then up | SiLU (SwiGLU) | none |
| Gemma-4 | fused `gate_up_proj` (E, 2I, H), `down_proj` (E, H, I) | concatenated: gate then up | `gelu_pytorch_tanh` (GeGLU) | none |
| gpt-oss | fused, MXFP4 `*_blocks` / `*_scales` | **interleaved**: gate `[::2]`, up `[1::2]` | clamped SwiGLU (α = 1.702, limit 7, `up + 1`) | `gate_up_proj_bias`, `down_proj_bias` |

**Qwen3.5 (122B-A10B, 35B-A3B).**
- The MoE FFN is unchanged from Qwen3.5-35B-A3B, and its layout is the one the released pruning code already handles.
- Layers alternate three GatedDeltaNet layers with one full-attention layer (122B: 36 + 12; 35B: 30 + 10).
- Importance is already restricted to routed experts: the pruning code freezes every parameter and enables gradients only on `gate_up_proj` and `down_proj`. No gradient accumulates on GatedDeltaNet parameters.
- Checkpoints include a vision tower and an MTP module. Transformers maps `qwen3_5_moe` to the text-only `Qwen3_5MoeForCausalLM`, which ignores `^model.visual.*` and `^mtp.*` at load time, so pruned checkpoints contain neither. Pruned models cannot use MTP speculative decoding.

**gpt-oss-120b.**
- Only the expert weights are MXFP4. Attention, router, embeddings, and `lm_head` are BF16.
- MXFP4 blocks have shape (E, rows, 90, 16), and 90 × 16 × 2 = 2880 values per row, so parameter counts are exact.
- Pruning needs a new path for fused, interleaved, biased expert tensors. Removing an intermediate dimension also removes two `gate_up_proj_bias` entries; `down_proj_bias` has hidden size and is never removed.
- Attention alternates sliding-window (128) and full layers, with biases and attention sinks.

**gemma-4-26B-A4B.**
- GeGLU keeps the gate/up/down triple, so the dimension grouping is unchanged. The fused layout matches Qwen3.5.
- Each layer sums a dense MLP (2112) and the MoE output. The dense MLP has no gate and is not pruned.
- The router adds a learned `per_expert_scale` to the top-k weights. It must be preserved but does not affect intermediate-dimension pruning.
- The checkpoint contains a vision tower but no audio weights, although the config lists `audio_config`. Use the text path.
- Attention is 25 sliding-window (1024) and 5 full layers, and embeddings are tied.

**Qwen3.5-9B.** Dense FFN in every layer, with the same hybrid attention as the MoE models. Evaluation-only reference for Table 5.

## Accounting convention

- **Language model** counts every text parameter and excludes the vision tower and the MTP module, because pruned checkpoints contain neither.
- **`p_model`** is `removed / language model`. Using the same basis before and after pruning keeps the ratio a true reduction.
- **Active parameters** include embeddings and `lm_head`. Vendor figures that exclude embeddings are smaller: gpt-oss-120b is 5.71B here and 5.13B without input embeddings, which matches OpenAI's 5.1B.

The paper's model-size table reports Qwen3.5-35B-A3B as 36.0B → 18.6B (`p_model` = 48.4%). The 36.0B is the full checkpoint (35.95B, including the 0.45B vision tower and 0.84B MTP module). The 18.6B is the text-only pruned model: 35.95 − 16.11 removed − 0.45 − 0.84 = 18.55B. On a language-model basis the reduction is **46.5%**. The Qwen3-30B-A3B row is unaffected because that model has no vision tower or MTP module. The scaling tables should use the language-model basis throughout.

## Parameter accounting

Output of `inspect_checkpoints.py` at `--ratio 0.5`.

### openai/gpt-oss-120b

| Component | Parameters | Share of language model |
| --- | ---: | ---: |
| embeddings | 579,133,440 | 0.5% |
| lm_head | 579,133,440 | 0.5% |
| routed experts | 114,701,598,720 | 98.2% |
| router | 13,275,648 | 0.0% |
| full / sliding attention | 955,805,184 | 0.8% |
| norms and scalars | 210,240 | 0.0% |
| **language model (excludes tower and MTP)** | **116,829,156,672** | 100.0% |
| checkpoint total | 116,829,156,672 | — |

### Qwen/Qwen3.5-122B-A10B

| Component | Parameters | Share of language model |
| --- | ---: | ---: |
| vision/audio tower | 451,290,864 | — |
| MTP module | 2,523,679,232 | — |
| embeddings | 762,839,040 | 0.6% |
| lm_head | 762,839,040 | 0.6% |
| routed experts | 115,964,116,992 | 95.0% |
| router | 37,896,192 | 0.0% |
| shared expert / dense FFN | 452,984,832 | 0.4% |
| full / sliding attention | 943,724,544 | 0.8% |
| linear attention (GatedDeltaNet) | 3,186,828,288 | 2.6% |
| norms and scalars | 297,984 | 0.0% |
| **language model (excludes tower and MTP)** | **122,111,526,912** | 100.0% |
| checkpoint total | 125,086,497,008 | — |

### Qwen/Qwen3.5-35B-A3B

| Component | Parameters | Share of language model |
| --- | ---: | ---: |
| vision/audio tower | 446,571,248 | — |
| MTP module | 844,640,768 | — |
| embeddings | 508,559,360 | 1.5% |
| lm_head | 508,559,360 | 1.5% |
| routed experts | 32,212,254,720 | 92.9% |
| router | 21,053,440 | 0.1% |
| shared expert / dense FFN | 125,829,120 | 0.4% |
| full / sliding attention | 272,634,880 | 0.8% |
| linear attention (GatedDeltaNet) | 1,011,553,920 | 2.9% |
| norms and scalars | 165,888 | 0.0% |
| **language model (excludes tower and MTP)** | **34,660,610,688** | 100.0% |
| checkpoint total | 35,951,822,704 | — |

### google/gemma-4-26B-A4B

| Component | Parameters | Share of language model |
| --- | ---: | ---: |
| vision/audio tower | 572,794,416 | — |
| embeddings | 738,197,504 | 2.9% |
| routed experts | 22,837,985,280 | 90.5% |
| router | 10,901,760 | 0.0% |
| shared expert / dense FFN | 535,265,280 | 2.1% |
| full / sliding attention | 1,110,197,760 | 4.4% |
| norms and scalars | 594,206 | 0.0% |
| **language model (excludes tower and MTP)** | **25,233,141,790** | 100.0% |
| checkpoint total | 25,805,936,206 | — |

### Qwen/Qwen3.5-9B

| Component | Parameters | Share of language model |
| --- | ---: | ---: |
| vision/audio tower | 456,010,480 | — |
| MTP module | 243,290,624 | — |
| embeddings | 1,017,118,720 | 11.4% |
| lm_head | 1,017,118,720 | 11.4% |
| shared expert / dense FFN | 4,831,838,208 | 54.0% |
| full / sliding attention | 469,766,144 | 5.2% |
| linear attention (GatedDeltaNet) | 1,617,695,232 | 18.1% |
| norms and scalars | 266,240 | 0.0% |
| **language model (excludes tower and MTP)** | **8,953,803,264** | 100.0% |
| checkpoint total | 9,653,104,368 | — |

### Budget at ratio 0.5 

| Model | Language model | Routed experts | Removed dims / expert | Removed params | Share of routed | p_model (language model) | p_model (checkpoint) | Active before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai/gpt-oss-120b | 116.83B | 114.70B | 1440 of 2880 | 57.34B | 49.99% | 49.1% | 49.1% | 5.71B → 3.92B |
| Qwen/Qwen3.5-122B-A10B | 122.11B | 115.96B | 512 of 1024 | 57.98B | 50.00% | 47.5% | 46.4% | 9.77B → 7.96B |
| Qwen/Qwen3.5-35B-A3B | 34.66B | 32.21B | 256 of 512 | 16.11B | 50.00% | 46.5% | 44.8% | 3.45B → 2.95B |
| google/gemma-4-26B-A4B | 25.23B | 22.84B | 352 of 704 | 11.42B | 50.00% | 45.3% | 44.2% | 3.82B → 3.11B |
| Qwen/Qwen3.5-9B | 8.95B | dense — not pruned | — | — | — | — | — | 8.95B (dense) |

"Removed dims / expert" is the IntDim-E allocation. IntDim-L and IntDim-G remove the same total number of parameters, allocated differently. The gpt-oss share is 49.99% because `down_proj_bias` is not removable.
