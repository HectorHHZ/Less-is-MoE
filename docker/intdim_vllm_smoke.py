"""GPU-only HF IntDim-E -> stock vLLM generation, using tiny random checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FAMILIES = ("qwen2_moe", "qwen3_moe", "olmoe", "qwen3_5_moe_35b", "qwen3_5_moe_122b", "gpt_oss", "gemma4")


def make_config(family):
    import transformers as hf

    common = dict(hidden_size=128, num_hidden_layers=2, vocab_size=256,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                  max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    moe = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256)
    if family == "qwen2_moe":
        return hf.Qwen2MoeConfig(**common, **moe, intermediate_size=512, shared_expert_intermediate_size=256, decoder_sparse_step=1)
    if family == "qwen3_moe":
        return hf.Qwen3MoeConfig(**common, **moe, intermediate_size=512, mlp_only_layers=[])
    if family == "olmoe":
        return hf.OlmoeConfig(**common, intermediate_size=256, num_experts=4, num_experts_per_tok=2)
    if family.startswith("qwen3_5_moe"):
        moe["moe_intermediate_size"] = 512 if family.endswith("35b") else 1024
        return hf.Qwen3_5MoeConfig(text_config=dict(
            **common, **moe, intermediate_size=512, shared_expert_intermediate_size=256,
            linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=64,
            linear_value_head_dim=64, linear_conv_kernel_dim=4, full_attention_interval=2,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                             "partial_rotary_factor": 0.25, "mrope_section": [2, 3, 3]}))
    if family == "gpt_oss":
        return hf.GptOssConfig(**common, intermediate_size=2880, num_local_experts=4,
                               num_experts_per_tok=2, sliding_window=128)
    return hf.Gemma4TextConfig(**common, intermediate_size=512, moe_intermediate_size=704,
                               num_experts=4, top_k_experts=2, enable_moe_block=True,
                               sliding_window=128, global_head_dim=64, num_global_key_value_heads=1,
                               hidden_size_per_layer_input=0, vocab_size_per_layer_input=256)


def prepare(family, directory):
    import torch
    from transformers import AutoModelForCausalLM
    from less_is_moe.intdim import discover, verify_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires a GPU")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(make_config(family), dtype=torch.bfloat16).cuda().eval()
    assert sum(p.numel() for p in model.parameters()) < 20_000_000
    model.save_pretrained(directory / "base")
    handles = discover(model)
    for h in handles:
        # Different retained units in each expert, with one uniform width.
        keep = torch.stack([torch.randperm(h.intermediate_size)[:h.intermediate_size // 2]
                            for _ in range(h.num_experts)])
        h.apply_units(keep)
    handles[0].intermediate_size_key.set(model.config, handles[0].intermediate_size)
    model.save_pretrained(directory / "pruned")
    report = verify_checkpoint(directory / "pruned")
    if not report.ok:
        raise RuntimeError(str(report))
    del model, handles
    gc.collect()
    torch.cuda.empty_cache()


def generate(checkpoint):
    from vllm import LLM, SamplingParams

    engine = LLM(model=str(checkpoint), skip_tokenizer_init=True, dtype="bfloat16",
                 enforce_eager=True, max_model_len=128, max_num_seqs=1,
                 max_num_batched_tokens=128, gpu_memory_utilization=0.08,
                 seed=0, trust_remote_code=False,
                 attention_config={"backend": "TRITON_ATTN"},
                 kernel_config={"moe_backend": "triton"})
    results = engine.generate([{"prompt_token_ids": [1, 3, 5, 7]}],
                              SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
                              use_tqdm=False)
    tokens = results[0].outputs[0].token_ids
    if len(tokens) != 2 or any(t < 0 or t >= 256 for t in tokens):
        raise RuntimeError(f"Invalid generation: {tokens}")
    print(json.dumps({"checkpoint": str(checkpoint), "tokens": list(tokens), "stock_vllm": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    # No project runtime replacements may participate in this check.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.checkpoint:
        generate(args.checkpoint)
        return
    for family in FAMILIES if args.family == "all" else (args.family,):
        with tempfile.TemporaryDirectory(prefix=f"intdim-{family}-") as temporary:
            directory = Path(temporary)
            prepare(family, directory)
            for stage in ("base", "pruned"):
                subprocess.run([sys.executable, __file__, "--checkpoint", str(directory / stage)], check=True, timeout=600)
        print(json.dumps({"family": family, "base_and_pruned": "passed"}), flush=True)


if __name__ == "__main__":
    main()
