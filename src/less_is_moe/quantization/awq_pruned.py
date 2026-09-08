"""
AWQ quantization for pruned MoE models (Qwen2-MoE / Qwen3-MoE).

Usage (run from project root):
    python -m less_is_moe.quantization.awq_pruned \
        --model_path checkpoints/qwen1.5-pruned \
        --output_path outputs/qwen1.5-pruned-awq \
        --model_type  qwen2_moe

Create the matching ``legacy`` or ``qwen3`` environment with ``setup.sh``;
Qwen3.5 is intentionally unsupported by the bundled AutoAWQ fork.
"""

import argparse
import json
from pathlib import Path


# ── 1. Patch for qwen3_moe ──────────────────────────────────────────────────

def patch_transformers(model_type: str) -> None:
    """Install the matching pruning-aware Transformers classes."""
    from less_is_moe.model_patches.registry import apply_hf_patch

    apply_hf_patch(model_type)
    print(f"[patch] transformers patched for {model_type}")


# ── 2. Post-quantization: restore custom config fields ──────────────────────

def restore_custom_config_fields(model_path: str, output_path: str):
    src_cfg_path = Path(model_path) / "config.json"
    dst_cfg_path = Path(output_path) / "config.json"

    if not src_cfg_path.exists() or not dst_cfg_path.exists():
        return

    with open(src_cfg_path) as f:
        src_cfg = json.load(f)
    with open(dst_cfg_path) as f:
        dst_cfg = json.load(f)

    custom_keys = [
        "num_experts_list",
        "mlp_only_layers",
        "decoder_sparse_step",
    ]
    merged = False
    for key in custom_keys:
        if key in src_cfg and key not in dst_cfg:
            dst_cfg[key] = src_cfg[key]
            merged = True
            print(f"[config] restored field: {key} = {src_cfg[key]}")

    if merged:
        with open(dst_cfg_path, "w") as f:
            json.dump(dst_cfg, f, indent=2)
        print(f"[config] saved merged config to {dst_cfg_path}")


def validate_expert_linear_alignment(model, group_size: int) -> None:
    """Fail early when a structurally pruned expert cannot use AWQ kernels."""
    import torch

    if group_size == 0 or group_size < -1:
        raise ValueError("--q_group_size must be a positive integer or -1")

    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if ".experts." not in name or not name.endswith(
            (".gate_proj", ".up_proj", ".down_proj")
        ):
            continue
        effective_group_size = module.in_features if group_size == -1 else group_size
        reasons = []
        if module.in_features % effective_group_size:
            reasons.append(
                f"in_features={module.in_features} is not divisible by "
                f"group_size={effective_group_size}"
            )
        if module.out_features % 8:
            reasons.append(f"out_features={module.out_features} is not divisible by 8")
        if reasons:
            failures.append(f"{name}: {', '.join(reasons)}")

    if failures:
        preview = "\n  - ".join(failures[:8])
        suffix = "" if len(failures) <= 8 else f"\n  ... and {len(failures) - 8} more"
        raise ValueError(
            "The pruned expert dimensions are incompatible with the bundled "
            "4-bit AWQ kernels:\n  - "
            f"{preview}{suffix}\nChoose a --q_group_size that divides every "
            "expert input dimension (32 works for the standard release ratios), "
            "or prune to dimensions aligned to both the group size and 8."
        )


# ── 3. Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AWQ quantization for pruned MoE checkpoints"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--model_type", default="qwen2_moe",
        choices=["qwen3_moe", "qwen2_moe"],
    )
    parser.add_argument("--w_bit", type=int, default=4, choices=[4])
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=32,
        help="AWQ group size (default: 32 for structurally pruned expert widths)",
    )
    parser.add_argument("--no_zero_point", action="store_true")
    parser.add_argument(
        "--version", default="GEMM",
        choices=["GEMM", "GEMV", "marlin"],
    )
    parser.add_argument(
        "--calib_dataset", default="pileval",
        choices=["pileval", "c4", "wikitext2"],
    )
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_seqlen", type=int, default=512)
    parser.add_argument(
        "--duo_scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply patches before AWQ imports Transformers model classes.
    patch_transformers(args.model_type)

    # Use pruned AWQ class matching model_type
    from transformers import AutoTokenizer
    if args.model_type == "qwen3_moe":
        from awq.models.qwen3_moe_pruned import Qwen3MoePrunedAWQForCausalLM as PrunedAWQCls
    else:
        from awq.models.qwen2_moe_pruned import Qwen2MoePrunedAWQForCausalLM as PrunedAWQCls

    quant_config = {
        "zero_point": not args.no_zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    print(f"[awq] Loading pruned model from: {args.model_path}")
    print(f"[awq] Model type: {args.model_type}")
    print(f"[awq] Quant config: {quant_config}")

    model = PrunedAWQCls.from_pretrained(
        args.model_path,
        args.model_type,
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    validate_expert_linear_alignment(model, args.q_group_size)

    print(f"[awq] Running quantization (calib_dataset={args.calib_dataset}, "
          f"samples={args.calib_samples}, seqlen={args.calib_seqlen}) ...")

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=args.calib_dataset,
        split="train",
        text_column="text",
        duo_scaling=args.duo_scaling,
        max_calib_samples=args.calib_samples,
        max_calib_seq_len=args.calib_seqlen,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[awq] Saving quantized model to: {output_path}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    restore_custom_config_fields(args.model_path, str(output_path))
    print("[awq] Done.")


if __name__ == "__main__":
    main()
