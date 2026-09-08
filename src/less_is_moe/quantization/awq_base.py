"""
AWQ quantization for base (non-pruned) models.

Usage (run from project root):
python -m less_is_moe.quantization.awq_base \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path outputs/Qwen1.5-MoE-A2.7B-awq

Install the bundled compatibility fork:
    pip install -e third_party/AutoAWQ --no-build-isolation
    pip install autoawq-kernels==0.0.9 --no-build-isolation
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AWQ quantization for base (non-pruned) models"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--w_bit", type=int, default=4, choices=[4])
    parser.add_argument("--q_group_size", type=int, default=128)
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

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    quant_config = {
        "zero_point": not args.no_zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    print(f"[awq] Loading model from: {args.model_path}")
    print(f"[awq] Quant config: {quant_config}")

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

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

    print("[awq] Done.")


if __name__ == "__main__":
    main()
