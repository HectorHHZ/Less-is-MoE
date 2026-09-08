"""
GPTQ quantization helper for pruned Qwen2-MoE checkpoints.

Features
--------
- Applies the pruned Qwen2-MoE block/decoder patch automatically unless
  `--base-model` is set.
- Runs GPTQModel quantization end-to-end with a small text calibration
  set pulled from `datasets` (default: wikitext-2, 128 samples).
- Saves a quantized checkpoint to `--output-path`.

Usage (from repo root)
----------------------
python -m less_is_moe.quantization.gptq_pruned \
  --model-path checkpoints/qwen1.5-pruned \
  --output-path outputs/qwen1.5-pruned-gptq \
  --bits 4 --group-size 128 \
  --calib-dataset wikitext --calib-name wikitext-2-raw-v1 --calib-split train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from datasets import load_dataset


def patch_pruned_qwen2_moe(use_base_model: bool) -> None:
    """
    Replace HF Qwen2-MoE blocks with the pruned versions.
    Must run before transformers imports its model classes.
    """
    if use_base_model:
        return

    from less_is_moe.model_patches.registry import apply_hf_patch

    apply_hf_patch("qwen2_moe")


def build_calibration_texts(
    dataset: str,
    name: str,
    split: str,
    field: str,
    nsamples: int,
) -> List[str]:
    if name:
        ds = load_dataset(dataset, name, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    # truncate early to keep RAM small
    column = ds.select(range(min(len(ds), nsamples)))[field]
    return column if isinstance(column, list) else list(column)


def parse_args():
    p = argparse.ArgumentParser(description="GPTQ quantization for pruned Qwen2-MoE")
    p.add_argument("--model-path", required=True, help="Path to pruned HF checkpoint")
    p.add_argument("--output-path", required=True, help="Where to save the GPTQ model")
    p.add_argument("--base-model", action="store_true", help="Skip pruning patch (use HF default blocks)")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="Weight bits (default: 4)")
    p.add_argument("--group-size", type=int, default=128, help="GPTQ group size (default: 128)")
    p.add_argument("--desc-act", action="store_true", help="Enable desc_act (act-order)")
    p.add_argument("--asym", action="store_true", help="Use asymmetric quantization (default: symmetric)")
    p.add_argument("--calib-dataset", default="wikitext", help="HF dataset name (default: wikitext)")
    p.add_argument("--calib-name", default="wikitext-2-raw-v1", help="HF dataset config/name (default: wikitext-2-raw-v1)")
    p.add_argument("--calib-split", default="train", help="HF dataset split (default: train)")
    p.add_argument("--calib-field", default="text", help="Text field name in dataset (default: text)")
    p.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples (default: 128)")
    p.add_argument("--calib-seqlen", type=int, default=2048, help="Max calibration sequence length")
    p.add_argument("--batch-size", type=int, default=1, help="Calibration batch size for GPTQModel.quantize")
    p.add_argument(
        "--backend",
        type=str.lower,
        default="torch",
        choices=[
            "auto",
            "torch",
            "triton",
            "exllama_v1",
            "exllama_v2",
            "marlin",
            "marlin_fp16",
            "ipex",
        ],
        help="GPTQModel 2.2 backend (default: torch)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    patch_pruned_qwen2_moe(use_base_model=args.base_model)

    # Import after patch so HF picks up pruned blocks
    # Some GPTQModel versions (e.g., 2.x) do not export FORMAT; rely on defaults.
    from gptqmodel import GPTQModel, QuantizeConfig
    try:
        from gptqmodel import BACKEND
    except Exception:
        BACKEND = None

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        sym=not args.asym,
    )

    print(f"[load] model: {args.model_path}")
    model = GPTQModel.from_pretrained(
        args.model_path,
        quantize_config=qcfg,
        trust_remote_code=True,
    )
    print(
        f"[calib] dataset={args.calib_dataset}/{args.calib_split} "
        f"samples={args.calib_samples} field={args.calib_field}"
    )
    calib_texts = build_calibration_texts(
        dataset=args.calib_dataset,
        name=args.calib_name,
        split=args.calib_split,
        field=args.calib_field,
        nsamples=args.calib_samples,
    )

    print(
        f"[quantize] bits={args.bits} group_size={args.group_size} "
        f"desc_act={args.desc_act} sym={not args.asym} backend={args.backend}"
    )
    backend_arg = BACKEND(args.backend) if BACKEND else args.backend
    model.quantize(
        calibration_dataset=calib_texts,
        batch_size=args.batch_size,
        calibration_dataset_concat_size=args.calib_seqlen,
        backend=backend_arg,
    )

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] saving quantized model to {out_dir}")
    model.save_quantized(str(out_dir))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(out_dir))
    print("[done] GPTQ quantization finished.")


if __name__ == "__main__":
    main()
