"""Prove that a saved checkpoint loads with stock Transformers classes.

A structurally pruned ``IntDim-E`` checkpoint is a standard model whose expert
intermediate size is smaller, so nothing in ``model_patches`` should be needed
to load it. This module checks that claim with the Transformers loader itself.

The checkpoint is loaded into the stock class on the meta device. The loader
applies its own checkpoint conversions first (for example, Transformers 5.x
merges per-expert ``experts.N.gate_proj`` tensors into a fused
``experts.gate_up_proj``), then reports missing, unexpected, and shape-mismatched
keys. Comparing names directly would get those conversions wrong, so the
loader's report is the ground truth.

Run it after every structural prune::

    python -m less_is_moe.intdim.verify /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerifyReport:
    checkpoint: str
    architecture: str = ""
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.mismatched or self.errors)

    def __str__(self) -> str:
        lines = [f"{'OK' if self.ok else 'FAIL'}: {self.checkpoint} as {self.architecture or '?'}"]
        for label, items in (
            ("missing from checkpoint", self.missing),
            ("unexpected in checkpoint", self.unexpected),
            ("shape mismatches", self.mismatched),
            ("loader errors", self.errors),
        ):
            if items:
                lines.append(f"  {label} ({len(items)}):")
                lines += [f"    {item}" for item in items[:20]]
                if len(items) > 20:
                    lines.append(f"    ... {len(items) - 20} more")
        return "\n".join(lines)


def _as_strings(items) -> list[str]:
    """Normalise loader info entries, which may be names or (name, shapes) tuples."""
    out = []
    for item in items or []:
        if isinstance(item, (tuple, list)):
            out.append(" ".join(str(part) for part in item))
        else:
            out.append(str(item))
    return sorted(out)


def verify_checkpoint(path: str | Path, device: str = "meta") -> VerifyReport:
    """Load ``path`` with the stock auto class and report any key or shape problem.

    ``device="meta"`` (the default) builds the model without materialising
    weights. Pass ``"cpu"`` for a full load.
    """
    from transformers import AutoModelForCausalLM

    path = Path(path)
    report = VerifyReport(str(path))
    try:
        model, info = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
    except Exception as exc:  # noqa: BLE001 - any load failure is a verification failure
        report.errors.append(f"{type(exc).__name__}: {exc}")
        return report
    report.architecture = type(model).__name__
    report.missing = _as_strings(info.get("missing_keys"))
    report.unexpected = _as_strings(info.get("unexpected_keys"))
    report.mismatched = _as_strings(info.get("mismatched_keys"))
    report.errors = _as_strings(info.get("error_msgs"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Directory containing config.json and *.safetensors")
    parser.add_argument("--device", default="meta", help="Device map for loading: meta (default, no weights) or cpu")
    args = parser.parse_args(argv)
    report = verify_checkpoint(args.checkpoint, device=args.device)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
