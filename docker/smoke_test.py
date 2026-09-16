"""Check the pinned environment without downloading models or activating patches."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def locked_versions(text: str) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", text, re.MULTILINE))


def check_versions(expected: dict[str, str], actual: dict[str, str]) -> None:
    mismatches = [
        f"{name}: expected {version}, found {actual.get(name, 'missing')}"
        for name, version in expected.items()
        if actual.get(name) != version
    ]
    if mismatches:
        raise RuntimeError("Environment drift:\n" + "\n".join(mismatches))


def check_gpu(torch) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; expose a GPU with NVIDIA Container Toolkit/CDI")
    # Exercise cuBLAS and a vLLM compiled CUDA extension on a tiny BF16 input.
    x = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(32, 128) / 4096
    x = x.to(torch.bfloat16)
    identity = torch.eye(128, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(x @ identity, x, rtol=0, atol=0)
    from vllm import _custom_ops as ops

    weight = torch.ones(128, device="cuda", dtype=x.dtype)
    output = torch.empty_like(x)
    ops.rms_norm(output, x, weight, 1e-6)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
    torch.cuda.synchronize()
    return {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "checks": ["bf16_matmul", "vllm_rms_norm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also test CUDA and the vLLM CUDA extension")
    args = parser.parse_args()
    expected = locked_versions((ROOT / "environments/unified/requirements.txt").read_text())
    if not {"torch", "transformers", "vllm", "tokenizers", "pip"} <= expected.keys():
        raise RuntimeError("The unified dependency lock is incomplete")
    actual = {name: importlib.metadata.version(name) for name in expected}
    check_versions(expected, actual)
    check_versions({"python": "3.12.14"}, {"python": platform.python_version()})
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    modules = {name: importlib.import_module(name) for name in ("torch", "transformers", "vllm", "tokenizers", "less_is_moe")}
    torch = modules["torch"]
    check_versions({"cuda": "13.0"}, {"cuda": torch.version.cuda})
    report = {
        "environment": "unified",
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "locked_packages": len(expected),
        "versions": {name: actual[name] for name in ("torch", "transformers", "vllm", "tokenizers")},
    }
    if args.gpu:
        report["gpu"] = check_gpu(torch)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
