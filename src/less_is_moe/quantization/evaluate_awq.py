"""Compatibility entry point for AWQ evaluation.

AWQ uses the same strict zero-shot protocol as the other Qwen evaluations.
This module only selects vLLM's AWQ backend and delegates to the canonical
evaluator so prompt construction and scoring cannot drift.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    argv = list(sys.argv)
    if "--max_new_tokens" in argv:
        argv[argv.index("--max_new_tokens")] = "--max_tokens"
    if "--quantization" not in argv:
        argv.extend(["--quantization", "awq"])
    sys.argv = argv
    runpy.run_module(
        "less_is_moe.evaluation.vllm_zero_shot",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
