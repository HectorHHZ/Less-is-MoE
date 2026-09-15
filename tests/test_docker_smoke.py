from __future__ import annotations

import runpy
from pathlib import Path

import pytest

SMOKE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docker/smoke_test.py"))


def test_lock_parser_preserves_cuda_wheel_build() -> None:
    text = "# generated lock\ntorch==2.13.0+cu130 \\\n    --hash=sha256:abc\ntransformers==5.17.0\n"
    assert SMOKE["locked_versions"](text) == {
        "torch": "2.13.0+cu130",
        "transformers": "5.17.0",
    }


@pytest.mark.parametrize("actual", ({}, {"torch": "2.13.0+cpu"}, {"torch": "2.14.0+cu130"}))
def test_version_check_rejects_missing_or_changed_wheels(actual: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="Environment drift"):
        SMOKE["check_versions"]({"torch": "2.13.0+cu130"}, actual)


def test_version_check_accepts_exact_versions() -> None:
    SMOKE["check_versions"]({"torch": "2.13.0+cu130"}, {"torch": "2.13.0+cu130"})
