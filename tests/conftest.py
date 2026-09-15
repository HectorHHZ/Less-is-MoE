"""Shared GPU fixtures; ordinary tests do not require torch or a GPU."""

import os

import pytest


@pytest.fixture(params=("float32", "bfloat16"))
def runtime(request):
    import torch

    device = "cuda"
    dtype = getattr(torch, request.param)
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("INTDIM_REQUIRE_GPU") == "1":
            pytest.fail("INTDIM_REQUIRE_GPU=1 requires a visible GPU")
        pytest.skip("GPU-only regression suite")
    return device, dtype
