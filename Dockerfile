# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:7d56ebe2b7cd864a60dca3c8b2d0a39f8fc110417e8253e32505c3387f59119c AS unified

ARG REVISION=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/HectorHHZ/Less-is-MoE" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      org.less-is-moe.environment="unified"

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    VLLM_PLUGINS=""

# Keep a compiler and CUDA toolkit for upstream kernels compiled at runtime.
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates git libnuma1 \
    && rm -rf /var/lib/apt/lists/* \
    && UV_PYTHON_DOWNLOADS=automatic uv python install 3.12.14 \
    && uv venv --python 3.12.14 /opt/venv \
    && uv cache clean

WORKDIR /opt/less-is-moe
COPY environments/unified/requirements.txt environments/unified/requirements.txt
# Hash-checked wheels only: no dependency overrides or unpinned source builds.
RUN uv pip sync --python /opt/venv/bin/python --torch-backend cu130 \
        --require-hashes --no-build --no-cache environments/unified/requirements.txt

COPY . .
# All third-party/build dependencies are already locked. Install only our code.
RUN uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation --no-cache . \
    && python docker/smoke_test.py \
    && python -m pip freeze --all > /opt/less-is-moe-environment.txt \
    && dpkg-query -W > /opt/less-is-moe-system-packages.txt

ENV MAX_JOBS=4 \
    LESS_IS_MOE_RUNTIME_PATCH=stock \
    INTDIM_REQUIRE_GPU=1

CMD ["bash"]
