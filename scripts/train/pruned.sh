#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/recipes/accelerate/zero2.yaml}"

exec "${ACCELERATE_BIN:-accelerate}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m less_is_moe.training.sft_pruned "$@"
