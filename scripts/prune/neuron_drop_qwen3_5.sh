#!/usr/bin/env bash
# Deprecated launcher, kept for one release. The per-family pruning modules were
# retired in favour of one implementation; see docs/MIGRATION.md.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[deprecated] neuron_drop_qwen3_5.sh now runs: python -m less_is_moe.intdim.prune --mode mask" >&2

exec "${PYTHON_BIN:-python}" -m less_is_moe.intdim.prune \
  --mode mask \
  "$@"
