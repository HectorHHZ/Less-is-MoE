#!/usr/bin/env bash
# Deprecated launcher, kept for one release. The per-family pruning modules were
# retired in favour of one implementation; see docs/MIGRATION.md.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[deprecated] neuron_structure_drop_qwen15_moe.sh now runs: python -m less_is_moe.intdim.prune --mode structural" >&2

exec "${PYTHON_BIN:-python}" -m less_is_moe.intdim.prune \
  --mode structural \
  --no-unwrap_message_content \
  "$@"
