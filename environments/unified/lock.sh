#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ "$(uv --version)" != 'uv 0.12.5'* ]]; then
  echo 'Regenerate the unified lock with uv 0.12.5.' >&2
  exit 1
fi
uv pip compile pyproject.toml environments/unified/requirements.in \
  --extra test --python-version 3.12.14 \
  --python-platform x86_64-manylinux_2_39 --torch-backend cu130 \
  --no-build --generate-hashes --no-annotate \
  --custom-compile-command './environments/unified/lock.sh' \
  --output-file environments/unified/requirements.txt "$@"
