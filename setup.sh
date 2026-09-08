#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./setup.sh PROFILE [OPTIONS]

Profiles:
  legacy   Qwen1.5-MoE and OLMoE (Transformers 4.49 / vLLM 0.8.4)
  qwen3    Qwen3-MoE (Transformers 4.53.1 / vLLM 0.8.4)
  qwen35   Qwen3.5-MoE (Transformers 5.2 / vLLM 0.19.1)

Options:
  --with-awq        Install the bundled AutoAWQ fork (legacy/qwen3 only)
  --with-gptq       Install GPTQModel (legacy only)
  --skip-flash-attn Skip flash-attn installation for legacy/qwen3
  -h, --help        Show this help

Environment overrides:
  PYTHON_BIN   Python interpreter used to create the venv (default: python3.11)
  VENV_DIR     Destination venv (default: .venv-PROFILE in this repository)
  MAX_JOBS     Parallel jobs used while compiling flash-attn
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

PROFILE="$1"
shift
WITH_AWQ=0
WITH_GPTQ=0
WITH_FLASH_ATTN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-awq) WITH_AWQ=1 ;;
    --with-gptq) WITH_GPTQ=1 ;;
    --skip-flash-attn) WITH_FLASH_ATTN=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${REPO_ROOT}/environments/${PROFILE}/requirements.txt"
TORCH_REQUIREMENTS="${REPO_ROOT}/environments/${PROFILE}/requirements-torch.txt"

case "${PROFILE}" in
  legacy)
    VLLM_VERSION="0.8.4"
    TRANSFORMERS_VERSION="4.49.0"
    METADATA_OVERRIDE_REASON="vLLM 0.8.4 declares Transformers >=4.51.1, but the legacy experiments require 4.49.0."
    ;;
  qwen3)
    VLLM_VERSION="0.8.4"
    TRANSFORMERS_VERSION="4.53.1"
    METADATA_OVERRIDE_REASON=""
    if [[ ${WITH_GPTQ} -eq 1 ]]; then
      echo "GPTQ is supported only by the legacy Qwen1.5/Qwen2-MoE profile." >&2
      exit 2
    fi
    ;;
  qwen35)
    VLLM_VERSION="0.19.1"
    TRANSFORMERS_VERSION="5.2.0"
    METADATA_OVERRIDE_REASON="vLLM 0.19.1 excludes Transformers 5.2.*, but the Qwen3.5 experiment patch requires 5.2.0."
    if [[ ${WITH_AWQ} -eq 1 || ${WITH_GPTQ} -eq 1 ]]; then
      echo "AWQ/GPTQ are not supported for Qwen3.5 in this release." >&2
      exit 2
    fi
    WITH_FLASH_ATTN=0
    ;;
  *) echo "Unknown profile: ${PROFILE}" >&2; usage >&2; exit 2 ;;
esac

PYTHON_COMMAND="${PYTHON_BIN:-python3.11}"
ENVIRONMENT_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-${PROFILE}}"

if ! command -v "${PYTHON_COMMAND}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_COMMAND}" >&2
  exit 1
fi

"${PYTHON_COMMAND}" -m venv "${ENVIRONMENT_DIR}"
ENVIRONMENT_PYTHON="${ENVIRONMENT_DIR}/bin/python"
"${ENVIRONMENT_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${ENVIRONMENT_PYTHON}" -m pip install -r "${TORCH_REQUIREMENTS}"
"${ENVIRONMENT_PYTHON}" -m pip install "vllm==${VLLM_VERSION}"

if [[ -n "${METADATA_OVERRIDE_REASON}" ]]; then
  cat >&2 <<EOF

WARNING: applying an intentional experimental dependency override.
${METADATA_OVERRIDE_REASON}
Transformers ${TRANSFORMERS_VERSION} will be restored with --no-deps after
vLLM installation. Keep this profile isolated; 'pip check' is expected to
report the vLLM/Transformers metadata disagreement.
EOF
fi

"${ENVIRONMENT_PYTHON}" -m pip install --no-deps \
  "transformers==${TRANSFORMERS_VERSION}"
"${ENVIRONMENT_PYTHON}" -m pip install -r "${REQUIREMENTS}"

if [[ ${WITH_FLASH_ATTN} -eq 1 ]]; then
  MAX_JOBS="${MAX_JOBS:-4}" \
    "${ENVIRONMENT_PYTHON}" -m pip install \
    "flash-attn==2.7.4.post1" --no-build-isolation
fi

"${ENVIRONMENT_PYTHON}" -m pip install -e "${REPO_ROOT}"

if [[ ${WITH_AWQ} -eq 1 ]]; then
  "${ENVIRONMENT_PYTHON}" -m pip install -e \
    "${REPO_ROOT}/third_party/AutoAWQ" --no-build-isolation
  "${ENVIRONMENT_PYTHON}" -m pip install \
    "autoawq-kernels==0.0.9" --no-build-isolation
fi

if [[ ${WITH_GPTQ} -eq 1 ]]; then
  # GPTQModel 2.2.0's source build imports torch during setup. Torch is already
  # installed in this profile, so expose it to the build instead of using an
  # empty isolated build environment.
  "${ENVIRONMENT_PYTHON}" -m pip install \
    "gptqmodel==2.2.0" --no-build-isolation
fi

cat <<EOF

Environment ready: ${ENVIRONMENT_DIR}
Activate it with:
  source "${ENVIRONMENT_DIR}/bin/activate"
EOF
