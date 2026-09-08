# Environment profiles

The three profiles are separate because their Transformers, vLLM, PyTorch, and
CUDA expectations are not interchangeable.

| Profile | Families | PyTorch / CUDA wheels | Transformers | vLLM | Other core pins |
| --- | --- | --- | --- | --- | --- |
| `legacy` | Qwen1.5-MoE, OLMoE | 2.6.0 / cu124 | 4.49.0 | 0.8.4 | tokenizers 0.21.x, DeepSpeed 0.15.3, flash-attn 2.7.4.post1 |
| `qwen3` | Qwen3-MoE | 2.6.0 / cu124 | 4.53.1 | 0.8.4 | tokenizers 0.21.x, DeepSpeed 0.15.3, flash-attn 2.7.4.post1 |
| `qwen35` | Qwen3.5-MoE | 2.10.0 / cu130 | 5.2.0 | 0.19.1 | tokenizers 0.22.2, DeepSpeed 0.16.7, xformers >= 0.0.35 |

Create an environment from the repository root:

```bash
./setup.sh legacy
source .venv-legacy/bin/activate
```

`setup.sh` creates `.venv-<profile>` by default. `VENV_DIR` changes the
destination and `PYTHON_BIN` selects an interpreter. The script contains no
cluster-specific cache directory, model path, authentication token, or GPU
selection.

## Intentional package-metadata overrides

The installer first installs the CUDA-specific PyTorch stack, then the selected
vLLM build. It subsequently restores the profile's Transformers version with
`pip install --no-deps` before installing the remaining requirements. This
order is deliberate:

- `legacy`: vLLM 0.8.4 declares `transformers>=4.51.1`, while the recorded
  Qwen1.5-MoE/OLMoE experiment stack uses Transformers 4.49.0.
- `qwen3`: vLLM 0.8.4 and Transformers 4.53.1 have compatible metadata; no
  override warning is expected.
- `qwen35`: vLLM 0.19.1 excludes Transformers 5.2.*, while the Qwen3.5 runtime
  patch was developed against Transformers 5.2.0.

Consequently, `pip check` is expected to report a vLLM/Transformers metadata
disagreement in `legacy` and `qwen35`. This is a documented reproduction choice,
not an unnoticed resolver result. Do not reuse either environment as a general
Python environment or run an unreviewed dependency upgrade inside it.

For the two cu124 profiles, flash-attn is installed after PyTorch with build
isolation disabled. Use `--skip-flash-attn` only if you intentionally select a
different attention backend and update the corresponding recipe. The Qwen3.5
profile uses SDPA/xformers and explicitly does not install flash-attn.

The NVIDIA driver must be new enough for the selected CUDA wheel. `setup.sh`
does not install a driver or CUDA toolkit. Verify the runtime before launching a
large job:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Optional AWQ and GPTQ dependencies are installed only when requested:

```bash
./setup.sh legacy --with-awq
./setup.sh legacy --with-gptq
```

GPTQ is limited to the `legacy` Qwen1.5/Qwen2-MoE profile. AWQ is available for
`legacy` and `qwen3`. Both extras are rejected by `qwen35` because this release
does not provide Qwen3.5 AWQ/GPTQ model adapters.
