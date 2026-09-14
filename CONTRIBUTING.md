# Contributing to Less is MoE

Thank you for helping improve Less is MoE. This is the research code for
[Less is MoE: Trimming Experts in Domain-Specialist Language Models](https://arxiv.org/abs/2606.05538)
(EMNLP 2026, Main Conference, Oral). Useful contributions include bug reports,
reproduction reports, support for new MoE model families, documentation, and
tests.

Because the repository backs published results, the bar for changes that
affect pruning, evaluation, or the reported protocol is higher than for
ordinary code. Please read [Protect the reported protocol](#protect-the-reported-protocol)
before changing either.

## Before you start

- Search existing issues and pull requests before opening a new one.
- Keep each issue and pull request focused on one problem or proposal.
- For a change whose direction or scope is uncertain — for example a new
  importance criterion or a change to an evaluator — open an issue before
  investing in an implementation.
- Never include credentials, private model URLs, private datasets, model
  weights, or other secrets in an issue, pull request, log, or fixture.

## Open an issue

Start the title with exactly one type prefix, then a short outcome:

| Prefix | Use it for |
| --- | --- |
| `[Bug]` | Reproducible incorrect behavior, such as a crash or an unloadable pruned checkpoint |
| `[Reproduction]` | A result that differs from the paper when following the release protocol |
| `[Model]` | Support for a new MoE family or a new version of a supported one |
| `[Feature]` | A new capability, such as a pruning option or quantization backend |
| `[Docs]` | Missing, unclear, or incorrect documentation |
| `[Question]` | A focused usage question the documentation does not answer |

For example: `[Bug] Structural Qwen3-MoE checkpoint fails to load in vLLM`.

Include the information needed to act on the report:

- **Bug:** the exact command, the full error, the environment profile
  (`legacy`, `qwen3`, or `qwen35`), the Python, PyTorch, Transformers, and vLLM
  versions, and the model family.
- **Reproduction:** everything in the [reproducibility checklist](docs/REPRODUCIBILITY.md) —
  base checkpoint revision, calibration data and sample count, sequence length,
  seed, dtype, drop ratio, the completed recipe, the evaluation protocol, and
  the metrics you obtained against the numbers you expected.
- **Model:** the Hugging Face model ID and how its MoE block differs from the
  closest supported family.

Security vulnerabilities do not belong in public issues. Follow
[SECURITY.md](SECURITY.md) and use a private GitHub security advisory.

## Protect the reported protocol

Several behaviors are deliberate, and a change that alters them changes what
the code measures. Do not modify them as a side effect of another change:

- **Importance criterion.** The released pruning code ranks neurons by
  `mean(abs(gradient))`, not squared-gradient Fisher information, because that
  is the criterion used for the released experiments. A different criterion is
  welcome as a new, explicitly named option; it must not silently replace the
  default.
- **Evaluation protocol.** Qwen1.5-MoE, Qwen3-MoE, and Qwen3.5-MoE use strict
  zero-shot evaluation. The multi-shot evaluator is only for OLMoE. Unknown
  protocol names fail closed and must continue to.
- **SFT paths.** Only base-model SFT and SFT of an already-pruned checkpoint
  are supported. The pruned recipes keep `router_prune_enable: false`.
- **Environment isolation.** The `legacy`, `qwen3`, and `qwen35` profiles pin
  incompatible Transformers and vLLM versions on purpose. Do not upgrade one
  profile to run another family, and do not broaden a pin without verifying
  every family that uses the profile.
- **Structural compaction.** Structural pruning requires one uniform surviving
  width per expert. The scripts reject layer and global masks instead of
  writing a checkpoint that cannot be loaded.

If you believe one of these behaviors is wrong, open an issue that explains
the evidence before sending code. A pull request that changes a reported
number must say so plainly and show the before and after results.

## Set up the repository

Python 3.11 and a CUDA-capable Linux host are recommended. Choose the profile
that matches the model family you are working on:

```bash
git clone https://github.com/HectorHHZ/Less-is-MoE.git
cd Less-is-MoE

./setup.sh legacy   # Qwen1.5-MoE and OLMoE
# ./setup.sh qwen3  # Qwen3-MoE
# ./setup.sh qwen35 # Qwen3.5-MoE

source .venv-legacy/bin/activate
pip install 'pytest>=8.0'
```

`setup.sh` already installs this package in editable mode. Install `pytest`
on its own rather than re-running `pip install -e '.[test]'`, which can
re-resolve the profile's pinned dependencies. See the
[environment matrix](docs/ENVIRONMENTS.md) for exact versions.

## Run the checks

The unit tests cover model-family detection, patch configuration validation,
and evaluation protocol selection. They use only the standard library and
`pytest`, so they run in seconds without a GPU:

```bash
pytest tests/
```

Pruning, SFT, quantization, and vLLM evaluation need GPUs, model weights, and
calibration data, so they are not part of the automated tests. When a change
touches one of those paths, run the affected launcher on a small model and
report the command and result in the pull request.

## Add support for a model family

A new MoE family is the most common substantial contribution. It usually needs
all of the following in the same pull request:

1. **Pruning.** Add `neuron_drop_<family>.py` and
   `neuron_structure_drop_<family>.py` under `src/less_is_moe/pruning/`,
   following the closest existing family. State whether shared experts are
   preserved.
2. **Runtime patches.** Structurally pruned checkpoints have layer-wise expert
   dimensions that stock implementations do not understand. Add the Hugging
   Face and vLLM patches under `src/less_is_moe/model_patches/` and register
   the family in `registry.py`. Detection must read the checkpoint
   configuration, not the directory name. See [model patches](docs/MODEL_PATCHES.md).
3. **Launchers.** Add thin wrappers under `scripts/prune/` that forward their
   arguments to the package module, matching the existing launchers.
4. **Environment.** Reuse a profile if its pins already support the family.
   Otherwise add a new profile instead of changing an existing one.
5. **Evaluation.** Say which protocol the family uses and why.
6. **Tests.** Add alias and configuration-detection cases to
   `tests/test_registry_and_protocols.py`, including a case that must be
   rejected.
7. **Documentation.** Update the support table in `README.md`, the
   [script reference](docs/SCRIPT_REFERENCE.md), and, if needed, the
   [environment matrix](docs/ENVIRONMENTS.md).

A family is supported only when a structurally pruned checkpoint loads and
evaluates in both Hugging Face and vLLM. Please include that evidence.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/) and match the style of the
  surrounding module.
- Add type annotations to new or changed functions.
- Fail early with a clear error instead of producing a checkpoint or result
  that is silently wrong. Existing examples are the rejection of uneven
  Qwen3.5 expert counts and of unknown evaluation protocols.
- Keep launchers thin. Logic belongs in the `src/less_is_moe/` package, where
  it can be imported and tested.
- Write comments that explain why a choice is necessary, especially where the
  code deviates from the stock model implementation.
- Keep the release free of hard-coded local paths, cluster names, and
  unpublished checkpoint locations. Recipes use `REPLACE_WITH_*` placeholders.

## AI-assisted contributions

AI-assisted work is welcome when it is directed, understood, and verified by
the human contributor. Do not submit autonomous or bulk-generated issues, pull
requests, or comments.

When an AI tool makes a non-trivial contribution:

- disclose the tool and how it was used in the pull request;
- review and understand every submitted line and factual claim;
- reproduce a bug yourself instead of trusting a generated diagnosis;
- run the relevant checks and report their actual results — never report
  experimental numbers you did not obtain; and
- remove speculative fixes, unrelated cleanup, and generated commentary.

AI assistance does not lower the bar for correctness, tests, or documentation.

## Make a pull request

Before requesting review:

- keep the diff as small as practical and remove unrelated changes;
- explain what changed, why, and how it was verified;
- link the relevant issue when one exists;
- add or update tests for behavior that can be tested without a GPU;
- update the documentation affected by the change;
- run `pytest tests/` and report the result, plus any GPU commands you ran; and
- state whether the change can alter a reported number.

Do not commit model weights, datasets, checkpoints, logs, experiment outputs,
or `.env` files; `.gitignore` excludes the common paths. Use a draft pull
request when the design or implementation is not ready for review.

## Review

This is a research repository maintained by the paper's authors, and review
time is limited. Maintainers may ask for changes to correctness, protocol
fidelity, tests, documentation, or scope, and may decline changes that add
long-term maintenance cost without a clear benefit to reproducing or extending
the work. A concise reminder on a pull request that has gone quiet is welcome.

## Citation

If your contribution builds on this work, please cite the paper. See the
[Citation section of the README](README.md#citation).

## License

By contributing, you agree that your contribution may be distributed under the
repository's [Apache License 2.0](LICENSE).
