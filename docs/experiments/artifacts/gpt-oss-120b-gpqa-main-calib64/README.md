# GPT-OSS-120B held-out GPQA-main artifacts

This directory contains the public, question-safe artifacts behind the
GPT-OSS-120B base and IntDim-E/L/G 50% results in the parent experiment record.

## Dataset

The access-controlled Hugging Face dataset
[`jayzou3773/less-is-moe-gpqa-main-calibration-64`](https://huggingface.co/datasets/jayzou3773/less-is-moe-gpqa-main-calibration-64),
revision `7134dfef5af4605eae0706c30efa9226f49aed96`, contains the exact 64-row
calibration set and 384-row held-out evaluation set. It comes from
`Idavidrein/gpqa`, config `gpqa_main`, revision
`633f5ee89ab8ad4522a9f850766b73f62147ffdd`.

GPQA asks users not to reveal examples online. This public repository therefore
commits only `dataset/selection-manifest.json` and `dataset/split-manifest.json`;
these contain provenance, selection rules, UUIDs, and hashes without question
text.

```text
calibration.jsonl SHA-256: 99e61c8c1bd4321162f1f3895d62e889b9a8b11be9627064207effce70e324c7
test.jsonl SHA-256:        c7475f3382e65a23a5f3c1da97947e1ac9c110686efcc23df16c44c9b65c44cb
selection manifest:       fd4bb8cd9be999a6ad780c14570af17fdd434b63da2199f12549f73aaf8b8f4d
split manifest:           ff0bf0307c875869ea9b90feab29413a9b6a7777ed844c5add4626b93d7b941a
```

## Results

Each result directory contains:

- `predictions.sanitized.jsonl`: all 3,072 per-sample decisions, with question
  text, options, answer text, raw generation, and final-answer text removed;
- `metrics.json`: aggregate, repeat-level, and hierarchy metrics from the frozen
  `graded-v4` scorer;
- `run.json`: model, runtime, prompt, sampling, hashes, and environment settings;
- `grading-audit.json`: raw-prediction hashes and the uniform regrading audit.

| Run | Correct | Total | Accuracy | Unparsed | Completion tokens |
|---|---:|---:|---:|---:|---:|
| Base BF16 | 2,365 | 3,072 | 76.99% | 3 | 28,039,222 |
| IntDim-E 50% | 779 | 3,072 | 25.36% | 188 | 1,586,117 |
| IntDim-L 50% | 1,935 | 3,072 | 62.99% | 12 | 15,567,420 |
| IntDim-G 50% | 1,796 | 3,072 | 58.46% | 44 | 40,743,679 |

Summing `correct`, `completion_tokens`, and null `prediction` fields in the
sanitized files reproduces these metrics. `artifact-manifest.json` records the
SHA-256 of every committed result and both private raw/graded prediction files.
