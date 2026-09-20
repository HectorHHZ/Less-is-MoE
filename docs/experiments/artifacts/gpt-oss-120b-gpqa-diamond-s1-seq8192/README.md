# GPT-OSS-120B GPQA-Diamond artifacts

This directory contains the public, question-safe artifacts behind the
GPT-OSS-120B base and IntDim-E/L/G results in the parent experiment record.

## Dataset

The run used all 198 questions from `Idavidrein/gpqa`, config
`gpqa_diamond`, revision
`633f5ee89ab8ad4522a9f850766b73f62147ffdd`. The source dataset asks users
not to reveal its examples online, so neither the question text nor the raw
model generations are committed to this public repository.

`dataset/split-manifest.json` records the exact selected UUIDs, source
revision, shuffle seed, and frozen data hashes. After obtaining access to the
official dataset, regenerate the evaluation input as described in
[`docs/supergpqa.md`](../../../supergpqa.md). The resulting file must match:

```text
evaluation.jsonl SHA-256: 9312a13f2c88c80082892fa4c60d761b5e6a04b943d36976f8904ac72e93f542
split-manifest.json SHA-256: 853ce946fbd9cd4309ce514ed69cbc571437df3d6bf30430f61d847a1350ff07
```

The older MoE-Honing `test.jsonl` contains the same 198 questions in a
different prompt/choice representation. It was not the frozen input used by
this experiment. Its SHA-256 is
`d5b0d6dad6c1993a8cb17fd7aa635fbc5e8b684ae42b5b6e80d15467b399eec3`.

## Results

Each result directory contains:

- `predictions.sanitized.jsonl`: all 1,584 per-sample decisions, with question
  text, options, answer text, raw generation, and final-answer text removed;
- `metrics.json`: complete aggregate, repeat-level, and discipline/field/
  subfield metrics from the frozen `graded-v4` scorer;
- `run.json`: the exact model, runtime, prompt, sampling, and environment
  settings;
- `grading-audit.json`: hashes and the uniform regrading audit.

The sanitized prediction rows retain the question UUID and SHA-256, repeat
index, gold and predicted letters, correctness, token counts, and finish
status. Summing these rows reproduces every headline metric:

| Run | Correct | Total | Accuracy | Unparsed | Completion tokens |
|---|---:|---:|---:|---:|---:|
| Base BF16 | 1,255 | 1,584 | 79.23% | 0 | 17,951,623 |
| IntDim-E 50% | 332 | 1,584 | 20.96% | 85 | 997,172 |
| IntDim-L 50% | 866 | 1,584 | 54.67% | 0 | 16,065,490 |
| IntDim-G 50% | 1,019 | 1,584 | 64.33% | 4 | 17,701,545 |

`artifact-manifest.json` records the SHA-256 of every committed artifact and
the SHA-256 of each private raw prediction file used by the grading pass.
