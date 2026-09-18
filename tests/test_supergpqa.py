import json

import pytest

from less_is_moe.evaluation.supergpqa import (
    build_prompt,
    extract_answer,
    final_answer_text,
    metrics,
    normalize_gpqa,
    prepare_dataset,
    repeat_questions,
)


def gpqa_source():
    return [{
        "Question": "Which choice is correct?",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
        "High-level domain": "Physics",
        "Subdomain": "Mechanics",
    }]


def test_gpqa_normalization_is_deterministic_and_preserves_answer():
    first = normalize_gpqa(gpqa_source(), seed=42)
    second = normalize_gpqa(gpqa_source(), seed=42)
    assert first == second
    row = first[0]
    assert row["options"][ord(row["answer_letter"]) - ord("A")] == "correct"
    assert row["benchmark"] == "gpqa_diamond"
    assert "A)" in build_prompt(row, profile="qwen35-mcq")


def test_prepare_gpqa_manifest_is_frozen(tmp_path):
    rows = normalize_gpqa(gpqa_source() * 1, seed=42)
    manifest = prepare_dataset(
        rows, tmp_path, calibration_size=0, dataset_name="Idavidrein/gpqa",
        dataset_revision="revision", seed=42,
    )
    assert manifest["dataset"] == "Idavidrein/gpqa"
    assert manifest["evaluation_count"] == 1
    saved = json.loads((tmp_path / "split-manifest.json").read_text())
    assert saved["evaluation_sha256"] == manifest["evaluation_sha256"]


@pytest.mark.parametrize(
    ("raw", "harmony", "thinking_prefix", "expected"),
    [
        ("analysis A<|channel|>final<|message|>{\"answer\":\"B\"}<|end|>", True, False, '{"answer":"B"}'),
        ("analysis A<|channel|>final <|constrain|>json<|message|>{\"answer\":\"C\"}", True, False, '{"answer":"C"}'),
        ("analysis says A", True, False, ""),
        ("reasoning</think>\n{\"answer\": \"D\"}", False, True, '{"answer": "D"}'),
        ("reasoning only", False, True, ""),
    ],
)
def test_final_answer_text_only_scores_final_channel(raw, harmony, thinking_prefix, expected):
    assert final_answer_text(raw, harmony=harmony, thinking_prefix=thinking_prefix) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "B"}', "B"),
        ("answer: C", "C"),
        ("The correct answer is D.", "D"),
        ("A", "A"),
        ('{"answer": "B"}\n{"answer": "C"}', None),
        ("I considered A and B.", None),
        ('{"answer": "outside"}', None),
    ],
)
def test_qwen_mcq_parser_is_strict_and_unambiguous(text, expected):
    assert extract_answer(text, ["a", "b", "c", "d"], profile="qwen35-mcq") == expected


def test_repeats_have_stable_ids_and_metrics_are_mean_pass_at_one():
    source = normalize_gpqa(gpqa_source(), seed=42)
    repeated = repeat_questions(source, 2)
    assert [row["sample_id"] for row in repeated] == [
        f"{source[0]['uuid']}/repeat-0", f"{source[0]['uuid']}/repeat-1"
    ]
    records = [
        {**row, "correct": correct, "prediction": "A", "finish_reason": "stop", "completion_tokens": 1}
        for row, correct in zip(repeated, (1, 0))
    ]
    result = metrics(records)
    assert result["accuracy"] == 0.5
    assert result["unique_questions_completed"] == 1
    assert result["by_repeat"]["0"]["accuracy"] == 1
    assert result["by_repeat"]["1"]["accuracy"] == 0


def test_invalid_profile_and_repeat_count_are_rejected():
    row = normalize_gpqa(gpqa_source(), seed=42)[0]
    with pytest.raises(ValueError):
        build_prompt(row, profile="unknown")
    with pytest.raises(ValueError):
        repeat_questions([row], 0)
