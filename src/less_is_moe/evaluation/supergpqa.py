"""SuperGPQA preparation and resumable GPU evaluation.

The question template follows SuperGPQA/SuperGPQA's zero-shot protocol.
Keep the generated split manifest with every base/pruned comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import unicodedata

DATASET_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf"
PROMPT = (
    "Answer the following multiple choice question. There is only one correct answer. "
    "The last line of your response should be in the format 'Answer: $LETTER' "
    "(without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n{}\n"
)
PROTOCOL_VERSION = 1


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def question_key(row):
    # Group repeated questions even when their answer options have been reordered.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", row["question"])).strip().casefold()


def validate_rows(rows):
    ids = set()
    for row in rows:
        if not row.get("uuid") or row["uuid"] in ids:
            raise ValueError("Every SuperGPQA row must have a unique uuid")
        ids.add(row["uuid"])
        options = row.get("options", [])
        letter = row.get("answer_letter", "")
        if not row.get("question") or not 2 <= len(options) <= 10:
            raise ValueError(f"Invalid question/options: {row['uuid']}")
        if len(letter) != 1 or letter not in "ABCDEFGHIJ"[:len(options)]:
            raise ValueError(f"Invalid answer_letter: {row['uuid']}")
        if options[ord(letter) - 65] != row.get("answer"):
            raise ValueError(f"Answer text and letter disagree: {row['uuid']}")
        if not all(row.get(key) for key in ("discipline", "field", "subfield")):
            raise ValueError(f"Missing hierarchy: {row['uuid']}")


def build_prompt(row):
    choices = "\n".join(f"{chr(65 + i)}) {option}" for i, option in enumerate(row["options"]))
    return PROMPT.format(row["question"] + "\n" + choices)


def fixed_chat(tokenizer, messages, *, reasoning_effort, prompt_date, generation=True):
    # GPT-OSS's template otherwise inserts the wall-clock date, changing prompts
    # between the base run and pruning experiments on subsequent days.
    template = tokenizer.get_chat_template()
    template = template.replace('strftime_now("%Y-%m-%d")', json.dumps(prompt_date))
    template = template.replace("strftime_now('%Y-%m-%d')", json.dumps(prompt_date))
    return tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=generation, reasoning_effort=reasoning_effort)


def prepare_dataset(rows, output_dir, *, calibration_size=128, seed=42,
                    dataset_revision=DATASET_REVISION, tokenizer=None,
                    reasoning_effort="high", prompt_date="2026-09-17"):
    validate_rows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[question_key(row)].append(row)
    strata = defaultdict(list)
    for key, group in groups.items():
        representative = min(group, key=lambda row: row["uuid"])
        stratum = representative["discipline"]
        strata[stratum].append((digest([seed, key]), key, representative))
    if not 0 <= calibration_size < len(groups):
        raise ValueError("Calibration size must be nonnegative and leave evaluation questions")
    for values in strata.values():
        values.sort(reverse=True)
    ordered = sorted(strata, key=lambda key: digest([seed, key]))
    calibration, selected = [], set()
    while len(calibration) < calibration_size:
        for key in ordered:
            if strata[key] and len(calibration) < calibration_size:
                _, question, representative = strata[key].pop()
                selected.add(question)
                calibration.append(representative)
    calibration_ids = {row["uuid"] for row in calibration}
    evaluation = sorted((r for r in rows if question_key(r) not in selected),
                        key=lambda r: digest([seed, r["uuid"]]))
    excluded = [r["uuid"] for r in rows if question_key(r) in selected and r["uuid"] not in calibration_ids]
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("calibration.jsonl", "evaluation.jsonl", "split-manifest.json"):
        if (folder / filename).exists():
            raise FileExistsError(f"Refusing to replace an existing split: {folder / filename}")
    calibration_output = []
    for row in calibration:
        prompt = build_prompt(row)
        answer = "Answer: " + row["answer_letter"]
        text = fixed_chat(tokenizer, [{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": answer}],
                          reasoning_effort=reasoning_effort, prompt_date=prompt_date,
                          generation=False) if tokenizer else prompt + "\n" + answer
        calibration_output.append({**row, "text": text})
    for filename, values in (("calibration.jsonl", calibration_output), ("evaluation.jsonl", evaluation)):
        (folder / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "dataset": "m-a-p/SuperGPQA",
        "dataset_revision": dataset_revision, "source_rows_sha256": digest(rows),
        "source_count": len(rows), "seed": seed, "calibration_count": len(calibration),
        "evaluation_count": len(evaluation), "calibration_uuids": [r["uuid"] for r in calibration],
        "evaluation_uuids": [r["uuid"] for r in evaluation], "excluded_duplicate_uuids": excluded,
        "selection": "deterministic discipline-balanced round robin; normalized questions stay in one partition",
        "calibration_subfields": len({tuple(r[k] for k in ("discipline", "field", "subfield")) for r in calibration}),
        "calibration_chat_template": tokenizer is not None,
        "reasoning_effort": reasoning_effort, "prompt_date": prompt_date,
        "calibration_sha256": file_digest(folder / "calibration.jsonl"),
        "evaluation_sha256": file_digest(folder / "evaluation.jsonl"),
    }
    write_json(folder / "split-manifest.json", manifest)
    return manifest


def final_answer_text(raw, *, harmony=False):
    if harmony:
        marker = "<|channel|>final<|message|>"
        if marker not in raw:
            return ""  # Never score an answer mentioned only inside reasoning.
        raw = raw.rsplit(marker, 1)[1]
    elif "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    elif "<think>" in raw:
        return ""
    return re.split(r"<\|(?:fim_suffix|im_end|end|start|return)\|>", raw, maxsplit=1)[0].strip()


def extract_answer(text, options):
    """Parse a final answer without searching arbitrary single letters in prose."""
    if not text:
        return None
    allowed = "ABCDEFGHIJ"[:len(options)]
    # Match formatting wrappers without accepting a prefix of another word.
    wrapper = r"(?:[\s*$({\[]|\\(?:boxed|mathbf|mathrm|text)\s*\{)*"
    patterns = [
        rf"(?:\banswer\s*\**\s*:|\b(?:final\s+|correct\s+|best\s+)?(?:answer|option)\s+is\s*:?)\s*{wrapper}([{allowed}])(?![A-Za-z])",
        rf"^\s*{wrapper}([{allowed}])(?:[\s*$)}}\].:]|$)",
    ]
    for candidate in (text.rstrip().split("\n")[-1], text):
        for pattern in patterns:
            matches = list(re.finditer(pattern, candidate, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).upper()
    tail = text.rstrip().split("\n")[-1].strip()
    tail = re.sub(r"^(?:the (?:correct |final )?answer is|answer\s*:)\s*", "", tail, flags=re.IGNORECASE)
    exact = [chr(65 + i) for i, option in enumerate(options) if tail.rstrip(".") == option.rstrip(".")]
    return exact[0] if len(exact) == 1 else None


def metrics(records):
    result = {"total": len(records), "correct": sum(r["correct"] for r in records),
              "unparsed": sum(r["prediction"] is None for r in records),
              "truncated": sum(r["finish_reason"] == "length" for r in records),
              "completion_tokens": sum(r["completion_tokens"] for r in records)}
    result["accuracy"] = result["correct"] / len(records) if records else 0.0
    for depth, name in enumerate(("discipline", "field", "subfield"), 1):
        groups = defaultdict(list)
        for row in records:
            key = "/".join(row[k] for k in ("discipline", "field", "subfield")[:depth])
            groups[key].append(row["correct"])
        scores = {key: {"total": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
                  for key, values in sorted(groups.items())}
        result[f"{name}_macro_accuracy"] = sum(v["accuracy"] for v in scores.values()) / len(scores) if scores else 0.0
        result[f"by_{name}"] = scores
    return result


def completed_requests(llm, prompts, sampling_params):
    """Keep the vLLM scheduler fed and yield complete requests as they finish."""
    if len(prompts) != len(sampling_params):
        raise ValueError("Each prompt requires its own sampling parameters")
    pending = {}
    for i, (prompt, params) in enumerate(zip(prompts, sampling_params)):
        # vLLM 0.29 returns randomized internal IDs from enqueue/add_request,
        # but RequestOutput carries the caller's external ID. Supply those IDs
        # explicitly rather than confusing the two namespaces.
        request_id = f"supergpqa-{i}"
        llm.llm_engine.add_request(request_id, prompt, params)
        pending[request_id] = i
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if output.finished:
                if output.request_id not in pending:
                    raise RuntimeError("Unexpected or duplicate completed request")
                yield pending.pop(output.request_id), output
    if pending:
        raise RuntimeError(f"The engine dropped {len(pending)} requests")


def run_evaluation(args):
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    if args.n_samples_per_problem != 1 or args.stop_sequences:
        raise ValueError("SuperGPQA currently requires one completion per question and native EOS stopping")
    if args.runtime_patch not in ("stock", "ragged"):
        raise ValueError("Use --runtime_patch stock (base/IntDim-E) or ragged (IntDim-L/G)")
    if not args.use_chat_template:
        raise ValueError("SuperGPQA requires --use_chat_template for consistent reasoning/chat formatting")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0 or args.max_tokens <= 0:
        raise ValueError("Batch size and generation token budget must be positive")
    data_path = Path(args.data_path)
    rows = read_rows(data_path)
    validate_rows(rows)
    split_path = data_path.parent / "split-manifest.json"
    if not split_path.exists():
        raise ValueError("Prepare a disjoint calibration/evaluation split first; split-manifest.json is required")
    split = json.loads(split_path.read_text())
    if file_digest(data_path) != split["evaluation_sha256"] or [r["uuid"] for r in rows] != split["evaluation_uuids"]:
        raise ValueError("Evaluation data does not match the frozen split manifest")
    if (args.reasoning_effort, args.prompt_date) != (split["reasoning_effort"], split["prompt_date"]):
        raise ValueError("Reasoning effort/date must match the frozen calibration protocol")
    if args.limit:
        rows = rows[:args.limit]
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    family = config.get_text_config().model_type
    if args.runtime_patch == "ragged":
        os.environ["VLLM_PLUGINS"] = "less_is_moe_ragged"
    else:
        os.environ["VLLM_PLUGINS"] = ""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts = [fixed_chat(tokenizer, [{"role": "user", "content": build_prompt(r)}],
                          reasoning_effort=args.reasoning_effort, prompt_date=args.prompt_date) for r in rows]
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    too_long = [rows[i]["uuid"] for i, tokens in enumerate(encoded) if len(tokens) + args.max_tokens > args.max_model_len]
    if too_long:
        raise ValueError(f"{len(too_long)} prompts exceed the context budget; increase --max_model_len. No rows were dropped.")
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in (
        "model_name_or_path", "dtype", "quantization", "tensor_parallel_size", "runtime_patch",
        "seed", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
        "max_tokens", "max_model_len", "batch_size", "gpu_memory_utilization", "reasoning_effort",
        "prompt_date", "limit", "enforce_eager", "max_num_batched_tokens")}
    settings.update(protocol_version=PROTOCOL_VERSION, dataset="supergpqa", tools=False,
                    scheduling="continuous-final-only",
                    split_sha256=file_digest(split_path), data_sha256=file_digest(data_path),
                    prompt_tokens_sha256=digest(encoded), question_count=len(rows),
                    max_prompt_tokens=max(map(len, encoded)), model_config_sha256=digest(config.to_dict()),
                    versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "tokenizers")})
    run_path, output_path = folder / "run.json", folder / "predictions.jsonl"
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text()) != settings:
            raise ValueError("Output directory contains a different run; use a fresh directory or --resume with identical settings")
    elif output_path.exists():
        raise ValueError("Predictions exist without run metadata")
    else:
        write_json(run_path, settings)
    records = read_rows(output_path) if output_path.exists() else []
    done = [r["uuid"] for r in records]
    if len(done) != len(set(done)) or not set(done).issubset({r["uuid"] for r in rows}):
        raise ValueError("Saved predictions contain duplicate or unexpected question UUIDs")
    start_time = time.monotonic()
    initial_count = len(records)
    if len(records) < len(rows):
        llm_kwargs = dict(model=args.model_name_or_path, dtype=args.dtype,
                          tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
                          max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                          max_num_batched_tokens=args.max_num_batched_tokens,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          enforce_eager=args.enforce_eager, enable_prefix_caching=False,
                          attention_config={"backend": "TRITON_ATTN"},
                          kernel_config={"moe_backend": "triton"})
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        if family == "qwen3_5_moe_text":
            llm_kwargs["hf_overrides"] = {"architectures": ["Qwen3_5MoeForCausalLM"]}
        llm = LLM(**llm_kwargs)
        completed_ids = set(done)
        pending_indices = [i for i, row in enumerate(rows) if row["uuid"] not in completed_ids]
        params = [SamplingParams(
            n=1, temperature=args.temperature, top_p=args.top_p if args.temperature > 0 else 1.0,
            top_k=args.top_k if args.temperature > 0 else -1, min_p=args.min_p if args.temperature > 0 else 0.0,
            repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens, skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY,
            seed=int(digest([args.seed, rows[i]["uuid"]])[:8], 16)) for i in pending_indices]
        pending_prompts = [{"prompt_token_ids": encoded[i]} for i in pending_indices]
        last_report = time.monotonic()
        with output_path.open("a", encoding="utf-8") as output_file:
            for pending_index, request_output in completed_requests(llm, pending_prompts, params):
                data_index = pending_indices[pending_index]
                row = rows[data_index]
                if len(request_output.outputs) != 1:
                    raise RuntimeError("Expected exactly one completion per question")
                if request_output.prompt_token_ids != encoded[data_index]:
                    raise RuntimeError("Completed request does not match the question's input tokens")
                completion = request_output.outputs[0]
                final = final_answer_text(completion.text, harmony=family == "gpt_oss")
                prediction = extract_answer(final, row["options"])
                record = {**row, "data_index": data_index, "raw_output": completion.text,
                          "final_answer": final, "prediction": prediction,
                          "correct": int(prediction == row["answer_letter"]),
                          "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                          "completion_tokens": len(completion.token_ids),
                          "prompt_tokens": len(request_output.prompt_token_ids)}
                records.append(record)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                if len(records) % args.batch_size == 0 or time.monotonic() - last_report >= 30 or len(records) == len(rows):
                    progress = metrics(records)
                    progress.update(complete=len(records) == len(rows), expected_total=len(rows),
                                    elapsed_this_run_seconds=time.monotonic() - start_time,
                                    resumed_from=initial_count)
                    write_json(folder / "metrics.json", progress)
                    print(json.dumps({k: v for k, v in progress.items() if not k.startswith("by_")}), flush=True)
                    last_report = time.monotonic()
    result = metrics(records)
    result.update(complete=True, expected_total=len(rows), run_settings=settings)
    write_json(folder / "metrics.json", result)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("by_") and k != "run_settings"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_size", type=int, default=128,
                        help="Use 0 for a complete-dataset base benchmark; use 64/128 for later pruning comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--reasoning_effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--prompt_date", default="2026-09-17")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    result = prepare_dataset(read_rows(args.input), args.output_dir, calibration_size=args.calibration_size,
                             seed=args.seed, dataset_revision=args.dataset_revision, tokenizer=tokenizer,
                             reasoning_effort=args.reasoning_effort, prompt_date=args.prompt_date)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("uuids")}, indent=2))


if __name__ == "__main__":
    main()
