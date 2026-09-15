"""Calibration data loaders shared by the pruning entry points.

The functions are copied from ``pruning/expert_drop_qwen15_moe.py`` so that the
generic ``intdim.prune`` command tokenizes exactly the same samples as the
per-family scripts. The only change is the ``unwrap_message_content`` switch on
``load_calib_data_hf``: the Qwen3 and Qwen3.5 scripts unwrap chat-message dicts
to their ``content``, while the Qwen1.5-MoE and OLMoE scripts do not.
"""

import json
import random


def load_calib_data(tokenizer, calib_data_path, n_samples, seq_len):
    """Load JSON/JSONL calibration file and tokenize into batches.

    Supports both JSON array (``[{"text": ...}, ...]``) and JSONL
    (one JSON object per line). Short samples are kept as-is so that
    small datasets like HumanEval work without padding or filtering.
    """
    with open(calib_data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        raw = json.loads(content)
        if isinstance(raw, dict):
            raw = [raw]
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in content.splitlines() if line.strip()]

    def _row_text(item):
        if "text" in item and item["text"]:
            return item["text"]
        if "prompt" in item and "completion" in item:
            return f"{item['prompt']}\n{item['completion']}"
        if "prompt" in item:
            return item["prompt"]
        if "instruction" in item and "output" in item:
            return f"{item['instruction']}\n{item['output']}"
        return None

    texts = [t for t in (_row_text(item) for item in raw) if t][:n_samples * 4]

    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches


def load_calib_data_hf(tokenizer, dataset_name, dataset_config, dataset_split,
                       n_samples, seq_len, text_column="prompt",
                       shuffle_seed=None, unwrap_message_content=True):
    """Load calibration data from a HuggingFace dataset.

    Args:
        text_column: Which column(s) to use as calibration text.
                     Comma-separated to concatenate multiple columns
                     (e.g. "prompt,completion").
        shuffle_seed: If set, shuffle the dataset with this seed before sampling.
        unwrap_message_content: If True, a dict field with a ``content`` key
            (a chat message) contributes only its content. The Qwen3 and
            Qwen3.5 scripts did this; the Qwen1.5-MoE and OLMoE scripts did not.
    """
    from datasets import load_dataset

    print(f"Loading HF dataset: {dataset_name} (config={dataset_config}, split={dataset_split})")

    # Try loading; some datasets have schema mismatch between splits
    # (e.g. train has 4 columns but dataset card declares only 2).
    # Fallback: load the raw JSON files directly, bypassing declared features.
    try:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    except Exception as first_err:
        print(f"  [WARN] Standard load failed, trying raw JSON fallback...")
        # Schema mismatch between splits — load the raw file directly.
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
        candidates = [f for f in repo_files
                      if dataset_split in f
                      and any(f.endswith(ext) for ext in (".json", ".jsonl"))]
        if candidates:
            data_urls = [f"hf://datasets/{dataset_name}/{f}" for f in candidates]
            ds = load_dataset("json", data_files=data_urls, split="train")
        else:
            raise first_err

    if shuffle_seed is not None:
        print(f"  Shuffling dataset with seed={shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    columns = ds.column_names
    text_cols = [c.strip() for c in text_column.split(",")]
    print(f"  Dataset columns: {columns}, num rows: {len(ds)}")
    print(f"  Using columns {text_cols} as calibration text")

    def _to_str(val):
        """Robustly convert any field value to a string."""
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, (list, tuple)):
            return " ".join(_to_str(x) for x in val)
        if isinstance(val, dict):
            if unwrap_message_content and "content" in val:
                return _to_str(val["content"])
            return " ".join(f"{k}: {_to_str(v)}" for k, v in val.items())
        return str(val)

    # Build text from dataset rows
    texts = []
    for row in ds:
        parts = [_to_str(row[c]) for c in text_cols if c in row]
        text = " ".join(p for p in parts if p)
        if not text:
            # Fallback: any non-empty stringifiable field
            for c in columns:
                v = _to_str(row[c])
                if v:
                    text = v
                    break
        if text:
            texts.append(text)

    print(f"  Extracted {len(texts)} text samples from dataset")

    # Tokenize
    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if ids.shape[1] < seq_len:
            # For short-sequence datasets (e.g. GSM), include shorter samples
            batches.append(ids)
        else:
            batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches


def _format_ceval(row):
    q = row.get("question", "") or ""
    a = row.get("A", "") or ""
    b = row.get("B", "") or ""
    c = row.get("C", "") or ""
    d = row.get("D", "") or ""
    ans = row.get("answer", "") or ""
    exp = row.get("explanation", "") or ""
    if not q:
        return ""
    text = f"题目：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
    if ans:
        text += f"\n答案：{ans}"
    if exp:
        text += f"\n解析：{exp}"
    return text


def _format_cmmlu(row):
    q = row.get("Question") or row.get("question") or ""
    a = row.get("A", "") or ""
    b = row.get("B", "") or ""
    c = row.get("C", "") or ""
    d = row.get("D", "") or ""
    ans = row.get("Answer") or row.get("answer") or ""
    if not q:
        return ""
    text = f"题目：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
    if ans:
        text += f"\n答案：{ans}"
    return text


def _format_math(row):
    problem = row.get("problem", "") or ""
    solution = row.get("solution", "") or ""
    if not problem:
        return ""
    text = f"Problem: {problem}"
    if solution:
        text += f"\nSolution: {solution}"
    return text


_PRESET_REGISTRY = {
    "ceval": {
        "datasets": ["ceval/ceval-exam"],
        "default_split": "dev",  # dev has ~5 per subject × 52 ≈ 260 rows, with explanation
        "format_fn": _format_ceval,
    },
    "cmmlu": {
        "datasets": ["haonan-li/cmmlu"],
        "default_split": "dev",  # dev has ~5 per subject × 67 ≈ 335 rows
        "format_fn": _format_cmmlu,
    },
    "math": {
        "datasets": ["EleutherAI/hendrycks_math", "lighteval/MATH", "hendrycks/competition_math"],
        "default_split": "train",  # train has 7500 rows
        "format_fn": _format_math,
    },
}


def _iter_preset_rows(dataset_name, split):
    """Yield (config_name, row) for every config of `dataset_name` on `split`.

    Falls back to a configless load if the dataset has no configs.
    """
    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [preset] get_dataset_config_names failed: {e}; trying default load")
        configs = []

    # Some datasets expose a single "default" config and work configless.
    if not configs:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        for row in ds:
            yield None, row
        return

    for cfg in configs:
        try:
            ds = load_dataset(dataset_name, cfg, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  [preset] skip config {cfg}: {e}")
            continue
        for row in ds:
            yield cfg, row


def load_calib_data_preset(tokenizer, preset, n_samples, seq_len,
                           split=None, shuffle_seed=None):
    """Load calibration data from a benchmark preset: ceval / math / cmmlu.

    These benchmarks are typically split across many per-subject configs;
    this loader enumerates every config, formats each row into a single
    calibration text, and tokenizes up to `n_samples` of them.
    """
    preset = preset.lower()
    if preset not in _PRESET_REGISTRY:
        raise ValueError(f"Unknown preset '{preset}'. Supported: {list(_PRESET_REGISTRY)}")

    info = _PRESET_REGISTRY[preset]
    split = split or info["default_split"]
    format_fn = info["format_fn"]

    # Try candidate dataset repos in order.
    texts = []
    last_err = None
    for dataset_name in info["datasets"]:
        print(f"Loading preset '{preset}' from {dataset_name} (split={split}) ...")
        try:
            local_texts = []
            for _cfg, row in _iter_preset_rows(dataset_name, split):
                t = format_fn(row)
                if t:
                    local_texts.append(t)
            if local_texts:
                texts = local_texts
                print(f"  Extracted {len(texts)} text samples from {dataset_name}")
                break
        except Exception as e:
            last_err = e
            print(f"  [preset] {dataset_name} failed: {e}")

    if not texts:
        raise RuntimeError(
            f"Failed to load any data for preset '{preset}'. Last error: {last_err}"
        )

    if shuffle_seed is not None:
        print(f"  Shuffling preset samples with seed={shuffle_seed}")
        random.Random(shuffle_seed).shuffle(texts)

    batches = []
    for text in texts:
        if len(batches) >= n_samples:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if ids.shape[1] == 0:
            continue
        batches.append(ids[:, :seq_len])

    if len(batches) < n_samples:
        print(f"[WARN] Only got {len(batches)} samples (requested {n_samples})")
    return batches
