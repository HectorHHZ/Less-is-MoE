"""Parameter accounting for the scaling backbones, read from checkpoint headers.

Downloads only each model's ``config.json`` and the JSON header of every
safetensors shard (via HTTP range requests), never the weights. Every number in
``model-configs.md`` is produced by this script.

Usage:
    python docs/scaling/verification/inspect_checkpoints.py [--cache DIR] [--ratio 0.5]

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODELS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-35B-A3B",
    "google/gemma-4-26B-A4B",
    "Qwen/Qwen3.5-9B",
]

# Order matters: the first matching rule wins.
CATEGORY_RULES = [
    ("vision/audio tower", r"^model\.(visual|vision_tower|embed_vision|audio_tower|embed_audio)\."),
    ("MTP module", r"^mtp\."),
    ("embeddings", r"embed_tokens\.weight$"),
    ("lm_head", r"^lm_head\.weight$"),
    ("routed experts", r"\.experts\.(gate_up_proj|down_proj)(_blocks|_bias)?$"),
    ("quantization scales", r"\.experts\.\w+_scales$"),
    ("router", r"\.(mlp\.gate\.weight|mlp\.router\.\w+|router\.[\w.]+|shared_expert_gate\.weight)$"),
    ("shared expert / dense FFN", r"\.(shared_expert|mlp)\.(gate|up|down)_proj\.weight$"),
    ("full / sliding attention", r"\.self_attn\."),
    ("linear attention (GatedDeltaNet)", r"\.linear_attn\."),
    ("norms and scalars", r"(norm(_\d)?\.weight|layer_scalar)$"),
]


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_model(model: str, cache: Path) -> tuple[dict, dict]:
    """Return (text config, tensor header map), using the cache when present."""
    model_dir = cache / model.replace("/", "__")
    config_path, header_path = model_dir / "config.json", model_dir / "headers.json"
    if not (config_path.exists() and header_path.exists()):
        model_dir.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{model}/resolve/main"
        config_path.write_bytes(fetch(f"{base}/config.json"))
        try:
            index = json.loads(fetch(f"{base}/model.safetensors.index.json"))
            shards = sorted(set(index["weight_map"].values()))
        except urllib.error.HTTPError:
            shards = ["model.safetensors"]

        def read_header(shard: str) -> dict:
            header_length = struct.unpack("<Q", fetch(f"{base}/{shard}", (0, 7)))[0]
            return json.loads(fetch(f"{base}/{shard}", (8, 8 + header_length - 1)))

        tensors: dict = {}
        with ThreadPoolExecutor(8) as pool:
            for header in pool.map(read_header, shards):
                header.pop("__metadata__", None)
                tensors.update({name: {"dtype": v["dtype"], "shape": v["shape"]} for name, v in header.items()})
        header_path.write_text(json.dumps(tensors))
    config = json.loads(config_path.read_text())
    return config.get("text_config", config), json.loads(header_path.read_text())


def parameter_count(name: str, info: dict) -> int:
    """Parameters represented by a tensor; MXFP4 blocks store two 4-bit values per byte."""
    shape = info["shape"]
    if name.endswith("_blocks") and info["dtype"] == "U8":
        return math.prod(shape[:-2]) * shape[-2] * shape[-1] * 2
    if name.endswith("_scales"):
        return 0
    return math.prod(shape) if shape else 1


def categorise(tensors: dict) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for name, info in tensors.items():
        category = next((label for label, rule in CATEGORY_RULES if re.search(rule, name)), None)
        if category is None:
            raise ValueError(f"Unclassified tensor: {name}")
        totals[category] += parameter_count(name, info)
    return dict(totals)


def moe_spec(config: dict) -> dict | None:
    experts = config.get("num_experts") or config.get("num_local_experts")
    if not experts:
        return None
    return {
        "experts": experts,
        "top_k": config.get("num_experts_per_tok") or config.get("top_k_experts"),
        "intermediate": config.get("moe_intermediate_size") or config["intermediate_size"],
        "hidden": config["hidden_size"],
    }


def billions(value: float) -> str:
    return f"{value / 1e9:.2f}B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path(".cache/scaling-verification"))
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction of routed-expert intermediate dimensions removed.")
    parser.add_argument("models", nargs="*", default=MODELS)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        config, tensors = load_model(model, args.cache)
        totals = categorise(tensors)
        checkpoint = sum(totals.values())
        language_model = checkpoint - totals.get("vision/audio tower", 0) - totals.get("MTP module", 0)
        spec = moe_spec(config)
        print(f"\n### {model}\n")
        print("| Component | Parameters | Share of language model |")
        print("| --- | ---: | ---: |")
        for label, _ in CATEGORY_RULES:
            if totals.get(label):
                share = "—" if label in ("vision/audio tower", "MTP module", "quantization scales") else f"{100 * totals[label] / language_model:.1f}%"
                print(f"| {label} | {totals[label]:,} | {share} |")
        print(f"| **language model (excludes tower and MTP)** | **{language_model:,}** | 100.0% |")
        print(f"| checkpoint total | {checkpoint:,} | — |")
        if spec is None:
            rows.append((model, checkpoint, language_model, None))
            continue

        routed = totals["routed experts"]
        layers = sum(1 for name in tensors if re.search(r"\.experts\.gate_up_proj(_blocks)?$", name) and not name.startswith("mtp."))
        has_bias = any(name.endswith("experts.gate_up_proj_bias") for name in tensors)
        e, k, i, h = spec["experts"], spec["top_k"], spec["intermediate"], spec["hidden"]
        expected = layers * e * (3 * h * i + ((2 * i + h) if has_bias else 0))
        assert routed == expected, f"{model}: routed experts {routed:,} != formula {expected:,}"

        removed_dims = round(args.ratio * i)
        removed = layers * e * removed_dims * (3 * h + (2 if has_bias else 0))
        active = language_model - routed + routed * k / e
        rows.append((model, checkpoint, language_model, {
            "layers": layers, "experts": e, "top_k": k, "intermediate": i, "hidden": h, "bias": has_bias,
            "routed": routed, "removed_dims": removed_dims, "removed": removed, "active": active,
            "active_after": active - removed * k / e,
        }))

    print("\n### Budget at ratio", args.ratio, "\n")
    print("| Model | Language model | Routed experts | Removed dims / expert | Removed params | Share of routed | p_model (language model) | p_model (checkpoint) | Active before → after |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model, checkpoint, language_model, budget in rows:
        if budget is None:
            print(f"| {model} | {billions(language_model)} | dense — not pruned | — | — | — | — | — | {billions(language_model)} (dense) |")
            continue
        print(
            f"| {model} | {billions(language_model)} | {billions(budget['routed'])} "
            f"| {budget['removed_dims']} of {budget['intermediate']} | {billions(budget['removed'])} "
            f"| {100 * budget['removed'] / budget['routed']:.2f}% | {100 * budget['removed'] / language_model:.1f}% "
            f"| {100 * budget['removed'] / checkpoint:.1f}% | {billions(budget['active'])} → {billions(budget['active_after'])} |"
        )


if __name__ == "__main__":
    main()
