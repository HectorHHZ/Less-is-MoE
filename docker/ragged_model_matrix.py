"""Run identical full-checkpoint L/G=50% export/reload/vLLM tests.

Example (inside the pinned GPU image):
  python -m docker.ragged_model_matrix --case olmoe \
    --model /models/OLMoE-1B-7B-0924 --output /results/olmoe

Use a separate, idle GPU for each concurrently running process. Stock masked
baselines are temporary; both compact checkpoints, logs and evidence survive.
Qwen3.5 tests the complete text tower and excludes vision and MTP weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

CASES = {
    "qwen15": ("Qwen/Qwen1.5-MoE-A2.7B", "qwen2_moe"),
    "olmoe": ("allenai/OLMoE-1B-7B-0924", "olmoe"),
    "qwen3": ("Qwen/Qwen3-30B-A3B", "qwen3_moe"),
    "qwen35": ("Qwen/Qwen3.5-35B-A3B", "qwen3_5_moe_text"),
}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    scores = args.output / "calibration-scores.pt"
    records = []
    env = dict(os.environ, VLLM_PLUGINS="less_is_moe_ragged", HF_HUB_OFFLINE="1")
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "src/less_is_moe/intdim").glob("*.py")),
               root / "docker/ragged_qwen_smoke.py", Path(__file__).resolve()]
    source_sha256 = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    config = json.loads((args.model / "config.json").read_text())
    from transformers import AutoConfig
    normalized = AutoConfig.from_pretrained(args.model).get_text_config()
    expected_family = CASES[args.case][1]
    if normalized.model_type != expected_family:
        raise ValueError(f"Expected {expected_family}, got {normalized.model_type}")
    manifest = {p.name: p.stat().st_size for p in sorted(args.model.glob("*.safetensors"))}
    if not manifest:
        raise ValueError("No complete pretrained safetensors checkpoint supplied")
    evidence = dict(case=args.case, source_repo=CASES[args.case][0], source_config=config,
                    source_weight_files=manifest, source_sha256=source_sha256,
                    source_config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                    calibration_samples=4, calibration_max_tokens=64, drop_ratio=0.5,
                    generated_tokens_per_prompt=args.max_tokens,
                    scope_results=records, status="running")
    report = args.output / "matrix.json"

    def record():
        report.write_text(json.dumps(evidence, indent=2) + "\n")

    def command(argv, logfile):
        with logfile.open("w") as stream:
            result = subprocess.run([sys.executable, "-m", "docker.ragged_qwen_smoke", *argv],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {logfile}")

    record()
    try:
        for scope in ("layer", "global"):
            out = args.output / scope
            out.mkdir(exist_ok=True)
            print(f"START {args.case} {scope}", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"ragged-{args.case}-{scope}-", dir=args.scratch) as scratch:
                command(["prepare", "--model", str(args.model), "--scope", scope,
                         "--output", str(out), "--scores", str(scores), "--masked-dir", scratch,
                         "--allow-logit-drift"], out / "prepare.log")
                prep = json.loads((out / "prepare.json").read_text())
                if prep["family"] != expected_family:
                    raise AssertionError("Wrong pretrained model family")
                for key in ("num_layers", "num_experts", "hidden_size"):
                    expected = getattr(normalized, "num_hidden_layers" if key == "num_layers" else key)
                    if prep[key] != expected:
                        raise AssertionError(f"Original model dimension changed: {key}")
                width = normalized.intermediate_size if args.case == "olmoe" else normalized.moe_intermediate_size
                full_count = normalized.num_hidden_layers * normalized.num_experts * width
                widths = prep["expert_intermediate_sizes"]
                if sum(sum(ws) for ws in widths.values()) != full_count // 2:
                    raise AssertionError("The checkpoint did not retain exactly 50% of routed intermediate units")
                if scope == "layer" and any(sum(ws) != normalized.num_experts * width // 2 for ws in widths.values()):
                    raise AssertionError("IntDim-L did not preserve the per-layer 50% budget")
                for checkpoint in ("masked", "compact"):
                    command(["generate", "--output", str(out), "--checkpoint", checkpoint,
                             "--max-tokens", str(args.max_tokens), "--gpu-memory-utilization", str(args.gpu_memory_utilization)],
                            out / f"{checkpoint}.log")
                baseline = json.loads((out / "vllm-masked.json").read_text())
                compact = json.loads((out / "vllm-compact.json").read_text())
                comparisons = []
                for reference, actual in zip(baseline["outputs"], compact["outputs"]):
                    a, b = reference["token_ids"], actual["token_ids"]
                    assert len(a) == len(b) == args.max_tokens
                    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                    comparisons.append(dict(matching_tokens=sum(x == y for x, y in zip(a, b)),
                                            total_tokens=len(a), exact_match=a == b,
                                            common_prefix_tokens=len(a) if first is None else first,
                                            first_mismatch_position_1based=None if first is None else first + 1))
                matching = sum(c["matching_tokens"] for c in comparisons)
                compared = sum(c["total_tokens"] for c in comparisons)
                records.append(dict(scope=scope, prepare=prep, stock_vllm=baseline,
                                    ragged_vllm=compact, matching_tokens=matching, compared_tokens=compared,
                                    per_prompt_comparison=comparisons))
                record()
            print(f"PASS {args.case} {scope}: inference completed; {matching}/{compared} matching tokens; "
                  f"BF16 numerical sanity passed={prep['hf_bf16_sanity_passed']}", flush=True)
        evidence["status"] = "passed"
        evidence["hf_bf16_sanity_passed"] = all(r["prepare"]["hf_bf16_sanity_passed"] for r in records)
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        record()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=128)
    run(parser.parse_args())
