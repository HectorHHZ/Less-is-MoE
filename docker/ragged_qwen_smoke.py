"""Reproducible GPU-only L/G export, HF reload and vLLM generation.

Run prepare and generate in separate processes so HF does not retain GPU memory
while vLLM starts. --model tiny is explicitly a random architecture fixture.
Any other --model is a complete local pretrained language-model checkpoint.
Qwen3.5 uses every language layer; its vision tower is outside this text test.
"""

import argparse
import gc
import json
import time
from pathlib import Path


CALIBRATION = [
    "Explain how binary search works and implement it in Python. Discuss the loop invariant and its time complexity.",
    "Write a Python function that merges two sorted arrays. Include examples with duplicate elements and empty inputs.",
    "A shop sells notebooks for three dollars each. How much do seven notebooks cost? Explain every step of the calculation.",
    "Describe the difference between a stack and a queue. Give one practical example of each data structure.",
]
PROMPTS = ["def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
           "Explain why the sky appears blue in two sentences."]


def prepare(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from less_is_moe.intdim import discover
    from less_is_moe.intdim import prune as P
    from less_is_moe.intdim.ragged import compact_model, load_checkpoint, save_checkpoint
    from less_is_moe.intdim.ragged_hf import load_source_model

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU/offload is not supported")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    started = time.time()
    if args.model == "tiny":
        from docker.intdim_vllm_smoke import make_config
        config = make_config(args.family).get_text_config()
        config._experts_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
        tokenizer = None
        calibration = [torch.tensor([[1, 3, 5, 7, 9, 11]], device="cuda"),
                       torch.tensor([[2, 4, 6, 8, 10, 12]], device="cuda")]
        prompts = [[1, 13, 15, 17], [2, 18, 20, 22, 24]]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = load_source_model(
            args.model, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa", experts_implementation="eager").eval()
        calibration = [tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.cuda()
                       for text in CALIBRATION]
        prompts = [tokenizer(text, add_special_tokens=True).input_ids for text in PROMPTS]
    assert all(p.is_cuda for p in model.parameters())
    model.config.use_cache = False
    original_parameters = sum(p.numel() for p in model.parameters())
    handles = discover(model)
    if args.scores and args.scores.exists():
        cached = torch.load(args.scores, weights_only=True)
        if cached["model"] != args.model or cached["calibration"] != CALIBRATION:
            raise ValueError("Score cache source does not match")
        scores = cached["scores"]
    else:
        scores = P.collect_neuron_gradient_scores(model, handles, calibration)
        if args.scores:
            torch.save(dict(model=args.model, calibration=CALIBRATION, scores=scores), args.scores)
    plan = P.pick_neurons_to_drop(scores, 0.5, args.scope)
    calibration_lengths = [batch.shape[-1] for batch in calibration]
    P.zero_dropped_neurons(handles, plan)
    masked = args.masked_dir or args.output / "masked"
    model.save_pretrained(masked, max_shard_size="4GB")
    if tokenizer is not None:
        tokenizer.save_pretrained(masked)
    inputs = [torch.tensor([prompt], device="cuda") for prompt in prompts]
    with torch.inference_mode():
        expected = [model(tokens, use_cache=False).logits[:, -1].float().cpu() for tokens in inputs]
    summary = compact_model(model, handles, plan)
    # Isolate structural correctness from BF16 rounding/routing amplification.
    # Check three experts in EVERY layer, using FP32 on the GPU and the same
    # inputs for the original zero-masked and compact matrices.
    fp32_error = 0.0
    fp32_checks = 0
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for h in handles:
                x = torch.randn(2, h.hidden_size, device="cuda", dtype=torch.float32)
                for expert in sorted({0, h.num_experts // 2, h.num_experts - 1}):
                    values = []
                    for container in (h, model.get_submodule(h.name)):
                        gate, up, down = [w.float() for w in container.expert_weights(expert)]
                        values.append(torch.nn.functional.linear(
                            torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
                            * torch.nn.functional.linear(x, up), down))
                    torch.testing.assert_close(values[0], values[1], rtol=1e-4, atol=1e-5)
                    fp32_error = max(fp32_error, (values[0] - values[1]).abs().max().item())
                    fp32_checks += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    del handles, scores, calibration
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        actual = [model(tokens, use_cache=False).logits[:, -1].float().cpu() for tokens in inputs]
    error = max((a - b).abs().max().item() for a, b in zip(actual, expected))
    # Full-model BF16 can amplify rounding differences through later routers.
    # Preserve these errors in the report; don't claim bitwise equivalence.
    logit_metrics = []
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        cosine = torch.nn.functional.cosine_similarity(a, b).item()
        kl = torch.nn.functional.kl_div(a.log_softmax(-1), b.softmax(-1), reduction="batchmean").item()
        logit_metrics.append(dict(max_abs=(a-b).abs().max().item(), rmse=(a-b).square().mean().sqrt().item(),
                                  cosine=cosine, reference_to_compact_kl=kl,
                                  argmax_equal=torch.equal(a.argmax(-1), b.argmax(-1))))
        if args.model == "tiny":
            torch.testing.assert_close(a, b, rtol=0.04, atol=0.015)
        elif (cosine < 0.995 or kl > 0.01) and not args.allow_logit_drift:
            raise AssertionError(f"Full BF16 numerical sanity check failed: {logit_metrics[-1]}")
    bf16_sanity = all(m["cosine"] >= 0.995 and m["reference_to_compact_kl"] <= 0.01 for m in logit_metrics)
    if not bf16_sanity:
        print("BF16 NUMERICAL CHECK FAILED: recording drift and continuing the explicitly requested inference check", flush=True)
    compact = args.output / "compact"
    save_checkpoint(model, compact, tokenizer)
    restored = load_checkpoint(compact, attn_implementation="sdpa")
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise AssertionError(f"Reload weight mismatch: {name}")
    with torch.inference_mode():
        reloaded = [restored(tokens, use_cache=False).logits[:, -1].float().cpu() for tokens in inputs]
    for a, b in zip(actual, reloaded):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    all_widths = [w for ws in summary["expert_intermediate_sizes"].values() for w in ws]
    summary.update(model=args.model, pretrained=args.model != "tiny", scope=args.scope, drop_ratio=0.5,
                   family=model.config.model_type, masked_checkpoint=str(masked),
                   calibration_texts=CALIBRATION if args.model != "tiny" else None,
                   prompt_texts=PROMPTS if args.model != "tiny" else None,
                   original_parameters=original_parameters, num_layers=model.config.num_hidden_layers,
                   num_experts=model.config.num_experts, hidden_size=model.config.hidden_size,
                   width_min=min(all_widths), width_max=max(all_widths), distinct_widths=len(set(all_widths)),
                   hf_logits_max_abs=error, hf_argmax_equal=all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(actual, expected)),
                   hf_logit_metrics=logit_metrics, fp32_expert_checks=fp32_checks, fp32_expert_max_abs=fp32_error,
                   hf_bf16_sanity_passed=bf16_sanity,
                   hf_bf16_sanity_thresholds=dict(min_cosine=0.995, max_reference_to_compact_kl=0.01),
                   reload_weights_exact=True, reload_logits_exact=True,
                   calibration_samples=len(CALIBRATION) if args.model != "tiny" else 2,
                   calibration_token_lengths=calibration_lengths,
                   prompt_token_ids=prompts, torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(), dtype="bfloat16",
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.time() - started)
    (args.output / "prepare.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(dict(masked=expected, compact=actual), args.output / "hf_logits.pt")
    print(json.dumps({k:v for k,v in summary.items() if k != "expert_intermediate_sizes"}), flush=True)


def generate(args):
    from vllm import LLM, SamplingParams
    import vllm
    metadata = json.loads((args.output / "prepare.json").read_text())
    checkpoint = Path(metadata.get("masked_checkpoint", args.output / "masked")) if args.checkpoint == "masked" else args.output / "compact"
    engine = LLM(model=str(checkpoint), skip_tokenizer_init=True, dtype="bfloat16",
                 enforce_eager=True, tensor_parallel_size=1, max_model_len=256,
                 max_num_seqs=2, max_num_batched_tokens=256, gpu_memory_utilization=args.gpu_memory_utilization,
                 seed=7, trust_remote_code=False, attention_config={"backend": "TRITON_ATTN"},
                 kernel_config={"moe_backend": "triton"})
    results = engine.generate([{"prompt_token_ids": ids} for ids in metadata["prompt_token_ids"]],
                              SamplingParams(max_tokens=args.max_tokens, temperature=0, ignore_eos=True, logprobs=5), use_tqdm=False)
    outputs = []
    for result in results:
        output = result.outputs[0]
        assert len(output.token_ids) == args.max_tokens
        outputs.append(dict(token_ids=list(output.token_ids),
                            first_logprobs={str(k): v.logprob for k,v in output.logprobs[0].items()}))
    if metadata["pretrained"]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        for record in outputs:
            record["text"] = tokenizer.decode(record["token_ids"])
    report = dict(checkpoint=str(checkpoint), vllm=vllm.__version__, backend="ragged_triton" if args.checkpoint == "compact" else "stock_triton",
                  dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True, max_model_len=256,
                  max_tokens=args.max_tokens, outputs=outputs)
    (args.output / f"vllm-{args.checkpoint}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "generate"])
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--family", choices=["qwen2_moe", "olmoe", "qwen3_moe", "qwen3_5_moe_35b"], default="qwen3_moe")
    parser.add_argument("--masked-dir", type=Path, help="Optional temporary baseline directory, e.g. on tmpfs")
    parser.add_argument("--allow-logit-drift", action="store_true",
                        help="Record a failed full-model BF16 sanity check but continue export/inference; FP32 and exact reload checks remain mandatory")
    parser.add_argument("--scope", choices=["layer", "global"], default="layer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--checkpoint", choices=["compact", "masked"], default="compact")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else generate)(args)
