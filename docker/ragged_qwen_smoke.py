"""Reproducible GPU-only L/G export, HF reload and vLLM generation.

Run prepare and generate in separate processes so HF does not retain GPU memory
while vLLM starts. --model tiny is explicitly a random architecture fixture.
Any other --model is a complete local pretrained Qwen3-MoE checkpoint.
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

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU/offload is not supported")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    started = time.time()
    if args.model == "tiny":
        from docker.intdim_vllm_smoke import make_config
        config = make_config("qwen3_moe")
        config._experts_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).cuda().eval()
        tokenizer = None
        calibration = [torch.tensor([[1, 3, 5, 7, 9, 11]], device="cuda"),
                       torch.tensor([[2, 4, 6, 8, 10, 12]], device="cuda")]
        prompts = [[1, 13, 15, 17], [2, 18, 20, 22, 24]]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
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
    P.zero_dropped_neurons(handles, plan)
    masked = args.output / "masked"
    model.save_pretrained(masked)
    if tokenizer is not None:
        tokenizer.save_pretrained(masked)
    inputs = [torch.tensor([prompt], device="cuda") for prompt in prompts]
    with torch.inference_mode():
        expected = [model(tokens, use_cache=False).logits[:, -1].float().cpu() for tokens in inputs]
    summary = compact_model(model, handles, plan)
    del handles, scores, calibration
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        actual = [model(tokens, use_cache=False).logits[:, -1].float().cpu() for tokens in inputs]
    error = max((a - b).abs().max().item() for a, b in zip(actual, expected))
    # BF16 layer roundoff can accumulate over all 48 layers; report the actual
    # errors and argmax separately. This is a numerical check, not a quality eval.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0.04, atol=0.25 if args.model != "tiny" else 0.015)
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
                   original_parameters=original_parameters, num_layers=model.config.num_hidden_layers,
                   num_experts=model.config.num_experts, hidden_size=model.config.hidden_size,
                   width_min=min(all_widths), width_max=max(all_widths), distinct_widths=len(set(all_widths)),
                   hf_logits_max_abs=error, hf_argmax_equal=all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(actual, expected)),
                   reload_weights_exact=True, reload_logits_exact=True,
                   calibration_samples=len(CALIBRATION) if args.model != "tiny" else 2,
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
    checkpoint = args.output / args.checkpoint
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
    parser.add_argument("--scope", choices=["layer", "global"], default="layer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--checkpoint", choices=["compact", "masked"], default="compact")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else generate)(args)
