"""GPU-only same-context BF16/FP32 diagnosis on saved full-model matrix cases.

Reuse the recorded pruning scores and compact checkpoint; do not recalibrate
or change a plan. FP32 promotes the same BF16 weights and disables TF32.
This is an HF structural/numerical control, not an FP32 vLLM benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def run(args):
    import torch
    from less_is_moe.intdim import discover
    from less_is_moe.intdim import prune as P
    from less_is_moe.intdim.ragged import load_checkpoint, expert_path
    from less_is_moe.intdim.ragged_hf import load_source_model

    assert torch.cuda.is_available(), "A GPU is required"
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.results / args.scope
    prep = json.loads((root / "prepare.json").read_text())
    a = json.loads((root / "vllm-masked.json").read_text())["outputs"]
    b = json.loads((root / "vllm-compact.json").read_text())["outputs"]
    contexts = []
    for i, prompt in enumerate(prep["prompt_token_ids"]):
        contexts.append(dict(prompt=i, label="initial", token_ids=prompt))
        first = next((j for j, (x, y) in enumerate(zip(a[i]["token_ids"], b[i]["token_ids"])) if x != y), None)
        if first is not None:
            contexts.append(dict(prompt=i, label="before_first_vllm_divergence",
                                 first_mismatch_position_1based=first + 1,
                                 stock_next_token=a[i]["token_ids"][first],
                                 compact_next_token=b[i]["token_ids"][first],
                                 token_ids=prompt + a[i]["token_ids"][:first]))

    def evaluate(model):
        assert all(p.is_cuda for p in model.parameters())
        outputs = []
        routes = {}
        hooks = []
        for i, layer in enumerate(model.model.layers):
            def capture(module, values, index=i):
                routes[index] = values[1].detach().sort(-1).values.cpu()
            hooks.append(model.get_submodule(expert_path(model.config, i)).register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                for context in contexts:
                    routes.clear()
                    ids = torch.tensor([context["token_ids"]], device="cuda")
                    logits = model(ids, use_cache=False).logits[:, -1].float().cpu()
                    outputs.append(dict(logits=logits, routes=dict(routes)))
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

    started = time.time()
    print(f"START {args.results.name} {args.scope}: full zero-mask BF16/FP32", flush=True)
    model = load_source_model(args.model, dtype=torch.bfloat16, local_files_only=True,
                              attn_implementation="sdpa", experts_implementation="eager")
    cached = torch.load(args.results / "calibration-scores.pt", weights_only=True)
    assert cached["model"] == str(args.model), "Score cache source differs"
    plan = P.pick_neurons_to_drop(cached["scores"], 0.5, args.scope)
    handles = discover(model)
    P.zero_dropped_neurons(handles, plan)
    del handles, cached
    if args.masked_dir:
        from transformers import AutoTokenizer
        model.save_pretrained(args.masked_dir, max_shard_size="4GB",
                              save_original_format=model.config.model_type != "gpt_oss")
        AutoTokenizer.from_pretrained(args.model, local_files_only=True).save_pretrained(args.masked_dir)
    masked_bf16 = evaluate(model)
    model.float()
    torch.cuda.empty_cache()
    masked_fp32 = evaluate(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"START {args.results.name} {args.scope}: saved compact BF16/FP32", flush=True)
    model = load_checkpoint(root / "compact", dtype=torch.bfloat16,
                             attn_implementation="sdpa", experts_implementation="eager")
    compact_bf16 = evaluate(model)
    model.float()
    torch.cuda.empty_cache()
    compact_fp32 = evaluate(model)

    def compare(reference, actual):
        records = []
        for ref, new in zip(reference, actual):
            # All arithmetic, including diagnostic reductions, stays on GPU.
            x, y = ref["logits"].cuda(), new["logits"].cuda()
            diff = x - y
            p = x.softmax(-1)
            kl = (p * (x.log_softmax(-1) - y.log_softmax(-1))).sum().item()
            mismatches = []
            for layer in ref["routes"]:
                changed = (ref["routes"][layer].cuda() != new["routes"][layer].cuda()).any(-1)
                mismatches.append(dict(layer=layer, changed_tokens=int(changed.sum()), total_tokens=changed.numel()))
            def top(values):
                scores, ids = values[0].topk(5)
                return dict(token_ids=ids.tolist(), logits=scores.tolist(), margin=(scores[0]-scores[1]).item())
            records.append(dict(max_abs=diff.abs().max().item(), rmse=diff.square().mean().sqrt().item(),
                                reference_to_compact_kl=kl,
                                argmax_equal=bool(x.argmax(-1) == y.argmax(-1)),
                                reference_top5=top(x), compact_top5=top(y),
                                changed_route_token_layers=sum(r["changed_tokens"] for r in mismatches),
                                total_route_token_layers=sum(r["total_tokens"] for r in mismatches),
                                layer_routes=mismatches))
        return records

    report = dict(model=str(args.model), scope=args.scope, contexts=contexts,
                  precision_control="Same BF16 weights promoted to FP32; TF32 disabled; full HF GPU models; no KV cache",
                  bf16=compare(masked_bf16, compact_bf16), fp32=compare(masked_fp32, compact_fp32),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  visible_gpus=torch.cuda.device_count(), hf_compact_device_map=getattr(model, "hf_device_map", None),
                  peak_allocated_bytes_per_gpu=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.time()-started)
    (root / "precision.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(dict(contexts=contexts, masked_bf16=masked_bf16, masked_fp32=masked_fp32,
                    compact_bf16=compact_bf16, compact_fp32=compact_fp32), root / "precision-tensors.pt")
    print(json.dumps({k: v for k, v in report.items() if k not in ("bf16", "fp32", "contexts")}), flush=True)
    for dtype in ("bf16", "fp32"):
        print(dtype, [(r["max_abs"], r["changed_route_token_layers"], r["argmax_equal"]) for r in report[dtype]], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scope", choices=["layer", "global"], required=True)
    parser.add_argument("--masked-dir", type=Path, help="Optional retained BF16 baseline for vLLM follow-up")
    run(parser.parse_args())
