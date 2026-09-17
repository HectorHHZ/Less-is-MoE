"""Repeat a full-model vLLM run, retaining top-five scores at every step.

Uses exactly the generation settings of ragged_model_matrix. Compare scores
at the first divergence only: both engines still have the same token history.
"""
import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    from vllm import LLM, SamplingParams
    import vllm
    prep = json.loads((args.results / "prepare.json").read_text())
    original = json.loads((args.results / f"vllm-{args.kind}.json").read_text())
    engine = LLM(model=str(args.checkpoint), skip_tokenizer_init=True, dtype="bfloat16",
                 enforce_eager=True, tensor_parallel_size=1, max_model_len=256,
                 max_num_seqs=2, max_num_batched_tokens=256, gpu_memory_utilization=0.6,
                 seed=7, trust_remote_code=False, attention_config={"backend": "TRITON_ATTN"},
                 kernel_config={"moe_backend": "triton"})
    results = engine.generate([{"prompt_token_ids": ids} for ids in prep["prompt_token_ids"]],
                              SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, logprobs=5), use_tqdm=False)
    outputs = []
    for result, previous in zip(results, original["outputs"]):
        out = result.outputs[0]
        assert len(out.token_ids) == 128
        outputs.append(dict(token_ids=list(out.token_ids), matches_previous_run=list(out.token_ids) == previous["token_ids"],
                            logprobs=[{str(k): v.logprob for k, v in step.items()} for step in out.logprobs]))
    report = dict(checkpoint=str(args.checkpoint), kind=args.kind, vllm=vllm.__version__, outputs=outputs,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.results / f"diagnostic-vllm-{args.kind}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(kind=args.kind, repeated_outputs_match=[o["matches_previous_run"] for o in outputs])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--kind", choices=["masked", "compact"], required=True)
    run(parser.parse_args())
