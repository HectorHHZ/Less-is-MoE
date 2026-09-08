import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Make the parent Less-is-MoE package available when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer


DEFAULT_PROMPT = [
    # {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {
        "role": "user",
        "content": (
            "Please choose the correct answer to the question: Which piece of safety equipment is used to keep mold spores from entering the respiratory system?\n\nAnswer1: safety goggles Answer2: breathing mask Answer3: rubber gloves Answer4: lead apron\n\nAnswer format: answer1/answer2/answer3/answer4"
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal text generation with an AWQ-quantized model.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="outputs/qwen15-pruned-awq",
        help="HF repo ID or local path to an AWQ-quantized model.",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Use base HF Qwen2-MoE implementation (skip pruned patch).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for loading the model (passed to transformers).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def apply_qwen2_moe_patch() -> None:
    """Replace HF Qwen2-MoE blocks with pruned implementations."""
    from less_is_moe.model_patches.registry import apply_hf_patch
    from less_is_moe.model_patches.hf.qwen2_moe.modeling_qwen2_moe import (
        Qwen2MoeSparseMoeBlock,
    )

    apply_hf_patch("qwen2_moe")
    logger.info("[Patch] Enabled Less-is-MoE Qwen2-MoE patch.")

    # Extra safety: monkey patch forward to avoid CUDA index_add_ issues seen in some environments.
    import torch
    import torch.nn.functional as F

    def _patched_forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        apply_mask = self.use_router_mask and (self.router_logits_mask is not None)
        if apply_mask:
            router_logits = router_logits + self.router_logits_mask.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.gate_num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            scatter_idx = top_x.to(final_hidden_states.device).unsqueeze(-1).expand_as(current_hidden_states)
            final_hidden_states.scatter_add_(0, scatter_idx, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    Qwen2MoeSparseMoeBlock.forward = _patched_forward
    logger.info(f"[Patch] Qwen2MoeSparseMoeBlock.forward monkey-patched (source: {Qwen2MoeSparseMoeBlock.__module__})")


def _make_safe_moe_forward():
    """稳健版 forward，用 scatter_add_ 代替 index_add_ 避免 CUDA 报错。
    兼容 stock transformers 和 pruned 模型的属性差异。"""
    import torch
    import torch.nn.functional as F

    def _safe_forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        if getattr(self, "use_router_mask", False) and getattr(self, "router_logits_mask", None) is not None:
            router_logits = router_logits + self.router_logits_mask.to(router_logits.dtype)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        num_classes = getattr(self, "gate_num_experts", self.num_experts)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_classes).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            scatter_idx = top_x.to(final_hidden_states.device).unsqueeze(-1).expand_as(current_hidden_states)
            final_hidden_states.scatter_add_(0, scatter_idx, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return _safe_forward


def patch_moe_instances(model) -> int:
    """无论类定义来源，按名字给所有 Qwen2MoeSparseMoeBlock 实例打补丁。"""
    from types import MethodType

    safe_forward = _make_safe_moe_forward()
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            module.forward = MethodType(safe_forward, module)
            patched += 1
    return patched


def _collect_eos_token_ids(tokenizer) -> list[int]:
    """Return a deduped list of usable EOS token ids, filtering out missing values."""

    def _add_candidate(store: list[int], token_id):
        if token_id is None:
            return
        if isinstance(token_id, list):
            for tid in token_id:
                _add_candidate(store, tid)
            return
        try:
            # Ensure it is an int and not a tensor.
            tid_int = int(token_id)
        except Exception:
            return
        store.append(tid_int)

    eos_ids: list[int] = []
    _add_candidate(eos_ids, getattr(tokenizer, "eos_token_id", None))

    # Common chat EOS variants across Qwen/Llama style tokenizers.
    for tok in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        _add_candidate(eos_ids, tok_id)

    # Deduplicate while preserving order.
    seen = set()
    unique_eos_ids = []
    for tid in eos_ids:
        if tid not in seen:
            unique_eos_ids.append(tid)
            seen.add(tid)
    return unique_eos_ids


def main():
    args = parse_args()
    device = get_best_device()

    if not args.base_model:
        apply_qwen2_moe_patch()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model = AutoAWQForCausalLM.from_quantized(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    try:
        patched = patch_moe_instances(model.model if hasattr(model, "model") else model)
        logger.info(f"[Patch] Patched {patched} Qwen2MoeSparseMoeBlock instances with safe forward.")
    except Exception as e:
        logger.warning(f"[Patch] Failed to patch MoE instances: {e}")

    inputs = tokenizer.apply_chat_template(
        DEFAULT_PROMPT,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    eos_ids = _collect_eos_token_ids(tokenizer)
    gen_kwargs = dict(
        **inputs,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    else:
        logger.warning("No EOS token ids found; generation will rely on default settings.")

    model.generate(**gen_kwargs)


if __name__ == "__main__":
    main()
