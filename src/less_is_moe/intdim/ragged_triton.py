"""BF16 ragged expert GEMMs. All token assignment and compute stay on GPU.

Weights are compact. Intermediate scratch uses the largest retained width;
out-of-range column tiles exit, and down GEMM loops only over each expert's I.
This correctness-first backend does not claim an end-to-end speedup.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, W, Bias, Widths, Offsets, Sorted, Experts, Padded, C,
             H: tl.constexpr, MAX_I: tl.constexpr, ROUTES: tl.constexpr,
             TOP_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             ACT: tl.constexpr, HAS_BIAS: tl.constexpr):
    block, col = tl.program_id(0), tl.program_id(1)
    if block * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + block)
    width = tl.load(Widths + expert)
    if col * BN >= width:
        return
    start = tl.load(Offsets + expert).to(tl.int64) * H * 2
    routes = tl.load(Sorted + block * BM + tl.arange(0, BM))
    rows = routes // TOP_K
    n = col * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(H, BK)):
        kk = step * BK + k
        x = tl.load(X + rows[:, None] * H + kk[None, :],
                    (routes[:, None] < ROUTES) & (kk[None, :] < H), 0)
        pos = start + n[None, :] * H + kk[:, None]
        mask = (n[None, :] < width) & (kk[:, None] < H)
        g = tl.load(W + pos, mask, 0)
        u = tl.load(W + pos + width * H, mask, 0)
        gate = tl.dot(x, g, gate)
        up = tl.dot(x, u, up)
    # Preserve HF eager BF16 rounding at the linear and activation boundaries.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:
        bstart = tl.load(Offsets + expert).to(tl.int64) * 2
        gb = tl.load(Bias + bstart + n, n < width, 0).to(tl.float32)
        ub = tl.load(Bias + bstart + width + n, n < width, 0).to(tl.float32)
        gate = (gate + gb[None, :]).to(tl.bfloat16).to(tl.float32)
        up = (up + ub[None, :]).to(tl.bfloat16).to(tl.float32)
    if ACT == 2:  # GPT-OSS: clipped, biased SwiGLU, alpha=1.702.
        gate = tl.minimum(gate, 7.0)
        up = tl.maximum(tl.minimum(up, 7.0), -7.0)
        scaled = (gate * 1.702).to(tl.bfloat16).to(tl.float32)
        sigmoid = tl.sigmoid(scaled).to(tl.bfloat16).to(tl.float32)
        act = (gate * sigmoid).to(tl.bfloat16).to(tl.float32)
        up = (up + 1.0).to(tl.bfloat16).to(tl.float32)
    elif ACT == 1:  # Gemma4: GELU with tanh approximation.
        z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        tanh = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        act = (0.5 * gate * (1.0 + tanh)).to(tl.bfloat16).to(tl.float32)
    else:
        act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    value = (act * up).to(tl.bfloat16)
    tl.store(C + routes[:, None] * MAX_I + n[None, :], value,
             (routes[:, None] < ROUTES) & (n[None, :] < width))


@triton.jit
def _down(C, W, Bias, Widths, Offsets, Sorted, Experts, Padded, Routing, Out,
          H: tl.constexpr, MAX_I: tl.constexpr, ROUTES: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr):
    block, col = tl.program_id(0), tl.program_id(1)
    if block * BM >= tl.load(Padded):
        return
    expert = tl.load(Experts + block)
    width = tl.load(Widths + expert)
    start = tl.load(Offsets + expert).to(tl.int64) * H
    routes = tl.load(Sorted + block * BM + tl.arange(0, BM))
    n = col * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(width, BK)):
        kk = step * BK + k
        values = tl.load(C + routes[:, None] * MAX_I + kk[None, :],
                         (routes[:, None] < ROUTES) & (kk[None, :] < width), 0)
        weights = tl.load(W + start + n[None, :] * width + kk[:, None],
                          (n[None, :] < H) & (kk[:, None] < width), 0)
        acc = tl.dot(values, weights, acc)
    route_weight = tl.load(Routing + routes, routes < ROUTES, 0).to(tl.float32)
    values = acc.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + expert * H + n, n < H, 0).to(tl.float32)
        values = (values + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    values = (values * route_weight[:, None]).to(tl.bfloat16)
    tl.store(Out + routes[:, None] * H + n[None, :], values,
             (routes[:, None] < ROUTES) & (n[None, :] < H))


def ragged_experts(hidden, gate_up, down, widths, offsets, max_width, topk_ids, topk_weights,
                   *, activation="silu", gate_up_bias=None, down_bias=None):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError("Ragged Triton experts require CUDA BF16")
    if gate_up.dtype != hidden.dtype or down.dtype != hidden.dtype:
        raise ValueError("Expert weights must have the activation dtype")
    hidden = hidden.contiguous()
    ids, routing = topk_ids.contiguous(), topk_weights.contiguous()
    tokens, hidden_size = hidden.shape
    top_k = ids.shape[1]
    if activation not in ("silu", "gelu_tanh", "swigluoai"):
        raise ValueError("Unknown ragged expert activation")
    if (gate_up_bias is None) != (down_bias is None):
        raise ValueError("Both expert bias tensors must be supplied together")
    if max_width == 0 and down_bias is None:
        return torch.zeros_like(hidden)
    if tokens == 0:
        return torch.empty_like(hidden)
    block_m = 16
    sorted_ids, expert_ids, padded = moe_align_block_size(ids, block_m, widths.numel())
    routes = tokens * top_k
    intermediate = torch.empty((routes, max_width), dtype=hidden.dtype, device=hidden.device)
    output = torch.empty((routes, hidden_size), dtype=hidden.dtype, device=hidden.device)
    common = dict(H=hidden_size, MAX_I=max_width, ROUTES=routes, BM=block_m, BN=64, BK=32)
    blocks = triton.cdiv(sorted_ids.numel(), block_m)
    if max_width:
        _gate_up[(blocks, triton.cdiv(max_width, 64))](
            hidden, gate_up, gate_up_bias if gate_up_bias is not None else gate_up,
            widths, offsets, sorted_ids, expert_ids, padded, intermediate,
            TOP_K=top_k, ACT={"silu": 0, "gelu_tanh": 1, "swigluoai": 2}[activation],
            HAS_BIAS=gate_up_bias is not None, **common)
    _down[(blocks, triton.cdiv(hidden_size, 64))](
        intermediate, down, down_bias if down_bias is not None else down,
        widths, offsets, sorted_ids, expert_ids, padded, routing, output,
        HAS_BIAS=down_bias is not None, **common)
    return output.view(tokens, top_k, hidden_size).sum(dim=1)
