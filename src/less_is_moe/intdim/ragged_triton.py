"""BF16 ragged expert GEMMs. All token assignment and compute stay on GPU.

Weights are compact. Intermediate scratch uses the largest retained width;
out-of-range column tiles exit, and down GEMM loops only over each expert's I.
This correctness-first backend does not claim an end-to-end speedup.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, W, Widths, Offsets, Sorted, Experts, Padded, C,
             H: tl.constexpr, MAX_I: tl.constexpr, ROUTES: tl.constexpr,
             TOP_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    value = (act * up).to(tl.bfloat16)
    tl.store(C + routes[:, None] * MAX_I + n[None, :], value,
             (routes[:, None] < ROUTES) & (n[None, :] < width))


@triton.jit
def _down(C, W, Widths, Offsets, Sorted, Experts, Padded, Routing, Out,
          H: tl.constexpr, MAX_I: tl.constexpr, ROUTES: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    values = (acc.to(tl.bfloat16).to(tl.float32) * route_weight[:, None]).to(tl.bfloat16)
    tl.store(Out + routes[:, None] * H + n[None, :], values,
             (routes[:, None] < ROUTES) & (n[None, :] < H))


def ragged_experts(hidden, gate_up, down, widths, offsets, max_width, topk_ids, topk_weights):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError("Ragged Triton experts require CUDA BF16")
    if gate_up.dtype != hidden.dtype or down.dtype != hidden.dtype:
        raise ValueError("Expert weights must have the activation dtype")
    hidden = hidden.contiguous()
    ids, routing = topk_ids.contiguous(), topk_weights.contiguous()
    tokens, hidden_size = hidden.shape
    top_k = ids.shape[1]
    if max_width == 0:
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
    _gate_up[(blocks, triton.cdiv(max_width, 64))](
        hidden, gate_up, widths, offsets, sorted_ids, expert_ids, padded, intermediate,
        TOP_K=top_k, **common)
    _down[(blocks, triton.cdiv(hidden_size, 64))](
        intermediate, down, widths, offsets, sorted_ids, expert_ids, padded, routing, output, **common)
    return output.view(tokens, top_k, hidden_size).sum(dim=1)
