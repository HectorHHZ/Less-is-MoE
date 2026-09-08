import os
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
try:
    from trl.models.modeling_base import PreTrainedModelWrapper
except ImportError:
    # TRL >= 1.0 removed this class. It is only used for compatibility/type
    # checks in the inherited trainer code, so a neutral fallback is enough.
    PreTrainedModelWrapper = object
import deepspeed
from copy import deepcopy
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


RouterMask = Union[torch.Tensor, Dict[int, torch.Tensor]]


def _format_cuda_memory_stats() -> str:
    if not torch.cuda.is_available():
        return "cuda not available"
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB"


_MOE_BLOCK_NAMES = {"Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock"}


class RouterScoreTracker:
    """Collect routing weight stats via forward hooks (outside the forward code)."""

    def __init__(self) -> None:
        self.handles = []
        self.step_sums = {}
        self.step_counts = {}
        self.total_sums = {}
        self.total_counts = {}
        # For sample-level mean: average token weights within a sample, then average across samples.
        self.step_sample_sums = {}
        self.step_sample_counts = {}
        self.total_sample_sums = {}
        self.total_sample_counts = {}
        self._registered = False

    def _make_hook(self, layer_idx: Optional[int]):
        def hook(_module, _inputs, output):
            if output is None or not isinstance(output, tuple) or len(output) < 2:
                return
            router_logits = output[1]
            if router_logits is None:
                return
            with torch.no_grad():
                routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float)

                # Token-average path (existing behavior).
                rw_flat = routing_weights
                if rw_flat.dim() > 2:
                    rw_flat = rw_flat.reshape(-1, rw_flat.shape[-1])
                rw_sum = rw_flat.sum(dim=0)
                if layer_idx not in self.step_sums or self.step_sums[layer_idx].shape[0] != rw_sum.shape[0]:
                    self.step_sums[layer_idx] = torch.zeros_like(rw_sum)
                    self.step_counts[layer_idx] = 0
                self.step_sums[layer_idx] += rw_sum.detach()
                self.step_counts[layer_idx] += int(rw_flat.shape[0])

                # Sample-average path: mean over tokens inside each sample, then mean across samples.
                batch_size = None
                rw_sample_mean = None
                if router_logits.dim() >= 3:
                    batch_size = router_logits.shape[0]
                    rw_sample_mean = routing_weights.mean(dim=1)  # (batch, num_experts)
                elif _inputs and isinstance(_inputs[0], torch.Tensor) and _inputs[0].dim() >= 2:
                    inp = _inputs[0]
                    batch_size = inp.shape[0]
                    seq_len = inp.shape[1]
                    if batch_size * seq_len == rw_flat.shape[0]:
                        rw_sample_mean = rw_flat.view(batch_size, seq_len, -1).mean(dim=1)

                if rw_sample_mean is not None and batch_size is not None:
                    sample_sum = rw_sample_mean.sum(dim=0)
                    if (
                        layer_idx not in self.step_sample_sums
                        or self.step_sample_sums[layer_idx].shape[0] != sample_sum.shape[0]
                    ):
                        self.step_sample_sums[layer_idx] = torch.zeros_like(sample_sum)
                        self.step_sample_counts[layer_idx] = 0
                    self.step_sample_sums[layer_idx] += sample_sum.detach()
                    self.step_sample_counts[layer_idx] += int(batch_size)

        return hook

    def register(self, model) -> None:
        if self._registered or model is None:
            return
        unwrapped = getattr(model, "module", model)
        for module in unwrapped.modules():
            if module.__class__.__name__ not in _MOE_BLOCK_NAMES:
                continue
            layer_idx = getattr(module, "layer_idx", None)
            self.handles.append(module.register_forward_hook(self._make_hook(layer_idx)))
        self._registered = True

    def _all_reduce_in_place(self, tensor: torch.Tensor, reduce_op=torch.distributed.ReduceOp.SUM):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=reduce_op)

    def pop_scores(self) -> Dict[str, Dict[str, Dict[int, list]]]:
        step_token_scores, cumulative_token_scores = {}, {}
        step_sample_scores, cumulative_sample_scores = {}, {}

        # all-reduce step sums across ranks
        for layer_idx, rw_sum in list(self.step_sums.items()):
            cnt = self.step_counts.get(layer_idx, 0)
            self._all_reduce_in_place(rw_sum)
            if cnt > 0:
                step_token_scores[layer_idx] = (rw_sum / cnt).detach().cpu().tolist()
            # accumulate to totals
            if layer_idx not in self.total_sums or self.total_sums[layer_idx].shape[0] != rw_sum.shape[0]:
                self.total_sums[layer_idx] = torch.zeros_like(rw_sum)
                self.total_counts[layer_idx] = 0
            self.total_sums[layer_idx] += rw_sum
            self.total_counts[layer_idx] += cnt
            # reset step buffers
            del self.step_sums[layer_idx]
            del self.step_counts[layer_idx]

        # compute cumulative averages
        for layer_idx, tot_sum in self.total_sums.items():
            tot_cnt = self.total_counts.get(layer_idx, 0)
            if tot_cnt > 0:
                cumulative_token_scores[layer_idx] = (tot_sum / tot_cnt).detach().cpu().tolist()

        # all-reduce sample-level sums across ranks
        for layer_idx, rw_sum in list(self.step_sample_sums.items()):
            cnt = self.step_sample_counts.get(layer_idx, 0)
            self._all_reduce_in_place(rw_sum)
            if cnt > 0:
                step_sample_scores[layer_idx] = (rw_sum / cnt).detach().cpu().tolist()
            if (
                layer_idx not in self.total_sample_sums
                or self.total_sample_sums[layer_idx].shape[0] != rw_sum.shape[0]
            ):
                self.total_sample_sums[layer_idx] = torch.zeros_like(rw_sum)
                self.total_sample_counts[layer_idx] = 0
            self.total_sample_sums[layer_idx] += rw_sum
            self.total_sample_counts[layer_idx] += cnt
            del self.step_sample_sums[layer_idx]
            del self.step_sample_counts[layer_idx]

        for layer_idx, tot_sum in self.total_sample_sums.items():
            tot_cnt = self.total_sample_counts.get(layer_idx, 0)
            if tot_cnt > 0:
                cumulative_sample_scores[layer_idx] = (tot_sum / tot_cnt).detach().cpu().tolist()

        return {
            "token": {"step": step_token_scores, "cumulative": cumulative_token_scores},
            "sample": {"step": step_sample_scores, "cumulative": cumulative_sample_scores},
        }

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self._registered = False


class RouterMaskCallback(TrainerCallback):
    """
    update router_logits_mask when on_step_end is called.
    """

    def __init__(
        self,
        mask_fn: Optional[Callable[..., Optional[RouterMask]]] = None,
        enabled: bool = True,
        prune_start_step: Optional[int] = None,
        prune_interval: int = 5,
        prune_min_keep: int = 1,
        prune_step_size: int = 32,
        prune_expert_per_layer: Optional[int] = None,
        use_prune_plan: bool = True,
    ) -> None:
        self.mask_fn = mask_fn
        self.enabled = enabled
        self.score_tracker = RouterScoreTracker()
        self.prune_start_step = prune_start_step
        self.prune_interval = prune_interval
        self.prune_min_keep = prune_min_keep
        self.prune_step_size = prune_step_size
        self.prune_expert_per_layer = prune_expert_per_layer
        self.use_prune_plan = use_prune_plan
        self._masked_experts_by_layer = {}
        self._prune_masks_by_layer = {}
        self._num_experts_by_layer = {}
        self._last_prune_step = None
        self._planned_prune_per_layer = {}
        self._remaining_prune_per_layer = {}
        self._total_prune_budget = 0
        self._total_prune_remaining = 0

    def _prune_target_per_layer(self) -> Optional[int]:
        return self.prune_expert_per_layer

    def _is_main(self, args) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return False
        return getattr(args, "local_rank", -1) in (-1, 0)

    def _prune_enabled(self) -> bool:
        return (
            self.prune_start_step is not None
            and self.prune_start_step >= 0
            and self.prune_interval is not None
            and self.prune_interval > 0
            and self._prune_target_per_layer() is not None
            and self._prune_target_per_layer() > 0
            and self.prune_step_size is not None
            and self.prune_step_size > 0
        )

    def _maybe_init_layer_info(self, model) -> None:
        if model is None:
            return
        unwrapped = getattr(model, "module", model)
        for module in unwrapped.modules():
            if module.__class__.__name__ not in _MOE_BLOCK_NAMES:
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None:
                continue
            num_experts = int(module.gate.out_features)
            if layer_idx not in self._num_experts_by_layer:
                self._num_experts_by_layer[layer_idx] = num_experts
                self._masked_experts_by_layer.setdefault(layer_idx, set())
            elif self._num_experts_by_layer[layer_idx] != num_experts:
                self._num_experts_by_layer[layer_idx] = num_experts

    def _should_prune(self, step: Optional[int]) -> bool:
        if not self._prune_enabled() or step is None:
            return False
        if step < self.prune_start_step:
            return False
        if (step - self.prune_start_step) % self.prune_interval != 0:
            return False
        if self._last_prune_step == step:
            return False
        return True

    def _pick_lowest_score(self, scores, masked):
        best_idx = None
        best_score = None
        for idx, score in enumerate(scores):
            if idx in masked:
                continue
            if best_idx is None or score < best_score or (score == best_score and idx < best_idx):
                best_idx = idx
                best_score = score
        return best_idx

    def _build_prune_mask(self, num_experts: int, masked) -> torch.Tensor:
        mask = torch.zeros((num_experts,), dtype=torch.float32)
        if masked:
            mask[torch.tensor(sorted(masked), dtype=torch.long)] = float("-inf")
        return mask

    def _ensure_prune_plan(self, model, args):
        if not self.use_prune_plan:
            return
        if self._planned_prune_per_layer:
            return
        self._maybe_init_layer_info(model)
        if not self._num_experts_by_layer:
            return
        per_layer_target = self._prune_target_per_layer()
        if per_layer_target is None or per_layer_target <= 0:
            return
        unwrapped = getattr(model, "module", model)
        entropy_delta = getattr(unwrapped, "_entropy_delta_mean", None)
        alpha = float(getattr(args, "entropy_slope_alpha", 1.0))
        beta = float(getattr(args, "entropy_slope_beta", 1.0))
        tau = float(getattr(args, "router_prune_score_tau", 1.0))
        tau = max(tau, 1e-6)

        layer_indices = sorted(self._num_experts_by_layer.keys())
        num_layers = len(layer_indices)
        total_budget = per_layer_target * num_layers

        sl = None
        if entropy_delta is not None and isinstance(entropy_delta, torch.Tensor):
            try:
                delta = entropy_delta.detach().float().to("cpu")
                if delta.numel() >= num_layers:
                    sl = beta * torch.relu(-delta[:num_layers]) + alpha * torch.relu(delta[:num_layers])
            except Exception:
                sl = None
        if sl is None:
            sl = torch.ones(num_layers, dtype=torch.float32)

        # Smoothly allocate prune budget: w_l = exp(-s_l / tau)
        weights = torch.exp(-sl / tau)
        if (not torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)

        caps = []
        for idx in layer_indices:
            num_exp = self._num_experts_by_layer[idx]
            caps.append(max(num_exp - self.prune_min_keep, 0))
        caps_tensor = torch.tensor(caps, dtype=torch.float32)
        max_total = int(caps_tensor.sum().item())
        total_budget = min(total_budget, max_total)
        self._total_prune_budget = total_budget
        self._total_prune_remaining = total_budget
        if total_budget == 0:
            return

        raw_alloc = weights / weights.sum() * float(total_budget)
        floors = torch.floor(raw_alloc).to(torch.int64)
        floors = torch.minimum(floors, caps_tensor.to(dtype=torch.int64))
        remainder = raw_alloc - floors.float()

        remaining = total_budget - int(floors.sum().item())
        room = (caps_tensor.to(torch.int64) - floors).tolist()
        remainder_list = remainder.tolist()
        while remaining > 0:
            best_idx = None
            best_val = None
            for i, r in enumerate(remainder_list):
                if room[i] <= 0:
                    continue
                if best_idx is None or r > best_val:
                    best_idx = i
                    best_val = r
            if best_idx is None:
                break
            floors[best_idx] += 1
            room[best_idx] -= 1
            remainder_list[best_idx] = 0.0
            remaining -= 1

        for i, layer_idx in enumerate(layer_indices):
            cnt = int(floors[i].item())
            self._planned_prune_per_layer[layer_idx] = cnt
            self._remaining_prune_per_layer[layer_idx] = cnt
        if self._is_main(args):
            # Debug: print per-layer slope (sl) and allocation weights at plan init.
            try:
                sl_list = sl.detach().cpu().tolist()
                wt_list = weights.detach().cpu().tolist()
                slope_str = ", ".join(f"L{layer_indices[i]}:{sl_list[i]:.6f}" for i in range(num_layers))
                weight_str = ", ".join(f"L{layer_indices[i]}:{wt_list[i]:.6f}" for i in range(num_layers))
                print(f"[RouterPrune] slopes={slope_str}")
                print(f"[RouterPrune] weight_raw=exp(-s_l/tau)={weight_str}")
            except Exception:
                pass
            plan_parts = [f"{idx}:{self._planned_prune_per_layer.get(idx, 0)}" for idx in layer_indices]
            print(f"[RouterPrune] plan initialized: total_budget={self._total_prune_budget}, per_layer={{" + ", ".join(plan_parts) + "}}")

    def _init_total_prune_budget(self, model, args):
        """
        初始化全局裁剪预算，用于不启用 per-layer 规划的情形。
        预算 = prune_expert_per_layer * 层数，且不超过各层可裁剪上限之和。
        """
        if self._total_prune_budget > 0:
            return
        self._maybe_init_layer_info(model)
        if not self._num_experts_by_layer:
            return
        per_layer_target = self._prune_target_per_layer()
        if per_layer_target is None or per_layer_target <= 0:
            return
        layer_indices = sorted(self._num_experts_by_layer.keys())
        caps = [max(self._num_experts_by_layer[idx] - self.prune_min_keep, 0) for idx in layer_indices]
        max_total = sum(caps)
        total_budget = per_layer_target * len(layer_indices)
        total_budget = min(total_budget, max_total)
        self._total_prune_budget = total_budget
        self._total_prune_remaining = total_budget
        if self._is_main(args):
            plan_parts = [f"{layer_indices[i]}:{caps[i]}" for i in range(len(layer_indices))]
            print(f"[RouterPrune] no-plan mode init: total_budget={total_budget}, caps={{" + ", ".join(plan_parts) + "}}")

    def _maybe_prune_with_scores(self, model, scores, step: Optional[int], args):
        if not self._prune_enabled() or step is None or step < self.prune_start_step:
            return
        if not self._should_prune(step):
            return
        # 初始化裁剪计划或预算
        if self.use_prune_plan:
            self._ensure_prune_plan(model, args)
        else:
            self._init_total_prune_budget(model, args)
        if self._total_prune_remaining <= 0:
            return
        cumulative_sample = scores.get("sample", {}).get("cumulative", {})
        if not cumulative_sample:
            return
        if self._is_main(args):
            try:
                sample_debug = []
                for layer_idx, layer_scores in sorted(cumulative_sample.items()):
                    vals = [f"{float(x):.6f}" for x in layer_scores]
                    sample_debug.append(f"L{layer_idx}:[{', '.join(vals)}]")
                if sample_debug:
                    print(f"[RouterPrune] step={step} cumulative_sample_scores " + "; ".join(sample_debug))
            except Exception:
                pass

        candidates = []
        pruned_list = []
        for layer_idx, layer_scores in cumulative_sample.items():
            masked = self._masked_experts_by_layer.setdefault(layer_idx, set())
            num_exp = self._num_experts_by_layer.get(layer_idx, len(layer_scores))
            # 按层限制保留的最少专家
            remaining_cap = max(num_exp - self.prune_min_keep - len(masked), 0)
            if self.use_prune_plan:
                remaining_plan = self._remaining_prune_per_layer.get(layer_idx, 0)
                if remaining_plan <= 0:
                    continue
                remaining_cap = min(remaining_cap, remaining_plan)
            if remaining_cap <= 0:
                continue
            for exp_idx, sc in enumerate(layer_scores):
                if exp_idx in masked:
                    continue
                candidates.append((float(sc), layer_idx, exp_idx))

        if not candidates:
            return
        candidates.sort(key=lambda x: x[0])
        target = min(self.prune_step_size, self._total_prune_remaining, len(candidates))
        pruned = 0
        for sc, layer_idx, exp_idx in candidates:
            if pruned >= target:
                break
            masked = self._masked_experts_by_layer.setdefault(layer_idx, set())
            num_exp = self._num_experts_by_layer.get(layer_idx, len(candidates))
            cap = max(num_exp - self.prune_min_keep, 0)
            if exp_idx in masked or len(masked) >= cap:
                continue
            if self.use_prune_plan:
                remaining_layer = self._remaining_prune_per_layer.get(layer_idx, 0)
                if remaining_layer <= 0:
                    continue
                self._remaining_prune_per_layer[layer_idx] = remaining_layer - 1
            masked.add(exp_idx)
            self._total_prune_remaining -= 1
            self._prune_masks_by_layer[layer_idx] = self._build_prune_mask(num_exp, masked)
            pruned_list.append((layer_idx, exp_idx))
            pruned += 1

        if pruned > 0:
            self._last_prune_step = step
            if self._is_main(args):
                pruned_str = ", ".join(f"L{l}-E{e}" for l, e in pruned_list)
                print(f"[RouterPrune] step={step} pruned={pruned} ({pruned_str})")
                # Report remaining experts per layer after pruning
                layer_summaries = []
                for layer_idx in sorted(self._num_experts_by_layer.keys()):
                    total = self._num_experts_by_layer[layer_idx]
                    masked = self._masked_experts_by_layer.get(layer_idx, set())
                    kept = total - len(masked)
                    layer_summaries.append(f"L{layer_idx}:{kept}/{total}")
                if layer_summaries:
                    print("[RouterPrune] remaining experts " + ", ".join(layer_summaries))

    def _resolve_layer_mask(self, mask, layer_idx):
        if mask is None:
            return None
        if isinstance(mask, dict):
            return mask.get(layer_idx)
        return mask

    def _combine_masks(self, mask_a, mask_b, device):
        if mask_a is None and mask_b is None:
            return None
        if mask_a is None:
            return mask_b.to(device)
        if mask_b is None:
            return mask_a.to(device)
        mask_a = mask_a.to(device)
        mask_b = mask_b.to(device)
        return torch.minimum(mask_a, mask_b)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not self.enabled:
            return
        self.score_tracker.register(model)
        self._maybe_init_layer_info(model)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not self.enabled:
            return
        self.score_tracker.remove()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if not self.enabled:
            return
        if model is None:
            return

        # DDP unwrap
        unwrapped = getattr(model, "module", model)
        mask = None
        if self.mask_fn is not None:
            mask = self.mask_fn(model=unwrapped, step=state.global_step, args=args, state=state)

        # Debug: summarize which experts are masked this step (only on main/local_rank 0 to reduce spam).
        prune_mask = self._prune_masks_by_layer if self._prune_masks_by_layer else None
        if self._is_main(args) and (mask is not None or prune_mask is not None):
            def _mask_summary(m):
                if isinstance(m, dict):
                    parts = []
                    for layer_idx, tensor in m.items():
                        if tensor is None:
                            continue
                        disabled = torch.isinf(tensor.detach().cpu()).nonzero(as_tuple=False).flatten().tolist()
                        parts.append(f"{layer_idx}:{disabled}")
                    return "; ".join(parts) if parts else "none"
                disabled = torch.isinf(m.detach().cpu()).nonzero(as_tuple=False).flatten().tolist()
                return f"all:{disabled}"
            # print(f"[RouterMaskCallback] on_step_end step={state.global_step} mask={_mask_summary(mask)}")
            # Also show router logits stats captured during the last forward.
            stats_parts = []
            for module in unwrapped.modules():
                if module.__class__.__name__ not in _MOE_BLOCK_NAMES:
                    continue
                stats = getattr(module, "_last_router_stats", None)
                if not stats:
                    continue
                min_v, max_v, inf_v = stats.get("min"), stats.get("max"), stats.get("inf")
                if min_v is None or max_v is None:
                    continue
                layer_idx = getattr(module, "layer_idx", None)
                part = f"{layer_idx}:min={min_v:.4f},max={max_v:.4f},#inf={int(inf_v or 0)}"
                masked_ids = stats.get("masked_ids")
                if masked_ids is not None:
                    part += f",masked={masked_ids}"
                stats_parts.append(part)
            if stats_parts:
                # print(f"[RouterMaskCallback] router_logits step={state.global_step} " + "; ".join(stats_parts))
                pass

        # Per-layer routing score (mean routing weight per expert over this step) collected via hooks.
        scores = self.score_tracker.pop_scores()
        token_scores = scores.get("token", {})
        sample_scores = scores.get("sample", {})
        if self._is_main(args):

            # Token-average: average over all tokens in this optimizer step (all micro-batches).
            step_token = token_scores.get("step", {})
            cum_token = token_scores.get("cumulative", {})
            if step_token:
                parts = [f"{layer_idx}:{vals}" for layer_idx, vals in step_token.items()]
                # print(f"[RouterMaskCallback] routing_scores_step_token_avg step={state.global_step} " + "; ".join(parts))
                pass
            if cum_token:
                parts = [f"{layer_idx}:{vals}" for layer_idx, vals in cum_token.items()]
                # print(f"[RouterMaskCallback] routing_scores_cumulative_token_avg step={state.global_step} " + "; ".join(parts))
                pass

            # Sample-average: per-sample mean over tokens, then averaged across samples.
            step_sample = sample_scores.get("step", {})
            cum_sample = sample_scores.get("cumulative", {})
            if step_sample:
                parts = [f"{layer_idx}:{vals}" for layer_idx, vals in step_sample.items()]
                # print(f"[RouterMaskCallback] routing_scores_step_sample_avg step={state.global_step} " + "; ".join(parts))
                pass
            if cum_sample:
                parts = [f"{layer_idx}:{vals}" for layer_idx, vals in cum_sample.items()]
                # print(f"[RouterMaskCallback] routing_scores_cumulative_sample_avg step={state.global_step} " + "; ".join(parts))
                pass

        if self._prune_enabled():
            self._maybe_prune_with_scores(unwrapped, scores, state.global_step, args)
            prune_mask = self._prune_masks_by_layer if self._prune_masks_by_layer else None

        if mask is None and prune_mask is None:
            return

        for module in unwrapped.modules():
            if not hasattr(module, "_set_router_logits_mask"):
                continue

            layer_idx = getattr(module, "layer_idx", None)
            target_mask = self._resolve_layer_mask(mask, layer_idx)
            target_prune = self._resolve_layer_mask(prune_mask, layer_idx)
            combined = self._combine_masks(target_mask, target_prune, module.gate.weight.device)
            if combined is None:
                continue
            module._set_router_logits_mask(combined)


class MemSnapshotCallback(TrainerCallback):
    """Dump CUDA memory snapshots on step_end (main process only)."""

    def __init__(self, interval: int = 2, snapshot_dir: Optional[str] = None) -> None:
        self.interval = interval
        self.snapshot_dir = snapshot_dir

    def _is_main(self, args) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return False
        return getattr(args, "local_rank", -1) in (-1, 0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.interval is None or self.interval <= 0:
            return
        if not self._is_main(args):
            return
        if not torch.cuda.is_available():
            return
        step = getattr(state, "global_step", None)
        if step is None or step % self.interval != 0:
            return
        # try:
        #     out_dir = self.snapshot_dir or os.path.join(args.output_dir, "mem_snapshots")
        #     os.makedirs(out_dir, exist_ok=True)
        #     torch.cuda.memory._dump_snapshot(os.path.join(out_dir, f"step{step}.cuda_snapshot"))
        # except Exception:
        #     pass


class CustomSFTTrainer(SFTTrainer):

    def __init__(
        self,
        *args,
        router_mask_fn: Optional[Callable[..., Optional[RouterMask]]] = None,
        callbacks=None,
        teacher_model_name_or_path: Optional[str] = None,
        teacher_model_init_kwargs: Optional[Dict[str, Any]] = None,
        disable_teacher_dropout: bool = True,
        layer_entropy_l1_weight: float = 0.0,
        layer_entropy_l1_layers: Optional[list[int]] = None,
        last_entropy_weight: float = 0.0,
        attn_kl_weight: float = 0.0,
        teacher_deepspeed_config: Optional[str] = None,
        entropy_eps: float = 1e-6,
        norm_name_override: Optional[str] = None,
        **kwargs,
    ):
        callbacks = list(callbacks) if callbacks is not None else []
        self.router_mask_enabled = getattr(self, "router_mask_enabled", False)
        if router_mask_fn is not None:
            callbacks.append(RouterMaskCallback(router_mask_fn, enabled=self.router_mask_enabled))
        self.teacher_model_name_or_path = teacher_model_name_or_path
        self.teacher_model_init_kwargs = teacher_model_init_kwargs or {}
        self.disable_teacher_dropout = disable_teacher_dropout
        self.teacher_model = None
        self._teacher_debug_checked = False
        self.layer_entropy_l1_weight = layer_entropy_l1_weight
        self.layer_entropy_l1_layers = set(layer_entropy_l1_layers) if layer_entropy_l1_layers else None
        self.last_entropy_weight = last_entropy_weight
        self.attn_kl_weight = attn_kl_weight
        self.teacher_deepspeed_config = teacher_deepspeed_config
        self.entropy_eps = entropy_eps
        self.norm_name_override = norm_name_override
        self._loss_debug_logged = False
        self._mem_step_logged = True
        self._mem_optim_logged = True
        self._student_device_logged = False
        self._entropy_delta_sum = None
        self._entropy_delta_count = 0
        self._mem_snapshot_interval = 2  # dump every N optimizer steps; set <=0 to disable
        self._mem_snapshot_dir = None
        if self._mem_snapshot_interval and self._mem_snapshot_interval > 0:
            callbacks.append(
                MemSnapshotCallback(
                    interval=self._mem_snapshot_interval,
                    snapshot_dir=self._mem_snapshot_dir,
                )
            )

        # Initialize the student before preparing the optional teacher model.

        super().__init__(*args, callbacks=callbacks, **kwargs)

        if self.accelerator.is_main_process:
            print(f"[Student] Memory after init: {_format_cuda_memory_stats()}")
            if self.accelerator.state.deepspeed_plugin is not None:
                try:
                    zero_stage = self.accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"]
                    print(f"[Student] DeepSpeed ZeRO stage: {zero_stage}")
                except Exception:
                    print("[Student] DeepSpeed ZeRO stage: unknown")

        if self.teacher_model_name_or_path is not None:
            self._init_teacher_model()

    def _freeze_teacher_parameters(self, model: nn.Module) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = False
        print("[Teacher] All parameters frozen (requires_grad=False).")
        return model

    def _disable_dropout_in_model(self, model: nn.Module) -> nn.Module:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.eval()
        print("[Teacher] Dropout disabled and model set to eval().")
        return model

    def _disable_router_mask_in_teacher(self, model: nn.Module) -> None:
        """Turn off router mask usage so teacher uses original gating."""
        if model is None:
            return
        disabled = 0
        for module in model.modules():
            if module.__class__.__name__ not in _MOE_BLOCK_NAMES:
                continue
            try:
                if hasattr(module, "disable_router_mask"):
                    module.disable_router_mask()
                else:
                    module.use_router_mask = False
                disabled += 1
            except Exception:
                continue
        if disabled > 0:
            print(f"[Teacher] Disabled router mask on {disabled} MoE SparseMoeBlock modules.")

    def _init_teacher_model(self) -> Optional[nn.Module]:
        teacher_model = self.teacher_model_name_or_path
        if teacher_model is None:
            return None

        print(f"[Teacher] Memory before init: {_format_cuda_memory_stats()}")
        init_kwargs = deepcopy(self.teacher_model_init_kwargs)
        dtype = init_kwargs.get("torch_dtype")
        if isinstance(dtype, str) and dtype not in ("auto", None):
            init_kwargs["torch_dtype"] = getattr(torch, dtype)
        init_kwargs.setdefault("use_cache", True)

        if isinstance(teacher_model, str):
            print(f"[Teacher] Initializing teacher model from {teacher_model} with kwargs: {init_kwargs}")
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **init_kwargs)
            print("[Teacher] Loaded teacher model weights.")
            print(f"[Teacher] Memory after load: {_format_cuda_memory_stats()}")

        if self.disable_teacher_dropout:
            teacher_model = self._disable_dropout_in_model(teacher_model)

        teacher_model = self._freeze_teacher_parameters(teacher_model)
        # Ensure teacher runs with unmasked router logits (original forward).
        self._disable_router_mask_in_teacher(teacher_model)

        if self.is_deepspeed_enabled:
            teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            print("[Teacher] Prepared with accelerator (non-DeepSpeed).")

        teacher_model.eval()
        print(f"[Teacher] Memory after prepare: {_format_cuda_memory_stats()}")
        self.teacher_model = teacher_model
        self.teacher_model_init_kwargs = init_kwargs
        try:
            param = next(teacher_model.parameters())
            print(f"[Teacher] Ready. device={param.device}, dtype={param.dtype}, deepspeed={self.is_deepspeed_enabled}")
        except Exception:
            print(f"[Teacher] Ready. (could not fetch param device/dtype)")
        return teacher_model

    # Prepare the optional teacher model with an inference-safe DeepSpeed setup.
    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # Teacher runs inference-only. If user config is ZeRO-1/2, fallback to stage-0
        # to avoid optimizer requirement (assertion in DeepSpeedEngine). ZeRO-3 can stay
        # because it shards params without needing optimizer state here.
        zero_stage = config_kwargs.get("zero_optimization", {}).get("stage", 0)
        config_kwargs.setdefault("zero_optimization", {})
        if zero_stage in (1, 2):
            if self.accelerator.is_main_process:
                print(f"[Teacher] ZeRO stage {zero_stage} requested; falling back to stage 0 for inference (no optimizer).")
            zero_stage = 0
        config_kwargs["zero_optimization"]["stage"] = zero_stage
        config_kwargs.pop("optimizer", None)
        config_kwargs.pop("scheduler", None)

        # ZeRO-3 needs model parameters to shard; ZeRO-0/1/2 skip.
        model_parameters = list(model.parameters()) if zero_stage == 3 else None

        model, *_ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            optimizer=None,
            lr_scheduler=None,
            config=config_kwargs,
        )
        model.eval()
        print(f"[Teacher] Prepared with DeepSpeed (no optimizer). zero_stage={zero_stage}")
        return model

    # compute_loss function:
    # 1. https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L1143
    # 2. https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3845
    # 3. https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3928

    # outputs attention and hidden states: https://huggingface.co/docs/transformers/en/main_classes/output
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        step = getattr(self.state, "global_step", None)
        prune_start_step = getattr(self.args, "router_prune_start_step", None)
        use_ce_only = (
            prune_start_step is not None
            and prune_start_step >= 0
            and step is not None
            and step < prune_start_step
        )

        need_layer_l1 = False if use_ce_only else (self.layer_entropy_l1_weight != 0)
        compute_entropy_for_plan = (
            prune_start_step is not None
            and prune_start_step >= 0
            and step is not None
            and step <= prune_start_step
        )

        teacher_outputs = None
        need_teacher = need_layer_l1
        if need_teacher:
            if self.teacher_model is None:
                raise ValueError(
                    "layer_entropy_l1_weight is non-zero but no "
                    "teacher_model_name_or_path was configured."
                )
            teacher_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask", "position_ids") if k in inputs}
            if teacher_inputs:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        **teacher_inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False,
                    )
        # Student forward with hidden states / attentions for auxiliary losses.
        model_inputs = dict(inputs)
        need_hidden_states = need_layer_l1 or compute_entropy_for_plan
        model_inputs.update(
            {
                "output_hidden_states": need_hidden_states,
                "use_cache": False,
            }
        )

        ce_loss, student_outputs = super().compute_loss(
            model,
            model_inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        # Auxiliary losses; keep teacher parts no-grad, student parts need grad.
        logits = getattr(student_outputs, "logits", None)
        attn_mask = inputs.get("attention_mask", None)
        device = logits.device

        norm_s, head_s = self._get_norm_and_head(model)
        norm_t, head_t = (None, None)
        if teacher_outputs is not None:
            norm_t, head_t = self._get_norm_and_head(self.teacher_model)

        student_layer_entropies: list[Optional[torch.Tensor]] = []
        plan_layer_entropies = None
        teacher_layer_entropies = []
        student_hiddens = None
        teacher_hiddens = None

        if need_layer_l1 or compute_entropy_for_plan:
            student_hiddens = getattr(student_outputs, "hidden_states", None)
            if student_hiddens is None or len(student_hiddens) == 0:
                raise ValueError("Student outputs missing hidden_states; cannot compute entropy losses.")

            num_layers = len(student_hiddens) - 1  # skip embedding
            student_layer_entropies = [None] * num_layers
            target_layers = self.layer_entropy_l1_layers
            selected_layers = (
                list(range(num_layers))
                if target_layers is None
                else [i for i in target_layers if 0 <= i < num_layers]
            )

            # Compute entropies with grad only for selected layers.
            if need_layer_l1:
                for idx in selected_layers:
                    hs = student_hiddens[idx + 1]
                    student_layer_entropies[idx] = self._compute_entropy_full(hs, norm_s, head_s, attn_mask)

            # For pruning plan stats, compute all-layer entropies in no_grad to avoid graph bloat.
            if compute_entropy_for_plan:
                with torch.no_grad():
                    plan_layer_entropies = self._compute_layer_entropies(
                        student_hiddens,
                        norm_s,
                        head_s,
                        attn_mask,
                    )

        if need_layer_l1 and teacher_outputs is not None:
            teacher_hiddens = getattr(teacher_outputs, "hidden_states", None)
            if teacher_hiddens is not None and len(teacher_hiddens) > 0:
                with torch.no_grad():
                    teacher_layer_entropies = self._compute_layer_entropies(
                        teacher_hiddens,
                        norm_t,
                        head_t,
                        attn_mask,
                    )

        # Layer-wise token entropy L1 (teacher vs student), using precomputed per-layer entropies.
        layer_l1 = torch.zeros(1, device=device, dtype=logits.dtype)
        used_layers = None
        if need_layer_l1 and student_layer_entropies and teacher_layer_entropies:
            target_layers = self.layer_entropy_l1_layers
            used_layers = []
            for idx, s_ent in enumerate(student_layer_entropies):
                if s_ent is None:
                    continue
                if target_layers is not None and idx not in target_layers:
                    continue
                if idx >= len(teacher_layer_entropies):
                    continue
                used_layers.append(idx)
                layer_l1 = layer_l1 + torch.abs(s_ent - teacher_layer_entropies[idx].detach())

        # Debug: log layer_l1 value only when we computed it to avoid UnboundLocalError.
        if used_layers is not None:
            try:
                print("layer_l1:", layer_l1.detach().item(), "layers:", used_layers if used_layers else "all")
            except Exception:
                # Fallback without formatting in case of device/precision issues
                print("layer_l1:", layer_l1, "layers:", used_layers if used_layers else "all")

        # Accumulate entropy deltas for pruning allocation (no loss contribution).
        if compute_entropy_for_plan and plan_layer_entropies:
            with torch.no_grad():
                deltas = self._compute_layer_entropy_deltas(plan_layer_entropies)
                if deltas is not None:
                    if self._entropy_delta_sum is None or self._entropy_delta_sum.numel() != deltas.numel():
                        self._entropy_delta_sum = torch.zeros_like(deltas)
                        self._entropy_delta_count = 0
                    self._entropy_delta_sum = self._entropy_delta_sum.to(deltas.device) + deltas.detach()
                    self._entropy_delta_count += 1
                if deltas is not None:
                    del deltas

        # Cache running mean of entropy deltas on the model for pruning allocation.
        if compute_entropy_for_plan and self._entropy_delta_sum is not None and self._entropy_delta_count > 0:
            with torch.no_grad():
                try:
                    mean_delta = (self._entropy_delta_sum / float(self._entropy_delta_count)).detach()
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        tmp = mean_delta.to(device)
                        torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM)
                        world = torch.distributed.get_world_size()
                        mean_delta = tmp / float(world)
                    unwrapped = getattr(model, "module", model)
                    setattr(unwrapped, "_entropy_delta_mean", mean_delta.to("cpu"))
                except Exception:
                    pass

        if plan_layer_entropies:
            del plan_layer_entropies
        if student_layer_entropies:
            del student_layer_entropies
        if teacher_layer_entropies:
            del teacher_layer_entropies
        # if student_hiddens is not None:
        #     del student_hiddens
        # if teacher_hiddens is not None:
        #     del teacher_hiddens
        loss = ce_loss + self.layer_entropy_l1_weight * layer_l1
        # print("ce_loss: ", ce_loss.item())

        return loss

    def _compute_layer_entropies(self, hidden_states, norm, head, mask):
        if hidden_states is None or len(hidden_states) <= 1:
            return []
        entropies = []
        for hs in hidden_states[1:]:  # skip embedding layer
            entropies.append(self._compute_entropy_full(hs, norm, head, mask))
        return entropies

    def _compute_layer_entropy_deltas(self, layer_entropies):
        if not layer_entropies:
            return None
        deltas = []
        for idx in range(len(layer_entropies) - 1):
            deltas.append(layer_entropies[idx + 1] - layer_entropies[idx])
        deltas.append(-layer_entropies[-1])
        return torch.stack(deltas, dim=0)


    def _compute_entropy_full(self, h, norm, head, mask):
        logits = self._logits_from_hidden(h, norm, head)
        ent = self._token_entropy(logits, mask)
        return self._masked_mean(ent, mask)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def optimizer_step(
        self,
        optimizer,
        model,
        trial=None,
        closure=None,
        **kwargs,
    ):
        main_proc = getattr(self.accelerator, "is_main_process", True)
        step = getattr(self.state, "global_step", None)
        result = super().optimizer_step(optimizer, model, trial=trial, closure=closure, **kwargs)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return result

    def _wrap_model(self, model, training=True, dataloader=None):
        model = super()._wrap_model(model, training=training, dataloader=dataloader)
        return model

    def _get_norm_and_head(self, model: nn.Module):
        """Best-effort fetch of final norm and lm_head."""
        unwrapped = getattr(model, "module", model)
        norm = None
        if self.norm_name_override:
            norm = getattr(unwrapped, self.norm_name_override, None)
        if norm is None and hasattr(unwrapped, "model"):
            base = getattr(unwrapped, "model")
            norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
        if norm is None:
            norm = getattr(unwrapped, "norm", None) or getattr(unwrapped, "final_layernorm", None)
        if norm is None:
            norm = nn.Identity()

        head = getattr(unwrapped, "lm_head", None)
        if head is None and hasattr(unwrapped, "model"):
            head = getattr(unwrapped.model, "lm_head", None)
        if head is None and hasattr(unwrapped, "score"):
            head = getattr(unwrapped, "score")
        if head is None and hasattr(unwrapped, "get_output_embeddings"):
            try:
                head = unwrapped.get_output_embeddings()
            except Exception:
                head = None
        if head is None:
            raise ValueError("Could not find lm_head / output embeddings to compute logits.")
        return norm, head

    def _logits_from_hidden(self, hidden_states: torch.Tensor, norm: nn.Module, head: nn.Module) -> torch.Tensor:
        return head(norm(hidden_states))

    def _token_entropy(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + self.entropy_eps)
        entropy = -(probs * log_probs).sum(dim=-1)
        if mask is not None:
            entropy = entropy * mask.to(entropy.dtype)
        return entropy

    def _masked_mean(self, tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return tensor.mean()
        mask = mask.to(tensor.dtype)
        mask = torch.broadcast_to(mask, tensor.shape)
        masked = tensor * mask
        denom = mask.sum().clamp_min(1e-8)
        return masked.sum() / denom

    def _attention_kl(
        self, student_attn: torch.Tensor, teacher_attn: torch.Tensor, attn_mask: Optional[torch.Tensor], eps: float
    ) -> torch.Tensor:
        # Align sequence lengths in case of minor mismatches.
        q_len = min(student_attn.shape[-2], teacher_attn.shape[-2])
        k_len = min(student_attn.shape[-1], teacher_attn.shape[-1])
        s = student_attn[..., :q_len, :k_len]
        t = teacher_attn[..., :q_len, :k_len]

        if attn_mask is not None:
            k_mask = attn_mask[:, :k_len].to(s.dtype).unsqueeze(1).unsqueeze(1)  # (b,1,1,k)
            s = s * k_mask
            t = t * k_mask

        # Renormalize after masking to obtain distributions.
        s_sum = s.sum(dim=-1, keepdim=True).clamp_min(eps)
        t_sum = t.sum(dim=-1, keepdim=True).clamp_min(eps)
        p = s / s_sum
        q = t / t_sum
        kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)  # (b, heads, q_len)

        if attn_mask is not None:
            q_mask = attn_mask[:, :q_len].to(kl.dtype).unsqueeze(1)  # (b,1,q)
            return self._masked_mean(kl, q_mask)
        return kl.mean()

    def _save_debug_info(self, debug_info: Dict[str, Any]) -> None:
        """Save debug information to a file in the output directory."""
        if hasattr(self, "args") and getattr(self.args, "output_dir", None):
            debug_file = os.path.join(self.args.output_dir, "loss_debug.log")
            try:
                with open(debug_file, "a") as f:
                    f.write(f"{debug_info}\n")
            except Exception:
                print(f"DEBUG(write_fail): {debug_info}")
        else:
            print(f"DEBUG: {debug_info}")


    # training_step(self, model, inputs, num_items_in_batch=None):
    # 1. https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3774


    # _inner_training_loop code:
    # 1. https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2229
    # (High priority)2. for loop for epoch, place to modify the mask:https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2448
    # (High priority)3. where to add hook after step end: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2611; https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L392

    def _materialize_router_masks(self):
        """
        Turn router_logits_mask (non-persistent buffers) into a pruned model so the saved
        checkpoint matches the masked structure. Follows the expert_drop flow: main rank
        prunes experts + gate weights, updates config, then saves.
        """
        unwrapped = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else getattr(
            self.model, "module", self.model
        )

        moe_blocks = []
        for module in unwrapped.modules():
            if hasattr(module, "_set_router_logits_mask") and hasattr(module, "gate"):
                moe_blocks.append(module)

        if not moe_blocks:
            return {"applied": False, "reason": "no_moe_blocks", "summary": []}

        # The release's main SFT path trains an already-pruned checkpoint. Do
        # not rewrite its expert layout unless an online/manual mask is active.
        # This also keeps the Qwen3.5 batched-expert representation untouched.
        if not any(
            getattr(module, "router_logits_mask", None) is not None
            for module in moe_blocks
        ):
            return {
                "applied": False,
                "reason": "no_active_router_masks",
                "summary": [],
            }

        if any(
            module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock"
            for module in moe_blocks
        ):
            raise ValueError(
                "Qwen3.5 router masks cannot be structurally materialized by "
                "this SFT callback; prune the checkpoint before SFT instead."
            )

        summary = []
        per_layer_kept = {}
        per_layer_orig = {}
        for module in moe_blocks:
            layer_idx = getattr(module, "layer_idx", None)
            gate = getattr(module, "gate", None)
            orig = gate.out_features if gate is not None else None
            mask = getattr(module, "router_logits_mask", None)

            if mask is None:
                keep_ids = list(range(orig))
            else:
                mask_flat = mask.detach()
                if mask_flat.dim() != 1 or mask_flat.shape[0] != orig:
                    raise ValueError(f"[RouterMaskSave] layer {layer_idx} mask shape {mask_flat.shape} != {orig}")
                keep_ids = torch.nonzero(mask_flat > float("-inf"), as_tuple=False).view(-1).tolist()

            if len(keep_ids) == 0:
                raise ValueError(f"[RouterMaskSave] Mask removed all experts in layer {layer_idx}; aborting save.")

            drop_ids = sorted(list(set(range(orig)) - set(keep_ids)))
            per_layer_kept[layer_idx] = keep_ids
            per_layer_orig[layer_idx] = orig
            summary.append(
                {"layer": layer_idx, "kept": keep_ids, "dropped": drop_ids, "original": orig, "mask_present": mask is not None}
            )

        # Prune in-place
        for module, info in zip(moe_blocks, summary):
            keep_ids, drop_ids = info["kept"], info["dropped"]
            gate: nn.Linear = module.gate
            if drop_ids:
                module.experts = nn.ModuleList([module.experts[i] for i in keep_ids])
                new_gate = nn.Linear(
                    in_features=gate.in_features,
                    out_features=len(keep_ids),
                    bias=False,
                    device=gate.weight.device,
                    dtype=gate.weight.dtype,
                )
                new_gate.weight.data = gate.weight.data[keep_ids].clone()
                module.gate = new_gate

            # Update runtime attributes so saved config matches structure.
            module.num_experts = len(keep_ids)
            module.gate_num_experts = len(keep_ids)
            module.top_k = min(getattr(module, "top_k", len(keep_ids)), len(keep_ids))
            module.router_logits_mask = None  # mask buffer is non-persistent; we materialize via pruning instead.

        # Update config so reloading builds matching modules.
        cfg = getattr(unwrapped, "config", None)
        if cfg is not None:
            num_layers = getattr(cfg, "num_hidden_layers", len(getattr(unwrapped.model, "layers", [])))
            # start from original config to preserve non-MoE layers if list provided
            def _init_list_from_cfg(field_name, fallback):
                val = getattr(cfg, field_name, None)
                if isinstance(val, list) and len(val) > 0:
                    # pad to num_layers if needed
                    if len(val) < num_layers:
                        val = val + [val[-1]] * (num_layers - len(val))
                    return list(val)
                return [fallback] * num_layers

            # original expert count (assuming uniform if int)
            orig_expert_default = None
            if isinstance(getattr(cfg, "num_experts", None), int):
                orig_expert_default = cfg.num_experts
            elif isinstance(getattr(cfg, "num_experts", None), list) and cfg.num_experts:
                orig_expert_default = cfg.num_experts[0]
            else:
                orig_expert_default = 1

            num_experts_list = _init_list_from_cfg("num_experts", orig_expert_default)
            gate_num_experts_list = _init_list_from_cfg("gate_num_experts", orig_expert_default)

            for layer_id, keep_ids in per_layer_kept.items():
                if layer_id is None or layer_id >= len(num_experts_list):
                    continue
                num_experts_list[layer_id] = len(keep_ids)
                gate_num_experts_list[layer_id] = len(keep_ids)

            cfg.num_experts = num_experts_list
            cfg.gate_num_experts = gate_num_experts_list

            if hasattr(cfg, "num_experts_per_tok") and isinstance(cfg.num_experts_per_tok, int):
                valid_counts = [c for c in num_experts_list if isinstance(c, int) and c > 0]
                if valid_counts:
                    cfg.num_experts_per_tok = min(cfg.num_experts_per_tok, min(valid_counts))

            cfg.router_mask_dropped = {str(item["layer"]): item["dropped"] for item in summary}
            cfg.router_mask_kept = {str(item["layer"]): item["kept"] for item in summary}
            # Persist the kept expert indices per layer so the saved config mirrors the pruned layout.
            existing_layer_experts_idx = getattr(cfg, "layer_experts_idx", None)
            if isinstance(existing_layer_experts_idx, list):
                layer_experts_idx = list(existing_layer_experts_idx)
                if len(layer_experts_idx) < num_layers:
                    layer_experts_idx += [None] * (num_layers - len(layer_experts_idx))
                else:
                    layer_experts_idx = layer_experts_idx[:num_layers]
            else:
                layer_experts_idx = [None] * num_layers

            for module in moe_blocks:
                layer_id = getattr(module, "layer_idx", None)
                if layer_id is None or layer_id >= num_layers:
                    continue
                if layer_id in per_layer_kept:
                    layer_experts_idx[layer_id] = list(per_layer_kept[layer_id])
                elif layer_experts_idx[layer_id] is None:
                    gate = getattr(module, "gate", None)
                    out_features = gate.out_features if gate is not None else len(getattr(module, "experts", []))
                    layer_experts_idx[layer_id] = list(range(out_features))

            # Fill any remaining gaps with the implicit full range for layers that still have MoE experts.
            for layer_id in range(num_layers):
                if layer_experts_idx[layer_id] is None:
                    experts_count = num_experts_list[layer_id] if layer_id < len(num_experts_list) else None
                    if isinstance(experts_count, int) and experts_count > 0:
                        layer_experts_idx[layer_id] = list(range(experts_count))

            cfg.layer_experts_idx = layer_experts_idx
            # Do not persist gate_num_experts in the saved config (use num_experts instead).
            if "gate_num_experts" in cfg.__dict__:
                cfg.__dict__.pop("gate_num_experts", None)
            # Persist dtype for reload parity.
            try:
                param_dtype = next(unwrapped.parameters()).dtype
                cfg.dtype = str(param_dtype).replace("torch.", "")
            except Exception:
                pass

        return {"applied": True, "reason": None, "summary": summary, "num_experts": per_layer_kept}

    # ── snapshot / restore helpers for _save_checkpoint ──

    _MOE_CONFIG_ATTRS = (
        "num_experts", "gate_num_experts", "num_experts_per_tok",
        "router_mask_dropped", "router_mask_kept", "layer_experts_idx", "dtype",
    )
    _NOT_SET = object()

    def _snapshot_moe_state(self):
        """Snapshot MoE module state so we can restore after materialization."""
        unwrapped = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else getattr(
            self.model, "module", self.model
        )
        blocks = []
        for module in unwrapped.modules():
            if hasattr(module, "_set_router_logits_mask") and hasattr(module, "gate"):
                blocks.append({
                    "module": module,
                    "experts": module.experts,
                    "gate": module.gate,
                    "num_experts": getattr(module, "num_experts", None),
                    "gate_num_experts": getattr(module, "gate_num_experts", None),
                    "top_k": getattr(module, "top_k", None),
                    "router_logits_mask": getattr(module, "router_logits_mask", None),
                })
        cfg = getattr(unwrapped, "config", None)
        cfg_snap = {}
        if cfg is not None:
            for attr in self._MOE_CONFIG_ATTRS:
                val = getattr(cfg, attr, self._NOT_SET)
                if val is not self._NOT_SET:
                    if isinstance(val, list):
                        val = list(val)
                    elif isinstance(val, dict):
                        val = dict(val)
                    cfg_snap[attr] = val
        return {"blocks": blocks, "cfg": cfg_snap, "config_obj": cfg}

    def _restore_moe_state(self, snapshot):
        """Restore MoE modules and config from snapshot after checkpoint save."""
        for info in snapshot["blocks"]:
            m = info["module"]
            m.experts = info["experts"]
            m.gate = info["gate"]
            if info["num_experts"] is not None:
                m.num_experts = info["num_experts"]
            if info["gate_num_experts"] is not None:
                m.gate_num_experts = info["gate_num_experts"]
            if info["top_k"] is not None:
                m.top_k = info["top_k"]
            m.router_logits_mask = info["router_logits_mask"]
        cfg = snapshot.get("config_obj")
        if cfg is not None:
            for attr, val in snapshot["cfg"].items():
                setattr(cfg, attr, val)

    def _save_checkpoint(self, model, trial, **kwargs):
        """Override to materialize masks before saving, then restore for continued training."""
        if self.args.should_save:
            snapshot = self._snapshot_moe_state()
            mask_info = self._materialize_router_masks()
            self._last_mask_save_info = mask_info
            try:
                super()._save_checkpoint(model, trial, **kwargs)
            finally:
                self._restore_moe_state(snapshot)
        else:
            super()._save_checkpoint(model, trial, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Materialize masks for the final save (end of training)."""
        if self.args.should_save and not _internal_call:
            snapshot = self._snapshot_moe_state()
            mask_info = self._materialize_router_masks()
            self._last_mask_save_info = mask_info
            if mask_info and not mask_info.get("applied", False):
                reason = mask_info.get("reason", "unknown")
                print(f"[RouterMaskSave] Skip pruning: {reason}")
            try:
                return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
            finally:
                self._restore_moe_state(snapshot)

        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
        **kwargs,
    ):
        """
        Wrap BaseTrainer.create_model_card without forwarding unsupported kwargs.
        The upstream TRL BaseTrainer only accepts model_name, dataset_name and tags,
        so swallow any extra fields (e.g. language, license) to avoid TypeErrors.
        """
        ds_name = dataset_name
        if ds_name is None:
            ds = kwargs.get("dataset")
            if isinstance(ds, str):
                ds_name = ds

        super().create_model_card(
            model_name=model_name,
            dataset_name=ds_name or kwargs.get("dataset_name"),
            tags=tags,
        )

        if not self.is_world_process_zero():
            return

        readme_path = os.path.join(self.args.output_dir, "README.md")
        mask_info = getattr(self, "_last_mask_save_info", None)
        if mask_info is None or not os.path.exists(readme_path):
            return

        lines = []
        lines.append("\n## Router mask / pruned experts\n")
        applied = mask_info.get("applied", False)
        if applied:
            per_layer_counts = mask_info.get("num_experts")
            if isinstance(per_layer_counts, dict):
                lines.append("- Mask materialized: per-layer expert counts recorded below.")
            else:
                lines.append(f"- Mask materialized: kept {per_layer_counts} experts per MoE layer.")
            for item in mask_info.get("summary", []):
                layer = item.get("layer")
                dropped = item.get("dropped", [])
                kept = item.get("kept", [])
                lines.append(f"- layer {layer}: kept {kept}; dropped {dropped}")
        else:
            lines.append(f"- Mask not materialized (reason: {mask_info.get('reason', 'unknown')}).")

        with open(readme_path, "a") as f:
            f.write("\n".join(lines) + "\n")
