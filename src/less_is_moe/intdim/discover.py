"""Structural discovery of Mixture-of-Experts expert weights.

Finds every MoE layer in a Hugging Face model from parameter *shapes*, not from
class or attribute names, so the same code covers Qwen1.5/Qwen3/Qwen3.5-MoE,
OLMoE, gpt-oss, and Gemma-4 without a per-family module.

Two storage layouts are recognised:

``fused``
    One module holding batched 3D parameters, ``gate_up`` with shape
    ``(E, 2I, H)`` or ``(E, H, 2I)`` and ``down`` with shape ``(E, H, I)`` or
    ``(E, I, H)``, optionally with biases ``(E, 2I)`` and ``(E, H)``.
    Used by Qwen3.5-MoE, Gemma-4, and gpt-oss.
``modulelist``
    An ``nn.ModuleList`` of ``E`` identical expert MLPs, each with three
    ``nn.Linear`` layers ``gate (I, H)``, ``up (I, H)``, ``down (H, I)``.
    Used by Qwen1.5/Qwen3-MoE and OLMoE on Transformers 4.x.

Shapes cannot distinguish a *concatenated* ``gate_up`` (gate rows then up rows)
from an *interleaved* one (alternating rows), and when ``I == H`` they cannot say
which ``down`` axis is the intermediate one. :func:`probe_fused_layout` settles
both empirically: routing a few tokens through one expert, zeroing intermediate
unit ``j`` in ``down`` must change the output exactly as zeroing its gate and up
rows does. A wrong pairing removes rows that belong to different units and
fails the check, so a mistaken layout raises instead of silently pruning
unpaired rows.
"""

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal

import torch
from torch import nn

from .registry import (
    DEFAULT_PAIRING,
    EXPERT_COUNT_KEYS,
    INTERMEDIATE_SIZE_ATTRS,
    INTERMEDIATE_SIZE_KEYS,
    LAYOUT_OVERRIDES,
    LINEAR_NAME_SETS,
    TOP_K_KEYS,
)

Pairing = Literal["concat", "interleaved"]
Kind = Literal["fused", "modulelist"]

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class DiscoveryError(RuntimeError):
    """Raised when a model's MoE structure cannot be identified unambiguously."""


@dataclass(frozen=True)
class ConfigKey:
    """Where a discovered quantity lives in the Hugging Face config."""

    scope: str  # "" for the root config, otherwise the nested attribute name
    name: str

    def get(self, config: Any) -> int:
        return getattr(self._target(config), self.name)

    def set(self, config: Any, value: int) -> None:
        setattr(self._target(config), self.name, value)

    def _target(self, config: Any) -> Any:
        return config if not self.scope else getattr(config, self.scope)

    def __str__(self) -> str:
        return f"{self.scope}.{self.name}" if self.scope else self.name


@dataclass(frozen=True)
class FusedLayout:
    """Axis conventions of a fused expert container."""

    gate_up_unit_axis: int  # axis of ``gate_up`` with size 2I
    down_unit_axis: int  # axis of ``down`` with size I
    pairing: Pairing



@dataclass
class MoeLayerHandle:
    """One MoE layer's expert weights, addressed by intermediate unit.

    ``select_units`` and ``apply_units`` are the only two operations a
    structural pruner needs; scoring reads ``expert_weights``.
    """

    name: str
    layer_index: int | None
    experts: nn.Module
    kind: Kind
    num_experts: int
    intermediate_size: int
    hidden_size: int
    expert_count_key: ConfigKey | None
    intermediate_size_key: ConfigKey | None
    top_k_key: ConfigKey | None
    fused: FusedLayout | None = None
    linear_names: tuple[str, str, str] | None = None

    # ------------------------------------------------------------------ fused
    @property
    def gate_up(self) -> nn.Parameter:
        dims = sorted((2 * self.intermediate_size, self.hidden_size))
        return self._fused_param(lambda p: p.ndim == 3 and sorted(p.shape[1:]) == dims)

    @property
    def down(self) -> nn.Parameter:
        return self._fused_param(lambda p: p.ndim == 3 and sorted(p.shape[1:]) == sorted((self.intermediate_size, self.hidden_size)))

    @property
    def gate_up_bias(self) -> nn.Parameter | None:
        return self._fused_param(lambda p: p.ndim == 2 and p.shape[1] == 2 * self.intermediate_size, optional=True, name_hint=("gate", "up"))

    @property
    def down_bias(self) -> nn.Parameter | None:
        return self._fused_param(lambda p: p.ndim == 2 and p.shape[1] == self.hidden_size, optional=True, name_hint=("down",))

    def _fused_param(self, predicate, optional: bool = False, name_hint: tuple[str, ...] = ()) -> nn.Parameter | None:
        matches = [(n, p) for n, p in self.experts.named_parameters(recurse=False) if p.shape[0] == self.num_experts and predicate(p)]
        if len(matches) > 1 and name_hint:
            # When 2I == H the two bias vectors have the same shape; fall back to the parameter name.
            matches = [(n, p) for n, p in matches if any(h in n for h in name_hint)]
        if len(matches) == 1:
            return matches[0][1]
        if optional and not matches:
            return None
        raise DiscoveryError(f"{self.name}: expected one matching fused parameter, found {[n for n, _ in matches]}")

    def gate_up_indices(self, units: torch.Tensor) -> torch.Tensor:
        """Indices along the ``gate_up`` unit axis covering gate and up rows of ``units``.

        ``units`` may be 1D (one index set) or ``(E, K)`` (one set per expert);
        the result has one extra trailing factor of 2 and is sorted per row.
        """
        assert self.fused is not None
        units = torch.as_tensor(units, dtype=torch.long)
        if self.fused.pairing == "concat":
            rows = torch.cat([units, units + self.intermediate_size], dim=-1)
        else:
            rows = torch.cat([2 * units, 2 * units + 1], dim=-1)
        return torch.sort(rows, dim=-1).values

    # ------------------------------------------------------------- modulelist
    def expert_linears(self, expert: int) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        assert self.linear_names is not None
        module = self.experts[expert]
        return tuple(getattr(module, n) for n in self.linear_names)  # type: ignore[return-value]

    # ---------------------------------------------------------------- generic
    def expert_weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gate (I, H), up (I, H), down (H, I))`` views for one expert."""
        if self.kind == "modulelist":
            gate, up, down = self.expert_linears(expert)
            return gate.weight, up.weight, down.weight
        assert self.fused is not None
        gu = self.gate_up[expert]
        if self.fused.gate_up_unit_axis == 2:
            gu = gu.transpose(0, 1)  # -> (2I, H)
        if self.fused.pairing == "concat":
            gate, up = gu[: self.intermediate_size], gu[self.intermediate_size :]
        else:
            gate, up = gu[0::2], gu[1::2]
        down = self.down[expert]
        if self.fused.down_unit_axis == 1:
            down = down.transpose(0, 1)  # -> (H, I)
        return gate, up, down

    def _per_expert(self, keep: torch.Tensor) -> torch.Tensor:
        keep = torch.as_tensor(keep)
        if keep.dtype == torch.bool or keep.is_floating_point() or keep.is_complex():
            raise ValueError("keep must contain integer unit indices")
        keep = keep.to(dtype=torch.long)
        if keep.ndim == 1:
            keep = keep.unsqueeze(0).expand(self.num_experts, -1)
        if keep.ndim != 2 or keep.shape[0] != self.num_experts:
            raise ValueError(f"keep must be 1D or (E={self.num_experts}, K), got {tuple(keep.shape)}")
        if keep.shape[1] == 0 or torch.any(keep < 0) or torch.any(keep >= self.intermediate_size):
            raise ValueError("keep must contain at least one in-range unit per expert")
        # A neuron selection is a set. All three projections must use the same
        # order; gate_up_indices already sorts the paired gate/up rows.
        keep = keep.sort(dim=-1).values
        if torch.any(keep[:, 1:] == keep[:, :-1]):
            raise ValueError("keep must not contain duplicate units")
        return keep

    def select_units(self, keep: torch.Tensor) -> dict[str, torch.Tensor]:
        """New expert tensors that retain only intermediate units ``keep``.

        ``keep`` is ``(E, K)``: expert ``e`` keeps units ``keep[e]``. A 1D
        ``keep`` applies the same units to every expert. Every expert keeps
        the same *count* ``K``, which is what ``IntDim-E`` needs for a
        structural (uniform-width) checkpoint; the units themselves differ per
        expert. Indices are normalized to ascending order. Returned tensors are detached copies keyed by parameter name
        (fused) or ``"{expert}.{gate|up|down}"`` (modulelist).
        """
        keep = self._per_expert(keep)
        out: dict[str, torch.Tensor] = {}
        if self.kind == "modulelist":
            for e in range(self.num_experts):
                gate, up, down = self.expert_linears(e)
                idx = keep[e].to(gate.weight.device)
                out[f"{e}.gate"] = gate.weight.detach()[idx].clone()
                out[f"{e}.up"] = up.weight.detach()[idx].clone()
                out[f"{e}.down"] = down.weight.detach()[:, idx].clone()
            return out
        assert self.fused is not None
        gate_up, down = self.gate_up, self.down
        rows = self.gate_up_indices(keep).to(gate_up.device)  # (E, 2K)
        keep = keep.to(down.device)
        names = {p: n for n, p in self.experts.named_parameters(recurse=False)}
        out[names[gate_up]] = _gather_units(gate_up.detach(), self.fused.gate_up_unit_axis, rows)
        out[names[down]] = _gather_units(down.detach(), self.fused.down_unit_axis, keep)
        if (bias := self.gate_up_bias) is not None:
            out[names[bias]] = bias.detach().gather(1, rows.to(bias.device)).clone()
        return out

    def apply_units(self, keep: torch.Tensor) -> None:
        """Shrink the expert modules in place to the units in ``keep``.

        Updates module attributes that mirror the intermediate size. The
        config is *not* updated; use ``intermediate_size_key.set``.
        """
        keep = self._per_expert(keep)
        new = self.select_units(keep)
        width = int(keep.shape[1])
        if self.kind == "modulelist":
            for e in range(self.num_experts):
                module = self.experts[e]
                for role, name in zip(("gate", "up", "down"), self.linear_names or ()):
                    old = getattr(module, name)
                    weight = new[f"{e}.{role}"]
                    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=old.bias is not None, device=weight.device, dtype=weight.dtype)
                    linear.weight = nn.Parameter(weight)
                    if old.bias is not None:
                        bias = old.bias.detach()
                        linear.bias = nn.Parameter(bias.clone() if role == "down" else bias[keep[e].to(bias.device)].clone())
                    setattr(module, name, linear)
                _set_size_attrs(module, width)
        else:
            for name, tensor in new.items():
                setattr(self.experts, name, nn.Parameter(tensor, requires_grad=getattr(self.experts, name).requires_grad))
        _set_size_attrs(self.experts, width)
        self.intermediate_size = width


def _gather_units(tensor: torch.Tensor, axis: int, index: torch.Tensor) -> torch.Tensor:
    """Gather per-expert unit indices ``index`` (E, K) along ``axis`` (1 or 2) of an (E, ., .) tensor."""
    index = index.to(tensor.device)
    if axis == 1:
        expanded = index[:, :, None].expand(-1, -1, tensor.shape[2])
    else:
        expanded = index[:, None, :].expand(-1, tensor.shape[1], -1)
    return tensor.gather(axis, expanded).clone()


def _set_size_attrs(module: nn.Module, width: int) -> None:
    for attr in INTERMEDIATE_SIZE_ATTRS:
        if isinstance(getattr(module, attr, None), int):
            setattr(module, attr, width)


# --------------------------------------------------------------------- config

def _config_scopes(config: Any) -> list[tuple[str, Any]]:
    scopes = [("", config)]
    text = getattr(config, "text_config", None)
    if text is not None and text is not config:
        scopes.append(("text_config", text))
    return scopes


def _find_config_key(config: Any, names: Iterable[str], value: int) -> ConfigKey | None:
    for scope, target in _config_scopes(config):
        for name in names:
            found = getattr(target, name, None)
            if isinstance(found, int) and not isinstance(found, bool) and found == value:
                return ConfigKey(scope, name)
    return None


def _first_config_key(config: Any, names: Iterable[str]) -> ConfigKey | None:
    for scope, target in _config_scopes(config):
        for name in names:
            if isinstance(getattr(target, name, None), int):
                return ConfigKey(scope, name)
    return None


def _hidden_size(config: Any) -> int:
    for _, target in _config_scopes(config):
        value = getattr(target, "hidden_size", None)
        if isinstance(value, int):
            return value
    raise DiscoveryError("config has no hidden_size")


def _model_type(config: Any) -> str:
    for _, target in _config_scopes(config):
        value = getattr(target, "model_type", None)
        if value:
            return str(value).removesuffix("_text")
    return ""


# ------------------------------------------------------------------ discovery

def _match_fused(module: nn.Module, hidden: int) -> tuple[int, int, int | None, int | None] | None:
    """Return ``(E, I, gate_up_unit_axis, down_unit_axis)`` if ``module`` holds fused experts.

    An axis is ``None`` when shapes cannot identify it: ``gate_up`` is square
    when ``2I == H`` (gpt-oss pruned to half width), and ``down`` is square when
    ``I == H`` (unpruned gpt-oss). :func:`probe_fused_layout` resolves both.
    """
    params = [p for p in module.parameters(recurse=False) if p.ndim == 3]
    if len(params) < 2:
        return None
    counts = {p.shape[0] for p in params}
    if len(counts) != 1:
        return None
    (num_experts,) = counts
    if num_experts < 2:
        return None
    for gu in params:
        dims = list(gu.shape[1:])
        if hidden not in dims:
            continue
        two_i = dims[1] if dims[0] == hidden else dims[0]
        if two_i % 2:
            continue
        inter = two_i // 2
        for down in params:
            if down is gu or sorted(down.shape[1:]) != sorted((hidden, inter)):
                continue
            gu_axis = None if dims[0] == dims[1] else (1 if gu.shape[1] == two_i else 2)
            down_axis = None if inter == hidden else (1 if down.shape[1] == inter else 2)
            return num_experts, inter, gu_axis, down_axis
    return None


def _match_modulelist(module: nn.Module, hidden: int) -> tuple[int, int, tuple[str, str, str]] | None:
    if not isinstance(module, nn.ModuleList) or len(module) < 2:
        return None
    first = module[0]
    if not all(type(child) is type(first) for child in module):
        return None
    for names in LINEAR_NAME_SETS:
        linears = [getattr(first, n, None) for n in names]
        if not all(isinstance(l, nn.Linear) for l in linears):
            continue
        gate, up, down = linears
        inter = gate.out_features
        if gate.in_features != hidden or up.weight.shape != (inter, hidden) or down.weight.shape != (hidden, inter):
            continue
        if all(getattr(c, names[0]).weight.shape == (inter, hidden) for c in module):
            return len(module), inter, names
    return None


def discover(model: nn.Module, config: Any | None = None, probe: bool = True) -> list[MoeLayerHandle]:
    """Find every MoE layer in ``model``.

    ``config`` defaults to ``model.config``. With ``probe=True`` fused layouts
    are verified (and resolved when unknown) by :func:`probe_fused_layout`,
    which runs a few forward passes through one expert per layer.
    """
    config = model.config if config is None else config
    hidden = _hidden_size(config)
    model_type = _model_type(config)
    override = LAYOUT_OVERRIDES.get(model_type, {})
    handles: list[MoeLayerHandle] = []
    claimed: set[str] = set()

    for name, module in model.named_modules():
        if any(name.startswith(prefix + ".") for prefix in claimed):
            continue
        fused = _match_fused(module, hidden)
        listed = None if fused else _match_modulelist(module, hidden)
        if fused is None and listed is None:
            continue
        claimed.add(name)
        match = _LAYER_INDEX.search(name)
        layer_index = int(match.group(1)) if match else None
        if fused is not None:
            num_experts, inter, gu_axis, down_axis = fused
            # Transformers 5.x records the layout on wrapped expert classes; use it as a hint only.
            hinted = getattr(module, "is_concatenated", None)
            preferred = override.get("pairing") or ({True: "concat", False: "interleaved"}.get(hinted) if isinstance(hinted, bool) else None)
            handle = MoeLayerHandle(
                name=name, layer_index=layer_index, experts=module, kind="fused",
                num_experts=num_experts, intermediate_size=inter, hidden_size=hidden,
                expert_count_key=_find_config_key(config, EXPERT_COUNT_KEYS, num_experts),
                intermediate_size_key=_find_config_key(config, INTERMEDIATE_SIZE_KEYS, inter),
                top_k_key=_first_config_key(config, TOP_K_KEYS),
                fused=FusedLayout(gu_axis or 1, down_axis or 2, preferred or DEFAULT_PAIRING),  # type: ignore[arg-type]
            )
            if probe:
                handle.fused = probe_fused_layout(
                    handle, known_gate_up_axis=gu_axis, known_down_axis=down_axis, preferred_pairing=preferred,
                )
            elif gu_axis is None or down_axis is None:
                raise DiscoveryError(f"{name}: square expert tensors make the unit axis ambiguous; run with probe=True")
        else:
            num_experts, inter, names = listed  # type: ignore[misc]
            handle = MoeLayerHandle(
                name=name, layer_index=layer_index, experts=module, kind="modulelist",
                num_experts=num_experts, intermediate_size=inter, hidden_size=hidden,
                expert_count_key=_find_config_key(config, EXPERT_COUNT_KEYS, num_experts),
                intermediate_size_key=_find_config_key(config, INTERMEDIATE_SIZE_KEYS, inter),
                top_k_key=_first_config_key(config, TOP_K_KEYS),
                linear_names=names,
            )
        if handle.intermediate_size_key is None:
            raise DiscoveryError(
                f"{name}: no config attribute equals the discovered intermediate size {inter}; "
                f"add its name to registry.INTERMEDIATE_SIZE_KEYS"
            )
        handles.append(handle)

    if not handles:
        raise DiscoveryError("no MoE expert containers found")
    return handles


# ---------------------------------------------------------------------- probe

@contextmanager
def _eager_experts(experts: nn.Module):
    """Run ``experts`` with its reference Python forward.

    Transformers 5.x dispatches wrapped expert classes to fused kernels
    (``grouped_mm``, ``batched_mm``) via ``config._experts_implementation``.
    The probe needs the plain per-expert math, and those kernels may not run on
    the probe's device, so switch to ``eager`` and restore afterwards.
    """
    config = getattr(experts, "config", None)
    previous = getattr(config, "_experts_implementation", None) if config is not None else None
    if previous is not None:
        config._experts_implementation = "eager"
    try:
        yield
    finally:
        if previous is not None:
            config._experts_implementation = previous


def _run_expert(experts: nn.Module, x: torch.Tensor, num_tokens: int, expert: int = 0) -> torch.Tensor:
    index = torch.full((num_tokens, 1), expert, dtype=torch.long, device=x.device)
    weights = torch.ones(num_tokens, 1, dtype=x.dtype, device=x.device)
    try:
        with _eager_experts(experts):
            out = experts(x, index, weights)
    except TypeError as exc:
        raise DiscoveryError(
            f"cannot call {type(experts).__name__}.forward(hidden_states, top_k_index, top_k_weights) "
            f"for the layout probe: {exc}"
        ) from exc
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def _consistent(handle: MoeLayerHandle, layout: FusedLayout, num_tokens: int, seed: int, max_units: int = 8) -> bool:
    """Whether zeroing a unit's gate/up rows matches zeroing its down column, under ``layout``.

    Only *live* units are informative: in a zero-masked checkpoint half the
    units may already be zero, and removing a dead unit changes nothing under
    any hypothesis. Up to ``max_units`` units whose hypothesised gate row is
    non-zero are tested; every informative unit must agree, and at least one
    must be informative.
    """
    gate_up = handle.gate_up
    inter = handle.intermediate_size

    def rows_of(unit: torch.Tensor) -> torch.Tensor:
        if layout.pairing == "concat":
            return torch.cat([unit, unit + inter])
        return torch.cat([2 * unit, 2 * unit + 1])

    # Probe the expert with the most live units under this hypothesis: a zero-mask
    # with the layer or global scope can remove every unit of some experts.
    gu = gate_up if layout.gate_up_unit_axis == 1 else gate_up.transpose(1, 2)  # (E, 2I, H)
    gates = gu[:, :inter] if layout.pairing == "concat" else gu[:, 0::2]
    # Bound temporary memory to one expert, including for large E/H/I.
    live_counts = torch.stack([gate.ne(0).any(dim=1).sum() for gate in gates])
    expert = int(live_counts.argmax())
    live = torch.nonzero(gates[expert].ne(0).any(dim=1), as_tuple=False).flatten().cpu()
    if live.numel() == 0:
        return False
    picks = live[torch.linspace(0, live.numel() - 1, min(max_units, live.numel())).round().long().unique()]

    # Keep #27's GPU precision and exception safety, combined with #26's
    # live-unit selection. Only the chosen expert is copied, in FP32 on GPU.
    memo = {}
    for parameter in handle.experts.parameters():
        value = parameter.detach()
        if value.ndim >= 2 and value.shape[0] == handle.num_experts:
            value = value[expert:expert + 1]
        memo[id(parameter)] = nn.Parameter(value.to(dtype=torch.float32).clone(), requires_grad=False)
    probe = copy.deepcopy(handle.experts, memo)
    for attr in EXPERT_COUNT_KEYS:
        if getattr(probe, attr, None) == handle.num_experts:
            setattr(probe, attr, 1)
    handle = replace(handle, experts=probe, num_experts=1)
    experts = handle.experts
    gate_up, down, bias = handle.gate_up, handle.down, handle.gate_up_bias
    expert = 0
    device = gate_up.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, handle.hidden_size, generator=gen).to(device=device)
    tol = 1e-6
    baseline = _run_expert(experts, x, num_tokens, expert).clone()
    informative = 0
    for unit in picks.tolist():
        u = torch.tensor([unit])
        rows = rows_of(u).to(device)

        saved = gate_up[expert].clone()
        saved_bias = bias[expert].clone() if bias is not None else None
        gate_up[expert].index_fill_(layout.gate_up_unit_axis - 1, rows, 0)
        if bias is not None:
            bias[expert].index_fill_(0, rows.to(bias.device), 0)
        zero_gate_up = _run_expert(experts, x, num_tokens, expert).clone()
        gate_up[expert].copy_(saved)
        if bias is not None:
            bias[expert].copy_(saved_bias)

        saved = down[expert].clone()
        down[expert].index_fill_(layout.down_unit_axis - 1, u.to(device), 0)
        zero_down = _run_expert(experts, x, num_tokens, expert).clone()
        down[expert].copy_(saved)

        down_changed = bool((zero_down - baseline).abs().max() > tol)
        gate_up_changed = bool((zero_gate_up - baseline).abs().max() > tol)
        if not (down_changed or gate_up_changed):
            continue
        informative += 1
        if not torch.allclose(zero_gate_up, zero_down, atol=tol, rtol=1e-4):
            return False
    return informative > 0


def probe_fused_layout(
    handle: MoeLayerHandle,
    *,
    known_gate_up_axis: int | None = None,
    known_down_axis: int | None = None,
    preferred_pairing: Pairing | None = None,
    num_tokens: int = 4,
    seed: int = 0,
) -> FusedLayout:
    """Verify or resolve the fused layout of ``handle`` empirically.

    Both pairings are tried; an axis is tried both ways only when square
    tensors leave it ambiguous. Exactly one hypothesis must pass, otherwise a
    :class:`DiscoveryError` lists which did.
    """
    pairings: list[Pairing] = ["concat", "interleaved"]
    if preferred_pairing in pairings:
        pairings.remove(preferred_pairing)
        pairings.insert(0, preferred_pairing)
    gate_up_axes = [known_gate_up_axis] if known_gate_up_axis is not None else [
        axis for axis in (1, 2) if handle.gate_up.shape[axis] == 2 * handle.intermediate_size
    ]
    down_axes = [known_down_axis] if known_down_axis is not None else [
        axis for axis in (1, 2) if handle.down.shape[axis] == handle.intermediate_size
    ]
    hypotheses = [FusedLayout(g, d, p) for p in pairings for g in gate_up_axes for d in down_axes]
    passed = [layout for layout in hypotheses if _consistent(handle, layout, num_tokens, seed)]
    if len(passed) == 1:
        return passed[0]
    describe_layout = lambda l: (l.pairing, l.gate_up_unit_axis, l.down_unit_axis)  # noqa: E731
    raise DiscoveryError(
        f"{handle.name}: layout probe found {len(passed)} consistent layouts out of "
        f"{[describe_layout(l) for l in hypotheses]}; passed={[describe_layout(l) for l in passed]}. "
        "Register this family's layout in registry.LAYOUT_OVERRIDES."
    )


def describe(handles: list[MoeLayerHandle]) -> str:
    lines = []
    for h in handles:
        layout = f"{h.kind}"
        if h.fused:
            layout += f"(pairing={h.fused.pairing}, gate_up_axis={h.fused.gate_up_unit_axis}, down_axis={h.fused.down_unit_axis})"
        else:
            layout += f"(linears={h.linear_names})"
        lines.append(
            f"layer {h.layer_index!s:>3} {h.name}: E={h.num_experts} I={h.intermediate_size} H={h.hidden_size} "
            f"{layout} keys: E={h.expert_count_key} I={h.intermediate_size_key} top_k={h.top_k_key}"
        )
    return "\n".join(lines)
