# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal next-layer MoE gate prediction with optional LoRA correction.

This module attaches side-channel predictors to ``MoERunner`` instances:

    current layer hidden states -> next layer gate -> optional LoRA delta
    -> next layer router._compute_routing()

The real inference path still uses the original router logits that are passed
to ``BaseRouter.select_experts`` by the model's MoE runner. Predicted routes
are cached separately on each runner and are never fed into the fused expert
kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

NextGatePredictor = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class NextGatePredictionBuildResult:
    predictors: dict[int, tuple[int, NextGatePredictor]]
    missing_gate_layers: list[tuple[int, int]]
    diagnostics: list[dict[str, Any]]
    lora_dir: str | None = None


def _gate_candidates(next_name: str) -> list[str]:
    candidates = []
    if next_name.endswith(".experts"):
        candidates.append(next_name.removesuffix(".experts") + ".gate")
    if ".experts" in next_name:
        candidates.append(next_name.rsplit(".experts", 1)[0] + ".gate")
    # Qwen-style layer names are commonly model.layers.N.mlp.experts.
    parts = next_name.split(".")
    if "layers" in parts:
        layer_pos = parts.index("layers")
        if layer_pos + 1 < len(parts):
            layer_id = parts[layer_pos + 1]
            candidates.append(f"model.layers.{layer_id}.mlp.gate")
            candidates.append(f"layers.{layer_id}.mlp.gate")
    return list(dict.fromkeys(candidates))


def _model_lookup_helpers(
    model: torch.nn.Module | None,
) -> tuple[
    Callable[[str], torch.nn.Module | None],
    Callable[[str], torch.nn.Parameter | None],
]:
    model_modules = dict(model.named_modules()) if model is not None else {}
    model_parameters = dict(model.named_parameters()) if model is not None else {}

    def _get_module_by_name(name: str) -> torch.nn.Module | None:
        if model is None:
            return None
        if name == "":
            return model
        if name in model_modules:
            return model_modules[name]
        if name.startswith("model."):
            stripped_name = name.removeprefix("model.")
            if stripped_name in model_modules:
                return model_modules[stripped_name]

        suffix_matches = [
            module
            for module_name, module in model_modules.items()
            if module_name.endswith(f".{name}") or name.endswith(f".{module_name}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    def _find_named_parameter(name: str) -> torch.nn.Parameter | None:
        if name in model_parameters:
            return model_parameters[name]
        if name.startswith("model."):
            stripped_name = name.removeprefix("model.")
            if stripped_name in model_parameters:
                return model_parameters[stripped_name]
        suffix_matches = [
            parameter
            for parameter_name, parameter in model_parameters.items()
            if (
                parameter_name.endswith(f".{name}")
                or name.endswith(f".{parameter_name}")
            )
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    return _get_module_by_name, _find_named_parameter


def _find_next_gate_projector(
    next_name: str,
    next_module: FusedMoE,
    get_module_by_name: Callable[[str], torch.nn.Module | None],
    find_named_parameter: Callable[[str], torch.nn.Parameter | None],
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, str]:
    runner = next_module.runner
    gate = getattr(runner, "gate", None)
    if gate is not None:
        if bool(getattr(runner, "_fse_fuse_gate", False)):

            def _project_with_fused_gate(hidden_states: torch.Tensor):
                runner._maybe_fuse_gate_weights()
                return torch.nn.functional.linear(
                    hidden_states, runner._combined_gate_weight
                )

            return _project_with_fused_gate, "runner.fused_gate"

        def _project_with_runner_gate(hidden_states: torch.Tensor):
            router_logits, _ = gate(hidden_states)
            return router_logits

        return _project_with_runner_gate, "runner.gate"

    for gate_name in _gate_candidates(next_name):
        parent = get_module_by_name(gate_name.rsplit(".", 1)[0])
        gate = getattr(parent, "gate", None) if parent is not None else None
        if gate is not None:

            def _project_with_parent_gate(
                hidden_states: torch.Tensor,
                _gate=gate,
            ):
                router_logits, _ = _gate(hidden_states)
                return router_logits

            return _project_with_parent_gate, gate_name

        weight = find_named_parameter(gate_name + ".weight")
        if weight is not None:
            bias = find_named_parameter(gate_name + ".bias")

            def _project_with_gate_weight(
                hidden_states: torch.Tensor,
                _weight=weight,
                _bias=bias,
            ):
                if hidden_states.dtype != _weight.dtype:
                    hidden_states = hidden_states.to(_weight.dtype)
                return torch.nn.functional.linear(hidden_states, _weight, _bias)

            return _project_with_gate_weight, gate_name + ".weight"

    return None, ""


def _load_lora_for_pair(
    lora_dir: str | None,
    source_layer_id: int,
    next_layer_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float] | None:
    if not lora_dir:
        return None
    root = Path(lora_dir).expanduser()
    candidates = [
        root / f"layer_{source_layer_id:04d}_to_{next_layer_id:04d}.pt",
        root / f"layer_{next_layer_id:04d}.pt",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    payload = torch.load(path, map_location=device, weights_only=True)
    lora_a = payload["lora_A"].to(device=device, dtype=dtype)
    lora_b = payload["lora_B"].to(device=device, dtype=dtype)
    rank = int(payload.get("rank", lora_a.shape[0]))
    alpha = float(payload.get("alpha", rank))
    return {
        "lora_A": lora_a,
        "lora_B": lora_b,
        "scale": alpha / rank,
    }


def _make_predictor(
    next_module: FusedMoE,
    gate_projector: Callable[[torch.Tensor], torch.Tensor],
    lora_payload: dict[str, torch.Tensor | float] | None,
) -> NextGatePredictor:
    @torch.no_grad()
    def _predict_next_topk(hidden_states: torch.Tensor):
        base_router_logits = gate_projector(hidden_states)
        router_logits = base_router_logits
        if lora_payload is not None:
            lora_a = lora_payload["lora_A"]
            lora_b = lora_payload["lora_B"]
            scale = lora_payload["scale"]
            assert isinstance(lora_a, torch.Tensor)
            assert isinstance(lora_b, torch.Tensor)
            assert isinstance(scale, float)
            lora_input = hidden_states.to(lora_a.dtype)
            lora_logits = (lora_input @ lora_a.T) @ lora_b.T
            router_logits = router_logits + lora_logits.to(router_logits.dtype) * scale

        router = next_module.router
        _, topk_ids = router._compute_routing(
            hidden_states,
            router_logits,
            router._get_indices_type(),
            input_ids=None,
        )
        return topk_ids, base_router_logits

    return _predict_next_topk


def build_next_gate_lora_predictors(
    *,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None,
    lora_dir: str | None,
) -> NextGatePredictionBuildResult:
    """Build side-channel next-layer gate predictors for trace/research use."""
    layers: list[tuple[str, FusedMoE]] = []
    for name, module in static_forward_context.items():
        if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
            layers.append((name, module))

    get_module_by_name, find_named_parameter = _model_lookup_helpers(model)
    predictors: dict[int, tuple[int, NextGatePredictor]] = {}
    missing_gate_layers: list[tuple[int, int]] = []
    diagnostics: list[dict[str, Any]] = []

    sorted_layers = sorted(layers, key=lambda item: item[1].layer_id)
    for (_, module), (next_name, next_module) in zip(
        sorted_layers,
        sorted_layers[1:],
    ):
        runner_gate = getattr(next_module.runner, "gate", None)
        gate_projector, gate_source = _find_next_gate_projector(
            next_name,
            next_module,
            get_module_by_name,
            find_named_parameter,
        )
        diagnostic = {
            "layer_id": module.layer_id,
            "next_layer_id": next_module.layer_id,
            "next_layer_name": next_name,
            "runner_gate": runner_gate is not None,
            "gate_found": gate_projector is not None,
            "gate_source": gate_source,
            "gate_candidates": _gate_candidates(next_name),
        }
        diagnostics.append(diagnostic)
        if gate_projector is None:
            missing_gate_layers.append((module.layer_id, next_module.layer_id))
            continue

        reference_param = next(next_module.parameters())
        lora_payload = _load_lora_for_pair(
            lora_dir,
            module.layer_id,
            next_module.layer_id,
            reference_param.device,
            reference_param.dtype,
        )
        predictors[module.layer_id] = (
            next_module.layer_id,
            _make_predictor(next_module, gate_projector, lora_payload),
        )

    return NextGatePredictionBuildResult(
        predictors=predictors,
        missing_gate_layers=missing_gate_layers,
        diagnostics=diagnostics,
        lora_dir=lora_dir,
    )


def _sc_eplb_lora_dir(env_value: str | None) -> tuple[bool, str | None]:
    if env_value is None:
        return False, None
    value = env_value.strip()
    if not value or value.lower() in ("0", "false", "off", "no"):
        return False, None
    if value.lower() in ("1", "true", "on", "yes"):
        return True, None
    return True, value


def maybe_attach_sc_eplb_next_gate_lora(
    *,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None,
    env_value: str | None,
) -> NextGatePredictionBuildResult | None:
    """Attach side-channel next-gate predictors to MoE runners if enabled."""
    enabled, lora_dir = _sc_eplb_lora_dir(env_value)
    if not enabled:
        return None

    result = build_next_gate_lora_predictors(
        static_forward_context=static_forward_context,
        model=model,
        lora_dir=lora_dir,
    )
    for module in static_forward_context.values():
        if not isinstance(module, FusedMoE):
            continue
        predictor = result.predictors.get(module.layer_id)
        if predictor is None:
            module.runner.clear_next_gate_predictor()
            continue
        next_layer_id, predict_next_gate = predictor
        module.runner.set_next_gate_predictor(next_layer_id, predict_next_gate)
    return result
