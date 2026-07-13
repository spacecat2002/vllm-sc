# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in traces for studying MoE routing and inter-layer activations.

This module is deliberately not part of the serving observability path. A
trace synchronizes device tensors to CPU and writes them to disk, so it is
intended for short, eager-mode research runs only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRACE_DIR_ENV = "VLLM_MOE_TRACE_DIR"
_MAX_STEPS_ENV = "VLLM_MOE_TRACE_MAX_STEPS"
_MAX_TOKENS_ENV = "VLLM_MOE_TRACE_MAX_TOKENS"
_ACTIVATIONS_ENV = "VLLM_MOE_TRACE_ACTIVATIONS"
_ACTIVATION_DTYPE_ENV = "VLLM_MOE_TRACE_ACTIVATION_DTYPE"
_TOKEN_SELECTION_ENV = "VLLM_MOE_TRACE_TOKEN_SELECTION"
_NEXT_GATE_ENV = "VLLM_MOE_TRACE_NEXT_GATE"
_NEXT_GATE_LORA_DIR_ENV = "VLLM_MOE_TRACE_NEXT_GATE_LORA_DIR"

ActivationMode = Literal["none", "input"]
TokenSelection = Literal["all", "prefill_last"]
NextGatePredictor = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _positive_int_env(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw in ("0", "false", "False"):
        return False
    if raw in ("1", "true", "True"):
        return True
    raise ValueError(f"{name} must be one of 0/1/false/true, got {raw!r}")


@dataclass(frozen=True)
class MoETraceConfig:
    output_dir: Path
    max_steps: int
    max_tokens: int
    activations: ActivationMode
    activation_dtype: torch.dtype
    token_selection: TokenSelection = "all"
    trace_next_gate: bool = False

    @classmethod
    def from_env(cls) -> MoETraceConfig | None:
        output_dir = envs.VLLM_MOE_TRACE_DIR
        if not output_dir:
            return None

        activations = envs.VLLM_MOE_TRACE_ACTIVATIONS
        if activations not in ("none", "input"):
            raise ValueError(
                f"{_ACTIVATIONS_ENV} must be 'none' or 'input', got {activations!r}"
            )

        dtype_name = envs.VLLM_MOE_TRACE_ACTIVATION_DTYPE
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_name not in dtypes:
            raise ValueError(
                f"{_ACTIVATION_DTYPE_ENV} must be one of {tuple(dtypes)}, "
                f"got {dtype_name!r}"
            )

        token_selection = envs.VLLM_MOE_TRACE_TOKEN_SELECTION
        if token_selection not in ("all", "prefill_last"):
            raise ValueError(
                f"{_TOKEN_SELECTION_ENV} must be 'all' or 'prefill_last', "
                f"got {token_selection!r}"
            )

        return cls(
            output_dir=Path(output_dir).expanduser().resolve(),
            max_steps=_positive_int_env(
                _MAX_STEPS_ENV, envs.VLLM_MOE_TRACE_MAX_STEPS
            ),
            max_tokens=_positive_int_env(
                _MAX_TOKENS_ENV, envs.VLLM_MOE_TRACE_MAX_TOKENS
            ),
            activations=activations,  # type: ignore[arg-type]
            activation_dtype=dtypes[dtype_name],
            token_selection=token_selection,  # type: ignore[arg-type]
            trace_next_gate=_bool_env(_NEXT_GATE_ENV),
        )


class MoETraceCollector:
    """Write one compact ``.pt`` record per MoE layer and forward step."""

    def __init__(
        self,
        config: MoETraceConfig,
        layer_names: dict[int, str],
        next_gate_predictors: dict[int, tuple[int, NextGatePredictor]]
        | None = None,
        next_gate_missing_gate_layers: list[tuple[int, int]] | None = None,
        next_gate_diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.layer_names = layer_names
        self.next_gate_predictors = next_gate_predictors or {}
        self.next_gate_missing_gate_layers = next_gate_missing_gate_layers or []
        self.next_gate_diagnostics = next_gate_diagnostics or []
        self.rank = _distributed_rank()
        self.rank_dir = config.output_dir / f"rank_{self.rank:05d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self._seen_layers: set[int] = set()
        self._selected_token_indices: list[int] | None = None
        self._selected_token_phases: list[int] | None = None
        self._selected_request_ids: list[str] | None = None
        self._write_metadata()

    def _write_metadata(self) -> None:
        metadata = {
            "format_version": 1,
            "rank": self.rank,
            "max_steps": self.config.max_steps,
            "max_tokens": self.config.max_tokens,
            "activations": self.config.activations,
            "activation_dtype": str(self.config.activation_dtype).removeprefix(
                "torch."
            ),
            "token_selection": self.config.token_selection,
            "trace_next_gate": self.config.trace_next_gate,
            "trace_next_gate_env": os.getenv(_NEXT_GATE_ENV),
            "next_gate_lora_dir": os.getenv(_NEXT_GATE_LORA_DIR_ENV),
            "next_gate_predictor_count": len(self.next_gate_predictors),
            "next_gate_predictor_pairs": [
                [layer_id, next_layer_id]
                for layer_id, (next_layer_id, _) in sorted(
                    self.next_gate_predictors.items()
                )
            ],
            "next_gate_missing_gate_layers": self.next_gate_missing_gate_layers,
            "next_gate_diagnostics": self.next_gate_diagnostics,
            "layers": self.layer_names,
            "notes": (
                "Each record contains logical expert IDs before EPLB mapping and "
                "the input to that MoE router. If trace_next_gate is enabled, "
                "a record for layer i also contains the top-k experts predicted "
                "by feeding layer i's selected router input to layer i+1's gate. "
                "A step is one observed forward or microbatch on this worker."
            ),
        }
        path = self.rank_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _begin_new_step_if_needed(self, layer_id: int) -> None:
        # A layer can only occur once in the usual transformer forward. Seeing
        # it again marks the next forward (or the next DBO microbatch).
        if layer_id in self._seen_layers:
            self.step += 1
            self._seen_layers.clear()

    def begin_forward(
        self,
        num_scheduled_tokens: list[int],
        num_computed_tokens: list[int],
        prefill_lengths: list[int],
        request_ids: list[str] | None = None,
    ) -> None:
        """Describe the packed request layout for the next model forward.

        The model runner packs each request into one contiguous token span.
        For every request, retain either all scheduled tokens or the final
        scheduled prefill token plus all decode tokens, according to the trace
        configuration. This makes batched prefill and batched decode
        unambiguous; tensor row count alone cannot distinguish them.
        """
        if not (
            len(num_scheduled_tokens)
            == len(num_computed_tokens)
            == len(prefill_lengths)
        ):
            raise ValueError("MoE trace request metadata lengths do not match")
        if request_ids is None:
            request_ids = [str(index) for index in range(len(num_scheduled_tokens))]
        if len(request_ids) != len(num_scheduled_tokens):
            raise ValueError("MoE trace request id count does not match batch size")

        indices: list[int] = []
        phases: list[int] = []
        selected_request_ids: list[str] = []
        request_start = 0
        for scheduled, computed, prefill_length, request_id in zip(
            num_scheduled_tokens,
            num_computed_tokens,
            prefill_lengths,
            request_ids,
        ):
            prompt_tokens = min(scheduled, max(prefill_length - computed, 0))
            if self.config.token_selection == "all":
                indices.extend(range(request_start, request_start + scheduled))
                phases.extend([0] * prompt_tokens)
                phases.extend([1] * (scheduled - prompt_tokens))
                selected_request_ids.extend([request_id] * scheduled)
            elif prompt_tokens > 0:
                indices.append(request_start + prompt_tokens - 1)
                phases.append(0)  # prefill
                selected_request_ids.append(request_id)
            if self.config.token_selection == "prefill_last":
                decode_start = request_start + prompt_tokens
                indices.extend(range(decode_start, request_start + scheduled))
                phases.extend([1] * (scheduled - prompt_tokens))  # decode
                selected_request_ids.extend(
                    [request_id] * (scheduled - prompt_tokens)
                )
            request_start += scheduled

        self._selected_token_indices = indices[: self.config.max_tokens]
        self._selected_token_phases = phases[: self.config.max_tokens]
        self._selected_request_ids = selected_request_ids[: self.config.max_tokens]

    @torch.no_grad()
    def capture(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        self._begin_new_step_if_needed(layer_id)
        if self.step >= self.config.max_steps:
            return
        self._seen_layers.add(layer_id)

        available_tokens = min(
            hidden_states.shape[0],
            router_logits.shape[0],
            topk_weights.shape[0],
            topk_ids.shape[0],
        )
        selected_indices = self._selected_token_indices
        selected_phases = self._selected_token_phases
        selected_request_ids = self._selected_request_ids
        if selected_indices is not None:
            assert selected_phases is not None
            assert selected_request_ids is not None
            valid = [
                (index, selected_phases[pos], selected_request_ids[pos])
                for pos, index in enumerate(selected_indices)
                if index < available_tokens
            ]
            indices = [index for index, _, _ in valid]
            token_phases = [phase for _, phase, _ in valid]
            request_ids = [request_id for _, _, request_id in valid]
            token_start = min(indices, default=0)
            token_end = max(indices, default=-1) + 1
            unique_phases = set(token_phases)
            if unique_phases == {0}:
                phase = "prefill"
            elif unique_phases == {1}:
                phase = "decode"
            else:
                phase = "mixed"
        elif self.config.token_selection == "prefill_last" and available_tokens > 1:
            # This mode is intended for the companion single-request script:
            # a multi-token forward is prefill, while decode forwards contain
            # one token. Chunked prefill is disabled by that script.
            token_start = available_tokens - 1
            token_end = available_tokens
            phase = "prefill"
            indices = [token_start]
            token_phases = [0]
            request_ids = [f"step_{self.step:06d}_token_{token_start:06d}"]
        else:
            token_start = 0
            token_end = min(available_tokens, self.config.max_tokens)
            phase = "decode" if self.config.token_selection == "prefill_last" else "all"
            indices = list(range(token_start, token_end))
            phase_code = 1 if phase == "decode" else -1
            token_phases = [phase_code] * len(indices)
            request_ids = [
                f"step_{self.step:06d}_token_{index:06d}" for index in indices
            ]

        index_tensor = torch.tensor(
            indices,
            dtype=torch.int64,
            device=hidden_states.device,
        )

        selected_hidden_states = hidden_states.index_select(0, index_tensor)
        selected_router_logits = router_logits.index_select(0, index_tensor)
        record: dict[str, Any] = {
            "format_version": 1,
            "rank": self.rank,
            "step": self.step,
            "layer_id": layer_id,
            "layer_name": self.layer_names.get(layer_id, ""),
            "phase": phase,
            "num_tokens_before_truncation": hidden_states.shape[0],
            "selected_token_start": token_start,
            "selected_token_end": token_end,
            "selected_token_indices": torch.tensor(indices, dtype=torch.int64),
            "token_phases": torch.tensor(token_phases, dtype=torch.int8),
            "request_ids": request_ids,
            "router_logits": selected_router_logits.to(
                device="cpu", dtype=self.config.activation_dtype
            ),
            "topk_ids": topk_ids.index_select(0, index_tensor).to(
                device="cpu", dtype=torch.int32
            ),
            "topk_weights": topk_weights.index_select(0, index_tensor).to(
                device="cpu", dtype=torch.float32
            ),
        }
        if self.config.activations == "input":
            record["activations"] = selected_hidden_states.to(
                device="cpu", dtype=self.config.activation_dtype
            )
        if self.config.trace_next_gate and layer_id in self.next_gate_predictors:
            next_layer_id, predict_next_gate = self.next_gate_predictors[layer_id]
            predicted_topk_ids, base_logits = predict_next_gate(
                selected_hidden_states
            )
            record["next_gate_layer_id"] = next_layer_id
            record["next_gate_base_logits"] = base_logits.to(
                device="cpu", dtype=self.config.activation_dtype
            )
            record["next_gate_predicted_topk_ids"] = predicted_topk_ids.to(
                device="cpu", dtype=torch.int32
            )

        path = self.rank_dir / (
            f"step_{self.step:06d}_layer_{layer_id:04d}.pt"
        )
        torch.save(record, path)


def maybe_attach_moe_trace(
    *,
    enforce_eager: bool,
    static_forward_context: dict[str, Any],
    model: torch.nn.Module | None = None,
) -> MoETraceCollector | None:
    """Attach a trace collector to every standard FusedMoE router if enabled."""
    config = MoETraceConfig.from_env()
    if config is None:
        return None
    if not enforce_eager:
        raise ValueError(
            f"{_TRACE_DIR_ENV} requires --enforce-eager because Python trace "
            "callbacks are not replayed by CUDA graphs"
        )

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    layers: list[tuple[str, FusedMoE]] = []
    for name, module in static_forward_context.items():
        if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
            layers.append((name, module))

    if not layers:
        logger.warning(
            "%s was set, but no traceable FusedMoE routers were found",
            _TRACE_DIR_ENV,
        )
        return None

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

    def _find_next_gate_projector(
        next_name: str,
        next_module: FusedMoE,
        fse_enabled: bool,
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, str]:
        gate = getattr(next_module.runner, "gate", None)
        if gate is not None:
            if fse_enabled:

                def _project_with_fused_gate(hidden_states: torch.Tensor):
                    next_module.runner._maybe_fuse_gate_weights()
                    return torch.nn.functional.linear(
                        hidden_states, next_module.runner._combined_gate_weight
                    )

                return _project_with_fused_gate, "runner.fused_gate"

            def _project_with_runner_gate(hidden_states: torch.Tensor):
                router_logits, _ = gate(hidden_states)
                return router_logits

            return _project_with_runner_gate, "runner.gate"

        for gate_name in _gate_candidates(next_name):
            parent = _get_module_by_name(gate_name.rsplit(".", 1)[0])
            gate = getattr(parent, "gate", None) if parent is not None else None
            if gate is not None:

                def _project_with_parent_gate(
                    hidden_states: torch.Tensor,
                    _gate=gate,
                ):
                    router_logits, _ = _gate(hidden_states)
                    return router_logits

                return _project_with_parent_gate, gate_name

            weight = _find_named_parameter(gate_name + ".weight")
            if weight is not None:
                bias = _find_named_parameter(gate_name + ".bias")

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
        source_layer_id: int,
        next_layer_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | float] | None:
        lora_dir = os.getenv(_NEXT_GATE_LORA_DIR_ENV)
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

    layer_names = {module.layer_id: name for name, module in layers}
    next_gate_predictors = {}
    next_gate_missing_gate_layers = []
    next_gate_diagnostics = []
    if config.trace_next_gate:
        sorted_layers = sorted(layers, key=lambda item: item[1].layer_id)
        for (_, module), (next_name, next_module) in zip(
            sorted_layers,
            sorted_layers[1:],
        ):
            runner = next_module.runner
            runner_gate = getattr(runner, "gate", None)
            gate_projector, gate_source = _find_next_gate_projector(
                next_name,
                next_module,
                bool(getattr(runner, "_fse_fuse_gate", False)),
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
            next_gate_diagnostics.append(diagnostic)
            if gate_projector is None:
                next_gate_missing_gate_layers.append(
                    (module.layer_id, next_module.layer_id)
                )
                continue
            reference_param = next(next_module.parameters())
            lora_payload = _load_lora_for_pair(
                module.layer_id,
                next_module.layer_id,
                reference_param.device,
                reference_param.dtype,
            )

            @torch.no_grad()
            def _predict_next_topk(
                hidden_states,
                _next_module=next_module,
                _gate_projector=gate_projector,
                _lora_payload=lora_payload,
            ):
                base_router_logits = _gate_projector(hidden_states)
                router_logits = base_router_logits
                if _lora_payload is not None:
                    lora_a = _lora_payload["lora_A"]
                    lora_b = _lora_payload["lora_B"]
                    scale = _lora_payload["scale"]
                    lora_input = hidden_states.to(lora_a.dtype)
                    lora_logits = (lora_input @ lora_a.T) @ lora_b.T
                    router_logits = router_logits + lora_logits.to(
                        router_logits.dtype
                    ) * scale
                router = _next_module.router
                _, topk_ids = router._compute_routing(
                    hidden_states,
                    router_logits,
                    router._get_indices_type(),
                    input_ids=None,
                )
                return topk_ids, base_router_logits

            next_gate_predictors[module.layer_id] = (
                next_module.layer_id,
                _predict_next_topk,
            )

        if not next_gate_predictors:
            logger.warning(
                "%s was set, but no next-layer gate predictors could be built",
                _NEXT_GATE_ENV,
            )
        elif next_gate_missing_gate_layers:
            logger.warning(
                "%s built %d next-layer gate predictors, but skipped %d layer "
                "pairs whose next layer has no runner.gate",
                _NEXT_GATE_ENV,
                len(next_gate_predictors),
                len(next_gate_missing_gate_layers),
            )

    collector = MoETraceCollector(
        config,
        layer_names,
        next_gate_predictors,
        next_gate_missing_gate_layers,
        next_gate_diagnostics,
    )
    for _, module in layers:
        layer_id = module.layer_id

        def _trace_fn(
            hidden_states,
            router_logits,
            topk_weights,
            topk_ids,
            _layer_id=layer_id,
        ):
            collector.capture(
                _layer_id,
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
            )

        module.router.set_trace_fn(_trace_fn)

    logger.warning(
        "MoE trace enabled for %d layers on rank %d; synchronous trace writes "
        "will reduce inference performance. Output: %s",
        len(layers),
        collector.rank,
        collector.rank_dir,
    )
    return collector
