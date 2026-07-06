# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in traces for studying MoE routing and inter-layer activations.

This module is deliberately not part of the serving observability path. A
trace synchronizes device tensors to CPU and writes them to disk, so it is
intended for short, eager-mode research runs only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

ActivationMode = Literal["none", "input"]
TokenSelection = Literal["all", "prefill_last"]


def _positive_int_env(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


@dataclass(frozen=True)
class MoETraceConfig:
    output_dir: Path
    max_steps: int
    max_tokens: int
    activations: ActivationMode
    activation_dtype: torch.dtype
    token_selection: TokenSelection = "all"

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
        )


class MoETraceCollector:
    """Write one compact ``.pt`` record per MoE layer and forward step."""

    def __init__(
        self,
        config: MoETraceConfig,
        layer_names: dict[int, str],
    ) -> None:
        self.config = config
        self.layer_names = layer_names
        self.rank = _distributed_rank()
        self.rank_dir = config.output_dir / f"rank_{self.rank:05d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self._seen_layers: set[int] = set()
        self._selected_token_indices: list[int] | None = None
        self._selected_token_phases: list[int] | None = None
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
            "layers": self.layer_names,
            "notes": (
                "Each record contains logical expert IDs before EPLB mapping and "
                "the input to that MoE router. A step is one observed forward or "
                "microbatch on this worker."
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
    ) -> None:
        """Describe the packed request layout for the next model forward.

        The model runner packs each request into one contiguous token span.
        For every request, retain the final scheduled prefill token and all
        scheduled decode tokens. This makes batched prefill and batched decode
        unambiguous; tensor row count alone cannot distinguish them.
        """
        if self.config.token_selection != "prefill_last":
            self._selected_token_indices = None
            self._selected_token_phases = None
            return
        if not (
            len(num_scheduled_tokens)
            == len(num_computed_tokens)
            == len(prefill_lengths)
        ):
            raise ValueError("MoE trace request metadata lengths do not match")

        indices: list[int] = []
        phases: list[int] = []
        request_start = 0
        for scheduled, computed, prefill_length in zip(
            num_scheduled_tokens,
            num_computed_tokens,
            prefill_lengths,
        ):
            prompt_tokens = min(scheduled, max(prefill_length - computed, 0))
            if prompt_tokens > 0:
                indices.append(request_start + prompt_tokens - 1)
                phases.append(0)  # prefill
            decode_start = request_start + prompt_tokens
            indices.extend(range(decode_start, request_start + scheduled))
            phases.extend([1] * (scheduled - prompt_tokens))  # decode
            request_start += scheduled

        self._selected_token_indices = indices[: self.config.max_tokens]
        self._selected_token_phases = phases[: self.config.max_tokens]

    @torch.no_grad()
    def capture(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        self._begin_new_step_if_needed(layer_id)
        if self.step >= self.config.max_steps:
            return
        self._seen_layers.add(layer_id)

        available_tokens = min(
            hidden_states.shape[0], topk_weights.shape[0], topk_ids.shape[0]
        )
        selected_indices = self._selected_token_indices
        selected_phases = self._selected_token_phases
        if (
            self.config.token_selection == "prefill_last"
            and selected_indices is not None
        ):
            assert selected_phases is not None
            valid = [
                (index, selected_phases[pos])
                for pos, index in enumerate(selected_indices)
                if index < available_tokens
            ]
            indices = [index for index, _ in valid]
            token_phases = [phase for _, phase in valid]
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
        else:
            token_start = 0
            token_end = min(available_tokens, self.config.max_tokens)
            phase = "decode" if self.config.token_selection == "prefill_last" else "all"
            indices = list(range(token_start, token_end))
            phase_code = 1 if phase == "decode" else -1
            token_phases = [phase_code] * len(indices)

        index_tensor = torch.tensor(
            indices,
            dtype=torch.int64,
            device=hidden_states.device,
        )

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
            "topk_ids": topk_ids.index_select(0, index_tensor).to(
                device="cpu", dtype=torch.int32
            ),
            "topk_weights": topk_weights.index_select(0, index_tensor).to(
                device="cpu", dtype=torch.float32
            ),
        }
        if self.config.activations == "input":
            record["activations"] = hidden_states.index_select(0, index_tensor).to(
                device="cpu", dtype=self.config.activation_dtype
            )

        path = self.rank_dir / (
            f"step_{self.step:06d}_layer_{layer_id:04d}.pt"
        )
        torch.save(record, path)


def maybe_attach_moe_trace(
    *,
    enforce_eager: bool,
    static_forward_context: dict[str, Any],
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

    layer_names = {module.layer_id: name for name, module in layers}
    collector = MoETraceCollector(config, layer_names)
    for _, module in layers:
        layer_id = module.layer_id

        def _trace_fn(hidden_states, topk_weights, topk_ids, _layer_id=layer_id):
            collector.capture(_layer_id, hidden_states, topk_weights, topk_ids)

        module.router.set_trace_fn(_trace_fn)

    logger.warning(
        "MoE trace enabled for %d layers on rank %d; synchronous trace writes "
        "will reduce inference performance. Output: %s",
        len(layers),
        collector.rank,
        collector.rank_dir,
    )
    return collector
