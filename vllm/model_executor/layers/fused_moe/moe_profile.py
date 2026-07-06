# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in runtime profiling for MoE dispatch, expert compute, and combine.

This is intended for short research runs. It synchronizes accelerator work
around measured regions and writes JSONL records, so it should not be enabled
for production serving.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import torch

_PROFILE_DIR_ENV = "VLLM_MOE_PROFILE_DIR"
_MAX_RECORDS_ENV = "VLLM_MOE_PROFILE_MAX_RECORDS"


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _jsonable_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().tolist()]


def _target_rank_ids(
    topk_ids: torch.Tensor,
    global_num_experts: int,
    num_dispatchers: int,
    round_robin: bool,
) -> torch.Tensor:
    valid = topk_ids >= 0
    expert_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    if round_robin:
        target = torch.remainder(expert_ids, num_dispatchers)
    else:
        base = global_num_experts // num_dispatchers
        remainder = global_num_experts % num_dispatchers
        if base == 0:
            target = expert_ids
        elif remainder == 0:
            target = torch.div(expert_ids, base, rounding_mode="floor")
        else:
            wide = base + 1
            cutoff = wide * remainder
            target = torch.empty_like(expert_ids)
            wide_mask = expert_ids < cutoff
            target[wide_mask] = torch.div(
                expert_ids[wide_mask], wide, rounding_mode="floor"
            )
            target[~wide_mask] = remainder + torch.div(
                expert_ids[~wide_mask] - cutoff, base, rounding_mode="floor"
            )
    return torch.where(valid, target, torch.full_like(target, -1))


def rank_distribution(
    topk_ids: torch.Tensor,
    global_num_experts: int,
    num_dispatchers: int,
    round_robin: bool,
) -> tuple[list[int], list[int]]:
    if num_dispatchers <= 1 or global_num_experts <= 0:
        tokens = topk_ids.shape[0]
        assignments = int((topk_ids >= 0).sum().item())
        return [assignments], [tokens]

    targets = _target_rank_ids(
        topk_ids.to(torch.int64), global_num_experts, num_dispatchers, round_robin
    )
    flat_targets = targets[targets >= 0]
    if flat_targets.numel() == 0:
        return [0] * num_dispatchers, [0] * num_dispatchers

    assignment_counts = torch.bincount(
        flat_targets, minlength=num_dispatchers
    ).to(torch.int64)
    unique_token_counts = torch.empty(
        num_dispatchers, dtype=torch.int64, device=targets.device
    )
    for rank in range(num_dispatchers):
        unique_token_counts[rank] = torch.any(targets == rank, dim=1).sum()
    return _jsonable_int_list(assignment_counts), _jsonable_int_list(
        unique_token_counts
    )


def _distribution_metrics(counts: list[int]) -> dict[str, float | int]:
    total = sum(counts)
    active = sum(1 for count in counts if count > 0)
    if total == 0:
        return {
            "active_ranks": 0,
            "max_share": 0.0,
            "imbalance": 0.0,
            "cv": 0.0,
        }
    mean = total / len(counts)
    variance = sum((count - mean) ** 2 for count in counts) / len(counts)
    return {
        "active_ranks": active,
        "max_share": max(counts) / total,
        "imbalance": max(counts) / mean if mean else 0.0,
        "cv": variance**0.5 / mean if mean else 0.0,
    }


@dataclass
class MoEProfileRecord:
    kernel_id: int
    call_index: int
    layer_name: str | None
    rank: int
    prepare_finalize: str
    fused_experts: str
    num_tokens: int
    top_k: int
    global_num_experts: int
    local_num_experts: int
    num_dispatchers: int
    round_robin: bool
    target_rank_assignments: list[int]
    target_rank_unique_tokens: list[int]
    dispatch_ms: float | None = None
    expert_compute_ms: float | None = None
    combine_ms: float | None = None
    local_expert_tokens: list[int] | None = None
    received_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def measure(
        self, field_name: str, device: torch.device
    ) -> Generator[None, None, None]:
        _sync_if_needed(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync_if_needed(device)
            setattr(self, field_name, (time.perf_counter() - start) * 1000.0)

    def set_local_expert_tokens(self, tokens: torch.Tensor | None) -> None:
        if tokens is None:
            return
        self.local_expert_tokens = _jsonable_int_list(tokens.to(torch.int64))
        self.received_tokens = sum(self.local_expert_tokens)

    def to_json(self) -> str:
        payload = {
            "rank": self.rank,
            "kernel_id": self.kernel_id,
            "call_index": self.call_index,
            "layer_name": self.layer_name,
            "prepare_finalize": self.prepare_finalize,
            "fused_experts": self.fused_experts,
            "num_tokens": self.num_tokens,
            "top_k": self.top_k,
            "global_num_experts": self.global_num_experts,
            "local_num_experts": self.local_num_experts,
            "num_dispatchers": self.num_dispatchers,
            "round_robin": self.round_robin,
            "target_rank_assignments": self.target_rank_assignments,
            "target_rank_unique_tokens": self.target_rank_unique_tokens,
            "target_rank_assignment_metrics": _distribution_metrics(
                self.target_rank_assignments
            ),
            "target_rank_unique_token_metrics": _distribution_metrics(
                self.target_rank_unique_tokens
            ),
            "local_expert_tokens": self.local_expert_tokens,
            "received_tokens": self.received_tokens,
            "dispatch_ms": self.dispatch_ms,
            "expert_compute_ms": self.expert_compute_ms,
            "combine_ms": self.combine_ms,
        }
        payload.update(self.extra)
        return json.dumps(payload, sort_keys=True)


class MoEProfiler:
    def __init__(self, output_dir: Path, max_records: int) -> None:
        self.rank = _distributed_rank()
        self.max_records = max_records
        self.num_records = 0
        rank_dir = output_dir / f"rank_{self.rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        self.path = rank_dir / "moe_profile.jsonl"
        self.file = self.path.open("a", encoding="utf-8")

    def enabled(self) -> bool:
        return self.num_records < self.max_records

    def write(self, record: MoEProfileRecord) -> None:
        if not self.enabled():
            return
        self.file.write(record.to_json() + "\n")
        self.file.flush()
        self.num_records += 1


_PROFILER: MoEProfiler | None = None
_PROFILER_INITIALIZED = False


def get_profiler() -> MoEProfiler | None:
    global _PROFILER, _PROFILER_INITIALIZED
    if _PROFILER_INITIALIZED:
        return _PROFILER
    _PROFILER_INITIALIZED = True
    output_dir = os.getenv(_PROFILE_DIR_ENV)
    if not output_dir:
        return None
    max_records = int(os.getenv(_MAX_RECORDS_ENV, "100000"))
    _PROFILER = MoEProfiler(Path(output_dir).expanduser().resolve(), max_records)
    return _PROFILER


def make_record(
    *,
    kernel_id: int,
    call_index: int,
    layer_name: str | None,
    prepare_finalize: object,
    fused_experts: object,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    local_num_experts: int,
    num_dispatchers: int,
    round_robin: bool,
) -> MoEProfileRecord | None:
    profiler = get_profiler()
    if profiler is None or not profiler.enabled():
        return None
    assignments, unique_tokens = rank_distribution(
        topk_ids, global_num_experts, num_dispatchers, round_robin
    )
    return MoEProfileRecord(
        kernel_id=kernel_id,
        call_index=call_index,
        layer_name=layer_name,
        rank=profiler.rank,
        prepare_finalize=prepare_finalize.__class__.__name__,
        fused_experts=fused_experts.__class__.__name__,
        num_tokens=hidden_states.shape[0],
        top_k=topk_ids.shape[1],
        global_num_experts=global_num_experts,
        local_num_experts=local_num_experts,
        num_dispatchers=num_dispatchers,
        round_robin=round_robin,
        target_rank_assignments=assignments,
        target_rank_unique_tokens=unique_tokens,
    )


def write_record(record: MoEProfileRecord | None) -> None:
    if record is None:
        return
    profiler = get_profiler()
    if profiler is not None:
        profiler.write(record)
