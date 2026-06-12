# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional CUDA-event profiler for KV-cache stage breakdown.

Works for both TurboQuant and standard (e.g. FlashAttention) backends so
quantized vs baseline runs produce the same stage table.

Enable with ``VLLM_KV_CACHE_STAGE_PROFILE=1`` (``VLLM_TURBOQUANT_PROFILE=1``
is a deprecated alias).  Results print every
``VLLM_KV_CACHE_STAGE_PROFILE_INTERVAL`` stage samples (default 100) and/or
append to ``VLLM_KV_CACHE_STAGE_PROFILE_FILE``.

Stage mapping (baseline FlashAttention vs TurboQuant):

+------------------+---------------------------+---------------------------+
| Stage            | Baseline (auto/bf16)      | TurboQuant                |
+==================+===========================+===========================+
| store_preprocess | N/A (0)                   | norm + rotate GEMM        |
| store_kernel     | reshape_and_cache_flash   | _tq_fused_store_*         |
| decode_q_rotate  | N/A (0)                   | q @ PiT                   |
| decode_stage1    | flash_attn (decode step)  | fused dequant + attn      |
| decode_stage2    | N/A (0)                   | LSE reduction             |
| dequant_bulk     | N/A (0)                   | _tq_full_dequant_kv       |
| inverse_rotate   | N/A (0)                   | k @ Pi                    |
| flash_attn       | flash_attn (prefill)      | flash_attn (prefill)      |
+------------------+---------------------------+---------------------------+

Pair with ``VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`` to see ``kv_stage:*`` scopes
in torch profiler Chrome traces.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

# Stage names shared across TurboQuant and baseline backends.
STORE_PREPROCESS = "store_preprocess"
STORE_KERNEL = "store_kernel"
DECODE_Q_ROTATE = "decode_q_rotate"
DECODE_STAGE1 = "decode_stage1"
DECODE_STAGE2 = "decode_stage2"
DEQUANT_BULK = "dequant_bulk"
INVERSE_ROTATE = "inverse_rotate"
FLASH_ATTN = "flash_attn"

_ALL_STAGES = (
    STORE_PREPROCESS,
    STORE_KERNEL,
    DECODE_Q_ROTATE,
    DECODE_STAGE1,
    DECODE_STAGE2,
    DEQUANT_BULK,
    INVERSE_ROTATE,
    FLASH_ATTN,
)

_enabled: bool | None = None
_profiler: KVCacheStageProfiler | None = None
_registered_global_kv_cache_dtype: str | None = None
_lock = threading.Lock()


def is_kv_cache_stage_profiling_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = envs.VLLM_KV_CACHE_STAGE_PROFILE
    return _enabled


# Backward-compatible alias.
is_turboquant_profiling_enabled = is_kv_cache_stage_profiling_enabled


def attention_stage_for_query_len(max_query_len: int) -> str:
    """Map a forward step to decode vs prefill attention stage."""
    return DECODE_STAGE1 if max_query_len == 1 else FLASH_ATTN


def register_global_kv_cache_dtype(kv_cache_dtype: str) -> None:
    """Cache the global KV cache dtype (V2 runner may lack config context)."""
    global _registered_global_kv_cache_dtype
    _registered_global_kv_cache_dtype = kv_cache_dtype


def _global_kv_cache_dtype() -> str:
    if _registered_global_kv_cache_dtype is not None:
        return _registered_global_kv_cache_dtype
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return vllm_config.cache_config.cache_dtype


def is_turboquant_serving() -> bool:
    return _global_kv_cache_dtype().startswith("turboquant")


def should_profile_flash_attention_kv_stage() -> bool:
    """Skip FA profiling during TQ serving (FA only runs on boundary layers)."""
    if not is_kv_cache_stage_profiling_enabled():
        return False
    return not is_turboquant_serving()


@dataclass
class KVCacheStageStats:
    """Aggregated milliseconds per stage (mean over recorded samples)."""

    store_preprocess_ms: float = 0.0
    store_kernel_ms: float = 0.0
    decode_q_rotate_ms: float = 0.0
    decode_stage1_ms: float = 0.0
    decode_stage2_ms: float = 0.0
    dequant_bulk_ms: float = 0.0
    inverse_rotate_ms: float = 0.0
    flash_attn_ms: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            STORE_PREPROCESS: self.store_preprocess_ms,
            STORE_KERNEL: self.store_kernel_ms,
            DECODE_Q_ROTATE: self.decode_q_rotate_ms,
            DECODE_STAGE1: self.decode_stage1_ms,
            DECODE_STAGE2: self.decode_stage2_ms,
            DEQUANT_BULK: self.dequant_bulk_ms,
            INVERSE_ROTATE: self.inverse_rotate_ms,
            FLASH_ATTN: self.flash_attn_ms,
            "sample_count": self.sample_count,
        }


# Backward-compatible alias.
TQStageStats = KVCacheStageStats


class KVCacheStageProfiler:
    """Thread-safe accumulator for per-stage CUDA event timings."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._engine_steps = 0
        self._interval = envs.VLLM_KV_CACHE_STAGE_PROFILE_INTERVAL

    def record(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        self._samples[name].append(elapsed_ms)

    def notify_engine_step(self) -> None:
        """Call once per engine forward step to flush aggregated stage stats."""
        self._engine_steps += 1
        if self._engine_steps % self._interval == 0:
            self.flush(log=True)

    def flush(self, *, log: bool = False) -> KVCacheStageStats:
        with _lock:
            stats = self._compute_stats()
            if stats.sample_count == 0:
                return stats
            counts = {name: len(vals) for name, vals in self._samples.items()}
            if log:
                self._emit(stats, counts)
            self._samples.clear()
            return stats

    def _compute_stats(self) -> KVCacheStageStats:
        def _mean(name: str) -> float:
            vals = self._samples.get(name)
            if not vals:
                return 0.0
            return sum(vals) / len(vals)

        counts = [len(v) for v in self._samples.values()]
        sample_count = max(counts) if counts else 0
        return KVCacheStageStats(
            store_preprocess_ms=_mean(STORE_PREPROCESS),
            store_kernel_ms=_mean(STORE_KERNEL),
            decode_q_rotate_ms=_mean(DECODE_Q_ROTATE),
            decode_stage1_ms=_mean(DECODE_STAGE1),
            decode_stage2_ms=_mean(DECODE_STAGE2),
            dequant_bulk_ms=_mean(DEQUANT_BULK),
            inverse_rotate_ms=_mean(INVERSE_ROTATE),
            flash_attn_ms=_mean(FLASH_ATTN),
            sample_count=sample_count,
        )

    def _emit(self, stats: KVCacheStageStats, counts: dict[str, int]) -> None:
        lines = [
            "KV cache stage breakdown (mean ms per layer-call, over last "
            f"{self._interval} engine step(s), max n={stats.sample_count}):",
        ]
        for stage in _ALL_STAGES:
            val = stats.to_dict()[stage]
            n = counts.get(stage, 0)
            lines.append(f"  {stage}: {val:.6f} ms (n={n})")
        store_total = stats.store_preprocess_ms + stats.store_kernel_ms
        decode_total = (
            stats.decode_q_rotate_ms + stats.decode_stage1_ms + stats.decode_stage2_ms
        )
        lines.append(f"  store_total: {store_total:.6f} ms")
        lines.append(f"  decode_total: {decode_total:.6f} ms")
        msg = "\n".join(lines)
        logger.info(msg)

        profile_file = envs.VLLM_KV_CACHE_STAGE_PROFILE_FILE
        if profile_file:
            record = {"stages": stats.to_dict()}
            try:
                with open(profile_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                logger.warning(
                    "Failed to write KV cache profile to %s: %s", profile_file, exc
                )


# Backward-compatible aliases.
TurboQuantProfiler = KVCacheStageProfiler


def get_kv_cache_stage_profiler() -> KVCacheStageProfiler:
    global _profiler
    if _profiler is None:
        _profiler = KVCacheStageProfiler()
    return _profiler


get_turboquant_profiler = get_kv_cache_stage_profiler


@contextmanager
def flash_attention_profile_stage(name: str) -> Iterator[None]:
    """Profile FA stages; disabled during TQ serving (boundary layers only)."""
    if not should_profile_flash_attention_kv_stage():
        yield
        return
    with kv_cache_profile_stage(name):
        yield


@contextmanager
def kv_cache_profile_stage(name: str) -> Iterator[None]:
    """Context manager: time one KV-cache stage on the current CUDA stream."""
    if not is_kv_cache_stage_profiling_enabled():
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    scope: AbstractContextManager = record_function_or_nullcontext(
        f"kv_stage:{name}"
    )
    with scope:
        start.record()
        try:
            yield
        finally:
            end.record()
            get_kv_cache_stage_profiler().record(name, start, end)


# Backward-compatible aliases.
tq_profile_stage = kv_cache_profile_stage


def kv_cache_profile_scope(name: str) -> AbstractContextManager:
    """Return a profiling context (nullcontext when disabled)."""
    if not is_kv_cache_stage_profiling_enabled():
        return nullcontext()
    return kv_cache_profile_stage(name)


tq_profile_scope = kv_cache_profile_scope


def notify_kv_cache_stage_engine_step() -> None:
    """Flush KV-cache stage stats once per completed engine forward step."""
    if not is_kv_cache_stage_profiling_enabled():
        return
    get_kv_cache_stage_profiler().notify_engine_step()
