# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional CUDA-event profiler for KV-cache stage breakdown.

Works for both TurboQuant and standard (e.g. FlashAttention) backends so
quantized vs baseline runs produce the same stage table.

Enable with ``VLLM_KV_CACHE_STAGE_PROFILE=N`` (``N > 0``).  The value is the
flush interval in engine steps (``1`` = every step).  Results append to
``VLLM_KV_CACHE_STAGE_PROFILE_FILE`` when set.

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
_registered_skip_layers: frozenset[str] = frozenset()
_lock = threading.Lock()


def kv_cache_stage_profile_interval() -> int:
    """Return flush interval; 0 means profiling is disabled."""
    return envs.VLLM_KV_CACHE_STAGE_PROFILE


def is_kv_cache_stage_profiling_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = kv_cache_stage_profile_interval() > 0
    return _enabled


def attention_stage_for_query_len(max_query_len: int) -> str:
    """Map a forward step to decode vs prefill attention stage."""
    return DECODE_STAGE1 if max_query_len == 1 else FLASH_ATTN


def register_global_kv_cache_dtype(kv_cache_dtype: str) -> None:
    """Cache the global KV cache dtype (V2 runner may lack config context)."""
    global _registered_global_kv_cache_dtype
    _registered_global_kv_cache_dtype = kv_cache_dtype


def register_kv_cache_skip_layers(skip_layers: list[str]) -> None:
    """Cache boundary layer indices excluded from profiling during TQ serving."""
    global _registered_skip_layers
    _registered_skip_layers = frozenset(skip_layers)


def _global_kv_cache_dtype() -> str:
    if _registered_global_kv_cache_dtype is not None:
        return _registered_global_kv_cache_dtype
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return vllm_config.cache_config.cache_dtype


def is_turboquant_serving() -> bool:
    return _global_kv_cache_dtype().startswith("turboquant")


def _resolve_layer_name(layer: object | None) -> str | None:
    if layer is None:
        return None
    if isinstance(layer, str):
        return layer
    return getattr(layer, "layer_name", None)


def should_profile_layer(layer: object | None = None) -> bool:
    """Return whether the current layer should contribute profiler samples.

    Baseline (non-TurboQuant) serving profiles every attention layer.
    TurboQuant serving profiles only compressed layers; boundary layers listed
    in ``kv_cache_dtype_skip_layers`` are excluded.
    """
    if not is_kv_cache_stage_profiling_enabled():
        return False
    if not is_turboquant_serving():
        return True

    layer_name = _resolve_layer_name(layer)
    if layer_name is None:
        return True

    from vllm.model_executor.models.utils import extract_layer_index

    try:
        layer_idx = str(extract_layer_index(layer_name))
    except ValueError:
        return True
    return layer_idx not in _registered_skip_layers


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


class KVCacheStageProfiler:
    """Thread-safe accumulator for per-stage CUDA event timings."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._engine_steps = 0
        self._interval = kv_cache_stage_profile_interval()

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
        scope = (
            "profiled TQ layer-call"
            if is_turboquant_serving()
            else "layer-call"
        )
        lines = [
            "KV cache stage breakdown (mean ms per "
            f"{scope}, over last {self._interval} engine step(s), "
            f"max n={stats.sample_count}):",
        ]
        active_stages = [stage for stage in _ALL_STAGES if counts.get(stage, 0) > 0]
        stats_dict = stats.to_dict()
        for stage in active_stages:
            val = stats_dict[stage]
            n = counts[stage]
            lines.append(f"  {stage}: {val:.6f} ms (n={n})")

        # store_stages = (STORE_PREPROCESS, STORE_KERNEL)
        # if any(counts.get(stage, 0) > 0 for stage in store_stages):
        #     store_total = stats.store_preprocess_ms + stats.store_kernel_ms
        #     lines.append(f"  store_total: {store_total:.6f} ms")

        # decode_stages = (DECODE_Q_ROTATE, DECODE_STAGE1, DECODE_STAGE2)
        # if any(counts.get(stage, 0) > 0 for stage in decode_stages):
        #     decode_total = (
        #         stats.decode_q_rotate_ms
        #         + stats.decode_stage1_ms
        #         + stats.decode_stage2_ms
        #     )
        #     lines.append(f"  decode_total: {decode_total:.6f} ms")

        msg = "\n".join(lines)
        logger.info(msg)

        profile_file = envs.VLLM_KV_CACHE_STAGE_PROFILE_FILE
        if profile_file:
            record = {
                "stages": {
                    stage: stats_dict[stage]
                    for stage in active_stages
                },
                "sample_count": stats.sample_count,
            }
            try:
                with open(profile_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                logger.warning(
                    "Failed to write KV cache profile to %s: %s", profile_file, exc
                )


def get_kv_cache_stage_profiler() -> KVCacheStageProfiler:
    global _profiler
    if _profiler is None:
        _profiler = KVCacheStageProfiler()
    return _profiler


@contextmanager
def flash_attention_profile_stage(
    name: str, layer: object | None = None
) -> Iterator[None]:
    """Profile FA / baseline attention stages for eligible layers."""
    with kv_cache_profile_stage(name, layer=layer):
        yield


@contextmanager
def kv_cache_profile_stage(
    name: str, layer: object | None = None
) -> Iterator[None]:
    """Context manager: time one KV-cache stage on the current CUDA stream."""
    if not should_profile_layer(layer):
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


def kv_cache_profile_scope(
    name: str, layer: object | None = None
) -> AbstractContextManager:
    """Return a profiling context (nullcontext when disabled or filtered)."""
    if not should_profile_layer(layer):
        return nullcontext()
    return kv_cache_profile_stage(name, layer=layer)


def notify_kv_cache_stage_engine_step() -> None:
    """Flush KV-cache stage stats once per completed engine forward step."""
    if not is_kv_cache_stage_profiling_enabled():
        return
    get_kv_cache_stage_profiler().notify_engine_step()
