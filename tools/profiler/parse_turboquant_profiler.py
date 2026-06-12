# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse torch profiler output and summarize TurboQuant kernel breakdown.

Usage:
    .venv/bin/python tools/profiler/parse_turboquant_profiler.py \\
        vllm_profile/profiler_out_0.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field


@dataclass
class KernelEntry:
    name: str
    self_cuda_ms: float
    cuda_total_ms: float
    num_calls: int


@dataclass
class TQBreakdown:
    store_kernel_ms: float = 0.0
    store_preprocess_ms: float = 0.0
    decode_stage1_ms: float = 0.0
    decode_stage2_ms: float = 0.0
    dequant_bulk_ms: float = 0.0
    kv_cache_update_ms: float = 0.0
    attention_ms: float = 0.0
    flash_attn_ms: float = 0.0
    entries: list[KernelEntry] = field(default_factory=list)

    @property
    def store_total_ms(self) -> float:
        return self.store_preprocess_ms + self.store_kernel_ms

    @property
    def decode_total_ms(self) -> float:
        return self.decode_stage1_ms + self.decode_stage2_ms


_STORE_KERNEL_PATTERNS = ("_tq_fused_store_mse", "_tq_fused_store_fp8")
_STORE_PREPROCESS_OPS = (
    "aten::linalg_vector_norm",
    "aten::div",
)
_DECODE_STAGE1 = "_tq_decode_stage1"
_DECODE_STAGE2 = "_fwd_kernel_stage2"
_DEQUANT_BULK = "_tq_full_dequant_kv"
_KV_UPDATE = "unified_kv_cache_update"
_ATTENTION = "unified_attention_with_output"
_FLASH_ATTN = ("_vllm_fa2_C::varlen_fwd", "flash::flash_fwd_kernel")


def _parse_ms(value: str) -> float:
    value = value.strip()
    if value.endswith("ms"):
        return float(value[:-2])
    if value.endswith("us"):
        return float(value[:-2]) / 1000.0
    if value.endswith("s"):
        return float(value[:-1]) * 1000.0
    return float(value)


def parse_profiler_table(path: str) -> tuple[list[KernelEntry], float]:
    """Parse vLLM torch profiler text table. Returns entries and total CUDA ms."""
    entries: list[KernelEntry] = []
    total_cuda_ms = 0.0
    in_table = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Self CUDA time total:" in line:
                match = re.search(r"([\d.]+)ms", line)
                if match:
                    total_cuda_ms = float(match.group(1))
                break

            if line.startswith("---"):
                in_table = not in_table
                continue
            if not in_table or not line.strip():
                continue
            if line.startswith("Name"):
                continue

            # Fixed-width columns; name may be truncated with "..."
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                # Last 11 fields are numeric columns; rest is name.
                tail = parts[-11:]
                name = " ".join(parts[:-11]).strip()
                self_cuda = _parse_ms(tail[6])
                cuda_total = _parse_ms(tail[8])
                num_calls = int(tail[10])
                entries.append(
                    KernelEntry(
                        name=name,
                        self_cuda_ms=self_cuda,
                        cuda_total_ms=cuda_total,
                        num_calls=num_calls,
                    )
                )
            except (ValueError, IndexError):
                continue

    return entries, total_cuda_ms


def summarize_turboquant(entries: list[KernelEntry]) -> TQBreakdown:
    breakdown = TQBreakdown(entries=entries)

    # GEMM time attributed to store/decode preprocess is approximate:
    # aten::mm is shared across the whole model.
    for e in entries:
        name = e.name
        cuda = e.self_cuda_ms

        if any(p in name for p in _STORE_KERNEL_PATTERNS):
            breakdown.store_kernel_ms += cuda
        elif any(op == name for op in _STORE_PREPROCESS_OPS):
            breakdown.store_preprocess_ms += cuda
        elif _DECODE_STAGE1 in name:
            breakdown.decode_stage1_ms += cuda
        elif _DECODE_STAGE2 in name:
            breakdown.decode_stage2_ms += cuda
        elif _DEQUANT_BULK in name:
            breakdown.dequant_bulk_ms += cuda
        elif _KV_UPDATE in name:
            breakdown.kv_cache_update_ms += cuda
        elif _ATTENTION in name:
            breakdown.attention_ms += cuda
        elif any(p in name for p in _FLASH_ATTN):
            breakdown.flash_attn_ms += cuda

    return breakdown


def print_report(path: str, breakdown: TQBreakdown, total_cuda_ms: float) -> None:
    print(f"TurboQuant breakdown from: {path}")
    print(f"Total Self CUDA time: {total_cuda_ms:.3f} ms\n")

    rows = [
        ("store_preprocess (norm+div)", breakdown.store_preprocess_ms),
        ("store_kernel (_tq_fused_store_*)", breakdown.store_kernel_ms),
        ("store_total", breakdown.store_total_ms),
        ("decode_stage1 (fused dequant+attn)", breakdown.decode_stage1_ms),
        ("decode_stage2 (_fwd_kernel_stage2)", breakdown.decode_stage2_ms),
        ("decode_total", breakdown.decode_total_ms),
        ("dequant_bulk (_tq_full_dequant_kv)", breakdown.dequant_bulk_ms),
        ("unified_kv_cache_update (op total)", breakdown.kv_cache_update_ms),
        ("unified_attention_with_output (op total)", breakdown.attention_ms),
        ("flash_attn (prefill)", breakdown.flash_attn_ms),
    ]

    print(f"{'Stage':<42} {'Self CUDA (ms)':>14} {'% of total':>10}")
    print("-" * 68)
    for label, ms in rows:
        pct = (ms / total_cuda_ms * 100) if total_cuda_ms > 0 else 0.0
        print(f"{label:<42} {ms:14.3f} {pct:9.1f}%")

    print("\nMatched TQ kernels (detail):")
    tq_names = (
        _STORE_KERNEL_PATTERNS
        + _STORE_PREPROCESS_OPS
        + (_DECODE_STAGE1, _DECODE_STAGE2, _DEQUANT_BULK)
    )
    for e in breakdown.entries:
        if any(p in e.name for p in tq_names):
            avg = e.self_cuda_ms / e.num_calls if e.num_calls else 0.0
            print(
                f"  {e.name}: {e.self_cuda_ms:.3f} ms total, "
                f"{e.num_calls} calls, {avg:.3f} ms/call"
            )

    print(
        "\nLimitation: decode_stage1 fuses dequantization and attention. "
        "aten::mm GEMM time is not attributed to TQ preprocess (shared "
        "with model matmuls)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TurboQuant stages from torch profiler output"
    )
    parser.add_argument(
        "profiler_file",
        nargs="?",
        default="vllm_profile/profiler_out_0.txt",
        help="Path to profiler_out_*.txt from TorchProfilerWrapper",
    )
    args = parser.parse_args()

    entries, total_cuda = parse_profiler_table(args.profiler_file)
    if not entries:
        print(f"No entries parsed from {args.profiler_file}", file=sys.stderr)
        sys.exit(1)

    breakdown = summarize_turboquant(entries)
    print_report(args.profiler_file, breakdown, total_cuda)


if __name__ == "__main__":
    main()
