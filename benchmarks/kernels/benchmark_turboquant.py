# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for TurboQuant stage-level GPU timing.

Uses CUDA events on the default stream to measure each stage independently.
Decode stage1 includes fused dequantization + attention (not separable).

Example:
    .venv/bin/python benchmarks/kernels/benchmark_turboquant.py \\
        --preset turboquant_k3v4_nc --seq-len 1024 --batch-size 32
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    solve_lloyd_max,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _fwd_kernel_stage2,
    _get_layout,
    _tq_decode_stage1,
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    _tq_fused_store_fp8,
    _tq_fused_store_mse,
    triton_turboquant_store,
)

DEVICE = torch.device(current_platform.device_type)


def _build_hadamard(d: int, device: torch.device) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device)


@dataclass
class StageTiming:
    mean_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def timed_stages(
    stages: dict[str, Callable[[], None]],
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> dict[str, StageTiming]:
    """Time ordered GPU stages with CUDA events on the default stream."""
    for fn in stages.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    accum: dict[str, list[float]] = {k: [] for k in stages}
    for _ in range(repeats):
        prev = torch.cuda.Event(enable_timing=True)
        prev.record()
        for name, fn in stages.items():
            fn()
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            torch.cuda.synchronize()
            accum[name].append(prev.elapsed_time(evt))
            prev = evt

    result: dict[str, StageTiming] = {}
    for name, vals in accum.items():
        sorted_vals = sorted(vals)
        p50 = sorted_vals[len(sorted_vals) // 2]
        p99 = sorted_vals[int(len(sorted_vals) * 0.99)]
        result[name] = StageTiming(
            mean_ms=statistics.mean(vals),
            p50_ms=p50,
            p99_ms=p99,
            min_ms=min(vals),
            max_ms=max(vals),
        )
    return result


def _print_table(title: str, timings: dict[str, StageTiming]) -> None:
    print(f"\n{title}")
    print("-" * 72)
    print(f"{'Stage':<22} {'mean':>8} {'p50':>8} {'p99':>8} {'min':>8} {'max':>8}")
    print(f"{'':22} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8}")
    print("-" * 72)
    total = 0.0
    for name, t in timings.items():
        print(
            f"{name:<22} {t.mean_ms:8.3f} {t.p50_ms:8.3f} "
            f"{t.p99_ms:8.3f} {t.min_ms:8.3f} {t.max_ms:8.3f}"
        )
        total += t.mean_ms
    print("-" * 72)
    print(f"{'TOTAL (sum of means)':<22} {total:8.3f}")


def _setup_tq_tensors(
    preset: str,
    head_dim: int,
    num_kv_heads: int,
    num_q_heads: int,
    seq_len: int,
    batch_size: int,
    block_size: int,
):
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)
    D = head_dim
    Hk = num_kv_heads
    Hq = num_q_heads
    device = DEVICE

    H = _build_hadamard(D, device)
    PiT = H
    Pi = H
    Pi_half = H.to(torch.float16)

    centroids, _ = solve_lloyd_max(D, cfg.centroid_bits)
    centroids = centroids.float().to(device)
    c_sorted, _ = centroids.sort()
    midpoints = ((c_sorted[:-1] + c_sorted[1:]) / 2).to(device)

    num_blocks = math.ceil(seq_len / block_size) + 1
    padded_slot = cfg.slot_size_aligned
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        Hk,
        padded_slot,
        device=device,
        dtype=torch.uint8,
    )

    torch.manual_seed(42)
    key = torch.randn(batch_size, Hk, D, device=device, dtype=torch.float16)
    value = torch.randn(batch_size, Hk, D, device=device, dtype=torch.float16)
    query = torch.randn(batch_size, Hq, D, device=device, dtype=torch.float16)

    slot_mapping = torch.arange(batch_size, device=device, dtype=torch.int32)
    block_table = torch.zeros(batch_size, num_blocks, device=device, dtype=torch.int32)
    for i in range(batch_size):
        block_table[i, 0] = i % num_blocks
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    return {
        "cfg": cfg,
        "D": D,
        "Hk": Hk,
        "Hq": Hq,
        "Pi": Pi,
        "PiT": PiT,
        "Pi_half": Pi_half,
        "centroids": centroids,
        "midpoints": midpoints,
        "kv_cache": kv_cache,
        "key": key,
        "value": value,
        "query": query,
        "slot_mapping": slot_mapping,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "block_size": block_size,
        "num_blocks": num_blocks,
    }


def benchmark_store(
    tensors: dict,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, StageTiming]:
    cfg = tensors["cfg"]
    key = tensors["key"]
    value = tensors["value"]
    kv_cache = tensors["kv_cache"]
    slot_mapping = tensors["slot_mapping"]
    PiT = tensors["PiT"]
    midpoints = tensors["midpoints"]

    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    n_centroids = 2**cfg.key_mse_bits
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1
    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)
    grid = (NH,)

    if cfg.key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()
        fp8_e4b15 = _use_fp8_e4b15(key.device.index or 0)

        def run_kernel():
            _tq_fused_store_fp8[grid](
                k_flat,
                v_flat,
                kv_cache.view(-1),
                slot_mapping,
                stride_cache_block=stride_block,
                stride_cache_pos=stride_pos,
                stride_cache_head=stride_head,
                D=D,
                H=H,
                BLOCK_SIZE=block_size,
                BLOCK_D=BLOCK_D,
                KPS=cfg.key_packed_size,
                VQB=cfg.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                BLOCK_VAL=BLOCK_VAL,
                BLOCK_GRP=block_grp,
                FP8_E4B15=fp8_e4b15,
                num_warps=4,
                num_stages=1,
            )

        return timed_stages({"store_kernel": run_kernel}, warmup=warmup, repeats=repeats)

    k_flat_buf = key.float().reshape(NH, D)
    v_flat_buf = value.float().reshape(NH, D)
    y_buf = torch.empty(NH, D, device=DEVICE, dtype=torch.float32)
    norms_buf = torch.empty(NH, 1, device=DEVICE, dtype=torch.float32)

    def preprocess():
        norms = k_flat_buf.norm(dim=1, keepdim=True)
        norms_buf.copy_(norms)
        y_buf.copy_(k_flat_buf / (norms + 1e-8) @ PiT)

    def run_kernel():
        _tq_fused_store_mse[grid](
            y_buf,
            norms_buf.squeeze(1),
            v_flat_buf,
            midpoints,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=cfg.key_mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )

    return timed_stages(
        {"store_preprocess": preprocess, "store_kernel": run_kernel},
        warmup=warmup,
        repeats=repeats,
    )


def benchmark_decode(
    tensors: dict,
    *,
    warmup: int,
    repeats: int,
    max_num_kv_splits: int,
) -> dict[str, StageTiming]:
    cfg = tensors["cfg"]
    D = tensors["D"]
    Hk = tensors["Hk"]
    Hq = tensors["Hq"]
    query = tensors["query"]
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    seq_lens = tensors["seq_lens"]
    Pi = tensors["Pi"]
    PiT = tensors["PiT"]
    centroids = tensors["centroids"]
    block_size = tensors["block_size"]

    B = query.shape[0]
    scale = 1.0 / math.sqrt(D)
    layout = _get_layout(
        D, cfg.key_mse_bits, cfg.effective_value_quant_bits, cfg.key_packed_size
    )
    NUM_KV_SPLITS = max_num_kv_splits
    kv_group_size = Hq // Hk
    fp8_e4b15 = _use_fp8_e4b15(query.device.index or 0)
    BLOCK_KV = 4
    grid = (B, Hq, NUM_KV_SPLITS)
    grid2 = (B, Hq)

    mid_o = torch.empty(
        B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=DEVICE
    )
    output = torch.empty(B, Hq, D, dtype=query.dtype, device=DEVICE)
    lse = torch.empty(B, Hq, dtype=torch.float32, device=DEVICE)

    if cfg.key_fp8:
        q_rot = query.contiguous()
        stages: dict[str, Callable[[], None]] = {}
    else:
        q_float = query.float()
        q_rot_buf = torch.empty(B, Hq, D, dtype=torch.float32, device=DEVICE)

        def q_rotate():
            q_rot_buf.copy_(q_float @ PiT)

        q_rot = q_rot_buf
        stages = {"decode_q_rotate": q_rotate}

    def stage1():
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=cfg.key_mse_bits,
            MSE_BYTES=layout["mse_bytes"],
            KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=layout["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=layout["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if cfg.key_fp8 else 0,
            NORM_CORRECTION=1 if cfg.norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )

    def stage2():
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=layout["BLOCK_D"],
            Lv=D,
            OUTPUT_FP16=1 if query.dtype == torch.float16 else 0,
            num_warps=4,
            num_stages=2,
        )

    stages["decode_stage1"] = stage1
    stages["decode_stage2"] = stage2
    return timed_stages(stages, warmup=warmup, repeats=repeats)


def benchmark_dequant_bulk(
    tensors: dict,
    *,
    cached_len: int,
    warmup: int,
    repeats: int,
) -> dict[str, StageTiming]:
    cfg = tensors["cfg"]
    D = tensors["D"]
    Hk = tensors["Hk"]
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"][:1]
    centroids = tensors["centroids"]
    Pi_half = tensors["Pi_half"]
    block_size = tensors["block_size"]

    mse_bytes = math.ceil(D * cfg.key_mse_bits / 8)
    val_data_bytes = math.ceil(D * cfg.effective_value_quant_bits / 8)
    BLOCK_D = triton.next_power_of_2(D)
    alloc_len = math.ceil(cached_len / block_size) * block_size

    k_buf = torch.empty(1, Hk, alloc_len, D, dtype=torch.float16, device=DEVICE)
    v_buf = torch.empty(1, Hk, alloc_len, D, dtype=torch.float16, device=DEVICE)
    grid = (alloc_len, Hk)

    def dequant():
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            centroids,
            k_buf,
            v_buf,
            k_buf.stride(0),
            k_buf.stride(1),
            k_buf.stride(2),
            v_buf.stride(0),
            v_buf.stride(1),
            v_buf.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            KPS=cfg.key_packed_size,
            VQB=cfg.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=cfg.key_mse_bits,
            KEY_FP8=1 if cfg.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if cfg.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(0),
            num_warps=4,
        )

    stages: dict[str, Callable[[], None]] = {"dequant_bulk": dequant}
    if not cfg.key_fp8:
        k_flat_buf = torch.empty(Hk * cached_len, D, dtype=torch.float16, device=DEVICE)

        def inverse_rotate():
            k_flat = k_buf[0, :, :cached_len, :].reshape(-1, D)
            k_flat_buf.copy_(k_flat @ Pi_half)

        stages["inverse_rotate"] = inverse_rotate

    return timed_stages(stages, warmup=warmup, repeats=repeats)


def benchmark_e2e(
    tensors: dict,
    *,
    warmup: int,
    repeats: int,
    max_num_kv_splits: int,
) -> dict[str, StageTiming]:
    cfg = tensors["cfg"]

    def store():
        triton_turboquant_store(
            tensors["key"],
            tensors["value"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["PiT"],
            tensors["midpoints"],
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
        )

    def decode():
        triton_turboquant_decode_attention(
            query=tensors["query"],
            kv_cache=tensors["kv_cache"],
            block_table=tensors["block_table"],
            seq_lens=tensors["seq_lens"],
            Pi=tensors["Pi"],
            centroids=tensors["centroids"],
            scale=1.0 / math.sqrt(tensors["D"]),
            mse_bits=cfg.key_mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.effective_value_quant_bits,
            key_fp8=cfg.key_fp8,
            norm_correction=cfg.norm_correction,
            PiT=tensors["PiT"],
            max_num_kv_splits=max_num_kv_splits,
        )

    return timed_stages(
        {"e2e_store": store, "e2e_decode": decode},
        warmup=warmup,
        repeats=repeats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant stage microbenchmark")
    parser.add_argument(
        "--preset",
        default="turboquant_k3v4_nc",
        choices=[
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-q-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cached-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-kv-splits", type=int, default=32)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["store", "decode", "dequant", "e2e"],
        choices=["store", "decode", "dequant", "e2e"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for TurboQuant benchmark")

    torch.set_default_device("cuda")
    tensors = _setup_tq_tensors(
        preset=args.preset,
        head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads,
        num_q_heads=args.num_q_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    # Pre-populate cache for decode/dequant benchmarks.
    cfg = tensors["cfg"]
    triton_turboquant_store(
        tensors["key"],
        tensors["value"],
        tensors["kv_cache"],
        tensors["slot_mapping"],
        tensors["PiT"],
        tensors["midpoints"],
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )
    torch.cuda.synchronize()

    header = (
        f"TurboQuant benchmark: preset={args.preset}, batch={args.batch_size}, "
        f"seq_len={args.seq_len}, head_dim={args.head_dim}"
    )
    print(header)
    print(
        "Note: decode_stage1 = fused dequant + attention (not separable without "
        "kernel changes)."
    )

    if "store" in args.paths:
        store_timings = benchmark_store(tensors, warmup=args.warmup, repeats=args.repeats)
        _print_table("Store path", store_timings)

    if "decode" in args.paths:
        decode_timings = benchmark_decode(
            tensors,
            warmup=args.warmup,
            repeats=args.repeats,
            max_num_kv_splits=args.max_kv_splits,
        )
        _print_table("Decode path", decode_timings)

    if "dequant" in args.paths:
        dequant_timings = benchmark_dequant_bulk(
            tensors,
            cached_len=args.cached_len,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _print_table(
            f"Continuation prefill dequant (cached_len={args.cached_len})",
            dequant_timings,
        )

    if "e2e" in args.paths:
        e2e_timings = benchmark_e2e(
            tensors,
            warmup=args.warmup,
            repeats=args.repeats,
            max_num_kv_splits=args.max_kv_splits,
        )
        _print_table("End-to-end launcher (store + decode)", e2e_timings)


if __name__ == "__main__":
    main()
