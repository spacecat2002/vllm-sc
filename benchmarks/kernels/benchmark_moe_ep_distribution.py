# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark isolated MoE EP dispatch/compute/combine stages.

Run with torchrun. This script bypasses the router and constructs topk_ids
directly, so token distribution across EP ranks is fully controlled.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_profile import rank_distribution
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.worker.workspace import init_workspace_manager


MODEL_PRESETS = {
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        "intermediate_size": 768,
        "num_experts": 128,
        "top_k": 8,
        "dtype": "bfloat16",
    },
    "qwen3-235b-a22b": {
        "hidden_size": 4096,
        "intermediate_size": 1536,
        "num_experts": 128,
        "top_k": 8,
        "dtype": "bfloat16",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated MoE EP distribution benchmark."
    )
    parser.add_argument(
        "--model-preset",
        choices=tuple(MODEL_PRESETS),
        help=(
            "Fill expert-related shape arguments from a known model preset. "
            "Currently supports Qwen3 MoE routed expert presets."
        ),
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
        ),
        default="deepep_low_latency",
    )
    parser.add_argument(
        "--pattern",
        choices=(
            "balanced",
            "concentrated",
            "local",
            "remote",
            "random",
            "skewed",
        ),
        default="balanced",
    )
    parser.add_argument(
        "--hot-rank",
        type=int,
        default=0,
        help="Target rank that receives the hot share for --pattern skewed.",
    )
    parser.add_argument(
        "--hot-share",
        type=float,
        default=0.5,
        help=(
            "Fraction of top-k assignments sent to --hot-rank for "
            "--pattern skewed. Use 1 / world_size for balanced-like routing "
            "and 1.0 for concentrated-like routing."
        ),
    )
    parser.add_argument(
        "--hot-shares",
        help=(
            "Comma-separated hot-share values to sweep in one distributed "
            "initialization, e.g. '0,0.25,0.5,0.75,1'. Overrides --hot-share."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of iterations to drop from each tail when computing "
            "summary means. Rows are trimmed by max_total_ms."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage-barrier",
        action="store_true",
        help=(
            "Synchronize all ranks before each measured stage. This removes "
            "arrival skew from dispatch/compute/combine timings, at the cost "
            "of making the benchmark less like the original pipeline."
        ),
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help=(
            "Number of local worker processes to spawn when not launched with "
            "torchrun. Defaults to torch.cuda.device_count(). Ignored when "
            "RANK/WORLD_SIZE/LOCAL_RANK are already set."
        ),
    )
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        help="Rendezvous address for internal multi-process launch.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Rendezvous port for internal multi-process launch.",
    )
    args = parser.parse_args()
    if args.model_preset is not None:
        for name, value in MODEL_PRESETS[args.model_preset].items():
            setattr(args, name, value)
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024 * 1024)


def parse_hot_share_values(args: argparse.Namespace) -> list[float]:
    if args.hot_shares is None:
        return [args.hot_share]
    values = []
    for raw_value in args.hot_shares.split(","):
        value = raw_value.strip()
        if value:
            values.append(float(value))
    if not values:
        raise ValueError("--hot-shares must contain at least one value.")
    return values


def trim_aggregates(
    aggregates: list[dict[str, Any]],
    trim_ratio: float,
) -> list[dict[str, Any]]:
    trim_count = int(len(aggregates) * trim_ratio)
    if trim_count == 0 or trim_count * 2 >= len(aggregates):
        return aggregates
    sorted_by_total = sorted(aggregates, key=lambda record: record["max_total_ms"])
    kept = sorted_by_total[trim_count:-trim_count]
    return sorted(kept, key=lambda record: record["iter"])


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sync_stage_start(device: torch.device, use_barrier: bool) -> None:
    sync(device)
    if use_barrier:
        dist.barrier()
        sync(device)


def time_stage(
    device: torch.device,
    fn,
    *,
    use_barrier: bool = False,
) -> tuple[Any, float]:
    sync_stage_start(device, use_barrier)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def contiguous_experts_for_rank(
    rank: int,
    num_local_experts: int,
) -> list[int]:
    start = rank * num_local_experts
    return list(range(start, start + num_local_experts))


def cycle_pool(pool: list[int], rows: int, cols: int, offset: int) -> torch.Tensor:
    values = [pool[(offset + i) % len(pool)] for i in range(rows * cols)]
    return torch.tensor(values, dtype=torch.int64).view(rows, cols)


def make_rank_skewed_ids(
    tokens: int,
    top_k: int,
    num_local_experts: int,
    world_size: int,
    source_rank: int,
    hot_rank: int,
    hot_share: float,
) -> torch.Tensor:
    total = tokens * top_k
    num_experts = num_local_experts * world_size
    if top_k > num_experts:
        raise ValueError(
            f"top_k={top_k} cannot be greater than num_experts={num_experts}."
        )
    normalized_hot_rank = hot_rank % world_size
    counts = [0] * world_size
    if world_size == 1:
        counts[0] = total
    else:
        hot_count = round(total * hot_share)
        counts[normalized_hot_rank] = hot_count
        other_ranks = [
            rank for rank in range(world_size) if rank != normalized_hot_rank
        ]
        for index in range(total - hot_count):
            counts[other_ranks[index % len(other_ranks)]] += 1

    targets = []
    remaining = counts[:]
    start_rank = source_rank % world_size
    while sum(remaining) > 0:
        for offset in range(world_size):
            target_rank = (start_rank + offset) % world_size
            if remaining[target_rank] > 0:
                targets.append(target_rank)
                remaining[target_rank] -= 1

    seen_per_rank = [0] * world_size
    expert_ids = []
    for token_start in range(0, total, top_k):
        used_experts: set[int] = set()
        token_targets = targets[token_start : token_start + top_k]
        for target_rank in token_targets:
            for _ in range(num_local_experts):
                local_expert = seen_per_rank[target_rank] % num_local_experts
                expert_id = target_rank * num_local_experts + local_expert
                seen_per_rank[target_rank] += 1
                if expert_id not in used_experts:
                    expert_ids.append(expert_id)
                    used_experts.add(expert_id)
                    break
            else:
                raise ValueError(
                    "Cannot generate unique top-k experts for one token. "
                    f"top_k={top_k}, target_rank={target_rank}, "
                    f"num_local_experts={num_local_experts}."
                )
    return torch.tensor(expert_ids, dtype=torch.int64).view(tokens, top_k)


def make_topk_ids(
    pattern: str,
    tokens: int,
    top_k: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
    hot_rank: int,
    hot_share: float,
) -> torch.Tensor:
    num_local_experts = num_experts // world_size
    if pattern == "balanced":
        pool = list(range(num_experts))
        ids = cycle_pool(pool, tokens, top_k, offset=rank * tokens * top_k)
    elif pattern == "concentrated":
        pool = contiguous_experts_for_rank(0, num_local_experts)
        ids = cycle_pool(pool, tokens, top_k, offset=rank * top_k)
    elif pattern == "local":
        pool = contiguous_experts_for_rank(rank, num_local_experts)
        ids = cycle_pool(pool, tokens, top_k, offset=0)
    elif pattern == "remote":
        target_rank = (rank + 1) % world_size
        pool = contiguous_experts_for_rank(target_rank, num_local_experts)
        ids = cycle_pool(pool, tokens, top_k, offset=0)
    elif pattern == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + rank)
        ids = torch.randint(num_experts, (tokens, top_k), generator=generator)
    elif pattern == "skewed":
        ids = make_rank_skewed_ids(
            tokens,
            top_k,
            num_local_experts,
            world_size,
            rank,
            hot_rank,
            hot_share,
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return ids.to(device=device)


def distribution_metrics(counts: list[int]) -> dict[str, float | int]:
    total = sum(counts)
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
        "active_ranks": sum(1 for count in counts if count > 0),
        "max_share": max(counts) / total,
        "imbalance": max(counts) / mean if mean else 0.0,
        "cv": variance**0.5 / mean if mean else 0.0,
    }


def local_expert_token_counts(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
) -> list[int]:
    valid_topk_ids = topk_ids >= 0
    clamped_topk_ids = torch.where(
        valid_topk_ids,
        topk_ids.to(torch.int64),
        torch.zeros_like(topk_ids, dtype=torch.int64),
    )
    local_ids = expert_map[clamped_topk_ids].to(torch.int64)
    valid_local_ids = valid_topk_ids & (local_ids >= 0)
    if not torch.any(valid_local_ids):
        return [0] * num_local_experts
    counts = torch.bincount(
        local_ids[valid_local_ids],
        minlength=num_local_experts,
    )
    return [int(x) for x in counts[:num_local_experts].detach().cpu().tolist()]


def make_expert_map(
    num_experts: int,
    num_local_experts: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    start = rank * num_local_experts
    end = start + num_local_experts
    expert_map[start:end] = torch.arange(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    return expert_map


def make_vllm_config(
    args: argparse.Namespace,
    world_size: int,
    rank: int,
    local_rank: int,
):
    vllm_config = VllmConfig()
    parallel_config = vllm_config.parallel_config
    parallel_config.data_parallel_size = world_size
    parallel_config.data_parallel_rank = rank
    parallel_config.enable_expert_parallel = True
    parallel_config.is_moe_model = True
    parallel_config.all2all_backend = args.backend
    parallel_config.distributed_executor_backend = "external_launcher"
    vllm_config.device_config.device = torch.device("cuda", local_rank)
    return vllm_config


def make_kernel(
    args: argparse.Namespace,
    vllm_config: VllmConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> mk.FusedMoEKernel:
    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        pcp_size_=get_pcp_group().world_size,
        dp_size_=get_dp_group().world_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    num_local_experts = args.num_experts // dist.get_world_size()
    moe_config = FusedMoEConfig(
        num_experts=args.num_experts,
        experts_per_token=args.top_k,
        hidden_dim=args.hidden_size,
        intermediate_size_per_partition=args.intermediate_size,
        num_local_experts=num_local_experts,
        num_logical_experts=args.num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=dtype,
        max_num_tokens=next_power_of_2(args.tokens),
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make()
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None
    fused_experts = TritonExperts(moe_config, quant_config)
    return mk.FusedMoEKernel(prepare_finalize, fused_experts)


def make_inputs(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + rank)
    num_local_experts = args.num_experts // world_size
    hidden_states = torch.randn(
        (args.tokens, args.hidden_size),
        device=device,
        dtype=dtype,
    )
    w1 = torch.randn(
        (num_local_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=dtype,
    ) / 10
    w2 = torch.randn(
        (num_local_experts, args.hidden_size, args.intermediate_size),
        device=device,
        dtype=dtype,
    ) / 10
    topk_ids = make_topk_ids(
        args.pattern,
        args.tokens,
        args.top_k,
        args.num_experts,
        rank,
        world_size,
        device,
        args.seed,
        args.hot_rank,
        args.hot_share,
    )
    topk_weights = torch.full(
        (args.tokens, args.top_k),
        1.0 / args.top_k,
        device=device,
        dtype=torch.float32,
    )
    expert_map = make_expert_map(args.num_experts, num_local_experts, rank, device)
    return {
        "hidden_states": hidden_states,
        "w1": w1.contiguous(),
        "w2": w2.contiguous(),
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "expert_map": expert_map,
    }


def run_one_iter(
    args: argparse.Namespace,
    kernel: mk.FusedMoEKernel,
    tensors: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    device: torch.device,
    iteration: int,
) -> dict[str, Any]:
    assert isinstance(kernel.impl, mk.FusedMoEKernelModularImpl)
    hidden_states = tensors["hidden_states"]
    topk_ids = tensors["topk_ids"]
    topk_weights = tensors["topk_weights"]
    requested_dtype = kernel.prepare_finalize.topk_indices_dtype()
    if requested_dtype is not None:
        topk_ids = topk_ids.to(requested_dtype)

    target_assignments, target_unique_tokens = rank_distribution(
        topk_ids,
        args.num_experts,
        world_size,
        round_robin=False,
    )
    total_assignments = sum(target_assignments)
    local_assignments = target_assignments[rank]
    remote_assignment_share = (
        (total_assignments - local_assignments) / total_assignments
        if total_assignments
        else 0.0
    )
    remote_unique_tokens = sum(target_unique_tokens) - target_unique_tokens[rank]
    token_bytes = args.hidden_size * dtype_nbytes(hidden_states.dtype)
    dispatch_remote_bytes = remote_unique_tokens * token_bytes
    combine_remote_bytes = dispatch_remote_bytes
    output = torch.empty_like(hidden_states)
    local_num_experts = tensors["w1"].shape[0]

    def prepare():
        return kernel.impl._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            args.num_experts,
            tensors["expert_map"],
            False,
        )

    (
        a1q,
        a1q_scale,
        expert_tokens_meta,
        dispatched_topk_ids,
        dispatched_topk_weights,
    ), dispatch_ms = time_stage(device, prepare, use_barrier=args.stage_barrier)

    def compute():
        return kernel.impl._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=tensors["w1"],
            w2=tensors["w2"],
            topk_weights=dispatched_topk_weights,
            topk_ids=dispatched_topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=args.num_experts,
            local_num_experts=local_num_experts,
            expert_map=tensors["expert_map"],
            apply_router_weight_on_input=False,
            expert_tokens_meta=expert_tokens_meta,
            output_alias=output,
        )

    fused_out, compute_ms = time_stage(
        device,
        compute,
        use_barrier=args.stage_barrier,
    )

    def finalize():
        return kernel.impl._finalize(
            output,
            fused_out,
            hidden_states,
            dispatched_topk_weights,
            dispatched_topk_ids,
            False,
            None,
            None,
        )

    _, combine_ms = time_stage(device, finalize, use_barrier=args.stage_barrier)

    if expert_tokens_meta is not None:
        local_expert_tokens = [
            int(x) for x in expert_tokens_meta.expert_num_tokens.detach().cpu().tolist()
        ]
    else:
        local_expert_tokens = local_expert_token_counts(
            dispatched_topk_ids,
            tensors["expert_map"],
            local_num_experts,
        )

    return {
        "record_type": "rank",
        "pattern": args.pattern,
        "backend": args.backend,
        "hot_share": args.hot_share,
        "stage_barrier": args.stage_barrier,
        "iter": iteration,
        "rank": rank,
        "world_size": world_size,
        "tokens": args.tokens,
        "top_k": args.top_k,
        "num_experts": args.num_experts,
        "num_local_experts": local_num_experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "dispatch_ms": dispatch_ms,
        "expert_compute_ms": compute_ms,
        "combine_ms": combine_ms,
        "total_ms": dispatch_ms + compute_ms + combine_ms,
        "source_target_assignments": target_assignments,
        "source_target_unique_tokens": target_unique_tokens,
        "source_distribution": distribution_metrics(target_assignments),
        "remote_assignment_share": remote_assignment_share,
        "remote_unique_tokens": remote_unique_tokens,
        "token_bytes": token_bytes,
        "dispatch_remote_bytes": dispatch_remote_bytes,
        "combine_remote_bytes": combine_remote_bytes,
        "a2a_remote_bytes": dispatch_remote_bytes + combine_remote_bytes,
        "local_expert_tokens": local_expert_tokens,
        "received_tokens": sum(local_expert_tokens),
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_iter: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_iter[int(record["iter"])].append(record)

    aggregates = []
    for iteration, iter_records in sorted(by_iter.items()):
        dispatch = [r["dispatch_ms"] for r in iter_records]
        compute = [r["expert_compute_ms"] for r in iter_records]
        combine = [r["combine_ms"] for r in iter_records]
        total = [r["total_ms"] for r in iter_records]
        received = [r["received_tokens"] for r in iter_records]
        remote_shares = [r["remote_assignment_share"] for r in iter_records]
        dispatch_remote_bytes = [r["dispatch_remote_bytes"] for r in iter_records]
        combine_remote_bytes = [r["combine_remote_bytes"] for r in iter_records]
        a2a_remote_bytes = [r["a2a_remote_bytes"] for r in iter_records]
        target_assignments = [0] * int(iter_records[0]["world_size"])
        remote_recv_bytes = [0] * int(iter_records[0]["world_size"])
        for record in iter_records:
            for target_rank, count in enumerate(record["source_target_assignments"]):
                target_assignments[target_rank] += count
            for target_rank, tokens in enumerate(record["source_target_unique_tokens"]):
                if target_rank != record["rank"]:
                    remote_recv_bytes[target_rank] += tokens * record["token_bytes"]
        mean_compute = statistics.mean(compute)
        mean_received = statistics.mean(received)
        mean_remote_recv_bytes = statistics.mean(remote_recv_bytes)
        target_metrics = distribution_metrics(target_assignments)
        aggregates.append(
            {
                "record_type": "aggregate",
                "iter": iteration,
                "hot_share": iter_records[0]["hot_share"],
                "max_dispatch_ms": max(dispatch),
                "mean_dispatch_ms": statistics.mean(dispatch),
                "max_expert_compute_ms": max(compute),
                "mean_expert_compute_ms": mean_compute,
                "max_combine_ms": max(combine),
                "mean_combine_ms": statistics.mean(combine),
                "max_total_ms": max(total),
                "mean_total_ms": statistics.mean(total),
                "compute_imbalance": max(compute) / mean_compute
                if mean_compute
                else 0.0,
                "received_tokens_max": max(received),
                "received_tokens_min": min(received),
                "received_tokens_imbalance": max(received) / mean_received
                if mean_received
                else 0.0,
                "mean_remote_assignment_share": statistics.mean(remote_shares),
                "target_assignment_imbalance": target_metrics["imbalance"],
                "target_assignment_max_share": target_metrics["max_share"],
                "dispatch_remote_bytes_total": sum(dispatch_remote_bytes),
                "dispatch_remote_bytes_max": max(dispatch_remote_bytes),
                "combine_remote_bytes_total": sum(combine_remote_bytes),
                "a2a_remote_bytes_total": sum(a2a_remote_bytes),
                "remote_recv_bytes_max": max(remote_recv_bytes),
                "remote_recv_bytes_imbalance": max(remote_recv_bytes)
                / mean_remote_recv_bytes
                if mean_remote_recv_bytes
                else 0.0,
            }
        )
    return aggregates


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def make_sweep_summary_row(
    args: argparse.Namespace,
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_aggregates = trim_aggregates(aggregates, args.trim_ratio)
    return {
        "hot_share": aggregates[0]["hot_share"],
        "iters": len(aggregates),
        "trimmed_iters": len(summary_aggregates),
        "mean_max_total_ms": statistics.mean(
            r["max_total_ms"] for r in summary_aggregates
        ),
        "mean_max_dispatch_ms": statistics.mean(
            r["max_dispatch_ms"] for r in summary_aggregates
        ),
        "mean_max_compute_ms": statistics.mean(
            r["max_expert_compute_ms"] for r in summary_aggregates
        ),
        "mean_max_combine_ms": statistics.mean(
            r["max_combine_ms"] for r in summary_aggregates
        ),
        "mean_compute_imbalance": statistics.mean(
            r["compute_imbalance"] for r in summary_aggregates
        ),
        "mean_remote_share": statistics.mean(
            r["mean_remote_assignment_share"] for r in summary_aggregates
        ),
        "mean_target_imbalance": statistics.mean(
            r["target_assignment_imbalance"] for r in summary_aggregates
        ),
        "mean_dispatch_mib": statistics.mean(
            bytes_to_mib(r["dispatch_remote_bytes_total"])
            for r in summary_aggregates
        ),
        "mean_a2a_mib": statistics.mean(
            bytes_to_mib(r["a2a_remote_bytes_total"]) for r in summary_aggregates
        ),
        "mean_max_recv_mib": statistics.mean(
            bytes_to_mib(r["remote_recv_bytes_max"]) for r in summary_aggregates
        ),
    }


def print_sweep_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print()
    print("sweep summary")
    print(
        "hot_share iters trimmed mean_total mean_dispatch mean_compute "
        "mean_combine compute_imbalance remote_share target_imbalance "
        "dispatch_mib a2a_mib max_recv_mib"
    )
    for row in rows:
        print(
            f"{row['hot_share']:>9.3f} "
            f"{row['iters']:>5} "
            f"{row['trimmed_iters']:>7} "
            f"{row['mean_max_total_ms']:>10.3f} "
            f"{row['mean_max_dispatch_ms']:>13.3f} "
            f"{row['mean_max_compute_ms']:>12.3f} "
            f"{row['mean_max_combine_ms']:>12.3f} "
            f"{row['mean_compute_imbalance']:>17.3f} "
            f"{row['mean_remote_share']:>12.3f} "
            f"{row['mean_target_imbalance']:>16.3f} "
            f"{row['mean_dispatch_mib']:>12.3f} "
            f"{row['mean_a2a_mib']:>7.3f} "
            f"{row['mean_max_recv_mib']:>12.3f}"
        )


def print_summary(args: argparse.Namespace, aggregates: list[dict[str, Any]]) -> None:
    if not aggregates:
        return
    summary_aggregates = trim_aggregates(aggregates, args.trim_ratio)
    print(
        "iter max_total max_dispatch max_compute max_combine "
        "compute_imbalance recv_min recv_max recv_imbalance "
        "remote_share target_imbalance dispatch_mib a2a_mib max_recv_mib"
    )
    for record in aggregates:
        print(
            f"{record['iter']:>4} "
            f"{record['max_total_ms']:>9.3f} "
            f"{record['max_dispatch_ms']:>12.3f} "
            f"{record['max_expert_compute_ms']:>11.3f} "
            f"{record['max_combine_ms']:>11.3f} "
            f"{record['compute_imbalance']:>17.3f} "
            f"{record['received_tokens_min']:>8} "
            f"{record['received_tokens_max']:>8} "
            f"{record['received_tokens_imbalance']:>14.3f} "
            f"{record['mean_remote_assignment_share']:>12.3f} "
            f"{record['target_assignment_imbalance']:>16.3f} "
            f"{bytes_to_mib(record['dispatch_remote_bytes_total']):>12.3f} "
            f"{bytes_to_mib(record['a2a_remote_bytes_total']):>7.3f} "
            f"{bytes_to_mib(record['remote_recv_bytes_max']):>12.3f}"
        )

    print()
    print(
        f"pattern={args.pattern} backend={args.backend} "
        f"hot_share={aggregates[0]['hot_share']} "
        f"iters={len(aggregates)} trimmed_iters={len(summary_aggregates)} "
        f"trim_ratio={args.trim_ratio}"
    )
    print(
        "mean(max_dispatch_ms)="
        f"{statistics.mean(r['max_dispatch_ms'] for r in summary_aggregates):.3f}, "
        "mean(max_compute_ms)="
        f"{statistics.mean(r['max_expert_compute_ms'] for r in summary_aggregates):.3f}, "
        "mean(max_combine_ms)="
        f"{statistics.mean(r['max_combine_ms'] for r in summary_aggregates):.3f}, "
        "mean(compute_imbalance)="
        f"{statistics.mean(r['compute_imbalance'] for r in summary_aggregates):.3f}"
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _set_distributed_env(
    rank: int,
    world_size: int,
    local_rank: int,
    master_addr: str,
    master_port: int,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)


def _run_worker(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by WORLD_SIZE.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = dtype_from_name(args.dtype)

    vllm_config = make_vllm_config(args, world_size, rank, local_rank)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        init_workspace_manager(device)

        kernel = make_kernel(args, vllm_config, dtype, device)
        num_tokens_across_dp = torch.full(
            (world_size,),
            args.tokens,
            device=device,
            dtype=torch.int,
        )
        hot_share_values = args.hot_share_values
        output_records: list[dict[str, Any]] = []
        sweep_summary_rows: list[dict[str, Any]] = []

        with set_forward_context(
            None,
            vllm_config,
            num_tokens=args.tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            def run_iterations(
                tensors: dict[str, torch.Tensor],
            ) -> list[dict[str, Any]]:
                for _ in range(args.warmup):
                    run_one_iter(
                        args,
                        kernel,
                        tensors,
                        rank,
                        world_size,
                        device,
                        -1,
                    )
                dist.barrier()

                records = []
                for iteration in range(args.iters):
                    records.append(
                        run_one_iter(
                            args,
                            kernel,
                            tensors,
                            rank,
                            world_size,
                            device,
                            iteration,
                        )
                    )
                dist.barrier()
                return records

            ctx = get_forward_context()

            for hot_share in hot_share_values:
                args.hot_share = hot_share
                tensors = make_inputs(args, rank, world_size, dtype, device)
                if ctx.dp_metadata is None:
                    local_records = run_iterations(tensors)
                else:
                    with ctx.dp_metadata.sp_local_sizes(sequence_parallel_size=1):
                        local_records = run_iterations(tensors)

                gathered: list[Any] | None = (
                    [None for _ in range(world_size)] if rank == 0 else None
                )
                dist.gather_object(local_records, gathered, dst=0)

                if rank == 0:
                    assert gathered is not None
                    records = [
                        record for rank_records in gathered for record in rank_records
                    ]
                    aggregates = aggregate_records(records)
                    output_records.extend(records + aggregates)
                    sweep_summary_rows.append(
                        make_sweep_summary_row(args, aggregates)
                    )
                    print_summary(args, aggregates)

            if rank == 0:
                print_sweep_summary(sweep_summary_rows)

        if rank == 0 and args.output_jsonl is not None:
            write_jsonl(args.output_jsonl, output_records)
            print(f"Wrote JSONL: {args.output_jsonl}")

    if dist.is_initialized():
        dist.destroy_process_group()


def _destroy_distributed_if_initialized() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn_worker(
    local_rank: int,
    args: argparse.Namespace,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    rank = local_rank
    _set_distributed_env(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    try:
        _run_worker(args, rank, world_size, local_rank)
    finally:
        _destroy_distributed_if_initialized()


def _launched_with_torchrun() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def main() -> None:
    args = parse_args()
    args.hot_share_values = parse_hot_share_values(args)

    for hot_share in args.hot_share_values:
        if not 0.0 <= hot_share <= 1.0:
            raise ValueError("--hot-share values must be between 0.0 and 1.0.")
    if not 0.0 <= args.trim_ratio < 0.5:
        raise ValueError("--trim-ratio must be in [0.0, 0.5).")

    if _launched_with_torchrun():
        try:
            _run_worker(
                args=args,
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
                local_rank=int(os.environ["LOCAL_RANK"]),
            )
        finally:
            _destroy_distributed_if_initialized()
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    world_size = args.nproc_per_node or torch.cuda.device_count()
    if world_size <= 0:
        raise ValueError("--nproc-per-node must be positive.")
    if world_size > torch.cuda.device_count():
        raise ValueError(
            f"--nproc-per-node={world_size} exceeds visible CUDA device count "
            f"{torch.cuda.device_count()}."
        )
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by --nproc-per-node.")

    master_port = args.master_port or _find_free_port()
    mp.spawn(
        _spawn_worker,
        args=(args, world_size, args.master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
