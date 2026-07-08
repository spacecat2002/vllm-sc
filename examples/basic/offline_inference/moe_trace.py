# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect and visualize MoE routes and router-input activations.

Examples::

    # Prompts are submitted together; the trace keeps request boundaries.
    .venv/bin/python examples/basic/offline_inference/moe_trace.py collect \
        --model Qwen/Qwen3-30B-A3B --prompts prompts.txt \
        --output-dir /tmp/qwen3_moe_trace --ep-size 4 --trace-next-gate

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-experts --trace-dir /tmp/qwen3_moe_trace --layers 8 31

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-activations --trace-dir /tmp/qwen3_moe_trace \
        --metric cka --phase prefill

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-route-similarity --trace-dir /tmp/qwen3_moe_trace --phase prefill

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-next-gate-similarity --trace-dir /tmp/qwen3_moe_trace

``collect`` intentionally uses eager mode and writes tensors synchronously.
It is a research utility, not a serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


MOE_BACKEND_CHOICES = (
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "flashinfer_b12x",
    "marlin",
    "humming",
    "triton_unfused",
    "aiter",
    "emulation",
)

DEFAULT_PROMPTS = [
    "Explain why the sky is blue.",
    "Write a short Python merge-sort function.",
    "Summarize the causes of the French Revolution.",
    "Translate 'good morning' into Chinese.",
    "What is the difference between TCP and UDP?",
]


def _read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def _num_experts(hf_config: Any) -> int:
    for name in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(hf_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine the number of experts from model config")


def _collect_dp_rank(
    args: argparse.Namespace,
    indexed_prompts: list[tuple[int, str]],
    global_dp_rank: int,
    dp_master_port: int,
    shard_dir: Path,
) -> None:
    """Run one offline SPMD rank and save its request-level route shard."""
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(global_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.ep_size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    # Request-level routed-expert capture is not supported by Model Runner V2.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm import LLM, SamplingParams

    prompts = [prompt for _, prompt in indexed_prompts]
    llm = LLM(
        model=args.model,
        # VLLM_DP_SIZE supplies DP=N, so EP_SIZE = TP_SIZE * DP_SIZE = N.
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=args.max_model_len,
        max_num_seqs=len(prompts),
        enable_chunked_prefill=False,
        enforce_eager=True,
        enable_return_routed_experts=True,
        moe_backend=args.moe_backend,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    routes: dict[str, np.ndarray] = {}
    prompt_token_counts: dict[str, int] = {}
    for (sample_id, _), request_output in zip(indexed_prompts, request_outputs):
        routed_experts = request_output.outputs[0].routed_experts
        if routed_experts is None:
            raise RuntimeError("vLLM did not return routed experts")
        sample_key = f"sample_{sample_id:06d}"
        routes[sample_key] = routed_experts
        prompt_token_counts[sample_key] = len(request_output.prompt_token_ids)

    np.savez_compressed(shard_dir / f"rank_{global_dp_rank:05d}.npz", **routes)
    shard_metadata = {
        "num_experts": _num_experts(llm.model_config.hf_text_config),
        "prompt_token_counts": prompt_token_counts,
    }
    (shard_dir / f"rank_{global_dp_rank:05d}.json").write_text(
        json.dumps(shard_metadata), encoding="utf-8"
    )

    # Match the offline DP example: let engine loops settle before process exit.
    from time import sleep

    sleep(1)


def _merge_route_shards(
    output_dir: Path,
    shard_dir: Path,
    num_samples: int,
) -> tuple[int, list[int], list[int]]:
    routes: dict[str, np.ndarray] = {}
    prompt_token_counts: dict[str, int] = {}
    sample_dp_ranks: dict[str, int] = {}
    num_experts: int | None = None
    for route_path in sorted(shard_dir.glob("rank_*.npz")):
        with np.load(route_path) as shard:
            routes.update({key: shard[key].copy() for key in shard.files})
            rank_id = int(route_path.stem.removeprefix("rank_"))
            sample_dp_ranks.update({key: rank_id for key in shard.files})
        shard_metadata = json.loads(route_path.with_suffix(".json").read_text())
        shard_num_experts = int(shard_metadata["num_experts"])
        if num_experts is not None and shard_num_experts != num_experts:
            raise ValueError("DP ranks reported different numbers of experts")
        num_experts = shard_num_experts
        prompt_token_counts.update(shard_metadata["prompt_token_counts"])

    expected_keys = [f"sample_{sample_id:06d}" for sample_id in range(num_samples)]
    if sorted(routes) != expected_keys:
        raise RuntimeError(
            f"Route shards are incomplete: expected {len(expected_keys)} samples, "
            f"found {len(routes)}"
        )
    assert num_experts is not None
    np.savez_compressed(
        output_dir / "routes.npz",
        **{key: routes[key] for key in expected_keys},
    )
    counts = [int(prompt_token_counts[key]) for key in expected_keys]
    ranks = [sample_dp_ranks[key] for key in expected_keys]
    return num_experts, counts, ranks


def collect(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    activation_dir = output_dir / "activations"
    if (output_dir / "routes.npz").exists() or any(
        activation_dir.glob("rank_*")
    ):
        raise FileExistsError(
            f"Trace output already exists under {output_dir}; choose a new directory"
        )

    from uuid import uuid4

    shard_dir = output_dir / "route_shards" / uuid4().hex
    activation_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts)
    if args.ep_size > len(prompts):
        raise ValueError("--ep-size cannot exceed the number of prompts")

    # Child ranks inherit trace configuration from this parent process.
    os.environ["VLLM_MOE_TRACE_DIR"] = str(activation_dir)
    os.environ["VLLM_MOE_TRACE_MAX_STEPS"] = str(
        len(prompts) * args.max_new_tokens
    )
    os.environ["VLLM_MOE_TRACE_MAX_TOKENS"] = str(args.max_tokens_per_sample)
    os.environ["VLLM_MOE_TRACE_ACTIVATIONS"] = "input"
    os.environ["VLLM_MOE_TRACE_ACTIVATION_DTYPE"] = args.activation_dtype
    os.environ["VLLM_MOE_TRACE_TOKEN_SELECTION"] = "prefill_last"
    os.environ["VLLM_MOE_TRACE_NEXT_GATE"] = "1" if args.trace_next_gate else "0"

    from multiprocessing import Process

    from vllm.platforms import current_platform
    from vllm.utils.network_utils import get_open_port

    if current_platform.is_rocm():
        from multiprocessing import set_start_method

        set_start_method("spawn", force=True)

    dp_master_port = get_open_port()
    indexed_prompts = list(enumerate(prompts))
    floor = len(indexed_prompts) // args.ep_size
    remainder = len(indexed_prompts) % args.ep_size

    def shard_start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    processes: list[Process] = []
    for global_dp_rank in range(args.ep_size):
        rank_prompts = indexed_prompts[
            shard_start(global_dp_rank) : shard_start(global_dp_rank + 1)
        ]
        process = Process(
            target=_collect_dp_rank,
            args=(args, rank_prompts, global_dp_rank, dp_master_port, shard_dir),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=args.timeout)
        if process.exitcode is None:
            process.kill()
            raise TimeoutError(f"DP process {process.pid} exceeded --timeout")
        if process.exitcode != 0:
            raise RuntimeError(
                f"DP process {process.pid} exited with code {process.exitcode}"
            )

    num_experts, prompt_token_counts, sample_dp_ranks = _merge_route_shards(
        output_dir,
        shard_dir,
        len(prompts),
    )

    metadata = {
        "model": args.model,
        "expert_parallel_size": args.ep_size,
        "num_samples": len(prompts),
        "num_experts": num_experts,
        "prompt_token_counts": prompt_token_counts,
        "sample_dp_ranks": sample_dp_ranks,
        "route_shape": "[token, layer, top_k]",
        "activation_point": "input to each MoE router",
        "activation_token_selection": (
            "last prompt token for prefill; each token for decode"
        ),
        "trace_next_gate": args.trace_next_gate,
        "moe_backend": args.moe_backend,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved routes and activations under {output_dir}")


def _load_route_samples(trace_dir: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    route_path = trace_dir / "routes.npz"
    with np.load(route_path) as data:
        samples = [data[key].copy() for key in sorted(data.files)]
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    return samples, metadata


def _expert_load_share(
    samples: list[np.ndarray],
    layer_id: int,
    num_experts: int,
    prompt_token_counts: list[int],
    phase: str,
) -> np.ndarray:
    shares = np.zeros((len(samples), num_experts), dtype=np.float64)
    for sample_id, routes in enumerate(samples):
        if routes.ndim != 3 or layer_id >= routes.shape[1]:
            raise ValueError(
                f"Layer {layer_id} is unavailable in route shape {routes.shape}"
            )
        prompt_tokens = prompt_token_counts[sample_id]
        if phase == "prefill":
            selected_routes = routes[:prompt_tokens]
        elif phase == "decode":
            selected_routes = routes[prompt_tokens:]
        else:
            selected_routes = routes
        ids = selected_routes[:, layer_id, :].reshape(-1)
        ids = ids[(ids >= 0) & (ids < num_experts)]
        counts = np.bincount(ids, minlength=num_experts)
        if counts.sum() > 0:
            shares[sample_id] = counts / counts.sum() * 100.0
    return shares


def _route_ids_for_phase(
    routes: np.ndarray,
    prompt_token_count: int,
    phase: str,
) -> np.ndarray:
    if phase == "prefill":
        return routes[:prompt_token_count]
    if phase == "decode":
        return routes[prompt_token_count:]
    return routes


def _expert_load_counts(
    layer_routes: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    ids = layer_routes.reshape(-1)
    ids = ids[(ids >= 0) & (ids < num_experts)]
    return np.bincount(ids, minlength=num_experts).astype(np.float64)


def _expert_load_cosine(
    layer_routes: np.ndarray,
    next_layer_routes: np.ndarray,
    num_experts: int,
) -> float:
    counts = _expert_load_counts(layer_routes, num_experts)
    next_counts = _expert_load_counts(next_layer_routes, num_experts)
    denominator = np.linalg.norm(counts) * np.linalg.norm(next_counts)
    if denominator == 0:
        return np.nan
    return float(np.dot(counts, next_counts) / denominator)


def _topk_overlap_ratio(
    layer_routes: np.ndarray,
    next_layer_routes: np.ndarray,
    num_experts: int,
) -> float:
    num_tokens = min(layer_routes.shape[0], next_layer_routes.shape[0])
    if num_tokens == 0:
        return np.nan

    similarities = _topk_overlap_values(
        layer_routes[:num_tokens],
        next_layer_routes[:num_tokens],
        num_experts,
    )
    if not similarities:
        return np.nan
    return float(np.mean(similarities))


def _topk_overlap_values(
    candidate_routes: np.ndarray,
    reference_routes: np.ndarray,
    num_experts: int,
) -> list[float]:
    num_tokens = min(candidate_routes.shape[0], reference_routes.shape[0])
    similarities: list[float] = []
    for token_id in range(num_tokens):
        current = {
            int(expert_id)
            for expert_id in candidate_routes[token_id]
            if 0 <= expert_id < num_experts
        }
        following = {
            int(expert_id)
            for expert_id in reference_routes[token_id]
            if 0 <= expert_id < num_experts
        }
        denominator = max(len(current), len(following))
        if denominator:
            similarities.append(len(current & following) / denominator)
    return similarities


def _adjacent_route_similarities(
    samples: list[np.ndarray],
    metadata: dict[str, Any],
    num_experts: int,
    phase: str,
) -> list[dict[str, float | int]]:
    prompt_token_counts = metadata["prompt_token_counts"]
    if not samples:
        raise ValueError("No route samples found")

    num_layers = min(routes.shape[1] for routes in samples)
    rows: list[dict[str, float | int]] = []
    for layer_id in range(num_layers - 1):
        topk_overlap_values = []
        load_cosine_values = []
        for sample_id, routes in enumerate(samples):
            if routes.ndim != 3:
                raise ValueError(
                    "Expected route shape [token, layer, top_k], "
                    f"got {routes.shape}"
                )
            selected_routes = _route_ids_for_phase(
                routes,
                int(prompt_token_counts[sample_id]),
                phase,
            )
            if selected_routes.shape[0] == 0:
                continue
            layer_routes = selected_routes[:, layer_id, :]
            next_layer_routes = selected_routes[:, layer_id + 1, :]
            topk_overlap = _topk_overlap_ratio(
                layer_routes,
                next_layer_routes,
                num_experts,
            )
            load_cosine = _expert_load_cosine(
                layer_routes,
                next_layer_routes,
                num_experts,
            )
            if not np.isnan(topk_overlap):
                topk_overlap_values.append(topk_overlap)
            if not np.isnan(load_cosine):
                load_cosine_values.append(load_cosine)

        rows.append(
            {
                "layer_i": layer_id,
                "layer_j": layer_id + 1,
                "num_samples": len(topk_overlap_values),
                "topk_overlap_mean": float(np.mean(topk_overlap_values))
                if topk_overlap_values
                else np.nan,
                "topk_overlap_std": float(np.std(topk_overlap_values))
                if topk_overlap_values
                else np.nan,
                "load_cosine_mean": float(np.mean(load_cosine_values))
                if load_cosine_values
                else np.nan,
                "load_cosine_std": float(np.std(load_cosine_values))
                if load_cosine_values
                else np.nan,
            }
        )
    return rows


def _categorical_expert_colors(num_experts: int) -> np.ndarray:
    """Generate discrete colors with strong contrast between adjacent IDs."""
    from matplotlib.colors import hsv_to_rgb

    expert_ids = np.arange(num_experts)
    # Golden-ratio hue stepping prevents adjacent IDs from receiving nearby
    # colors. Alternating saturation and value further separates dense bands.
    hues = np.mod(expert_ids * 0.618033988749895, 1.0)
    saturations = np.where(expert_ids % 2 == 0, 0.68, 0.92)
    values = np.where((expert_ids // 2) % 2 == 0, 0.92, 0.72)
    return hsv_to_rgb(np.column_stack((hues, saturations, values)))


def plot_experts(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    samples, metadata = _load_route_samples(trace_dir)
    num_experts = args.num_experts or int(metadata["num_experts"])
    layers = args.layers or [0]
    prompt_token_counts = metadata["prompt_token_counts"]
    colors = _categorical_expert_colors(num_experts)

    fig, axes = plt.subplots(
        len(layers),
        2,
        figsize=(13, max(4.0, 3.8 * len(layers))),
        squeeze=False,
        sharey=True,
    )
    x = np.arange(len(samples))
    for row, layer_id in enumerate(layers):
        shares = _expert_load_share(
            samples,
            layer_id,
            num_experts,
            prompt_token_counts,
            args.phase,
        )
        sorted_shares = np.sort(shares, axis=1)[:, ::-1]

        axes[row, 0].stackplot(
            x,
            shares.T,
            colors=colors,
            edgecolor="#202020",
            linewidth=0.15,
        )
        axes[row, 0].set_title(f"Layer {layer_id}: fixed expert IDs")
        axes[row, 1].stackplot(
            x,
            sorted_shares.T,
            colors=colors,
            edgecolor="#202020",
            linewidth=0.15,
        )
        axes[row, 1].set_title(f"Layer {layer_id}: experts sorted by load")
        axes[row, 0].set_ylabel("Expert load share (%)")
        for axis in axes[row]:
            axis.set_xlim(0, max(len(samples) - 1, 1))
            axis.set_ylim(0, 100)
            axis.set_xlabel("Sample")
            axis.grid(alpha=0.15)

    title = metadata.get("model", "MoE expert load distribution")
    fig.suptitle(f"{title} ({args.phase})", fontsize=14)
    fig.tight_layout()
    output = args.output or trace_dir / "expert_distribution.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def _write_route_similarity_data(
    output: Path,
    rows: list[dict[str, float | int]],
    metadata: dict[str, Any],
    phase: str,
) -> None:
    csv_header = (
        "layer_i,layer_j,num_samples,topk_overlap_mean,topk_overlap_std,"
        "load_cosine_mean,load_cosine_std"
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            ",".join(
                [
                    str(row["layer_i"]),
                    str(row["layer_j"]),
                    str(row["num_samples"]),
                    f"{row['topk_overlap_mean']:.6f}",
                    f"{row['topk_overlap_std']:.6f}",
                    f"{row['load_cosine_mean']:.6f}",
                    f"{row['load_cosine_std']:.6f}",
                ]
            )
        )
    output.with_suffix(".csv").write_text(
        csv_header + "\n" + "\n".join(csv_rows) + "\n",
        encoding="utf-8",
    )

    layer_pairs = np.array(
        [[row["layer_i"], row["layer_j"]] for row in rows],
        dtype=np.int64,
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        layer_pairs=layer_pairs,
        topk_overlap_mean=np.array(
            [row["topk_overlap_mean"] for row in rows],
            dtype=np.float64,
        ),
        topk_overlap_std=np.array(
            [row["topk_overlap_std"] for row in rows],
            dtype=np.float64,
        ),
        load_cosine_mean=np.array(
            [row["load_cosine_mean"] for row in rows],
            dtype=np.float64,
        ),
        load_cosine_std=np.array(
            [row["load_cosine_std"] for row in rows],
            dtype=np.float64,
        ),
    )

    json_payload = {
        "phase": phase,
        "metrics": {
            "topk_overlap": (
                "Mean per-token overlap ratio between adjacent layers' top-k "
                "expert sets: same experts divided by the per-token top-k size."
            ),
            "load_cosine": (
                "Cosine similarity between adjacent layers' aggregate expert "
                "assignment count vectors."
            ),
        },
        "model": metadata.get("model"),
        "num_experts": metadata.get("num_experts"),
        "rows": rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )


def plot_route_similarity(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    samples, metadata = _load_route_samples(trace_dir)
    num_experts = args.num_experts or int(metadata["num_experts"])
    rows = _adjacent_route_similarities(
        samples,
        metadata,
        num_experts,
        args.phase,
    )
    if not rows:
        raise ValueError("Need at least two routed MoE layers to compare")

    layer_labels = [f"{row['layer_i']}→{row['layer_j']}" for row in rows]
    x = np.arange(len(rows))
    metric_specs = {
        "topk-overlap": (
            "topk_overlap_mean",
            "topk_overlap_std",
            "Same expert / top-k",
        ),
        "load-cosine": ("load_cosine_mean", "load_cosine_std", "Expert load cosine"),
    }
    selected_metrics = (
        list(metric_specs)
        if args.metric == "all"
        else [args.metric]
    )

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.55), 4.8))
    for metric in selected_metrics:
        mean_key, std_key, label = metric_specs[metric]
        means = np.array([row[mean_key] for row in rows], dtype=np.float64)
        stds = np.array([row[std_key] for row in rows], dtype=np.float64)
        ax.plot(x, means, marker="o", linewidth=1.8, label=label)
        ax.fill_between(
            x,
            np.maximum(means - stds, 0.0),
            np.minimum(means + stds, 1.0),
            alpha=0.15,
        )

    ax.set_xticks(x, labels=layer_labels, rotation=90)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Adjacent layer pair")
    ax.set_ylabel("Similarity")
    ax.grid(alpha=0.2)
    ax.legend()
    title = metadata.get("model", "MoE route similarity")
    ax.set_title(f"{title}: adjacent routed-expert overlap ({args.phase})")
    fig.tight_layout()

    output = args.output or trace_dir / f"route_similarity_{args.phase}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    _write_route_similarity_data(output, rows, metadata, args.phase)
    print(f"Saved {output} and matching .csv/.json/.npz data")


def _phase_mask(record: dict[str, Any], phase: str) -> np.ndarray | None:
    if phase == "all":
        return None
    token_phases = record.get("token_phases")
    if token_phases is None:
        return None
    phase_code = 0 if phase == "prefill" else 1
    return token_phases.numpy() == phase_code


def _load_next_gate_overlap_rows(
    trace_dir: Path,
    rank: int | None,
    num_experts: int,
    phase: str,
) -> list[dict[str, float | int]]:
    import torch

    activation_root = trace_dir / "activations"
    if rank is None:
        rank_dirs = sorted(activation_root.glob("rank_*"))
    else:
        rank_dirs = [activation_root / f"rank_{rank:05d}"]

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rank_dir in rank_dirs:
        if not rank_dir.exists():
            continue
        rank_id = int(rank_dir.name.removeprefix("rank_"))
        for path in sorted(rank_dir.glob("step_*_layer_*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            key = (rank_id, int(record["step"]), int(record["layer_id"]))
            records[key] = record
    if not records:
        raise ValueError(f"No activation records found under {activation_root}")

    grouped_values: dict[tuple[int, int], list[float]] = {}
    grouped_steps: dict[tuple[int, int], int] = {}
    for (rank_id, step, layer_id), record in records.items():
        if "next_gate_predicted_topk_ids" not in record:
            continue
        next_layer_id = int(record["next_gate_layer_id"])
        next_record = records.get((rank_id, step, next_layer_id))
        if next_record is None:
            continue

        predicted = record["next_gate_predicted_topk_ids"].numpy()
        actual = next_record["topk_ids"].numpy()
        mask = _phase_mask(record, phase)
        if mask is not None:
            predicted = predicted[mask]
            actual = actual[mask]
        if predicted.shape[0] == 0 or actual.shape[0] == 0:
            continue

        pair = (layer_id, next_layer_id)
        values = _topk_overlap_values(predicted, actual, num_experts)
        if values:
            grouped_values.setdefault(pair, []).extend(values)
            grouped_steps[pair] = grouped_steps.get(pair, 0) + 1

    if not grouped_values:
        raise ValueError(
            "No next-gate predictions found. Re-run collect with "
            "--trace-next-gate to save next_gate_predicted_topk_ids."
        )

    rows: list[dict[str, float | int]] = []
    for layer_i, layer_j in sorted(grouped_values):
        values = grouped_values[(layer_i, layer_j)]
        rows.append(
            {
                "layer_i": layer_i,
                "layer_j": layer_j,
                "num_steps": grouped_steps[(layer_i, layer_j)],
                "num_tokens": len(values),
                "topk_overlap_mean": float(np.mean(values)),
                "topk_overlap_std": float(np.std(values)),
            }
        )
    return rows


def _write_next_gate_similarity_data(
    output: Path,
    rows: list[dict[str, float | int]],
    metadata: dict[str, Any],
    phase: str,
) -> None:
    csv_header = (
        "layer_i,layer_j,num_steps,num_tokens,topk_overlap_mean,topk_overlap_std"
    )
    csv_rows = []
    for row in rows:
        csv_rows.append(
            ",".join(
                [
                    str(row["layer_i"]),
                    str(row["layer_j"]),
                    str(row["num_steps"]),
                    str(row["num_tokens"]),
                    f"{row['topk_overlap_mean']:.6f}",
                    f"{row['topk_overlap_std']:.6f}",
                ]
            )
        )
    output.with_suffix(".csv").write_text(
        csv_header + "\n" + "\n".join(csv_rows) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        output.with_suffix(".npz"),
        layer_pairs=np.array(
            [[row["layer_i"], row["layer_j"]] for row in rows],
            dtype=np.int64,
        ),
        topk_overlap_mean=np.array(
            [row["topk_overlap_mean"] for row in rows],
            dtype=np.float64,
        ),
        topk_overlap_std=np.array(
            [row["topk_overlap_std"] for row in rows],
            dtype=np.float64,
        ),
        num_tokens=np.array([row["num_tokens"] for row in rows], dtype=np.int64),
    )

    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "phase": phase,
                "metric": (
                    "For each layer pair i->i+1, feed layer i's traced MoE "
                    "router input into layer i+1's gate/router, then compare "
                    "that predicted top-k set with layer i+1's actual top-k "
                    "set. The value is same experts divided by top-k."
                ),
                "model": metadata.get("model"),
                "num_experts": metadata.get("num_experts"),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_next_gate_similarity(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    num_experts = args.num_experts or int(metadata["num_experts"])
    rows = _load_next_gate_overlap_rows(
        trace_dir,
        args.rank,
        num_experts,
        args.phase,
    )

    layer_labels = [f"{row['layer_i']}→{row['layer_j']}" for row in rows]
    x = np.arange(len(rows))
    means = np.array([row["topk_overlap_mean"] for row in rows], dtype=np.float64)
    stds = np.array([row["topk_overlap_std"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.55), 4.8))
    ax.plot(x, means, marker="o", linewidth=1.8, label="Same expert / top-k")
    ax.fill_between(
        x,
        np.maximum(means - stds, 0.0),
        np.minimum(means + stds, 1.0),
        alpha=0.15,
    )
    ax.set_xticks(x, labels=layer_labels, rotation=90)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer i activation -> layer i+1 gate")
    ax.set_ylabel("Overlap ratio")
    ax.grid(alpha=0.2)
    ax.legend()
    title = metadata.get("model", "MoE next-gate overlap")
    ax.set_title(f"{title}: predicted vs actual next-layer top-k ({args.phase})")
    fig.tight_layout()

    output = args.output or trace_dir / f"next_gate_similarity_{args.phase}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    _write_next_gate_similarity_data(output, rows, metadata, args.phase)
    print(f"Saved {output} and matching .csv/.json/.npz data")


def _load_activations(
    trace_dir: Path,
    rank: int | None,
    max_tokens: int,
    phase: str,
) -> tuple[list[int], np.ndarray]:
    import torch

    activation_root = trace_dir / "activations"
    if rank is None:
        rank_dirs = sorted(activation_root.glob("rank_*"))
    else:
        rank_dirs = [activation_root / f"rank_{rank:05d}"]

    records: dict[tuple[int, int, int], torch.Tensor] = {}
    for rank_dir in rank_dirs:
        rank_id = int(rank_dir.name.removeprefix("rank_"))
        for path in sorted(rank_dir.glob("step_*_layer_*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if "activations" not in record:
                continue
            activations = record["activations"]
            record_phase = record.get("phase", "all")
            if phase != "all":
                token_phases = record.get("token_phases")
                if token_phases is not None:
                    phase_code = 0 if phase == "prefill" else 1
                    activations = activations[token_phases == phase_code]
                elif record_phase != phase:
                    continue
            if activations.shape[0] == 0:
                continue
            key = (rank_id, int(record["step"]), int(record["layer_id"]))
            records[key] = activations
    if not records:
        rank_label = "all ranks" if rank is None else f"rank {rank}"
        raise ValueError(f"No activation records found for {rank_label}")

    rank_steps = sorted({(rank_id, step) for rank_id, step, _ in records})
    layers = sorted({layer for _, _, layer in records})
    chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    remaining = max_tokens
    for rank_id, step in rank_steps:
        if remaining <= 0 or any(
            (rank_id, step, layer) not in records for layer in layers
        ):
            continue
        common_tokens = min(
            records[(rank_id, step, layer)].shape[0] for layer in layers
        )
        common_tokens = min(common_tokens, remaining)
        for layer in layers:
            chunks[layer].append(
                records[(rank_id, step, layer)][:common_tokens].float()
            )
        remaining -= common_tokens

    if any(not chunks[layer] for layer in layers):
        raise ValueError("No complete step containing every traced MoE layer")
    activations = torch.stack(
        [torch.cat(chunks[layer], dim=0) for layer in layers], dim=0
    )
    return layers, activations.numpy()


def _pairwise_cosine(activations: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(activations, axis=-1, keepdims=True)
    normalized = activations / np.maximum(norms, 1e-12)
    return np.einsum("lnd,mnd->lm", normalized, normalized) / activations.shape[1]


def _pairwise_linear_cka(activations: np.ndarray) -> np.ndarray:
    # With centered features, normalized Frobenius products of token Gram
    # matrices are the standard biased linear CKA estimator.
    gram_vectors = []
    for layer_activations in activations:
        centered = layer_activations - layer_activations.mean(axis=0, keepdims=True)
        gram = centered @ centered.T
        vector = gram.reshape(-1)
        gram_vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
    normalized_grams = np.stack(gram_vectors)
    return normalized_grams @ normalized_grams.T


def _write_similarity_data(
    output: Path, layers: list[int], similarity: np.ndarray
) -> None:
    np.save(output.with_suffix(".npy"), similarity)
    header = "layer," + ",".join(str(layer) for layer in layers)
    rows = [
        str(layer) + "," + ",".join(f"{value:.6f}" for value in row)
        for layer, row in zip(layers, similarity)
    ]
    output.with_suffix(".csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )


def plot_activations(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    layers, activations = _load_activations(
        trace_dir,
        args.rank,
        args.max_tokens,
        args.phase,
    )
    if args.metric == "cka":
        similarity = _pairwise_linear_cka(activations)
        metric_title = "Linear CKA"
    else:
        similarity = _pairwise_cosine(activations)
        metric_title = "Mean token-wise cosine similarity"

    num_layers = len(layers)
    side = max(8.0, num_layers * 0.42)
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    image = ax.imshow(
        similarity,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )
    ax.set_xticks(range(num_layers), labels=layers, rotation=90)
    ax.set_yticks(range(num_layers), labels=layers)
    ax.set_xlabel("Compared layer")
    ax.set_ylabel("Target layer")
    ax.set_title(
        f"All-pairs MoE activation similarity ({metric_title}, {args.phase})"
    )

    font_size = max(3.0, min(8.0, 180.0 / num_layers))
    for row in range(num_layers):
        for col in range(num_layers):
            value = similarity[row, col]
            text_color = "black" if value > 0.62 else "white"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Similarity")
    fig.tight_layout()
    output = args.output or trace_dir / (
        f"activation_similarity_{args.metric}_{args.phase}.png"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    _write_similarity_data(output, layers, similarity)
    print(f"Saved {output} and matching .npy/.csv data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--prompts", type=Path)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser.add_argument(
        "--ep-size",
        dest="ep_size",
        type=int,
        default=1,
        help=(
            "Number of expert-parallel ranks. The script uses TP=1, DP=N, "
            "and enable_expert_parallel=True, so EP size is N."
        ),
    )
    collect_parser.add_argument("--max-model-len", type=int, default=4096)
    collect_parser.add_argument("--max-new-tokens", type=int, default=16)
    collect_parser.add_argument("--max-tokens-per-sample", type=int, default=4096)
    collect_parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum seconds to wait for each offline DP rank.",
    )
    collect_parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    collect_parser.add_argument(
        "--trace-next-gate",
        action="store_true",
        help=(
            "Also save, for each layer i, the top-k experts predicted by "
            "feeding layer i's traced router input to layer i+1's gate/router."
        ),
    )
    collect_parser.add_argument(
        "--moe-backend",
        choices=MOE_BACKEND_CHOICES,
        default="auto",
        help="MoE expert-kernel backend to pass to vLLM, e.g. triton.",
    )
    collect_parser.set_defaults(func=collect)

    expert_parser = subparsers.add_parser("plot-experts")
    expert_parser.add_argument("--trace-dir", type=Path, required=True)
    expert_parser.add_argument("--layers", type=int, nargs="+")
    expert_parser.add_argument("--num-experts", type=int)
    expert_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="prefill"
    )
    expert_parser.add_argument("--output", type=Path)
    expert_parser.set_defaults(func=plot_experts)

    route_similarity_parser = subparsers.add_parser("plot-route-similarity")
    route_similarity_parser.add_argument("--trace-dir", type=Path, required=True)
    route_similarity_parser.add_argument("--num-experts", type=int)
    route_similarity_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="all"
    )
    route_similarity_parser.add_argument(
        "--metric",
        choices=("topk-overlap", "load-cosine", "all"),
        default="topk-overlap",
    )
    route_similarity_parser.add_argument("--output", type=Path)
    route_similarity_parser.set_defaults(func=plot_route_similarity)

    next_gate_parser = subparsers.add_parser("plot-next-gate-similarity")
    next_gate_parser.add_argument("--trace-dir", type=Path, required=True)
    next_gate_parser.add_argument(
        "--rank",
        type=int,
        help="Analyze one EP rank; by default traces from all ranks are merged.",
    )
    next_gate_parser.add_argument("--num-experts", type=int)
    next_gate_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="all"
    )
    next_gate_parser.add_argument("--output", type=Path)
    next_gate_parser.set_defaults(func=plot_next_gate_similarity)

    activation_parser = subparsers.add_parser("plot-activations")
    activation_parser.add_argument("--trace-dir", type=Path, required=True)
    activation_parser.add_argument(
        "--rank",
        type=int,
        help="Analyze one EP rank; by default traces from all ranks are merged.",
    )
    activation_parser.add_argument("--max-tokens", type=int, default=512)
    activation_parser.add_argument("--metric", choices=("cka", "cosine"), default="cka")
    activation_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="all"
    )
    activation_parser.add_argument("--output", type=Path)
    activation_parser.set_defaults(func=plot_activations)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
