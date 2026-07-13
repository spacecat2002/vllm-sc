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


_ASSISTANT_ROLES = {"assistant", "gpt"}
_ROLE_LABELS = {
    "assistant": "Assistant",
    "gpt": "Assistant",
    "human": "User",
    "system": "System",
    "user": "User",
}


def _iter_json_prompt_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "prompts", "instances", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        conversations = payload.get("conversations") or payload.get("messages")
        if isinstance(conversations, list):
            return [payload]
    return []


def _format_sharegpt_prompt(item: Any) -> str | None:
    if isinstance(item, str):
        prompt = item.strip()
        return prompt or None
    if not isinstance(item, dict):
        return None

    conversations = item.get("conversations") or item.get("messages")
    if not isinstance(conversations, list):
        prompt = item.get("prompt") or item.get("text")
        if isinstance(prompt, str):
            prompt = prompt.strip()
            return prompt or None
        return None

    turns: list[tuple[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").strip().lower()
        text = turn.get("value")
        if text is None:
            text = turn.get("content")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        turns.append((role, text))

    if not turns:
        return None
    if turns[-1][0] in _ASSISTANT_ROLES:
        turns = turns[:-1]
    if not turns:
        return None

    lines = []
    for role, text in turns:
        label = _ROLE_LABELS.get(role, role.title() if role else "User")
        lines.append(f"{label}: {text}")
    if turns[-1][0] not in _ASSISTANT_ROLES:
        lines.append("Assistant:")
    return "\n".join(lines)


def _read_json_prompts(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
    else:
        items = _iter_json_prompt_items(json.loads(path.read_text(encoding="utf-8")))

    prompts = []
    for item in items:
        prompt = _format_sharegpt_prompt(item)
        if prompt:
            prompts.append(prompt)
    return prompts


def _read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    if path.suffix.lower() in (".json", ".jsonl"):
        prompts = _read_json_prompts(path)
    else:
        prompts = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def _sample_prompts(
    prompts: list[str],
    *,
    num_prompts: int | None,
    sample_mode: str,
    seed: int,
) -> tuple[list[str], list[int]]:
    source_indices = list(range(len(prompts)))
    if num_prompts is None or num_prompts >= len(prompts):
        return prompts, source_indices
    if num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")

    if sample_mode == "first":
        selected_indices = source_indices[:num_prompts]
    elif sample_mode == "random":
        rng = np.random.default_rng(seed)
        selected_indices = sorted(
            int(index)
            for index in rng.choice(len(prompts), size=num_prompts, replace=False)
        )
    else:
        raise ValueError(f"Unsupported --prompt-sample-mode: {sample_mode}")

    return [prompts[index] for index in selected_indices], selected_indices


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
    os.environ["VLLM_MOE_TRACE_NEXT_GATE"] = "1" if args.trace_next_gate else "0"
    if args.next_gate_lora_dir is not None:
        os.environ["VLLM_MOE_TRACE_NEXT_GATE_LORA_DIR"] = str(
            args.next_gate_lora_dir.resolve()
        )
    else:
        os.environ.pop("VLLM_MOE_TRACE_NEXT_GATE_LORA_DIR", None)

    from vllm import LLM, SamplingParams

    prompts = [prompt for _, prompt in indexed_prompts]
    collect_batch_size = args.collect_batch_size or len(prompts)
    llm = LLM(
        model=args.model,
        # VLLM_DP_SIZE supplies DP=N, so EP_SIZE = TP_SIZE * DP_SIZE = N.
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=args.max_model_len,
        max_num_seqs=min(len(prompts), collect_batch_size),
        enforce_eager=True,
        enable_return_routed_experts=True,
        moe_backend=args.moe_backend,
        load_format=args.load_format,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    routes: dict[str, np.ndarray] = {}
    prompt_token_counts: dict[str, int] = {}
    generated_texts: dict[str, str] = {}
    for start in range(0, len(indexed_prompts), collect_batch_size):
        batch_indexed_prompts = indexed_prompts[start : start + collect_batch_size]
        batch_prompts = [prompt for _, prompt in batch_indexed_prompts]
        request_outputs = llm.generate(
            batch_prompts,
            sampling_params,
            use_tqdm=True,
        )
        for (sample_id, _), request_output in zip(
            batch_indexed_prompts,
            request_outputs,
        ):
            completion_output = request_output.outputs[0]
            routed_experts = completion_output.routed_experts
            if routed_experts is None:
                raise RuntimeError("vLLM did not return routed experts")
            sample_key = f"sample_{sample_id:06d}"
            routes[sample_key] = routed_experts
            prompt_token_counts[sample_key] = len(request_output.prompt_token_ids)
            generated_texts[sample_key] = completion_output.text

    np.savez_compressed(shard_dir / f"rank_{global_dp_rank:05d}.npz", **routes)
    shard_metadata = {
        "num_experts": _num_experts(llm.model_config.hf_text_config),
        "prompt_token_counts": prompt_token_counts,
        "generated_texts": generated_texts,
        "trace_next_gate_arg": args.trace_next_gate,
        "trace_next_gate_env": os.environ.get("VLLM_MOE_TRACE_NEXT_GATE"),
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
) -> tuple[int, list[int], list[int], list[str]]:
    routes: dict[str, np.ndarray] = {}
    prompt_token_counts: dict[str, int] = {}
    generated_texts: dict[str, str] = {}
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
        generated_texts.update(shard_metadata["generated_texts"])

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
    texts = [generated_texts[key] for key in expected_keys]
    return num_experts, counts, ranks, texts


def _check_next_gate_trace(output_dir: Path) -> None:
    import torch

    activation_dir = output_dir / "activations"
    rank_metadata_paths = sorted(activation_dir.glob("rank_*/metadata.json"))
    if not rank_metadata_paths:
        raise RuntimeError(f"No rank metadata found under {activation_dir}")

    has_predictor = False
    for metadata_path in rank_metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictor_count = int(metadata.get("next_gate_predictor_count", 0))
        trace_next_gate = bool(metadata.get("trace_next_gate", False))
        print(
            f"{metadata_path.parent.name}: trace_next_gate={trace_next_gate}, "
            f"next_gate_predictor_count={predictor_count}"
        )
        if predictor_count > 0:
            has_predictor = True
        elif trace_next_gate:
            diagnostics = metadata.get("next_gate_diagnostics", [])
            print(
                f"{metadata_path.parent.name}: next_gate_diagnostics sample="
                f"{diagnostics[:3]}"
            )

    if not has_predictor:
        raise RuntimeError(
            "--trace-next-gate was set, but no rank built any next-gate "
            "predictor. Check activations/rank_*/metadata.json for "
            "next_gate_diagnostics."
        )

    records = []
    for rank_dir in sorted(activation_dir.glob("rank_*")):
        records.extend(_iter_activation_records(rank_dir, torch))
        if len(records) >= 32:
            break
    if not records:
        raise RuntimeError(f"No activation records found under {activation_dir}")

    for record in records[: min(len(records), 32)]:
        if "next_gate_predicted_topk_ids" in record:
            return

    raise RuntimeError(
        "Next-gate predictors were built, but the sampled .pt records do not "
        "contain next_gate_predicted_topk_ids. Check whether only final-layer "
        "records were sampled or inspect all activation records."
    )


def _packed_activation_paths(rank_dir: Path) -> list[Path]:
    return sorted(rank_dir.glob("records_*.pt"))


def _iter_activation_records(
    rank_dir: Path,
    torch_module: Any,
) -> list[dict[str, Any]]:
    packed_paths = _packed_activation_paths(rank_dir)
    if packed_paths:
        records = []
        for path in packed_paths:
            payload = torch_module.load(
                path, map_location="cpu", weights_only=True
            )
            records.extend(payload["records"])
        return records

    records = []
    for path in sorted(rank_dir.glob("step_*_layer_*.pt")):
        records.append(
            torch_module.load(path, map_location="cpu", weights_only=True)
        )
    return records


def _load_activation_records(
    activation_root: Path,
    rank: int | None,
    torch_module: Any,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    if rank is None:
        rank_dirs = sorted(activation_root.glob("rank_*"))
    else:
        rank_dirs = [activation_root / f"rank_{rank:05d}"]

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rank_dir in rank_dirs:
        if not rank_dir.exists():
            continue
        rank_id = int(rank_dir.name.removeprefix("rank_"))
        for record in _iter_activation_records(rank_dir, torch_module):
            key = (rank_id, int(record["step"]), int(record["layer_id"]))
            records[key] = record
    return records


def _pack_activation_records(
    activation_root: Path,
    *,
    delete_unpacked: bool,
) -> None:
    import torch

    for rank_dir in sorted(activation_root.glob("rank_*")):
        record_paths = sorted(rank_dir.glob("step_*_layer_*.pt"))
        if not record_paths:
            continue
        records = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in record_paths
        ]
        rank_id = int(rank_dir.name.removeprefix("rank_"))
        packed_path = rank_dir / f"records_rank_{rank_id:05d}.pt"
        torch.save(
            {
                "format_version": 1,
                "rank": rank_id,
                "num_records": len(records),
                "records": records,
            },
            packed_path,
        )
        if delete_unpacked:
            for path in record_paths:
                path.unlink()
        print(f"Packed {len(records)} records into {packed_path}")


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
    all_prompts = _read_prompts(args.prompts)
    prompts, source_prompt_indices = _sample_prompts(
        all_prompts,
        num_prompts=args.num_prompts,
        sample_mode=args.prompt_sample_mode,
        seed=args.prompt_sample_seed,
    )
    if args.ep_size > len(prompts):
        raise ValueError("--ep-size cannot exceed the number of prompts")
    if args.collect_batch_size is not None and args.collect_batch_size <= 0:
        raise ValueError("--collect-batch-size must be positive")

    # Child ranks inherit trace configuration from this parent process.
    os.environ["VLLM_MOE_TRACE_DIR"] = str(activation_dir)
    os.environ["VLLM_MOE_TRACE_MAX_STEPS"] = str(
        len(prompts) * args.max_new_tokens
    )
    os.environ["VLLM_MOE_TRACE_MAX_TOKENS"] = str(args.max_tokens_per_sample)
    os.environ["VLLM_MOE_TRACE_ACTIVATIONS"] = "input"
    os.environ["VLLM_MOE_TRACE_ACTIVATION_DTYPE"] = args.activation_dtype
    # Keep every scheduled prefill and decode token. The collector also saves
    # per-token phase labels, so plotting can isolate prefill without sampling
    # it down to the final prompt token.
    os.environ["VLLM_MOE_TRACE_TOKEN_SELECTION"] = "all"
    os.environ["VLLM_MOE_TRACE_NEXT_GATE"] = "1" if args.trace_next_gate else "0"
    if args.next_gate_lora_dir is not None:
        os.environ["VLLM_MOE_TRACE_NEXT_GATE_LORA_DIR"] = str(
            args.next_gate_lora_dir.resolve()
        )
    else:
        os.environ.pop("VLLM_MOE_TRACE_NEXT_GATE_LORA_DIR", None)

    import multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    dp_master_port = get_open_port()
    mp_context = mp.get_context("spawn")
    indexed_prompts = list(enumerate(prompts))
    floor = len(indexed_prompts) // args.ep_size
    remainder = len(indexed_prompts) % args.ep_size

    def shard_start(rank: int) -> int:
        return rank * floor + min(rank, remainder)

    processes: list[mp.Process] = []
    for global_dp_rank in range(args.ep_size):
        rank_prompts = indexed_prompts[
            shard_start(global_dp_rank) : shard_start(global_dp_rank + 1)
        ]
        process = mp_context.Process(
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

    (
        num_experts,
        prompt_token_counts,
        sample_dp_ranks,
        generated_texts,
    ) = _merge_route_shards(
        output_dir,
        shard_dir,
        len(prompts),
    )

    metadata = {
        "model": args.model,
        "expert_parallel_size": args.ep_size,
        "num_input_prompts": len(all_prompts),
        "num_samples": len(prompts),
        "source_prompt_indices": source_prompt_indices,
        "prompt_sample_mode": args.prompt_sample_mode,
        "prompt_sample_seed": args.prompt_sample_seed,
        "collect_batch_size": args.collect_batch_size,
        "num_experts": num_experts,
        "prompt_token_counts": prompt_token_counts,
        "sample_dp_ranks": sample_dp_ranks,
        "route_shape": "[token, layer, top_k]",
        "activation_point": "input to each MoE router",
        "activation_token_selection": "all scheduled prefill and decode tokens",
        "trace_next_gate": args.trace_next_gate,
        "next_gate_lora_dir": str(args.next_gate_lora_dir)
        if args.next_gate_lora_dir is not None
        else None,
        "moe_backend": args.moe_backend,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    generations = [
        {
            "sample_id": sample_id,
            "source_prompt_index": source_prompt_indices[sample_id],
            "dp_rank": sample_dp_ranks[sample_id],
            "prompt": prompt,
            "generated_text": generated_texts[sample_id],
        }
        for sample_id, prompt in enumerate(prompts)
    ]
    (output_dir / "generations.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # for item in generations:
    #     print(
    #         f"\n[sample {item['sample_id']:06d} | dp_rank {item['dp_rank']}]\n"
    #         f"Prompt:\n{item['prompt']}\n"
    #         f"Generated:\n{item['generated_text']}"
    #     )
    if args.trace_next_gate:
        _check_next_gate_trace(output_dir)
    if args.pack_activations:
        _pack_activation_records(
            activation_dir,
            delete_unpacked=args.delete_unpacked_activations,
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


def _expert_load_share_by_step(
    trace_dir: Path,
    layer_id: int,
    num_experts: int,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate expert load across ranks for each model-forward step."""
    import torch

    activation_root = trace_dir / "activations"
    records = _load_activation_records(activation_root, None, torch)
    counts_by_step: dict[int, np.ndarray] = {}
    for (_, step, record_layer_id), record in records.items():
        if record_layer_id != layer_id:
            continue

        ids = record["topk_ids"].numpy()
        mask = _phase_mask(record, phase)
        if mask is not None:
            ids = ids[mask]
        elif phase != "all":
            raise ValueError(
                "The activation trace has no token phase metadata; "
                f"cannot select phase {phase!r}"
            )

        ids = ids.reshape(-1)
        ids = ids[(ids >= 0) & (ids < num_experts)]
        counts = np.bincount(ids, minlength=num_experts).astype(np.float64)
        if step not in counts_by_step:
            counts_by_step[step] = counts
        else:
            counts_by_step[step] += counts

    if not counts_by_step:
        raise ValueError(
            f"Layer {layer_id} is unavailable under {activation_root}"
        )

    steps = np.asarray(sorted(counts_by_step), dtype=np.int64)
    shares = np.stack([counts_by_step[int(step)] for step in steps])
    totals = shares.sum(axis=1, keepdims=True)
    np.divide(shares, totals, out=shares, where=totals != 0)
    shares *= 100.0
    return steps, shares


def plot_experts(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    metadata = json.loads(
        (trace_dir / "metadata.json").read_text(encoding="utf-8")
    )
    num_experts = args.num_experts or int(metadata["num_experts"])
    layers = args.layers or [0]
    colors = _categorical_expert_colors(num_experts)
    if args.columns <= 0:
        raise ValueError("--columns must be positive")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        if args.x_axis != "step":
            raise ValueError("--max-steps requires --x-axis step")

    samples = None
    if args.x_axis == "sample":
        samples, metadata = _load_route_samples(trace_dir)

    num_columns = min(args.columns, len(layers))
    num_rows = (len(layers) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5.2 * num_columns, 3.8 * num_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.ravel()
    for plot_id, layer_id in enumerate(layers):
        if args.x_axis == "step":
            x, shares = _expert_load_share_by_step(
                trace_dir,
                layer_id,
                num_experts,
                args.phase,
            )
            if args.max_steps is not None:
                x = x[: args.max_steps]
                shares = shares[: args.max_steps]
            x_label = "Model-forward iteration step"
        else:
            assert samples is not None
            x = np.arange(len(samples))
            shares = _expert_load_share(
                samples,
                layer_id,
                num_experts,
                metadata["prompt_token_counts"],
                args.phase,
            )
            x_label = "Sample"
        axis = flat_axes[plot_id]
        axis.stackplot(
            x,
            shares.T,
            colors=colors,
            edgecolor="#202020",
            linewidth=0.15,
        )
        axis.set_title(f"Layer {layer_id}")
        if plot_id % num_columns == 0:
            axis.set_ylabel("Expert load share (%)")
        if len(x) == 1:
            axis.set_xlim(float(x[0]) - 0.5, float(x[0]) + 0.5)
        else:
            axis.set_xlim(float(x[0]), float(x[-1]))
        axis.set_ylim(0, 100)
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.15)

    for axis in flat_axes[len(layers) :]:
        axis.set_visible(False)

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
    records = _load_activation_records(activation_root, rank, torch)
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


def _multihot_topk(
    topk_ids: "torch.Tensor",
    num_experts: int,
    device: "torch.device",
) -> "torch.Tensor":
    import torch

    target = torch.zeros(
        (topk_ids.shape[0], num_experts),
        device=device,
        dtype=torch.float32,
    )
    safe_ids = topk_ids.clamp(min=0, max=num_experts - 1)
    return target.scatter_(1, safe_ids, 1.0)


def _topk_overlap_from_logits(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    top_k: int,
    num_experts: int,
) -> float:
    import torch

    predicted = torch.topk(logits, k=top_k, dim=-1).indices
    pred_hot = _multihot_topk(predicted, num_experts, logits.device)
    label_hot = _multihot_topk(labels, num_experts, logits.device)
    overlap = (pred_hot * label_hot).sum(dim=-1) / top_k
    return float(overlap.mean().item())


def _request_level_split_indices(
    request_ids: list[str],
    val_fraction: float,
    seed: int,
    device: "torch.device",
) -> tuple["torch.Tensor", "torch.Tensor", int, int]:
    import torch

    if not 0 <= val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")

    unique_request_ids = sorted(set(request_ids))
    num_requests = len(unique_request_ids)
    if num_requests == 0:
        raise ValueError("No request ids found for LoRA training split")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if val_fraction > 0 and num_requests > 1:
        val_request_count = int(num_requests * val_fraction)
        val_request_count = max(1, val_request_count)
        val_request_count = min(val_request_count, num_requests - 1)
        request_order = torch.randperm(num_requests, generator=generator).tolist()
        val_request_ids = {
            unique_request_ids[index]
            for index in request_order[:val_request_count]
        }
    else:
        val_request_ids = set()

    train_indices = [
        index
        for index, request_id in enumerate(request_ids)
        if request_id not in val_request_ids
    ]
    val_indices = [
        index
        for index, request_id in enumerate(request_ids)
        if request_id in val_request_ids
    ]
    if not train_indices:
        raise ValueError(
            "--val-fraction leaves no training requests; lower it or collect "
            "more trace data"
        )

    return (
        torch.tensor(train_indices, dtype=torch.long, device=device),
        torch.tensor(val_indices, dtype=torch.long, device=device),
        len(set(request_ids) - val_request_ids),
        len(val_request_ids),
    )


def _load_next_gate_lora_training_data(
    trace_dir: Path,
    rank: int | None,
    num_experts: int,
    phase: str,
    max_examples_per_layer: int | None,
) -> dict[int, dict[str, Any]]:
    import torch

    activation_root = trace_dir / "activations"
    records = _load_activation_records(activation_root, rank, torch)
    if not records:
        raise ValueError(f"No activation records found under {activation_root}")

    chunks: dict[int, dict[str, Any]] = {}
    for (rank_id, step, layer_id), record in sorted(records.items()):
        if "next_gate_predicted_topk_ids" not in record:
            continue
        if "activations" not in record:
            raise ValueError("LoRA training requires activation records")
        if "next_gate_base_logits" not in record:
            raise ValueError(
                "LoRA training requires next_gate_base_logits. Re-run collect "
                "with --trace-next-gate after this change."
            )
        next_layer_id = int(record["next_gate_layer_id"])
        next_record = records.get((rank_id, step, next_layer_id))
        if next_record is None:
            continue
        if "router_logits" not in next_record:
            raise ValueError(
                "LoRA logits-target training requires router_logits in the "
                "next-layer record. Re-run collect after this change."
            )

        activations = record["activations"]
        base_logits = record["next_gate_base_logits"]
        target_logits = next_record["router_logits"]
        labels = next_record["topk_ids"]
        request_ids = record.get("request_ids")
        if request_ids is None:
            raise ValueError(
                "Prompt-level LoRA validation split requires request_ids in "
                "activation records. Re-run collect after this change."
            )
        if len(request_ids) != activations.shape[0]:
            raise ValueError("Activation record request_ids length mismatch")
        mask = _phase_mask(record, phase)
        if mask is not None:
            activations = activations[mask]
            base_logits = base_logits[mask]
            target_logits = target_logits[mask]
            labels = labels[mask]
            mask_list = mask.cpu().tolist()
            request_ids = [
                request_id
                for request_id, keep in zip(request_ids, mask_list)
                if keep
            ]
        if activations.shape[0] == 0:
            continue

        entry = chunks.setdefault(
            next_layer_id,
            {
                "source_layer_id": layer_id,
                "next_layer_id": next_layer_id,
                "activations": [],
                "base_logits": [],
                "target_logits": [],
                "labels": [],
                "request_ids": [],
                "count": 0,
            },
        )
        remaining = None
        if max_examples_per_layer is not None:
            remaining = max_examples_per_layer - int(entry["count"])
            if remaining <= 0:
                continue
            activations = activations[:remaining]
            base_logits = base_logits[:remaining]
            target_logits = target_logits[:remaining]
            labels = labels[:remaining]
            request_ids = request_ids[:remaining]

        entry["activations"].append(activations.to(torch.float32))
        entry["base_logits"].append(base_logits.to(torch.float32))
        entry["target_logits"].append(target_logits.to(torch.float32))
        entry["labels"].append(labels.to(torch.long).clamp(0, num_experts - 1))
        entry["request_ids"].extend(str(request_id) for request_id in request_ids)
        entry["count"] = int(entry["count"]) + int(activations.shape[0])

    datasets: dict[int, dict[str, Any]] = {}
    for next_layer_id, entry in chunks.items():
        activations = torch.cat(entry["activations"], dim=0)
        request_ids = list(entry["request_ids"])
        if len(request_ids) != activations.shape[0]:
            raise ValueError("LoRA training request_ids length mismatch")
        datasets[next_layer_id] = {
            "source_layer_id": entry["source_layer_id"],
            "next_layer_id": next_layer_id,
            "activations": activations,
            "base_logits": torch.cat(entry["base_logits"], dim=0),
            "target_logits": torch.cat(entry["target_logits"], dim=0),
            "labels": torch.cat(entry["labels"], dim=0),
            "request_ids": request_ids,
        }
    if not datasets:
        raise ValueError("No next-gate LoRA training examples were found")
    return datasets


def _summarize_lora_results(
    results: list[dict[str, Any]],
) -> dict[str, float | int]:
    total_examples = sum(int(row["val_examples"]) for row in results)
    if total_examples == 0:
        return {
            "val_examples": 0,
            "baseline_overlap": float("nan"),
            "lora_overlap": float("nan"),
            "overlap_delta": float("nan"),
            "baseline_mse": float("nan"),
            "lora_mse": float("nan"),
            "mse_delta": float("nan"),
        }
    baseline_overlap = sum(
        float(row["baseline_val_overlap"]) * int(row["val_examples"])
        for row in results
    ) / total_examples
    lora_overlap = sum(
        float(row["lora_val_overlap"]) * int(row["val_examples"])
        for row in results
    ) / total_examples
    baseline_mse = sum(
        float(row["baseline_val_mse"]) * int(row["val_examples"])
        for row in results
    ) / total_examples
    lora_mse = sum(
        float(row["lora_val_mse"]) * int(row["val_examples"])
        for row in results
    ) / total_examples
    return {
        "val_examples": total_examples,
        "baseline_overlap": baseline_overlap,
        "lora_overlap": lora_overlap,
        "overlap_delta": lora_overlap - baseline_overlap,
        "baseline_mse": baseline_mse,
        "lora_mse": lora_mse,
        "mse_delta": lora_mse - baseline_mse,
    }


def train_next_gate_lora(args: argparse.Namespace) -> list[dict[str, Any]]:
    import math
    import torch

    trace_dir = args.trace_dir.resolve()
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    num_experts = args.num_experts or int(metadata["num_experts"])
    datasets = _load_next_gate_lora_training_data(
        trace_dir,
        args.rank,
        num_experts,
        args.phase,
        args.max_examples_per_layer,
    )

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for next_layer_id, data in sorted(datasets.items()):
        source_layer_id = int(data["source_layer_id"])
        x = data["activations"].to(device=device)
        base_logits = data["base_logits"].to(device=device)
        target_logits = data["target_logits"].to(device=device)
        labels = data["labels"].to(device=device)
        request_ids = data["request_ids"]
        num_examples, hidden_size = x.shape
        num_experts = target_logits.shape[1]
        top_k = labels.shape[1]
        rank = min(args.rank_dim, hidden_size, num_experts)
        scale = args.alpha / rank

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + next_layer_id)
        train_indices, val_indices, train_requests, val_requests = (
            _request_level_split_indices(
                request_ids,
                args.val_fraction,
                args.seed + next_layer_id,
                device,
            )
        )

        lora_a = torch.nn.Parameter(
            torch.empty((rank, hidden_size), device=device, dtype=torch.float32)
        )
        lora_b = torch.nn.Parameter(
            torch.zeros((num_experts, rank), device=device, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        optimizer = torch.optim.AdamW(
            [lora_a, lora_b],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        for epoch in range(args.epochs):
            order = train_indices[
                torch.randperm(
                    train_indices.numel(),
                    generator=generator,
                    device=device,
                )
            ]
            total_loss = 0.0
            total_seen = 0
            for start in range(0, order.numel(), args.batch_size):
                batch = order[start : start + args.batch_size]
                batch_x = x[batch]
                batch_base = base_logits[batch]
                batch_target = target_logits[batch]
                lora_logits = (batch_x @ lora_a.T) @ lora_b.T
                logits = batch_base + lora_logits * scale
                loss = loss_fn(logits, batch_target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * batch.numel()
                total_seen += int(batch.numel())

            eval_indices = val_indices if val_indices.numel() else train_indices
            with torch.no_grad():
                baseline_eval_logits = base_logits[eval_indices]
                baseline_loss = loss_fn(
                    baseline_eval_logits,
                    target_logits[eval_indices],
                )
                baseline_overlap = _topk_overlap_from_logits(
                    baseline_eval_logits,
                    labels[eval_indices],
                    top_k,
                    num_experts,
                )
                eval_logits = (
                    base_logits[eval_indices]
                    + ((x[eval_indices] @ lora_a.T) @ lora_b.T) * scale
                )
                eval_loss = loss_fn(
                    eval_logits,
                    target_logits[eval_indices],
                )
                eval_overlap = _topk_overlap_from_logits(
                    eval_logits,
                    labels[eval_indices],
                    top_k,
                    num_experts,
                )
            if args.verbose:
                mean_loss = total_loss / max(total_seen, 1)
                print(
                    f"layer {source_layer_id}->{next_layer_id} "
                    f"epoch {epoch + 1}/{args.epochs}: "
                    f"loss={mean_loss:.6f}, val_mse={eval_loss.item():.6f}, "
                    f"baseline_overlap={baseline_overlap:.4f}, "
                    f"lora_overlap={eval_overlap:.4f}"
                )

        path = output_dir / (
            f"layer_{source_layer_id:04d}_to_{next_layer_id:04d}.pt"
        )
        torch.save(
            {
                "source_layer_id": source_layer_id,
                "next_layer_id": next_layer_id,
                "rank": rank,
                "alpha": float(args.alpha),
                "hidden_size": hidden_size,
                "num_experts": num_experts,
                "lora_A": lora_a.detach().cpu(),
                "lora_B": lora_b.detach().cpu(),
            },
            path,
        )
        results.append(
            {
                "source_layer_id": source_layer_id,
                "next_layer_id": next_layer_id,
                "num_examples": num_examples,
                "train_examples": int(train_indices.numel()),
                "val_examples": int(eval_indices.numel()),
                "train_requests": train_requests,
                "val_requests": val_requests,
                "rank": rank,
                "alpha": args.alpha,
                "baseline_val_mse": float(baseline_loss.item()),
                "lora_val_mse": float(eval_loss.item()),
                "mse_delta": float(eval_loss.item() - baseline_loss.item()),
                "baseline_val_overlap": baseline_overlap,
                "lora_val_overlap": eval_overlap,
                "overlap_delta": eval_overlap - baseline_overlap,
                "path": str(path),
            }
        )
        print(
            f"Saved {path} "
            f"(train={train_indices.numel()}, val={eval_indices.numel()}, "
            f"baseline_overlap={baseline_overlap:.4f}, "
            f"lora_overlap={eval_overlap:.4f}, "
            f"delta={eval_overlap - baseline_overlap:+.4f})"
        )

    summary = _summarize_lora_results(results)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "trace_dir": str(trace_dir),
                "phase": args.phase,
                "num_experts": num_experts,
                "rank_dim": args.rank_dim,
                "alpha": args.alpha,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "target": "next_layer_router_logits",
                "loss": "mse",
                "validation_split": "request_level",
                "val_fraction": args.val_fraction,
                "summary": summary,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "LoRA validation summary: "
        f"baseline_overlap={summary['baseline_overlap']:.4f}, "
        f"lora_overlap={summary['lora_overlap']:.4f}, "
        f"delta={summary['overlap_delta']:+.4f}; "
        f"baseline_mse={summary['baseline_mse']:.6f}, "
        f"lora_mse={summary['lora_mse']:.6f}, "
        f"delta={summary['mse_delta']:+.6f}"
    )
    return results


def run_next_gate_lora_pipeline(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    train_trace_dir = work_dir / "train_trace"
    lora_dir = work_dir / "lora"
    eval_trace_dir = work_dir / "eval_trace"
    work_dir.mkdir(parents=True, exist_ok=True)

    train_prompts = args.prompts
    eval_prompts = args.eval_prompts or args.prompts

    common_collect = {
        "model": args.model,
        "ep_size": args.ep_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_tokens_per_sample": args.max_tokens_per_sample,
        "num_prompts": args.num_prompts,
        "prompt_sample_mode": args.prompt_sample_mode,
        "prompt_sample_seed": args.prompt_sample_seed,
        "collect_batch_size": args.collect_batch_size,
        "timeout": args.timeout,
        "activation_dtype": args.activation_dtype,
        "trace_next_gate": True,
        "load_format": args.load_format,
        "moe_backend": args.moe_backend,
        "pack_activations": args.pack_activations,
        "delete_unpacked_activations": args.delete_unpacked_activations,
    }

    skip_train_collect = args.skip_collect or args.skip_train_collect
    skip_eval_collect = args.skip_collect or args.skip_eval_collect

    if skip_train_collect:
        print(f"[1/4] Reusing existing training trace under {train_trace_dir}")
        if not (train_trace_dir / "metadata.json").exists():
            raise FileNotFoundError(
                f"--skip-train-collect requires {train_trace_dir}/metadata.json"
            )
    else:
        print(f"[1/4] Collecting training trace under {train_trace_dir}")
        collect(
            argparse.Namespace(
                **common_collect,
                prompts=train_prompts,
                output_dir=train_trace_dir,
                next_gate_lora_dir=None,
            )
        )

    print(f"[2/4] Training next-gate LoRA adapters under {lora_dir}")
    lora_results = train_next_gate_lora(
        argparse.Namespace(
            trace_dir=train_trace_dir,
            output_dir=lora_dir,
            rank=args.train_rank,
            num_experts=args.num_experts,
            phase=args.train_phase,
            rank_dim=args.rank_dim,
            alpha=args.alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=args.device,
            max_examples_per_layer=args.max_examples_per_layer,
            verbose=args.verbose,
        )
    )
    lora_summary = _summarize_lora_results(lora_results)

    if skip_eval_collect:
        print(f"[3/4] Skipping LoRA eval collect under {eval_trace_dir}")
    else:
        print(f"[3/4] Collecting LoRA eval trace under {eval_trace_dir}")
        collect(
            argparse.Namespace(
                **common_collect,
                prompts=eval_prompts,
                output_dir=eval_trace_dir,
                next_gate_lora_dir=lora_dir,
            )
        )

    print("[4/4] Writing baseline and LoRA eval plots/data")
    baseline_output = work_dir / f"baseline_next_gate_similarity_{args.eval_phase}.png"
    lora_output = work_dir / f"lora_next_gate_similarity_{args.eval_phase}.png"
    plot_next_gate_similarity(
        argparse.Namespace(
            trace_dir=train_trace_dir,
            rank=args.eval_rank,
            num_experts=args.num_experts,
            phase=args.eval_phase,
            output=baseline_output,
        )
    )
    lora_plot = None
    if (eval_trace_dir / "metadata.json").exists():
        plot_next_gate_similarity(
            argparse.Namespace(
                trace_dir=eval_trace_dir,
                rank=args.eval_rank,
                num_experts=args.num_experts,
                phase=args.eval_phase,
                output=lora_output,
            )
        )
        lora_plot = str(lora_output)
    else:
        print(
            f"Skipping LoRA plot because {eval_trace_dir}/metadata.json "
            "does not exist"
        )

    summary = {
        "model": args.model,
        "train_trace_dir": str(train_trace_dir),
        "lora_dir": str(lora_dir),
        "eval_trace_dir": str(eval_trace_dir),
        "baseline_plot": str(baseline_output),
        "lora_plot": lora_plot,
        "offline_validation": lora_summary,
        "train_prompts": str(train_prompts) if train_prompts is not None else None,
        "eval_prompts": str(eval_prompts) if eval_prompts is not None else None,
    }
    comparison_payload = {
        "summary": lora_summary,
        "per_layer": lora_results,
    }
    (work_dir / "accuracy_comparison.json").write_text(
        json.dumps(comparison_payload, indent=2), encoding="utf-8"
    )
    comparison_rows = [
        (
            "source_layer_id,next_layer_id,val_examples,"
            "baseline_overlap,lora_overlap,overlap_delta,"
            "baseline_mse,lora_mse,mse_delta"
        )
    ]
    for row in lora_results:
        comparison_rows.append(
            ",".join(
                [
                    str(row["source_layer_id"]),
                    str(row["next_layer_id"]),
                    str(row["val_examples"]),
                    f"{row['baseline_val_overlap']:.6f}",
                    f"{row['lora_val_overlap']:.6f}",
                    f"{row['overlap_delta']:.6f}",
                    f"{row['baseline_val_mse']:.6f}",
                    f"{row['lora_val_mse']:.6f}",
                    f"{row['mse_delta']:.6f}",
                ]
            )
        )
    comparison_rows.append(
        ",".join(
            [
                "overall",
                "overall",
                str(lora_summary["val_examples"]),
                f"{lora_summary['baseline_overlap']:.6f}",
                f"{lora_summary['lora_overlap']:.6f}",
                f"{lora_summary['overlap_delta']:.6f}",
                f"{lora_summary['baseline_mse']:.6f}",
                f"{lora_summary['lora_mse']:.6f}",
                f"{lora_summary['mse_delta']:.6f}",
            ]
        )
    )
    (work_dir / "accuracy_comparison.csv").write_text(
        "\n".join(comparison_rows) + "\n", encoding="utf-8"
    )
    (work_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "Offline validation comparison: "
        f"before={lora_summary['baseline_overlap']:.4f}, "
        f"after={lora_summary['lora_overlap']:.4f}, "
        f"delta={lora_summary['overlap_delta']:+.4f}"
    )
    print(f"Accuracy comparison: {work_dir / 'accuracy_comparison.csv'}")
    print(f"Pipeline complete. Summary: {work_dir / 'pipeline_summary.json'}")


def _load_activations(
    trace_dir: Path,
    rank: int | None,
    max_tokens: int,
    phase: str,
) -> tuple[list[int], np.ndarray]:
    import torch

    activation_root = trace_dir / "activations"
    loaded_records = _load_activation_records(activation_root, rank, torch)

    records: dict[tuple[int, int, int], torch.Tensor] = {}
    for key, record in loaded_records.items():
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
        "--num-prompts",
        type=int,
        help=(
            "Use only this many prompts from --prompts. By default all prompts "
            "are used."
        ),
    )
    collect_parser.add_argument(
        "--prompt-sample-mode",
        choices=("first", "random"),
        default="first",
        help="How to choose prompts when --num-prompts is set.",
    )
    collect_parser.add_argument(
        "--prompt-sample-seed",
        type=int,
        default=0,
        help="Random seed used when --prompt-sample-mode=random.",
    )
    collect_parser.add_argument(
        "--collect-batch-size",
        type=int,
        help=(
            "Number of prompts passed to each llm.generate call per DP rank. "
            "Also caps vLLM max_num_seqs for collection."
        ),
    )
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
        "--next-gate-lora-dir",
        type=Path,
        help=(
            "Directory containing LoRA adapters produced by "
            "train-next-gate-lora. The adapters are used only for trace "
            "next-gate prediction, not for normal MoE inference."
        ),
    )
    collect_parser.add_argument(
        "--load-format",
        default="auto",
        help=(
            "Model weight load format passed to vLLM. Use 'dummy' only for "
            "plumbing tests; dummy weights make route-overlap plots meaningless."
        ),
    )
    collect_parser.add_argument(
        "--moe-backend",
        choices=MOE_BACKEND_CHOICES,
        default="auto",
        help="MoE expert-kernel backend to pass to vLLM, e.g. triton.",
    )
    collect_parser.add_argument(
        "--pack-activations",
        action="store_true",
        help="Pack per-step activation .pt files into one shard per rank.",
    )
    collect_parser.add_argument(
        "--delete-unpacked-activations",
        action="store_true",
        help="Delete per-step activation .pt files after packing.",
    )
    collect_parser.set_defaults(func=collect)

    expert_parser = subparsers.add_parser("plot-experts")
    expert_parser.add_argument("--trace-dir", type=Path, required=True)
    expert_parser.add_argument("--layers", type=int, nargs="+")
    expert_parser.add_argument("--num-experts", type=int)
    expert_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="all"
    )
    expert_parser.add_argument(
        "--x-axis",
        choices=("step", "sample"),
        default="step",
        help=(
            "Plot actual model-forward iteration steps by default. Use "
            "'sample' for the original per-prompt view."
        ),
    )
    expert_parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Maximum number of layer subplots per row.",
    )
    expert_parser.add_argument(
        "--max-steps",
        type=int,
        help="Plot only the first N recorded model-forward iteration steps.",
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

    train_lora_parser = subparsers.add_parser("train-next-gate-lora")
    train_lora_parser.add_argument("--trace-dir", type=Path, required=True)
    train_lora_parser.add_argument("--output-dir", type=Path, required=True)
    train_lora_parser.add_argument(
        "--rank",
        type=int,
        help="Train from one EP rank; by default traces from all ranks are used.",
    )
    train_lora_parser.add_argument("--num-experts", type=int)
    train_lora_parser.add_argument(
        "--phase", choices=("prefill", "decode", "all"), default="all"
    )
    train_lora_parser.add_argument("--rank-dim", type=int, default=8)
    train_lora_parser.add_argument("--alpha", type=float, default=16.0)
    train_lora_parser.add_argument("--epochs", type=int, default=5)
    train_lora_parser.add_argument("--batch-size", type=int, default=1024)
    train_lora_parser.add_argument("--lr", type=float, default=1e-3)
    train_lora_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_lora_parser.add_argument("--val-fraction", type=float, default=0.1)
    train_lora_parser.add_argument("--seed", type=int, default=0)
    train_lora_parser.add_argument("--device", help="Training device, e.g. cuda:0")
    train_lora_parser.add_argument("--max-examples-per-layer", type=int)
    train_lora_parser.add_argument("--verbose", action="store_true")
    train_lora_parser.set_defaults(func=train_next_gate_lora)

    pipeline_parser = subparsers.add_parser("run-next-gate-lora-pipeline")
    pipeline_parser.add_argument("--model", required=True)
    pipeline_parser.add_argument("--prompts", type=Path)
    pipeline_parser.add_argument(
        "--eval-prompts",
        type=Path,
        help="Optional held-out prompts for LoRA validation collection.",
    )
    pipeline_parser.add_argument("--work-dir", type=Path, required=True)
    pipeline_parser.add_argument(
        "--skip-collect",
        action="store_true",
        help=(
            "Reuse existing train_trace/eval_trace under --work-dir and skip "
            "both collection phases."
        ),
    )
    pipeline_parser.add_argument(
        "--skip-train-collect",
        action="store_true",
        help="Reuse --work-dir/train_trace and skip baseline training collection.",
    )
    pipeline_parser.add_argument(
        "--skip-eval-collect",
        action="store_true",
        help=(
            "Skip LoRA eval collection. Offline before/after validation from "
            "the train trace split is still reported."
        ),
    )
    pipeline_parser.add_argument(
        "--ep-size",
        dest="ep_size",
        type=int,
        default=1,
        help=(
            "Number of expert-parallel ranks. The script uses TP=1, DP=N, "
            "and enable_expert_parallel=True, so EP size is N."
        ),
    )
    pipeline_parser.add_argument("--max-model-len", type=int, default=4096)
    pipeline_parser.add_argument("--max-new-tokens", type=int, default=16)
    pipeline_parser.add_argument("--max-tokens-per-sample", type=int, default=4096)
    pipeline_parser.add_argument(
        "--num-prompts",
        type=int,
        help=(
            "Use only this many prompts for each collection phase. By default "
            "all prompts are used."
        ),
    )
    pipeline_parser.add_argument(
        "--prompt-sample-mode",
        choices=("first", "random"),
        default="first",
        help="How to choose prompts when --num-prompts is set.",
    )
    pipeline_parser.add_argument(
        "--prompt-sample-seed",
        type=int,
        default=0,
        help="Random seed used when --prompt-sample-mode=random.",
    )
    pipeline_parser.add_argument(
        "--collect-batch-size",
        type=int,
        help=(
            "Number of prompts passed to each llm.generate call per DP rank. "
            "Also caps vLLM max_num_seqs for collection."
        ),
    )
    pipeline_parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum seconds to wait for each offline DP rank.",
    )
    pipeline_parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    pipeline_parser.add_argument("--load-format", default="auto")
    pipeline_parser.add_argument(
        "--moe-backend",
        choices=MOE_BACKEND_CHOICES,
        default="auto",
    )
    pipeline_parser.add_argument(
        "--pack-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack activation records into one shard per rank.",
    )
    pipeline_parser.add_argument(
        "--delete-unpacked-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete per-step activation .pt files after packing.",
    )
    pipeline_parser.add_argument("--num-experts", type=int)
    pipeline_parser.add_argument(
        "--train-rank",
        type=int,
        help="Train from one EP rank; by default traces from all ranks are used.",
    )
    pipeline_parser.add_argument(
        "--eval-rank",
        type=int,
        help="Plot one EP rank; by default traces from all ranks are merged.",
    )
    pipeline_parser.add_argument(
        "--train-phase", choices=("prefill", "decode", "all"), default="all"
    )
    pipeline_parser.add_argument(
        "--eval-phase", choices=("prefill", "decode", "all"), default="all"
    )
    pipeline_parser.add_argument("--rank-dim", type=int, default=8)
    pipeline_parser.add_argument("--alpha", type=float, default=16.0)
    pipeline_parser.add_argument("--epochs", type=int, default=5)
    pipeline_parser.add_argument("--batch-size", type=int, default=1024)
    pipeline_parser.add_argument("--lr", type=float, default=1e-3)
    pipeline_parser.add_argument("--weight-decay", type=float, default=0.0)
    pipeline_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of trace examples held out per layer for offline "
            "before/after validation during LoRA training."
        ),
    )
    pipeline_parser.add_argument("--seed", type=int, default=0)
    pipeline_parser.add_argument("--device", help="Training device, e.g. cuda:0")
    pipeline_parser.add_argument("--max-examples-per-layer", type=int)
    pipeline_parser.add_argument("--verbose", action="store_true")
    pipeline_parser.set_defaults(func=run_next_gate_lora_pipeline)

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
