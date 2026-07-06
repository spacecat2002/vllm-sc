# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect and visualize MoE routes and router-input activations.

Examples::

    # Prompts are submitted together; the trace keeps request boundaries.
    .venv/bin/python examples/basic/offline_inference/moe_trace.py collect \
        --model Qwen/Qwen3-30B-A3B --prompts prompts.txt \
        --output-dir /tmp/qwen3_moe_trace --ep-size 4

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-experts --trace-dir /tmp/qwen3_moe_trace --layers 8 31

    .venv/bin/python examples/basic/offline_inference/moe_trace.py \
        plot-activations --trace-dir /tmp/qwen3_moe_trace \
        --metric cka --phase prefill

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


def collect(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    activation_dir = output_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts)

    # Workers inherit these variables when LLM starts its executor processes.
    os.environ["VLLM_MOE_TRACE_DIR"] = str(activation_dir)
    # Generating N tokens performs one prefill forward and up to N - 1 decode
    # forwards per prompt.
    os.environ["VLLM_MOE_TRACE_MAX_STEPS"] = str(
        len(prompts) * args.max_new_tokens
    )
    os.environ["VLLM_MOE_TRACE_MAX_TOKENS"] = str(args.max_tokens_per_sample)
    os.environ["VLLM_MOE_TRACE_ACTIVATIONS"] = "input"
    os.environ["VLLM_MOE_TRACE_ACTIVATION_DTYPE"] = args.activation_dtype
    os.environ["VLLM_MOE_TRACE_TOKEN_SELECTION"] = "prefill_last"

    # Import after setting the environment so spawned workers see the config.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        # Pure EP topology for load-distribution experiments: attention is
        # replicated over DP ranks, while experts are sharded over
        # EP_SIZE = TP_SIZE * DP_SIZE = args.ep_size ranks.
        tensor_parallel_size=1,
        data_parallel_size=args.ep_size,
        enable_expert_parallel=True,
        max_model_len=args.max_model_len,
        max_num_seqs=len(prompts),
        enable_chunked_prefill=False,
        enforce_eager=True,
        enable_return_routed_experts=True,
        load_format="dummy",
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
    )

    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    routes: dict[str, np.ndarray] = {}
    prompt_token_counts: list[int] = []
    for sample_id, request_output in enumerate(request_outputs):
        routed_experts = request_output.outputs[0].routed_experts
        if routed_experts is None:
            raise RuntimeError("vLLM did not return routed experts")
        routes[f"sample_{sample_id:06d}"] = routed_experts
        prompt_token_counts.append(len(request_output.prompt_token_ids))

    np.savez_compressed(output_dir / "routes.npz", **routes)
    hf_config = llm.model_config.hf_text_config
    metadata = {
        "model": args.model,
        "expert_parallel_size": args.ep_size,
        "num_samples": len(prompts),
        "num_experts": _num_experts(hf_config),
        "prompt_token_counts": prompt_token_counts,
        "route_shape": "[token, layer, top_k]",
        "activation_point": "input to each MoE router",
        "activation_token_selection": (
            "last prompt token for prefill; each token for decode"
        ),
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


def plot_experts(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    trace_dir = args.trace_dir.resolve()
    samples, metadata = _load_route_samples(trace_dir)
    num_experts = args.num_experts or int(metadata["num_experts"])
    layers = args.layers or [0]
    prompt_token_counts = metadata["prompt_token_counts"]
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, num_experts))

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

        axes[row, 0].stackplot(x, shares.T, colors=colors, linewidth=0)
        axes[row, 0].set_title(f"Layer {layer_id}: fixed expert IDs")
        axes[row, 1].stackplot(x, sorted_shares.T, colors=colors, linewidth=0)
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
        "--activation-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
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
