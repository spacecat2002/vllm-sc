#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Capture and visualize GDN (Gated Delta Network) beta and state-delta statistics
during autoregressive decoding.

Usage:
    python benchmarks/benchmark_gdn_stats.py \
        --model <model_name_or_path> \
        --prompt "Long input text here..." \
        --max-new-tokens 200 \
        --output gdn_stats.pkl \
        --plot gdn_analysis.png

The script enables the gdn_stats_collector, runs generation, then plots:
  1. Mean beta per decode step (per layer, first request)
  2. State-delta ||ΔS|| per decode step (per layer, first request)
  3. Beta value distribution (histogram, all layers, all steps)
  4. State cosine similarity per decode step (per layer, first request)
"""

import argparse
import os
import pickle
import sys

import numpy as np

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GDN beta/state-delta analysis")
    p.add_argument("--model", required=True, help="Model name or local path")
    p.add_argument(
        "--prompt",
        default=(
            "Prefix caching is a key latency optimization for autoregressive LLM serving."
        ),
        help="Input prompt for generation",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=5300,
        help="Number of tokens to generate (= number of decode steps captured)",
    )
    p.add_argument(
        "--output", default="gdn_stats.pkl",
        help="Path to save raw stats (pickle)",
    )
    p.add_argument(
        "--plot", default="gdn_analysis.png",
        help="Path to save the output figure",
    )
    p.add_argument(
        "--max-layers", type=int, default=24,
        help="Max number of GDN layers to plot (sorted by layer index)",
    )
    p.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size for vLLM",
    )
    p.add_argument(
        "--load-stats", default=None,
        help="Skip inference; load existing stats pkl and just plot",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference + capture
# ---------------------------------------------------------------------------

def run_capture(args: argparse.Namespace) -> dict:
    """Run vLLM inference with stats capture enabled."""
    from vllm import LLM, SamplingParams
    from vllm import envs
    from vllm.model_executor.layers.mamba.gdn import gdn_stats_collector as gsc

    print(f"[capture] Loading model: {args.model}")
    print(
        "[capture] VLLM_ENABLE_V1_MULTIPROCESSING="
        f"{int(envs.VLLM_ENABLE_V1_MULTIPROCESSING)}"
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        async_scheduling=False,
        enforce_eager=True,  # easier to hook; remove for speed
        # load_format="dummy"
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,  # greedy – deterministic
    )

    print(f"[capture] Enabling GDN stats collector")
    gsc.enable()

    print(f"[capture] Running generation ({args.max_new_tokens} new tokens)...")
    outputs = llm.generate([args.prompt], sampling_params)

    gsc.disable()
    generated_text = outputs[0].outputs[0].text
    print(f"[capture] Generated: {generated_text[:120]}{'...' if len(generated_text) > 120 else ''}")

    collector = gsc.get_collector()
    stats = gsc.get_stats()
    print(
        "[capture] Collector calls: "
        f"beta={collector.beta_calls}, delta={collector.delta_calls}, "
        f"prefill_fwds={collector.prefill_forwards}, "
        f"decode_fwds={collector.decode_forwards}, "
        f"disabled={collector.disabled_calls}"
    )
    print(f"[capture] Captured stats for {len(stats)} GDN layers")
    for name, s in list(stats.items())[:3]:
        print(
            f"  {name}: {len(s.betas)} beta steps, "
            f"{len(s.state_deltas)} delta steps, "
            f"{len(s.state_cosines)} cosine steps"
        )

    gsc.save(args.output)
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sort_layer_keys(keys: list[str]) -> list[str]:
    """Sort layer keys by the first integer found in the name."""
    import re
    def _key(s: str) -> int:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0
    return sorted(keys, key=_key)


def _layer_label(key: str) -> str:
    import re
    m = re.search(r"\.layers\.(\d+)\.", key)
    return f"layer {m.group(1)}" if m else key


def plot_stats(stats: dict, args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    layer_keys = _sort_layer_keys(list(stats.keys()))
    # Limit to max_layers evenly spaced
    if len(layer_keys) > args.max_layers:
        idxs = np.linspace(0, len(layer_keys) - 1, args.max_layers, dtype=int)
        layer_keys = [layer_keys[i] for i in idxs]

    n_layers = len(layer_keys)
    colors = plt.cm.tab20(np.linspace(0, 1, n_layers))

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax_beta   = fig.add_subplot(gs[0, 0])
    ax_delta  = fig.add_subplot(gs[0, 1])
    ax_hist   = fig.add_subplot(gs[1, 0])
    ax_log    = fig.add_subplot(gs[1, 1])

    all_betas_flat = []
    all_deltas_flat = []
    all_cosines_flat = []

    for i, key in enumerate(layer_keys):
        s = stats[key]
        color = colors[i]
        label = _layer_label(key)

        # ── beta over decode steps ──────────────────────────────────────────
        if s.betas:
            # Each element: [num_requests] – take first request
            beta_series = np.array([b[0] if b.ndim > 0 and len(b) > 0 else b
                                    for b in s.betas])
            steps = np.arange(len(beta_series))
            ax_beta.plot(steps, beta_series, color=color, alpha=0.8,
                         linewidth=1.2, label=label)
            all_betas_flat.extend(beta_series.tolist())

        # ── state delta over decode steps ───────────────────────────────────
        if s.state_deltas:
            delta_series = np.array([d[0] if d.ndim > 0 and len(d) > 0 else d
                                     for d in s.state_deltas])
            steps = np.arange(len(delta_series))
            ax_delta.plot(steps, delta_series, color=color, alpha=0.8,
                          linewidth=1.2, label=label)
            all_deltas_flat.extend(delta_series.tolist())

        # ── state cosine similarity over decode steps ───────────────────────
        if s.state_cosines:
            cosine_series = np.array([
                c[0] if c.ndim > 0 and len(c) > 0 else c
                for c in s.state_cosines
            ])
            steps = np.arange(len(cosine_series))
            ax_log.plot(steps, cosine_series, color=color, alpha=0.8,
                        linewidth=1.2, label=label)
            all_cosines_flat.extend(cosine_series.tolist())

    # ── axis decoration ─────────────────────────────────────────────────────
    ax_beta.set_title("Mean Beta (gate) per Decode Step", fontsize=13)
    ax_beta.set_xlabel("Decode step (token index)")
    ax_beta.set_ylabel("Mean β across heads (sigmoid output)")
    ax_beta.set_ylim(0, 1.05)
    ax_beta.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_beta.legend(fontsize=7, ncol=2)
    ax_beta.grid(True, alpha=0.3)

    ax_delta.set_title("State Delta ||S_t − S_{t−1}|| per Decode Step", fontsize=13)
    ax_delta.set_xlabel("Decode step (token index)")
    ax_delta.set_ylabel("Frobenius norm of state change")
    ax_delta.legend(fontsize=7, ncol=2)
    ax_delta.grid(True, alpha=0.3)

    # Beta histogram
    if all_betas_flat:
        ax_hist.hist(all_betas_flat, bins=50, color="steelblue", edgecolor="none",
                     alpha=0.8)
        ax_hist.axvline(np.mean(all_betas_flat), color="red", linestyle="--",
                        linewidth=1.5, label=f"mean={np.mean(all_betas_flat):.3f}")
        ax_hist.set_title("Beta Value Distribution (all layers & steps)", fontsize=13)
        ax_hist.set_xlabel("β value")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

    # State cosine similarity
    ax_log.set_ylim(-0.05, 1.05)
    ax_log.set_title("State Cosine Similarity per Decode Step", fontsize=13)
    ax_log.set_xlabel("Decode step (token index)")
    ax_log.set_ylabel("cos(S_t, S_{t-1})")
    ax_log.legend(fontsize=7, ncol=2)
    ax_log.grid(True, alpha=0.3)

    # ── summary stats annotation ─────────────────────────────────────────────
    if all_betas_flat:
        beta_arr = np.array(all_betas_flat)
        fig.text(
            0.01, 0.01,
            f"Beta  mean={beta_arr.mean():.4f}  std={beta_arr.std():.4f}  "
            f"p10={np.percentile(beta_arr, 10):.4f}  p90={np.percentile(beta_arr, 90):.4f}",
            fontsize=8, color="gray",
        )

    fig.suptitle(
        f"GDN Layer Analysis — model: {os.path.basename(args.model)}\n"
        f"prompt_len≈{len(args.prompt.split())} words, "
        f"decode_steps={args.max_new_tokens}",
        fontsize=14,
    )

    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved figure to {args.plot}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.load_stats:
        print(f"[main] Loading pre-saved stats from {args.load_stats}")
        with open(args.load_stats, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = run_capture(args)

    if not stats:
        print("[main] No stats captured – is the model a hybrid GDN model?")
        sys.exit(1)

    plot_stats(stats, args)


if __name__ == "__main__":
    main()
