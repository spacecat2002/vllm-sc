# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot MoE EP stage-sum latency across locality/skew sweep points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="moe_ep_distribution",
        help="Filename prefix for generated PNG files.",
    )
    return parser.parse_args()


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No summary rows found in {path}.")
    for row in rows:
        row["local_share"] = float(row["local_share"])
        row["rank_skew"] = float(row["rank_skew"])
        row["mean_max_total_ms"] = float(row["mean_max_total_ms"])
    return rows


def plot_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with "
            "`uv pip install matplotlib`."
        ) from error

    local_shares = sorted({row["local_share"] for row in rows})
    rank_skews = sorted({row["rank_skew"] for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for local_share in local_shares:
        selected = sorted(
            (row for row in rows if row["local_share"] == local_share),
            key=lambda row: row["rank_skew"],
        )
        axis.plot(
            [row["rank_skew"] for row in selected],
            [row["mean_max_total_ms"] for row in selected],
            marker="o",
            label=f"local_share={local_share:g}",
        )
    axis.set(
        title="Stage-sum latency by rank skew",
        xlabel="Configured rank skew",
        ylabel="Mean max-rank stage-sum latency (ms)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    by_rank_skew_path = output_dir / f"{prefix}_by_rank_skew.png"
    fig.savefig(by_rank_skew_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for rank_skew in rank_skews:
        selected = sorted(
            (row for row in rows if row["rank_skew"] == rank_skew),
            key=lambda row: row["local_share"],
        )
        axis.plot(
            [row["local_share"] for row in selected],
            [row["mean_max_total_ms"] for row in selected],
            marker="o",
            label=f"rank_skew={rank_skew:g}",
        )
    axis.set(
        title="Stage-sum latency by local share",
        xlabel="Configured local share",
        ylabel="Mean max-rank stage-sum latency (ms)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    by_local_share_path = output_dir / f"{prefix}_by_local_share.png"
    fig.savefig(by_local_share_path, dpi=180)
    plt.close(fig)

    return [by_rank_skew_path, by_local_share_path]


def main() -> None:
    args = parse_args()
    output_paths = plot_summary(
        read_summary_rows(args.input_csv),
        args.output_dir,
        args.prefix,
    )
    for output_path in output_paths:
        print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
