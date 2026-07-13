# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot locality/skew sweeps from benchmark_moe_ep_distribution.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skew-local-share",
        type=float,
        default=0.5,
        help="Local-share slice used by the rank-skew line plot.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No summary rows found in {path}.")
    numeric_fields = set(rows[0]) - {"record_type"}
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def nearest_value(values: set[float], requested: float) -> float:
    return min(values, key=lambda value: abs(value - requested))


def plot(rows: list[dict[str, Any]], output: Path, skew_local_share: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with "
            "`uv pip install matplotlib`."
        ) from error

    local_shares = sorted({row["local_share"] for row in rows})
    rank_skews = sorted({row["rank_skew"] for row in rows})
    zero_skew = nearest_value(set(rank_skews), 0.0)
    selected_local_share = nearest_value(set(local_shares), skew_local_share)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    communication_rows = sorted(
        (row for row in rows if row["rank_skew"] == zero_skew),
        key=lambda row: row["mean_remote_share"],
    )
    for field, label in (
        ("mean_max_dispatch_ms", "dispatch"),
        ("mean_max_compute_ms", "compute"),
        ("mean_max_combine_ms", "combine"),
        ("mean_max_end_to_end_ms", "end to end"),
    ):
        axes[0, 0].plot(
            [row["mean_remote_share"] for row in communication_rows],
            [row[field] for row in communication_rows],
            marker="o",
            label=label,
        )
    axes[0, 0].set(
        title=f"Communication sweep (rank_skew={zero_skew:g})",
        xlabel="Measured remote assignment share",
        ylabel="Latency (ms)",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    skew_rows = sorted(
        (
            row
            for row in rows
            if row["local_share"] == selected_local_share
        ),
        key=lambda row: row["rank_skew"],
    )
    for field, label in (
        ("mean_max_dispatch_ms", "dispatch"),
        ("mean_max_compute_ms", "compute"),
        ("mean_max_combine_ms", "combine"),
        ("mean_max_end_to_end_ms", "end to end"),
    ):
        axes[0, 1].plot(
            [row["rank_skew"] for row in skew_rows],
            [row[field] for row in skew_rows],
            marker="o",
            label=label,
        )
    axes[0, 1].set(
        title=f"Load-skew sweep (local_share={selected_local_share:g})",
        xlabel="Configured rank skew",
        ylabel="Latency (ms)",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    row_by_point = {
        (row["rank_skew"], row["local_share"]): row for row in rows
    }
    for axis, field, title in (
        (axes[1, 0], "mean_max_end_to_end_ms", "End-to-end latency (ms)"),
        (axes[1, 1], "mean_token_imbalance", "Received-token imbalance"),
    ):
        matrix = [
            [row_by_point[(skew, share)][field] for share in local_shares]
            for skew in rank_skews
        ]
        image = axis.imshow(matrix, origin="lower", aspect="auto")
        axis.set_xticks(range(len(local_shares)), [f"{x:g}" for x in local_shares])
        axis.set_yticks(range(len(rank_skews)), [f"{x:g}" for x in rank_skews])
        axis.set(
            title=title,
            xlabel="Local share",
            ylabel="Rank skew",
        )
        fig.colorbar(image, ax=axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote plot: {output}")


def main() -> None:
    args = parse_args()
    plot(read_rows(args.input_csv), args.output, args.skew_local_share)


if __name__ == "__main__":
    main()
