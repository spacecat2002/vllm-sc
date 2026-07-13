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

    for local_share in local_shares:
        selected = sorted(
            (row for row in rows if row["local_share"] == local_share),
            key=lambda row: row["rank_skew"],
        )
        axes[1, 0].plot(
            [row["rank_skew"] for row in selected],
            [row["mean_max_total_ms"] for row in selected],
            marker="o",
            label=f"local_share={local_share:g}",
        )
    axes[1, 0].set(
        title="All local shares by rank skew",
        xlabel="Configured rank skew",
        ylabel="Stage-sum latency (ms)",
    )
    axes[1, 0].legend(fontsize="small")
    axes[1, 0].grid(alpha=0.25)

    for rank_skew in rank_skews:
        selected = sorted(
            (row for row in rows if row["rank_skew"] == rank_skew),
            key=lambda row: row["local_share"],
        )
        axes[1, 1].plot(
            [row["local_share"] for row in selected],
            [row["mean_max_total_ms"] for row in selected],
            marker="o",
            label=f"rank_skew={rank_skew:g}",
        )
    axes[1, 1].set(
        title="All rank skews by local share",
        xlabel="Configured local share",
        ylabel="Stage-sum latency (ms)",
    )
    axes[1, 1].legend(fontsize="small")
    axes[1, 1].grid(alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote plot: {output}")


def main() -> None:
    args = parse_args()
    plot(read_rows(args.input_csv), args.output, args.skew_local_share)


if __name__ == "__main__":
    main()
