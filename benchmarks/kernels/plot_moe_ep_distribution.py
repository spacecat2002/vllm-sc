# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot MoE EP stage-sum latency across locality/skew sweep points."""

from __future__ import annotations

import argparse
import csv
import math
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
        row["mean_max_dispatch_ms"] = float(row["mean_max_dispatch_ms"])
        row["mean_max_compute_ms"] = float(row["mean_max_compute_ms"])
        row["mean_max_combine_ms"] = float(row["mean_max_combine_ms"])
    return rows


def plot_facets(
    plt,
    rows: list[dict[str, Any]],
    *,
    facet_field: str,
    x_field: str,
    facet_values: list[float],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    num_cols = 2
    num_rows = math.ceil(len(facet_values) / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(13, 4 * num_rows),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    stage_fields = (
        ("mean_max_dispatch_ms", "dispatch"),
        ("mean_max_compute_ms", "compute"),
        ("mean_max_combine_ms", "combine"),
    )
    for axis, facet_value in zip(axes.flat, facet_values):
        selected = sorted(
            (row for row in rows if row[facet_field] == facet_value),
            key=lambda row: row[x_field],
        )
        for field, label in stage_fields:
            axis.plot(
                [row[x_field] for row in selected],
                [row[field] for row in selected],
                marker="o",
                label=label,
            )
        axis.set(
            title=f"{facet_field}={facet_value:g}",
            xlabel=xlabel,
            ylabel="Mean max-rank stage latency (ms)",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in list(axes.flat)[len(facet_values) :]:
        axis.set_visible(False)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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

    by_rank_skew_path = output_dir / f"{prefix}_by_rank_skew.png"
    by_local_share_path = output_dir / f"{prefix}_by_local_share.png"

    plot_facets(
        plt,
        rows,
        facet_field="local_share",
        x_field="rank_skew",
        facet_values=local_shares,
        title="Stage latency by rank skew for each local share",
        xlabel="Configured rank skew",
        output_path=by_local_share_path,
    )
    plot_facets(
        plt,
        rows,
        facet_field="rank_skew",
        x_field="local_share",
        facet_values=rank_skews,
        title="Stage latency by local share for each rank skew",
        xlabel="Configured local share",
        output_path=by_rank_skew_path,
    )

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
