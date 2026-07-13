# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot one iteration-latency figure per MoE EP sweep point."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="moe_ep",
        help="Filename prefix for generated PNG files.",
    )
    return parser.parse_args()


def read_aggregate_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("record_type") == "aggregate":
                records.append(record)
    if not records:
        raise ValueError(f"No aggregate records found in {path}.")
    return records


def filename_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_sweep_points(
    records: list[dict[str, Any]],
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

    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (float(record["local_share"]), float(record["rank_skew"]))
        grouped[key].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for (local_share, rank_skew), point_records in sorted(grouped.items()):
        point_records.sort(key=lambda record: int(record["iter"]))
        iterations = [int(record["iter"]) for record in point_records]

        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for field, label in (
            ("max_dispatch_ms", "dispatch"),
            ("max_expert_compute_ms", "compute"),
            ("max_combine_ms", "combine"),
        ):
            axis.plot(
                iterations,
                [float(record[field]) for record in point_records],
                linewidth=1.2,
                label=label,
            )
        axis.set(
            title=(
                f"MoE EP latency: local_share={local_share:g}, "
                f"rank_skew={rank_skew:g}"
            ),
            xlabel="Iteration",
            ylabel="Max-rank stage latency (ms)",
        )
        axis.grid(alpha=0.25)
        axis.legend()

        filename = (
            f"{prefix}_local_{filename_value(local_share)}"
            f"_skew_{filename_value(rank_skew)}.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    output_paths = plot_sweep_points(
        read_aggregate_records(args.input_jsonl),
        args.output_dir,
        args.prefix,
    )
    print(f"Wrote {len(output_paths)} plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
