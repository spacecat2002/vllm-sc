# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Global statistics collector for GDN (Gated Delta Network) layer analysis.

Usage:
    from vllm.model_executor.layers.mamba.gdn import gdn_stats_collector as gsc

    gsc.enable()
    # ... run inference ...
    gsc.disable()
    stats = gsc.get_stats()
    gsc.save("gdn_stats.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class LayerStats:
    # Per-decode-step mean beta across heads; shape: list of [num_requests]
    betas: list[np.ndarray] = field(default_factory=list)
    # Per-decode-step state delta Frobenius norm; shape: list of [num_requests]
    state_deltas: list[np.ndarray] = field(default_factory=list)
    # Per-decode-step cosine similarity between old and new state.
    state_cosines: list[np.ndarray] = field(default_factory=list)


class _GDNStatsCollector:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._stats: dict[str, LayerStats] = {}
        self.beta_calls: int = 0
        self.delta_calls: int = 0
        self.prefill_forwards: int = 0
        self.decode_forwards: int = 0
        self.disabled_calls: int = 0

    def enable(self) -> None:
        self._enabled = True
        self._stats.clear()
        self.beta_calls = 0
        self.delta_calls = 0
        self.prefill_forwards = 0
        self.decode_forwards = 0
        self.disabled_calls = 0

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def record_beta(
        self,
        layer_name: str,
        beta: torch.Tensor,
    ) -> None:
        """
        Record beta (gate) values for one forward step.

        Args:
            layer_name: identifier for the GDN layer (e.g. self.prefix)
            beta: float tensor, shape [num_requests, num_heads] or
                  [num_heads] for a single request.
                  Values should already be after sigmoid (range 0-1).
        """
        self.beta_calls += 1
        if not self._enabled:
            self.disabled_calls += 1
            return
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)
        # Average across heads → one scalar per request
        mean_per_req = beta.float().mean(dim=-1).cpu().numpy()
        self._stats.setdefault(layer_name, LayerStats()).betas.append(mean_per_req)

    def record_state_delta(
        self,
        layer_name: str,
        delta_norm: torch.Tensor,
    ) -> None:
        """
        Record ||S_t - S_{t-1}|| (Frobenius norm) for one forward step.

        Args:
            layer_name: identifier for the GDN layer
            delta_norm: float tensor, shape [num_requests], one norm per request
        """
        self.delta_calls += 1
        if not self._enabled:
            self.disabled_calls += 1
            return
        if delta_norm.dim() == 0:
            delta_norm = delta_norm.unsqueeze(0)
        self._stats.setdefault(layer_name, LayerStats()).state_deltas.append(
            delta_norm.float().cpu().numpy()
        )

    def record_state_cosine(
        self,
        layer_name: str,
        cosine: torch.Tensor,
    ) -> None:
        if not self._enabled:
            self.disabled_calls += 1
            return
        if cosine.dim() == 0:
            cosine = cosine.unsqueeze(0)
        self._stats.setdefault(layer_name, LayerStats()).state_cosines.append(
            cosine.float().cpu().numpy()
        )

    def note_prefill_forward(self) -> None:
        if self._enabled:
            self.prefill_forwards += 1

    def note_decode_forward(self) -> None:
        if self._enabled:
            self.decode_forwards += 1

    def get_stats(self) -> dict[str, LayerStats]:
        return dict(self._stats)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._stats, f)
        print(f"[GDNStats] Saved stats for {len(self._stats)} layers to {path}")


# Module-level singleton – import this module and call enable()/disable()
_collector = _GDNStatsCollector()


def enable() -> None:
    _collector.enable()


def disable() -> None:
    _collector.disable()


def get_collector() -> _GDNStatsCollector:
    return _collector


def get_stats() -> dict[str, LayerStats]:
    return _collector.get_stats()


def save(path: str) -> None:
    _collector.save(path)
