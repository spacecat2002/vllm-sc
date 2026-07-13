# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_trace import (
    MoETraceCollector,
    MoETraceConfig,
)

pytestmark = pytest.mark.cpu_test


def test_moe_trace_collector_writes_truncated_records(tmp_path):
    config = MoETraceConfig(
        output_dir=tmp_path,
        max_steps=2,
        max_tokens=2,
        activations="input",
        activation_dtype=torch.float16,
    )
    collector = MoETraceCollector(config, {3: "model.layers.3.mlp.experts"})
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[1, 2], [2, 3], [3, 4]])
    topk_weights = torch.full((3, 2), 0.5)

    collector.capture(3, hidden_states, topk_weights, topk_ids)

    rank_dir = tmp_path / "rank_00000"
    record = torch.load(
        rank_dir / "step_000000_layer_0003.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert record["num_tokens_before_truncation"] == 3
    assert record["topk_ids"].shape == (2, 2)
    assert record["topk_ids"].dtype == torch.int32
    assert record["activations"].shape == (2, 4)
    assert record["activations"].dtype == torch.float16

    metadata = json.loads((rank_dir / "metadata.json").read_text())
    assert metadata["layers"] == {"3": "model.layers.3.mlp.experts"}


def test_moe_trace_collector_starts_new_step_on_repeated_layer(tmp_path):
    config = MoETraceConfig(
        output_dir=tmp_path,
        max_steps=2,
        max_tokens=4,
        activations="none",
        activation_dtype=torch.float16,
    )
    collector = MoETraceCollector(config, {1: "layer.1", 2: "layer.2"})
    hidden_states = torch.zeros((1, 4))
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    topk_weights = torch.ones((1, 1))

    collector.capture(1, hidden_states, topk_weights, topk_ids)
    collector.capture(2, hidden_states, topk_weights, topk_ids)
    collector.capture(1, hidden_states, topk_weights, topk_ids)

    rank_dir = tmp_path / "rank_00000"
    assert (rank_dir / "step_000000_layer_0002.pt").exists()
    assert (rank_dir / "step_000001_layer_0001.pt").exists()
    second = torch.load(
        rank_dir / "step_000001_layer_0001.pt", weights_only=True
    )
    assert "activations" not in second


def test_moe_trace_collector_selects_last_prefill_token(tmp_path):
    config = MoETraceConfig(
        output_dir=tmp_path,
        max_steps=1,
        max_tokens=8,
        activations="input",
        activation_dtype=torch.float16,
        token_selection="prefill_last",
    )
    collector = MoETraceCollector(config, {0: "layer.0"})
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]])
    topk_weights = torch.full((3, 2), 0.5)

    collector.capture(0, hidden_states, topk_weights, topk_ids)

    record = torch.load(
        tmp_path / "rank_00000" / "step_000000_layer_0000.pt",
        weights_only=True,
    )
    assert record["selected_token_start"] == 2
    assert record["selected_token_end"] == 3
    assert record["phase"] == "prefill"
    assert torch.equal(record["activations"].float(), hidden_states[-1:])
    assert torch.equal(record["topk_ids"], topk_ids[-1:].int())


def test_moe_trace_collector_selects_batched_prefill_and_decode(tmp_path):
    config = MoETraceConfig(
        output_dir=tmp_path,
        max_steps=1,
        max_tokens=8,
        activations="input",
        activation_dtype=torch.float16,
        token_selection="prefill_last",
    )
    collector = MoETraceCollector(config, {0: "layer.0"})
    # Request 0 schedules three remaining prompt tokens; request 1 schedules
    # one decode token. Packed row indices are therefore [0, 1, 2, 3].
    collector.begin_forward(
        num_scheduled_tokens=[3, 1],
        num_computed_tokens=[2, 5],
        prefill_lengths=[5, 5],
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    topk_ids = torch.arange(8).reshape(4, 2)
    topk_weights = torch.full((4, 2), 0.5)

    collector.capture(0, hidden_states, topk_weights, topk_ids)

    record = torch.load(
        tmp_path / "rank_00000" / "step_000000_layer_0000.pt",
        weights_only=True,
    )
    assert record["phase"] == "mixed"
    assert torch.equal(record["selected_token_indices"], torch.tensor([2, 3]))
    assert torch.equal(record["token_phases"], torch.tensor([0, 1], dtype=torch.int8))
    assert torch.equal(record["activations"].float(), hidden_states[[2, 3]])


def test_moe_trace_collector_selects_all_batched_tokens(tmp_path):
    config = MoETraceConfig(
        output_dir=tmp_path,
        max_steps=1,
        max_tokens=8,
        activations="input",
        activation_dtype=torch.float16,
        token_selection="all",
    )
    collector = MoETraceCollector(config, {0: "layer.0"})
    collector.begin_forward(
        num_scheduled_tokens=[3, 1],
        num_computed_tokens=[2, 5],
        prefill_lengths=[5, 5],
        request_ids=["prefill-request", "decode-request"],
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    router_logits = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    topk_ids = torch.arange(8).reshape(4, 2)
    topk_weights = torch.full((4, 2), 0.5)

    collector.capture(
        0,
        hidden_states,
        router_logits,
        topk_weights,
        topk_ids,
    )

    record = torch.load(
        tmp_path / "rank_00000" / "step_000000_layer_0000.pt",
        weights_only=True,
    )
    assert record["phase"] == "mixed"
    assert torch.equal(record["selected_token_indices"], torch.arange(4))
    assert torch.equal(
        record["token_phases"], torch.tensor([0, 0, 0, 1], dtype=torch.int8)
    )
    assert record["request_ids"] == [
        "prefill-request",
        "prefill-request",
        "prefill-request",
        "decode-request",
    ]
    assert torch.equal(record["activations"].float(), hidden_states)
