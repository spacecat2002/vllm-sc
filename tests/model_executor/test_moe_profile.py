# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.moe_profile import rank_distribution


def test_rank_distribution_linear_counts_assignments_and_unique_tokens():
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 3], [-1, 2]])

    assignments, unique_tokens = rank_distribution(
        topk_ids,
        global_num_experts=4,
        num_dispatchers=2,
        round_robin=False,
    )

    assert assignments == [3, 4]
    assert unique_tokens == [2, 3]


def test_rank_distribution_round_robin_counts_by_expert_owner():
    topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]])

    assignments, unique_tokens = rank_distribution(
        topk_ids,
        global_num_experts=6,
        num_dispatchers=3,
        round_robin=True,
    )

    assert assignments == [2, 2, 2]
    assert unique_tokens == [2, 2, 2]
