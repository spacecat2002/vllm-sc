# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import ProcessGroup, all_gather

from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TransferMetadata:
    """Metadata describing a completed EPLB buffer transfer."""

    is_unchanged: np.ndarray
    """Mask of (num_local_experts,) indicating experts unchanged after rebalance."""
    is_received_locally: np.ndarray
    """Mask of (num_local_experts,) indicating experts received from local data."""
    recv_primary_mask: np.ndarray
    """Mask of (num_local_experts,) indicating primary experts received."""
    recv_count: int
    """Number of received experts for the layer."""
    recv_expert_ids: np.ndarray
    """Expert ids (num_local_experts,) of remote primary experts."""
    recv_dst_rows: np.ndarray
    """Target expert indices (num_local_experts,) in local tensors to send."""


@dataclass
class AsyncEplbLayerResult:
    """
    The result of one completed async EPLB layer transfer.
    """

    layer_idx: int
    """Index of the MoE layer that was transferred."""
    new_physical_to_logical_map: torch.Tensor
    """
    New physical→logical mapping for layers_idx, on CPU.
    Shape: (num_physical_experts)
    """
    transfer_metadata: TransferMetadata
    """Metadata describing what was received during transfer_layer."""
    consumed_event: CpuGpuEvent
    """
    Event used to synchronize access to the intermediate buffer. The async worker calls
    wait() after it finishes transferring weights to the intermediate buffer. The main
    thread calls record() after it finishes transferring weights out of the intermediate
    buffer in _move_to_workspace()
    """


def get_ep_ranks_with_experts_batch(
    expert_ids: np.ndarray,
    num_local_experts: int,
    old_indices: np.ndarray, # eg. [0,1,2,3,0,1,2,3]
    new_indices: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        expert_ids: 1D array of expert indices to query.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two dictionaries mapping expert_id to:
        - ranks_to_send: The ranks that have this expert and need to send.
        - ranks_to_recv: The ranks that need to receive this expert.
    """
    ranks_to_send_map: dict[int, list[int]] = {}
    ranks_to_recv_map: dict[int, list[int]] = {}

    # Fast path: if no experts, return empty dicts
    if expert_ids.size == 0:
        return ranks_to_send_map, ranks_to_recv_map

    unique_experts = np.unique(expert_ids) # [2,0,2] -> [0,2]
    num_positions = len(old_indices) # len([0,1,2,3,0,1,2,3]) = 8
    position_indices = np.arange(num_positions, dtype=np.int32) # [0,1,2,3,4,5,6,7]

    # Vectorized approach: find all positions matching any query expert in one pass
    # Use np.isin to get boolean masks for all relevant positions at once
    # 如果某个专家在旧布局中出现在某个 rank 的物理槽位上，就说明这个 rank 可以发送该专家的权重。
    old_relevant_mask = np.isin(old_indices, unique_experts) # [0,1,2,3,0,1,2,3] vs [0,2] -> [True, False, True, False, True, False, True, False]
    new_relevant_mask = np.isin(new_indices, unique_experts)

    # Process old_indices (send ranks)
    if np.any(old_relevant_mask):
        old_relevant_positions = position_indices[old_relevant_mask] # [0,2,4,6]
        old_relevant_experts = old_indices[old_relevant_mask] # [0,2,0,2]
        old_relevant_ranks = old_relevant_positions // num_local_experts # [0,2,4,6] // 4 = [0,0,1,1]

        # Sort by expert first, then by position (to maintain first-appearance order) 
        # [0,2,0,2] -> [0,0,2,2], [0,0,1,1] -> [0,1,0,1]
        sort_order = np.lexsort((old_relevant_positions, old_relevant_experts))
        sorted_experts = old_relevant_experts[sort_order] # [0,0,2,2]
        sorted_ranks = old_relevant_ranks[sort_order] # [0,1,0,1]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        ) # 将专家边界的索引存储在一个数组中，例如[0,2,4]表示专家0的边界是[0,2), 专家2的边界是[2,4)

        # For each expert, extract unique ranks in order of first appearance
        for i in range(len(expert_boundaries) - 1):  # 例如当前i=0，边界为[0,2,4]
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start]) # expert = 0
            expert_ranks = sorted_ranks[start:end] # expert_ranks = [0,1]  # 对应的rank是[0,1]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True) # 保留顺序的唯一rank索引，例如[0,1] -> [0,1]
            unique_ranks = expert_ranks[np.sort(unique_idx)]
            ranks_to_send_map[expert] = unique_ranks.tolist()

    # Process new_indices (recv ranks)
    if np.any(new_relevant_mask):
        new_relevant_positions = position_indices[new_relevant_mask]
        new_relevant_experts = new_indices[new_relevant_mask]
        new_relevant_ranks = new_relevant_positions // num_local_experts

        # Sort by expert first, then by position
        sort_order = np.lexsort((new_relevant_positions, new_relevant_experts))
        sorted_experts = new_relevant_experts[sort_order]
        sorted_ranks = new_relevant_ranks[sort_order]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        )

        # For each expert, extract unique ranks and exclude local copies
        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]

            # Remove ranks that have local copies (in send map)
            send_ranks_set = set(ranks_to_send_map.get(expert, []))
            recv_ranks_actual = [
                int(r) for r in unique_ranks if r not in send_ranks_set
            ]
            ranks_to_recv_map[expert] = recv_ranks_actual

    # Handle experts that only appear in old (send only) or new (recv only)
    for expert in unique_experts:
        expert = int(expert)
        if expert not in ranks_to_send_map:
            ranks_to_send_map[expert] = []
        if expert not in ranks_to_recv_map:
            ranks_to_recv_map[expert] = []

    return ranks_to_send_map, ranks_to_recv_map


def move_to_buffer(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: Sequence[torch.Tensor],
    cuda_stream: torch.cuda.Stream | None,
    ep_rank: int,
    communicator: EplbCommunicator,
    layer_idx: int = 0,
) -> TransferMetadata:
    """
    Rearranges expert weights during EPLB rebalancing.

    Args:
        num_local_experts: Number of local experts.
        old_indices: (num_experts_total,) ndarray of current (old)
            global-to-local expert assignments.
        new_indices: (num_experts_total,) ndarray of desired (new)
            global-to-local assignments after rebalance.
        expert_weights: Original expert weights for the layer.
        expert_weights_buffers: Intermediate buffers (one per tensor).
        cuda_stream: CUDA stream for async copies (can be None for sync mode).
        ep_rank: Rank of this process in expert parallel group.
        communicator: EplbCommunicator instance for P2P communication.
        layer_idx: Index of the MoE layer being transferred.

    Returns:
        TransferMetadata: Metadata needed for completing remote weight transfers.
    """
    assert old_indices.shape == new_indices.shape
    # 表示当前 rank 的每个本地槽位是否是某个远程专家的“主接收槽位”。
    # 如果一个远程专家需要在当前 rank 上保存多个副本，只接收一次，其他副本稍后从主接收槽位本地复制。
    recv_primary_mask = np.zeros((num_local_experts,), dtype=np.bool_)
    # 保存当前 rank 需要发送的逻辑专家 ID。
    send_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    # 保存每个待发送专家在当前 rank 本地权重张量中的源行。
    send_src_rows = np.full((num_local_experts,), -1, dtype=np.int32)
    # 保存当前 rank 需要接收的逻辑专家 ID。
    recv_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    # 保存每个待接收专家在当前 rank 本地权重张量中的目标行。
    recv_dst_rows = np.full((num_local_experts,), -1, dtype=np.int32)

    # base是当前 rank 的第一个本地专家在全局专家索引中的位置。例如ep_rank=2, num_local_experts=4, 那么base=8
    base = ep_rank * num_local_experts
    # 得到当前 rank 的本地权重行, 例如num_local_experts=4, 那么local_rows=[0,1,2,3]
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    # 把本地行号映射到全局行号, 例如base=8, local_rows=[0,1,2,3], 那么local_global=[8,9,10,11]
    local_global = base + local_rows

    old_local_expert_ids = old_indices[local_global]
    new_local_expert_ids = new_indices[local_global]

    # Unchanged mask
    is_unchanged = old_local_expert_ids == new_local_expert_ids

    # Local receive eligibility
    new_valid = new_local_expert_ids != -1
    # 检查新的需要接收的专家是否在旧的本地专家列表中出现过，如果出现过，说明可以直接本地复制
    can_recv_local = np.isin(
        new_local_expert_ids, old_local_expert_ids, assume_unique=False
    )
    # 所有未改变的和可以本地复制的专家都被标记为本地接收
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local)
    )

    # Send map: first src row per unique expert present locally in old mapping
    send_count = 0
    valid_old = old_local_expert_ids != -1
    if np.any(valid_old): # 只有当前rank至少保存一个有效专家时才可以作为发送者
        uniq_experts, first_idx = np.unique(
            old_local_expert_ids[valid_old], return_index=True
        ) # 找到当前 rank 的本地专家中每个唯一逻辑专家的第一个出现位置
        filtered_rows = local_rows[valid_old] # 过滤掉无效的本地行，只保留有效的本地行
        src_rows = filtered_rows[first_idx] # 对于每个唯一逻辑专家，找到其在当前 rank 本地权重张量中的源行
        send_count = int(uniq_experts.shape[0]) # 计算当前 rank 需要发送的唯一逻辑专家数量
        send_expert_ids[:send_count] = uniq_experts # 保存当前 rank 需要发送的逻辑专家 ID
        send_src_rows[:send_count] = src_rows # 保存每个待发送专家在当前 rank 本地权重张量中的源行

    # Recv map: primary dst per unique expert needed remotely
    recv_count = 0
    need_recv_mask = np.logical_and(~is_received_locally, new_valid)
    if np.any(need_recv_mask):
        desired_experts = new_local_expert_ids[need_recv_mask]
        desired_dsts = local_rows[need_recv_mask]
        uniq_recv_experts, uniq_indices = np.unique(
            desired_experts, return_index=True
        ) # 如果同一个远程专家在多个目标槽位出现，只选择第一个主目标槽位
        dst_rows = desired_dsts[uniq_indices]
        recv_count = int(uniq_recv_experts.shape[0])
        recv_expert_ids[:recv_count] = uniq_recv_experts
        recv_dst_rows[:recv_count] = dst_rows
        recv_primary_mask[dst_rows] = True

    # 目标槽位发生变化，并且当前 rank 需要接收的专家是本地接收的，才可以直接从本地复制到中间缓冲区
    eligible_local_buffer_mask = np.logical_and(~is_unchanged, is_received_locally)

    # 1. Local moves into tmp buffers
    if bool(eligible_local_buffer_mask.any()) and send_count > 0:
        dest_indices = np.nonzero(eligible_local_buffer_mask)[0].tolist()
        expert_to_src_map = dict(
            zip(send_expert_ids[:send_count], send_src_rows[:send_count])
        )
        for dst in dest_indices:
            expert = new_local_expert_ids[dst]
            src_local = expert_to_src_map.get(expert, -1)
            if src_local != -1:
                with torch.cuda.stream(cuda_stream):
                    for w, b in zip(expert_weights, expert_weights_buffers):
                        b[dst].copy_(w[src_local], non_blocking=True)

    communicator.set_transfer_context(old_indices, layer_idx)

    # 2. Post sends
    if send_count > 0:
        experts = send_expert_ids[:send_count]
        srcs = send_src_rows[:send_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]

        # send_map表示旧布局中拥有该专家的rank，recv_map表示新布局中需要该专家的rank
        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, src in zip(experts.tolist(), srcs.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            # 计算每个发送者至少需要负责多少个接收者
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender # 计算剩余的接收者起始位置
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv): # 如果当前发送者还有剩余的接收者需要负责
                recv_ranks.append(ranks_to_recv[recver_pos])
            expert_tensors = [w[src] for w in expert_weights]
            for dst in recv_ranks:
                communicator.add_send(expert_tensors, dst, expert_id=int(expert))

    # 3. Post recvs
    if recv_count > 0:
        experts = recv_expert_ids[:recv_count]
        dsts = recv_dst_rows[:recv_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, dst in zip(experts.tolist(), dsts.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank) # 按照上面send的逻辑寻找当前rank对应的发送者
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src = ranks_to_send[recver_pos - remainder_start]
            communicator.add_recv(
                [b[dst] for b in expert_weights_buffers],
                src,
                expert_id=int(expert),
            )

    # 4. Execute transfers and wait for completion.
    communicator.execute()
    return TransferMetadata(
        is_unchanged=is_unchanged,
        is_received_locally=is_received_locally,
        recv_primary_mask=recv_primary_mask,
        recv_count=recv_count,
        recv_expert_ids=recv_expert_ids,
        recv_dst_rows=recv_dst_rows,
    )


def move_from_buffer(
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: list[torch.Tensor],
    transfer_metadata: TransferMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """
    Copies expert weights from communication buffers back to the target weight tensors
    after EPLB rebalancing.

    Args:
        expert_weights: List of the actual MoE layer weights used in the execution.
        expert_weights_buffers: Intermediate buffers containing the experts weights
            after the transfer is completed.
        transfer_metadata: TransferMetadata containing transfer metadata.
        new_indices: (num_experts_total,) mapping from local rows to desired
            (possibly global) expert id, after rebalance.
        ep_rank: Rank of the process in the expert parallel group.
    """
    is_unchanged = transfer_metadata.is_unchanged
    is_received_locally = transfer_metadata.is_received_locally
    recv_primary_mask = transfer_metadata.recv_primary_mask
    recv_count = transfer_metadata.recv_count
    recv_expert_ids = transfer_metadata.recv_expert_ids
    recv_dst_rows = transfer_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    # Mask for rows to copy back from buffers:
    # copy if locally received OR remote primary recv
    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask_np = np.logical_and(~is_unchanged, copy_mask)
    if bool(dest_mask_np.any()):
        dest_indices = np.nonzero(dest_mask_np)[0].tolist()
        for dst in dest_indices:
            for w, b in zip(expert_weights, expert_weights_buffers):
                w[dst].copy_(b[dst], non_blocking=True)

    if recv_count == 0:
        return

    # Duplicate remote received rows to non-primary duplicate dsts
    base = ep_rank * num_local_experts
    local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1),
    )
    # All received experts are unique in the destination, so no need to copy duplicates
    if not bool(duplicate_mask.any()):
        return

    dup_dst_rows = np.nonzero(duplicate_mask)[0]
    dup_experts = local_experts[dup_dst_rows]

    prim_experts = recv_expert_ids[:recv_count]
    prim_dsts = recv_dst_rows[:recv_count]
    order = np.argsort(prim_experts, kind="stable")
    prim_experts_sorted = prim_experts[order]
    prim_dsts_sorted = prim_dsts[order]
    pos = np.searchsorted(prim_experts_sorted, dup_experts)
    valid = np.logical_and(
        pos < prim_experts_sorted.shape[0],
        prim_experts_sorted[np.minimum(pos, prim_experts_sorted.shape[0] - 1)]
        == dup_experts,
    )
    if not bool(valid.any()):
        return

    matched_dst_rows = dup_dst_rows[valid]
    matched_src_rows = prim_dsts_sorted[pos[valid]]

    for dst, src in zip(matched_dst_rows.tolist(), matched_src_rows.tolist()):
        for w in expert_weights:
            w[dst].copy_(w[src], non_blocking=True)


def transfer_layer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
    layer_idx: int = 0,
) -> TransferMetadata:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_layer_indices: Shape (num_physical_experts,).
        new_layer_indices: Shape (num_physical_experts,).
        expert_weights: Iterable of weight tensors for this layer, each with shape
            (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection.
        expert_weights_buffer: Intermediate buffers (one per weight tensor).
        ep_group: The device process group for expert parallelism.
        communicator: EplbCommunicator instance for P2P communication.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        cuda_stream: CUDA stream for async copies (can be None for sync mode).
        rank_mapping: Optional rank mapping for elastic expert parallelism.
        layer_idx: Index of the MoE layer being transferred.

    Returns:
        TransferMetadata: Metadata needed for completing remote weight transfers,
            including is_unchanged and is_received_locally masks.
    """
    ep_size = ep_group.size()
    if rank_mapping is not None:
        # Add a layer dimension for compatibility with mapping functions
        old_layer_indices_2d = old_layer_indices.unsqueeze(0)
        new_layer_indices_2d = new_layer_indices.unsqueeze(0)

        if len(rank_mapping) == ep_group.size():
            # scale down
            new_layer_indices_2d = _map_new_expert_indices_with_rank_mapping(
                new_layer_indices_2d,
                rank_mapping,
            )
        else:
            # scale up
            old_layer_indices_2d = _map_old_expert_indices_with_rank_mapping(
                old_layer_indices_2d,
                rank_mapping,
                ep_group.size(),
            )

        # Remove the layer dimension
        old_layer_indices = old_layer_indices_2d.squeeze(0)
        new_layer_indices = new_layer_indices_2d.squeeze(0)

    assert old_layer_indices.shape == new_layer_indices.shape
    num_physical_experts = old_layer_indices.shape[0]
    assert len(expert_weights[0]) >= 1
    num_local_physical_experts = expert_weights[0].shape[0]
    assert num_physical_experts == ep_size * num_local_physical_experts

    old_layer_indices_np = old_layer_indices.cpu().numpy()
    new_layer_indices_np = new_layer_indices.cpu().numpy()

    return move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices=old_layer_indices_np,
        new_indices=new_layer_indices_np,
        expert_weights=expert_weights,
        expert_weights_buffers=expert_weights_buffer,
        cuda_stream=cuda_stream,
        ep_rank=ep_group.rank(),
        communicator=communicator,
        layer_idx=layer_idx,
    )


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        expert_buffer: Pre-allocated receive buffer tensors (one per
            weight tensor in a single layer).
        ep_group: The device process group for expert parallelism.
        communicator: EplbCommunicator instance for P2P communication.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    assert len(expert_weights[0]) >= 1

    # 第0维: MoE层数, 第1维: 不同的权重，比如gate_up_proj, gate_up_proj的shape[0]是物理专家的个数
    num_local_physical_experts = expert_weights[0][0].shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    ep_rank = ep_group.rank()
    assert num_physical_experts == ep_size * num_local_physical_experts

    first_layer_weights = list(expert_weights[0])

    if is_profile:
        if communicator.needs_profile_buffer_reservation:
            # Reserve NCCL communication buffers via a dummy all_gather.
            # Backends that pre-allocate their own transfer buffers
            # skip this to avoid the extra memory spike during profiling.
            profile_buffer: list[torch.Tensor] = [
                torch.empty_like(w) for w in first_layer_weights
            ]
            for weight, buffer in zip(expert_weights[0], profile_buffer):
                dummy_recv_buffer = [buffer for _ in range(ep_size)]
                torch.distributed.barrier()
                all_gather(
                    dummy_recv_buffer,
                    weight,
                    group=ep_group,
                )
        return

    weights_buffer = list(expert_buffer)

    old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    for layer_idx in range(num_moe_layers):
        transfer_metadata = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices=old_global_expert_indices_cpu[layer_idx],
            new_indices=new_global_expert_indices_cpu[layer_idx],
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            cuda_stream=None,
            ep_rank=ep_rank,
            communicator=communicator,
            layer_idx=layer_idx,
        )

        move_from_buffer(
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            transfer_metadata=transfer_metadata,
            new_indices=new_global_expert_indices_cpu[layer_idx],
            ep_rank=ep_rank,
        )


def _map_old_expert_indices_with_rank_mapping(
    old_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
    new_ep_size: int,
) -> torch.Tensor:
    """
    Map the old global expert indices to the new global expert indices.

    Args:
        old_global_expert_indices:
            Shape (num_layers, old_ep_size * num_local_physical_experts).
        rank_mapping: Mapping from old rank to new rank.
        new_ep_size: New expert parallelism size.

    Returns:
        Mapped expert indices with shape
        (num_layers, new_ep_size * num_local_physical_experts).
    """
    num_layers, old_num_physical_experts = old_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    num_local_physical_experts = old_num_physical_experts // old_ep_size
    new_num_physical_experts = new_ep_size * num_local_physical_experts

    # Create mapped tensor with new shape, initialized to -1
    mapped_expert_indices = torch.full(
        (num_layers, new_num_physical_experts),
        fill_value=-1,
        dtype=old_global_expert_indices.dtype,
        device=old_global_expert_indices.device,
    )

    # Handle rank mapping (scale up/down with rank changes)
    for old_rank in range(old_ep_size):
        new_rank = rank_mapping.get(old_rank)
        if new_rank is not None and new_rank >= 0 and new_rank < new_ep_size:
            # This old rank exists in the new configuration
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, new_start_idx:new_end_idx] = (
                old_global_expert_indices[:, old_start_idx:old_end_idx]
            )
        # If new_rank is None or >= new_ep_size, the experts remain -1
        # (scale down case)

    return mapped_expert_indices


def _map_new_expert_indices_with_rank_mapping(
    new_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
) -> torch.Tensor:
    num_layers, new_num_physical_experts = new_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    new_ep_size = sum(new_rank != -1 for new_rank in rank_mapping.values())
    num_local_physical_experts = new_num_physical_experts // new_ep_size
    old_num_physical_experts = old_ep_size * num_local_physical_experts

    mapped_expert_indices = torch.full(
        (num_layers, old_num_physical_experts),
        fill_value=-1,
        dtype=new_global_expert_indices.dtype,
        device=new_global_expert_indices.device,
    )

    for old_rank in range(old_ep_size):
        new_rank = rank_mapping[old_rank]
        if new_rank >= 0 and new_rank < new_ep_size:
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, old_start_idx:old_end_idx] = (
                new_global_expert_indices[:, new_start_idx:new_end_idx]
            )

    return mapped_expert_indices


__all__ = ["transfer_layer", "move_from_buffer", "TransferMetadata"]
