# Hybrid Attention 中 GDN 线性注意力与 Hybrid Cache 流程

本文按当前 vLLM 实现梳理 hybrid attention 里的 GDN 线性注意力计算，以及 hybrid KV cache 的 block size 解析、组织管理和 prefix cache 流程。相关实现主要在 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`、`vllm/v1/attention/backends/gdn_attn.py`、`vllm/platforms/interface.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/single_type_kv_cache_manager.py`。

## GDN 线性注意力的计算过程

GDN 在 vLLM 里被归到 SSM/Mamba 类后端，但计算形式是 linear attention 风格的递推状态更新，而不是 softmax attention。`GatedDeltaNetAttention` 基类把该层声明为 `MambaAttentionBackendEnum.GDN_ATTN`，所以调度和 cache 管理侧把它当作需要状态 cache 的 SSM 层处理。以 Qwen3-Next/Qwen3.5 的 `QwenGatedDeltaNetAttention` 为例，一个 forward 分成三段：输入投影、GDN core attention、自带门控归一化后的输出投影。

输入投影阶段先从 hidden states 产生两组张量。`in_proj_qkvz` 生成 query、key、value 和输出门控 `z`；`in_proj_ba` 生成每个 value head 对应的 `b` 和 `a`，它们会进入 GDN 的 gating 公式。Qwen3-Next 和 Qwen3.5 的 checkpoint 排布不同：Qwen3-Next 使用 interleaved GQA layout，代码通过 `fix_query_key_value_ordering()` 或编译路径里的 `prepare_gdn_attention_core_inputs()` 把交错布局拆成连续的 q/k/v/z/b/a；Qwen3.5 的权重本身按 `[q, k, v, z]` 和 `[b, a]` 排列，直接 split 即可。最终传给 core op 的 `mixed_qkv` 是 `[q, k, v]` 拼接后的扁平张量，`z` 保留给最后的 `RMSNormGated`。

core attention 阶段由 `torch.ops.vllm.qwen_gdn_attention_core` 或 CPU/XPU 对应 op 承担。它从 forward context 里取当前层的 `GDNAttentionMetadata` 和该层的 `kv_cache`，其中 `kv_cache[0]` 是 causal conv state，`kv_cache[1]` 是 GDN/SSM recurrent state。core 先对 `mixed_qkv` 做一维因果卷积：prefill 使用 `causal_conv1d_fn` 处理变长序列并把最后的卷积窗口写回 conv state；decode 使用 `causal_conv1d_update` 对每个请求的当前 token 增量更新 conv state。卷积输出再被 `rearrange_mixed_qkv()` 拆成形状近似为 `(1, seq, heads, dim)` 的 query、key、value。

GDN 的状态更新可以理解为对每个 token、每个 head 维护一个矩阵状态 `S`。prefill 时，`a`、`b`、`A_log`、`dt_bias` 先生成门控量 `g` 和 `beta`；CPU 路径里能直接看到 `ops.fused_gdn_gating_cpu(A_log, a, b, dt_bias)`，CUDA/ROCm 路径则在 fused kernel 或 FLA/FlashInfer/CuteDSL kernel 内做同类计算。随后执行 chunk gated delta rule：`q`、`k` 通常先做 L2 norm，`g` 表示状态衰减，`beta` 表示写入强度，`k/v` 形成对状态的增量写入，`q` 再从更新后的状态读出输出。直观地说，每个 token 都在做 `S_t = gate(S_{t-1}, k_t, v_t, beta_t)`，`o_t = q_t @ S_t`，只是实际 kernel 会按 chunk 并行扫描来避免逐 token Python 循环。prefill kernel 返回所有 token 的 `attn_out`，并在 `output_final_state=True` 时返回最后 recurrent state；vLLM 把这个 final state 写回对应请求的 SSM cache block。

decode 是单 token 增量路径，核心区别是不用 chunk scan，而是直接对每个请求取当前 state 做一次 recurrent update。CPU 路径调用 `fused_sigmoid_gating_delta_rule_update_cpu()`，CUDA 路径会走 `fused_sigmoid_gating_delta_rule_update` 或 packed recurrent decode。它读取 `state_indices_tensor` 定位每个请求的当前状态槽位，用当前 token 的 q/k/v/a/b 更新 state 并产出当前 token 的 core attention 输出。spec decode 存在时，metadata 会把 speculative decode 和普通 prefill/decode 拆开：`spec_state_indices_tensor` 可以包含一个请求的多个 speculative state block，`num_accepted_tokens` 用来决定哪些 speculative 状态最终可承接。

prefill kernel 的选择在 `ChunkGatedDeltaRule` 里完成。默认是 Triton/FLA 的 `fla_chunk_gated_delta_rule`；CUDA Hopper 或满足限制的 Blackwell 可以选择 FlashInfer；Blackwell 且显式请求时也可能选择 in-tree CuteDSL。FlashInfer 路径会把 `g` 转为 `exp(g)` 后调用 `flashinfer.gdn_prefill.chunk_gated_delta_rule`，并把输出 reshape 回 FLA 风格。CuteDSL 路径要求 metadata 里提前准备好 `chunk_indices` 和 `chunk_offsets`。这些差异只影响 kernel 实现和性能，不改变上层语义：prefill 都消费一段 token、初始 state 和 chunk metadata，输出 token 级结果以及最后 state。

最后输出阶段把 core attention 输出和 `z` 一起送入 `RMSNormGated`。也就是说 GDN 的 value-head 输出会先按 head_v_dim 做 gated RMSNorm，再 flatten head 维度，经过 `out_proj` 回到 hidden size。这个 `z` 是输出门，不是 recurrent state；它只参与当前 forward 的输出调制，不进入 hybrid cache。

## Hybrid Cache 的 block size 确认

vLLM 里要区分三种粒度。第一种是 attention KV cache 的 `cache_config.block_size`，默认来自 `CacheConfig.DEFAULT_BLOCK_SIZE=16`，也可能由 attention backend 的 preferred block size 改写。第二种是 Mamba/GDN state cache 的 `cache_config.mamba_block_size`，只有 prefix caching 相关路径需要显式关心；不指定时由平台对齐逻辑推导。第三种是 scheduler/hash 粒度：`scheduler_block_size` 是多个 KV cache group block size 的 LCM，保证调度出的 `num_computed_tokens` 能同时对齐所有 group；`hash_block_size` 是请求 block hash 的计算粒度，在多 group 且 prefix cache/KV connector 启用时通常取各 group block size 的 GCD，用户也可用 `cache_config.hash_block_size` 指定更细粒度，但每个 group block size 必须能被它整除。

平台初始化会先为普通 attention backend 选 block size，再对 hybrid 模型做 `_align_hybrid_block_size()`。这一步先计算 attention 每 token 的 page size：普通 full attention 约为 `num_kv_heads * (K + V head dim) * dtype_size`，MLA、TurboQuant 等特殊 cache dtype 有各自公式。然后通过模型类的 `get_mamba_state_shape_from_config()` 和 `get_mamba_state_dtype_from_config()` 构造 `MambaSpec(block_size=-1)`，计算单个 GDN/Mamba state page 的真实字节数。GDN 的 state shape 来自 `MambaStateShapeCalculator.gated_delta_net_state_shape()`，包含 conv state、SSM/recurrent state，以及 speculative decode 需要的额外状态形状。

对 hybrid attention+GDN 来说，一个关键约束是：单个物理 block pool 只能用统一 page size，所以 attention page 必须大于等于 GDN/Mamba state page，并且最终要把 GDN/Mamba page padding 到和 attention page 完全一致。代码里的核心关系是：

```text
attn_page_size = attention_block_size * attn_page_size_1_token
mamba_page_size = sum(prod(state_shape_i) * dtype_size_i)
attn_page_size >= mamba_page_size
mamba_page_size_padded = attn_page_size
```

如果启用 `mamba_cache_mode == "all"`，也就是支持按 block 缓存所有 GDN/Mamba checkpoint，vLLM 会优先按 Mamba chunk 和 kernel block 对齐来选 attention block size。具体是先取 `base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()`，再和 attention backend 支持的 kernel block alignment 做 LCM，得到 `chunk_size`；然后计算一个 GDN state 至少等价于多少个 attention token：`attn_tokens_per_mamba_state = ceil(mamba_page_size / attn_page_size_1_token)`；最终 `attn_block_size` 会向上取整到 `chunk_size` 的倍数。此时 `cache_config.mamba_block_size` 也被设成这个对齐后的 `attn_block_size`，这样 full attention 和 GDN group 的 block 粒度一致，prefix cache 命中长度天然对齐。

如果没有用 `"all"`，平台逻辑只要求 attention block size 满足 backend kernel alignment 且 page 能容纳 GDN state：`attn_block_size = kernel_block_alignment_size * ceil(mamba_page_size / (kernel_block_alignment_size * attn_page_size_1_token))`。如果 `mamba_cache_mode == "align"`，`cache_config.mamba_block_size` 会直接等于最终 attention block size；`"none"` 则不做细粒度 Mamba prefix checkpoint，只保留运行态 state。XPU 还有一个额外修正：GDN kernel 只支持 block size 被 64 整除，所以会把最终 block size 再向上对齐到 64，并同步更新 `mamba_page_size_padded`。

生成 scheduler 用的 block size 时，`resolve_kv_cache_block_sizes()` 会看最终 KV cache groups。单 group 时 `scheduler_block_size = cache_config.block_size * dcp * pcp`，`hash_block_size` 相同。多 group 时不支持 DCP/PCP，`scheduler_block_size = lcm(group.block_size...)`。如果 prefix cache 和 connector 都没启用，hash 粒度直接等于 scheduler 粒度；如果有 Mamba/GDN group 的 block size 和 `cache_config.block_size` 不一致，也会退回 scheduler 粒度，避免 finer hash 与 Mamba block 无法整除。否则默认 `hash_block_size = gcd(group.block_size...)`，用于在请求创建时生成最细公共 block hash，后续按 group 需要再合并成更大 block hash。

## Hybrid Cache 的组织与管理

KV cache 的静态结构由 `KVCacheConfig` 描述，包含 `num_blocks`、`kv_cache_tensors` 和 `kv_cache_groups`。`KVCacheGroupSpec` 是逻辑分组：同一组里的层共享同一张 block table，并且必须有相同的 cache 行为，比如 full attention、sliding window、Mamba/GDN。`KVCacheTensor` 是 worker 实际初始化的物理 tensor 描述，`shared_by` 说明这个 tensor 被哪些层共享。

hybrid 模型的核心是把不同类型 layer 的 page size 统一。`get_kv_cache_groups()` 会先按 spec 判断是否能作为一个 uniform group；如果模型有多种 cache 类型，就走 page-size unification。对于 attention+GDN，前面的 `_align_hybrid_block_size()` 已经把 GDN/Mamba state padding 到 attention page size，因此 `unify_kv_cache_spec_page_size()` 可以得到统一 page size。之后 `_get_kv_cache_groups_uniform_page_size()` 会按 attention 类型和 page size 把层切成若干 KV cache group。最终 `get_kv_cache_config()` 取所有 group 中最大的 layer 数作为 `group_size`，创建 `group_size` 个物理 `KVCacheTensor`；每个 tensor 被来自不同 group、相同 group 内序号的一组 layer 共享。

运行时并不是每个 layer 单独分配 block，而是每个 KV cache group 维护一份 block 列表。`KVCacheBlocks.blocks[i][j]` 表示第 `i` 个 group 的第 `j` 个逻辑 block。对 full attention group，这些 block 覆盖从 prompt 开始到当前 token 的所有 token；对 sliding window group，过期 block 会被替换成 null block；对 GDN/Mamba group，由于只需要最后 recurrent state，`get_num_skipped_tokens()` 返回 `num_computed_tokens - 1`，所以旧 state block 会尽早释放或置空，只保留当前可继续递推的 state checkpoint。物理 block 的复用由 `BlockPool` 管理，空闲 block 在双向链表里排队，prefix-cached block 即使 ref_cnt 为 0 也可留在 LRU 队列中等待命中或被驱逐。

调度侧只和 `KVCacheManager` 交互，`KVCacheManager` 把实际工作委托给 coordinator。prefix cache 关闭时用 `KVCacheCoordinatorNoPrefixCache`，只做分配释放；只有一个 group 时用 `UnitaryKVCacheCoordinator`；多个 group 且启用 prefix cache 时用 `HybridKVCacheCoordinator`。coordinator 内部为每个 group 创建一个 `SingleTypeKVCacheManager` 子类实例：full attention 使用 `FullAttentionManager`，sliding window 使用 `SlidingWindowManager`，GDN/Mamba 使用 `MambaManager`。所有 manager 共享同一个 `BlockPool`，因此全局显存块数、LRU 驱逐和 ref count 是统一的。

Mamba/GDN manager 的分配策略取决于 `mamba_cache_mode`。`"none"` 只需要当前 running state，加 speculative 时再多留 speculative blocks；它不能提供真正的 Mamba prefix checkpoint。`"all"` 会按 block 保存每个 `i * block_size` 位置的 state，因此一条长 prompt 可能拥有 `ceil(max_model_len / block_size) + speculative_blocks` 个 GDN state blocks，prefix cache 可以命中任意已缓存边界。`"align"` 只保留每个 scheduler step 的最后 token state，内存上只按 `2 + speculative_blocks` 个 page 估算：一个当前 running state、一个上一步 state 用来拷贝/承接，再加 speculative blocks；它的 block table 输入在 attention metadata 阶段会被 `mamba_get_block_table_tensor()` gather 成最后 `1 + speculative_blocks` 个 block。

每次 `allocate_slots()` 大致分三步。第一步，先根据已经 computed 的 token 数调用 `remove_skipped_blocks()`，释放 full attention 以外那些不再参与计算的旧 block；对 GDN/Mamba 来说就是释放不再需要的旧 recurrent state。第二步，如果 prefix cache 或外部 KV connector 命中了一段前缀，`allocate_new_computed_blocks()` 会 touch 命中的 cached blocks，避免它们被 LRU 驱逐，并把它们加入请求的 group block list；被 attention window/GDN state 跳过的前缀位置用 null block 占位。第三步，按本轮要计算的新 token 和 lookahead token 补齐新 block。GDN/Mamba 的 `"align"` 模式不为 lookahead 打破对齐；已有请求通常每步最多再拿一个新 running-state block，并复用前一步 speculative block。

## Prefix Cache 过程

请求进入 scheduler 前会按 `hash_block_size` 计算 `request.block_hashes`。每个 block hash 不是只 hash 当前 block token，而是把 parent block hash、当前 block token ids、以及多模态/tenant salt 等 extra keys 一起 hash，形成链式前缀 hash。真正放进 `BlockPool.cached_block_hash_to_block` 时，vLLM 会再把 group id 追加到 hash 后面形成 `BlockHashWithGroupId`，因此相同 token 前缀在不同 KV cache group 中独立缓存、独立驱逐，不会把 full attention 的 KV block 和 GDN state block 混用。

命中查询从 `KVCacheManager.get_computed_blocks()` 开始。它会把最大命中长度限制为 `request.num_tokens - 1`，因为即使整段 prompt 都命中，也必须重算最后一个 token 来拿 logits。然后 coordinator 调用各 attention group 的 `find_longest_cache_hit()`。full attention 是左到右扫描：从第一个 block hash 开始，只要某个 block miss，后续一定不能作为完整前缀命中，所以立即停止。GDN/Mamba 是右到左扫描：它只需要某个 block 边界上的 recurrent state 作为初始 state，所以从允许的最右侧边界向左找第一个已缓存 state；命中后在返回列表前面补 null block，让 `len(hit_blocks) * block_size` 仍然表示命中的 token 长度。

hybrid prefix cache 的难点是多个 group 必须得到同一个可复用前缀长度。`HybridKVCacheCoordinator` 会先把相同 spec 的 group 合并成 attention groups，并把 full attention group 排在前面。随后执行一个单调递减的 fixed-point 流程：先用当前候选长度查 full attention，再用 full attention 给出的上界查 GDN/Mamba 或其他 efficient attention；如果后者只能命中更短长度，就把候选长度降下来并重新检查，直到所有 group 都接受同一个长度。简单的 “1 个 full attention + 1 个其他类型” hybrid 通常一轮就够。最终返回的是每个 KV cache group 各自的 hit blocks，以及公共 hit token 数。

当 group 的实际 `block_size` 大于 `hash_block_size` 时，查询和缓存都通过 `BlockHashListWithBlockSize` 懒合并 hash。例如 hash 粒度是 16，某个 group block size 是 32，那么 group 第 0 个 block hash 实际由请求的第 0、1 个 16-token hash 拼接/再 hash 语义等价地组成。这样请求可以保留最细公共 hash 粒度，而每个 group 仍按自己的物理 block size 查 cache。

命中后的 block 不会马上重新写入 cache；它们先被 `allocate_new_computed_blocks()` touch 并挂到请求上，增加 ref count，保证本次运行期间不会被驱逐。本轮新计算结束后，`cache_blocks()` 才会把完整 block 提交到 prefix cache。`BlockPool.cache_full_blocks()` 会跳过 null block 和 mask 掉的 block，为每个可缓存 block 写入 `block_hash_with_group_id` 并插入 cached map。Hybrid coordinator 会把可缓存长度先向下对齐到 `scheduler_block_size`，保证后续所有 group 的命中长度都落在公共对齐边界上。

GDN/Mamba prefix cache 在 `"all"` 模式下最直观：每个满 block 边界都可能保存一个可恢复的 recurrent state。下一条请求如果 full attention 前缀也命中，并且 GDN group 在相同或更短的公共边界找到 state，就可以跳过此前 token 的 GDN 递推，从该 state 继续 prefill 或 decode。`MambaManager` 还有一个保护：如果命中的 GDN state block 是同一个 scheduler step 中刚由其他请求产生的，它会让 `get_num_blocks_to_allocate()` 返回超过 GPU block 总数的值，迫使调度器不要在同一步复用它；这样避免读取尚未稳定提交的 recurrent state。

`"align"` 模式更省内存，但 prefix cache 能力更受限。它只围绕 scheduler step 对齐点保存/承接最后 state，block table 会被裁剪成最后 `1 + speculative_blocks` 个 state block；旧 state block 在下一步被释放或置 null。这个模式适合用更少 GDN state 内存换取有限的对齐 checkpoint。`"none"` 则基本没有 GDN prefix state 复用能力，只保留当前请求继续生成所需的运行态 state；hybrid 模型若还启用 full attention prefix cache，也只能在 GDN 侧重新递推或按 coordinator 结果退化。

整体上，hybrid prefix cache 的正确性来自三个约束：所有 group 的 page size 被统一，所有命中 token 长度对齐到 `scheduler_block_size`，所有 cached block key 都带 group id。第一个约束保证同一个 block pool 能安全切给不同类型 layer；第二个约束保证 full attention KV 和 GDN state 表示的是同一个前缀边界；第三个约束保证不同 cache 语义不会发生 hash 冲突式误用。
