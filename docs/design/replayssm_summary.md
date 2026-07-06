# ReplaySSM：缓存 SSM 输入而不是状态

本文整理自 Tri Dao 的博客《ReplaySSM: Cache SSM Inputs, Not State》（2026），并按 vLLM hybrid attention/cache 语境总结。ReplaySSM 的核心判断是：当前 SSM 解码每步都把 recurrent state 写回 HBM，但这个 state 的唯一消费者通常只是下一步 recurrent update；如果改为缓存最近的 SSM 输入，在需要时 replay 出 state，绝大多数 decode step 就不必写回完整 state。该方法不改变模型输出，只改变 state 的计算和缓存方式。

## 背景与问题

SSM、GDN、Mamba-2、Kimi Delta Attention 这类结构把历史 token 压缩进固定大小的 recurrent state，避免 Transformer KV cache 随上下文长度线性增长。hybrid 模型通常把多数 SSM 层和少数 attention 层交错起来，用 SSM 降低长上下文 memory traffic，用 attention 保留更好的精确召回能力。问题在于，SSM 的常数内存不是免费的：每个 decode step 都要从 HBM 读取 state、用当前 token 的输入更新 state、读出输出、再把更新后的 state 写回 HBM。state update 的算术强度很低，主要是向量外积、矩阵向量读出和 state load/store，不像 GEMM 那样能充分利用 Tensor Core，因此在服务 batch size 下容易成为 memory-bound bottleneck。

标准 SSM 解码还有两个实际问题。第一，state 是历史的摘要，更新后无法像 attention KV cache 那样通过移动指针撤销 token；speculative decoding 一旦 draft token 被拒绝，必须恢复到最后接受 token 的 state。vLLM 这类实现通常为每个 speculative 位置保存一份完整 state snapshot，窗口大小为 `T` 时就引入接近 `T` 倍的 state traffic 和容量开销。第二，SSM state 递推有顺序依赖，验证多个 draft token 时，baseline 需要按 draft 顺序逐个更新 state 并产生中间输出，不能像 Transformer attention 那样自然把 verification 变成更大的 batched computation。

ReplaySSM 的思路是把 “急切总结历史” 改成 “缓存最近输入，必要时再总结”。设 checkpoint state 为 `S0`，buffer 保存 checkpoint 之后的最近 SSM 输入。普通 decode step 不再把每步的新 state 写回 HBM，而是把当前 token 对 state update 所需的小向量追加到 buffer。只有 buffer 满了，才把 buffer 中的输入汇总进 `S0`，清空 buffer，开始下一轮缓存。这样，state write-back 从每步一次变成每 `L` 步一次，`L` 是 buffer capacity；rollback 也从恢复完整 state 变成移动 buffer 指针。

## Mamba-2 中的计算方式

以 Mamba-2 为例，baseline recurrence 可以写成：

```text
S_t = a_t * S_{t-1} + Delta_t * (v_t k_t^T)
y_t = S_t q_t
```

这个 recurrence 有两种等价路线。baseline 走 summary route：加载 `S_{t-1}`，构造新 state `S_t`，再用 `q_t` 读输出，并写回 `S_t`。ReplaySSM 利用 history route：把 `S_t` 展开成 checkpoint state 加上 buffer 内所有历史输入的加权外积。对于多数 decode step，实际只需要输出 `y_t`，不需要物化完整 `S_t`，因此可以把 `(v k^T) q` 改写成 `v (k^T q)`。也就是说，先算 key 与 query 的内积，再用这个标量加权 value，直接得到输出贡献，而不构造 `d x n` 的 state 矩阵。

带 checkpoint 和 buffer 时，输出可以理解为两部分相加：一部分是 checkpoint state 对当前 query 的读出，另一部分是 buffer 中每个 cached input 对当前 query 的贡献。ReplaySSM 仍然读取 checkpoint state，但不写回完整 state；它读取 buffer 里的 `v/k/Delta` 和衰减因子，计算加权和并输出。只有 flush step 需要走 state-and-output route，把 buffer 的外积贡献物化进 `S0` 并写回一次完整 state。

这种改写减少了主要 memory traffic。baseline 每步既 load state 又 store state，主项是完整 state 的读写；ReplaySSM 多数步骤只 load checkpoint state 和小 buffer，store 当前输入向量。flush step 更贵，但成本被 buffer 长度摊薄。buffer 太小会频繁 flush，buffer 太大又会让每步读取和计算更多 cached input，因此存在中等容量的最佳点；博客中的评测里 Nemotron-3 使用 8、Qwen3.5 使用 16 是较好的设置。

## GDN 中的计算方式

GDN 属于 delta-rule family，更新式比 Mamba-2 多了擦除旧内容的修正项。baseline 可理解为：

```text
alpha = exp(g)
S = alpha * S
u = beta * (v - S k)
S = S + u k^T
y = S q
```

这里 `u = beta * (v - S k)` 是关键。它先用当前 state 在 key `k` 上读出已有内容，再从 value `v` 中扣掉这部分，形成 correction，再写回 `u k^T`。因此 GDN 不能像 Mamba-2 那样简单缓存原始 `v`。如果只缓存 `v`，replay 时仍然需要每一步的中间 state 才能重新算出每个 `u`，顺序依赖没有消失。

ReplaySSM 在 GDN 中缓存的是 `(u, k, g)`，不是 `(v, k)`。一旦当前步的 `u` 被算出，GDN 的后续 state 递推就可以写成 `S_t = alpha_t * S_{t-1} + u_t k_t^T`，和 Mamba-2 一样可以展开成 decayed checkpoint 加 buffer 中若干外积的加权和。标准 decode 时，ReplaySSM 先从 checkpoint 和 buffer 重建临时 state `S_h`，再算当前步的 `alpha`、`u` 和输出，最后把 `(u, k, g)` 追加到 buffer。buffer 满时才把 `alpha * S_h + u k^T` 写回 checkpoint state。

GDN 和 Mamba-2 的另一个差别是输出路径。Mamba-2 可以用 output-only route 避免物化 state，因为它主要需要 `q` 方向上的读出；GDN 当前步还需要 `S_h k` 来计算 correction，又需要 `S_h q` 来输出，因此需要在标准 decode 中走 state-and-output route，先根据 checkpoint 和 buffer 重建 state，再读 `k` 和 `q` 两个方向。博客指出 GDN 不使用 Mamba-2 那种共享 `k^T q` 的预计算 kernel；GDN 的重点是把缓存对象从 raw value 换成 correction `u`，并在 speculative verification 中使用 chunk-wise delta-rule parallelism。

## Speculative Decoding

baseline SSM speculative decoding 的代价来自两个方面：每个 draft token 要保存一份 full state snapshot，便于拒绝时恢复；verification 本身还要按 draft 顺序执行长度为 `T` 的 recurrent scan。ReplaySSM 让 draft token 的 SSM 输入进入同一个 ring buffer。rollback 时不恢复 full state，只移动 buffer pointer，保留 accepted draft 对应的 entries，丢弃 rejected entries。这样，恢复成本从 state copy/restore 变成 buffer 元数据更新。

对 Mamba-2，ReplaySSM 的 output-only form 让 verification 不必逐个物化中间 state。每个 draft query 都从同一个 checkpoint state 和同一个 buffer window 读取，只是 causal mask 不同：第 `s` 个 draft 只能看到自己之前的 cached entries。key-query 部分变成 cached keys 与 draft queries 的矩阵乘法，value 汇总也变成带 causal mask 的矩阵计算。也就是说，ReplaySSM 把原先长度 `T` 的 serial state update 改成更适合 GPU 的 batched inner-product computation。

对 GDN，draft correction `u_s` 仍然依赖之前 draft 的效果，但这个依赖可以用训练中 chunked delta-rule 的形式并行化。ReplaySSM 先从 checkpoint 和 buffer 重建 `S_h`，计算每个 draft 的 `S_h q_s` 和 `S_h k_s`，再构造严格下三角矩阵，表达 draft 之间的 correction 依赖。随后通过一个 `T x T` 的 triangular solve 一次性求出所有 correction `U_s`，再计算所有 draft 输出并把 `(U_s, k_s, g_s)` 追加到 buffer。除这个小规模三角求解外，主要操作可以组织成 GEMM，从而消除 per-draft state snapshot 和 serial delta-rule scan。

flush 策略在 speculative decoding 里要更保守。若 buffer 当前已有 `h` 个条目，spec window 为 `T`，capacity 为 `L`，自然条件是 `h + T > L` 时 flush；但这会导致某一步接受较多 draft 后，下一步可用 buffer slot 不足，实际 spec window 被截短。ReplaySSM 选择提前一个窗口 flush，即 `h + 2T > L` 时就汇总 checkpoint，保证每一步开始时至少有 `T` 个可写 slot。

## Kernel 与 CUDA Graph 设计

ReplaySSM 的 kernel 设计围绕两个目标：减少重复计算和避免数据相关拷贝。Mamba-2 output-only decode 里，不同 value heads 共享同一组 `k^T q` 内积；如果把这些内积放在主 SSM kernel 内部，会跨 head 重复计算并增加寄存器压力。因此 ReplaySSM 用一个小 precompute kernel 按 group 计算共享内积，再让主 kernel 读取 scratch buffer。GDN 不采用这个预计算，因为它需要 state 在 `k` 与 `q` 两个方向上的读出，并采用 state-and-output route。

speculative decoding 下，不同序列每步 accepted draft 数不同，buffer commit/rollback 也不同。如果把 accepted tokens 搬回 buffer 头部，会在连续 batching 下引入大量数据相关 copy；ReplaySSM 改用 ring buffer，通过 kernel 内索引保证逻辑顺序正确，rollback 和 commit 都成为 pointer 操作。tree-based speculative decoding 中 accepted tokens 甚至可能不连续，ring buffer 的索引方式也更自然。

CUDA Graph 支持的难点是 batch divergence 和 host-device sync。连续 batching 下，同一个 batch 里有的序列需要 flush，有的不需要；spec decode 里每条序列 accepted token 数也不同。ReplaySSM 把 flush decision 作为 per-sequence runtime data，由 kernel 内部分支处理，而不是为 flush/non-flush 捕获不同图。commit/rollback 也用小 kernel 在 device 侧更新 buffer pointer，避免把 accepted counts 拉回 host 造成同步。

## 评测结论

博客在 vLLM 上实现 ReplaySSM，并在 CUDA Graph 开启的条件下评测 Nemotron-3（Mamba-2）和 Qwen3.5（GDN）两类 hybrid 模型。模型覆盖 4B dense 到 550B MoE，SSM state 使用 FP32，buffer 中缓存的向量使用 BF16；speculative decoding 使用各模型的 MTP heads 作为 drafter。

标准 decoding 中，ReplaySSM 主要加速 SSM kernel，因此端到端收益小于 kernel 层收益。博客给出的结果是：Nemotron-3 上 SSM kernel 约 1.43x 到 1.84x，端到端约 1.20x 到 1.48x；Qwen3.5 上 SSM kernel 约 1.43x 到 1.64x，端到端约 1.20x 到 1.27x。端到端收益被 attention、GEMM、采样和其他框架开销稀释，但在 hybrid 模型里仍然可观，因为 SSM 层数量通常多于 attention 层。

speculative decoding 是 ReplaySSM 的更大收益点。博客在 GSM8K prompt、spec window 为 4、temperature 为 0 的设置下扫 batch size。ReplaySSM 保持和 baseline 相同的 draft acceptance 行为，但在大 batch 下达到相对 vLLM 标准 decoding 约 1.87x 到 1.96x 的端到端吞吐；相对 vLLM baseline speculative path 最高约 2.14x。原因一是 verification step 更便宜，baseline 的 GDN/Qwen3.5 speculative kernel 随窗口增大接近线性变贵，而 ReplaySSM 的 state traffic 主要是 checkpoint load 加偶发 flush；原因二是容量占用更低，baseline 为每个 draft 预留 full state snapshot 会把最大 decode batch 大约压到标准 serving 的四分之一，而 ReplaySSM 通过缓存小输入向量恢复约 3.0x 到 3.3x 的并发能力。

## 对 vLLM Hybrid Cache 的启示

当前 vLLM 的 hybrid cache 以 “state block” 为核心：GDN/Mamba layer 的 `MambaSpec` 描述 conv/recurrent state 的 page size，`MambaManager` 管理 state blocks，并通过 `mamba_cache_mode` 决定是否按 block 保存 checkpoint。ReplaySSM 提出的方向是把一部分 state block 写回压力转移为 input buffer 管理：普通 step 缓存 `(v,k,Delta)` 或 GDN 的 `(u,k,g)`，flush step 才写回 full recurrent state。它不会替代 attention KV cache，也不会改变 full attention group 的 prefix cache 语义；它主要改变 SSM group 内部的运行态缓存对象和 speculative rollback 方式。

如果把 ReplaySSM 接入类似 vLLM 的 hybrid cache，需要在 cache 设计上显式区分三类数据：长期 checkpoint state、短期 replay input buffer、以及 attention KV blocks。prefix cache 也要明确命中边界：full attention 仍按 token block hash 查 KV；SSM/GDN 则可能命中某个 checkpoint state 加一段 replay buffer，或者在 flush 边界上形成可复用 state。speculative decoding 还需要把 buffer pointer、accepted count、flush flag 做成 device-side metadata，避免破坏 CUDA Graph。

ReplaySSM 的本质不是 “SSM 不需要 state”，而是 “不必每个 token 都把 state 物化并写回 HBM”。它把 state 从每步提交的主数据结构，降级为周期性 checkpoint；把最近历史从不可逆摘要，改成可回放的输入 buffer。对 GDN 来说，关键细节是缓存 correction `u` 而不是 raw value `v`，否则无法摆脱 correction 项带来的 serial dependency。

来源：https://tridao.me/blog/2026/replayssm/
