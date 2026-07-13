# vLLM Dual Batch Overlap（DBO）代码导读

本文以当前仓库代码为准，说明 vLLM 的 Dual Batch Overlap（DBO）如何工作、
何时生效，以及涉及的主要实现文件。现有的英文设计概览见
[Dual Batch Overlap](dbo.md)；本文着重补充配置校验、DP 协调、CUDA Graph、
MoE 通信时序、限制和测试。

## 1. DBO 解决的问题

在带有 Expert Parallelism（EP）的 MoE 模型中，一层 MoE 的主干可抽象为：

```text
router -> dispatch(all-to-all) -> local experts compute -> combine(all-to-all)
```

`dispatch` 与 `combine` 的稀疏 all-to-all 往往会让计算流等待通信，尤其是在
Data Parallelism（DP）和 EP 共同部署、每个 rank 上的工作量足够大时。DBO 将
一个模型执行 batch 沿 token 维度划分为两个 microbatch（代码中常缩写为
`ubatch`），让两个 CPU 线程以预设的交接点轮流提交 GPU 工作：一个 ubatch 的
通信正在进行时，另一个 ubatch 尽量在计算流上执行 attention、router、专家 MLP
或 shared expert。目标是缩短端到端关键路径，而不是改变模型数值语义。

这里的 “Dual” 是固定的两个微批，不是把两个独立请求并发执行；每个原 batch
仍然只产生一次按原 token 顺序拼接的输出。DBO 与一般化的 `ubatch_size` 基础设施
共用切分与上下文代码，但 `enable_dbo=True` 时微批数被固定为 2。

一个典型的 DeepEP 高吞吐时序可写为（下标为 ubatch 编号）：

```text
计算流:  A0_0 -> A1_0 -> MLP_1 -> shared_1 -> MLP_0 -> shared_0 -> A0_1 -> A1_1
通信流:          dispatch_1 -> dispatch_0 -> combine_1 -> combine_0
```

其中 `A0/A1` 表示 MoE 前后的普通模型计算，`MLP` 是 routed experts，`shared`
是 shared experts。真实顺序由 MoE kernel 插入的 yield/hook 决定，并会随
all-to-all 后端和模型结构变化；上图表示“通信与另一微批计算重叠”的意图，而
不是所有模型的逐指令保证。

## 2. 启用条件、命令行和配置语义

面向用户的开关定义在
[`ParallelConfig`](../../vllm/config/parallel.py) 中，并由
[`EngineArgs`](../../vllm/engine/arg_utils.py) 暴露为命令行参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--enable-dbo` | `False` | 启用 DBO；`num_ubatches` 固定返回 2。 |
| `--dbo-decode-token-threshold` | 32 | 纯 uniform-decode batch 达到此 token 数后才尝试切分。 |
| `--dbo-prefill-token-threshold` | 512 | 只要 batch 含 prefill（或不满足 uniform decode）即使用此阈值。 |
| `--ubatch-size` | 0 | 通用 microbatch 开关；大于 1 也会走 ubatching 基础设施，但不等同于 DBO 的双批策略。 |
| `VLLM_DBO_COMM_SMS` | CUDA 为 20、ROCm 为 64 | DBO 时预留给通信 kernel 的 SM/CU 数；剩余资源给计算。 |

建议的启动形式如下。`--data-parallel-size` 必须大于 1，且应与
`--enable-expert-parallel` 一起使用；否则 DP 协调路径不会生成 ubatch，DBO 不会在
模型 forward 中实际生效。

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --all2all-backend deepep_low_latency
```

当前配置初始化会要求所有 ubatching 使用 `deepep_low_latency` 或
`deepep_high_throughput`；其他 all-to-all backend 会触发断言。通常 decode 为主时
优先从 `deepep_low_latency` 开始，prefill 为主时从 `deepep_high_throughput` 开始，
但最终选择与阈值都应以目标模型、GPU、网络和真实请求分布的 benchmark 为准。DeepEP
后端必须已经安装。`SMControlContextManager` 只会在相应 all-to-all manager 和
DeepGEMM 能力可用时真正设置通信/计算 SM 数，并将通信 SM 数限制到后端允许的最大值。

DBO 会关闭 cascade attention，因为当前 microbatch attention metadata 路径不支持
它。CPU 平台会记录 warning 后将 `enable_dbo` 置为 `False`。DBO 也是 V2 model runner
的未支持特性，因此自动选择时会回退到 V1 model runner；这不是报错，而是为走本实现
所做的兼容性回退。

## 3. 一次 batch 如何决定是否真的使用 DBO

仅传入 `--enable-dbo` 不意味着每一步都切成两份。核心入口为
[`GPUModelRunner._determine_batch_execution_and_padding`](../../vllm/v1/worker/gpu_model_runner.py)：

1. 它通过 `_is_uniform_decode` 判断 batch 是否为每个请求 query 长度相同的 decode
   batch；随后选 decode 或 prefill 阈值。
2. [`check_ubatch_thresholds`](../../vllm/v1/worker/ubatch_utils.py) 先检查
   `ParallelConfig.use_ubatching`，再用未填充 token 数和对应阈值作比较。低于阈值时，
   单 batch 正常执行，避免线程、切分和同步开销反而伤害小 batch 延迟。
3. DP 大小大于 1 时，
   [`coordinate_batch_across_dp`](../../vllm/v1/worker/dp_utils.py) 对各 DP rank 的
   原始 token 数、已有 CUDA-graph/TP 填充后的 token 数、是否愿意 ubatch、当前
   CUDA Graph mode 做一次 all-reduce。
4. 只有**所有** DP rank 都愿意切分时才继续。这一点很关键：MoE all-to-all 的各 rank
   必须使用一致的 microbatch 边界，不能由某个 rank 单独决定。
5. 代码还会检查最小原始 token 数与最大填充 token 数的组合不会让最后一个 ubatch
   为空；一旦任一 rank 会出现空的第二份，所有 rank 都放弃本次 ubatching。生效后，
   DP ranks 会被填充到相同总 token 数。

因此，阈值是“尝试条件”而非硬保证。动态负载不均、DP padding 或某 rank 过小，都可能让
一次理论上达到阈值的请求回退为单 batch。这是正确性所需的集体决策，而非调度失败。

## 4. 切分与 attention metadata

[`maybe_create_ubatch_slices`](../../vllm/v1/worker/ubatch_utils.py) 以累计
`num_scheduled_tokens` 构造 `UBatchSlice`：每个 slice 同时保存 request slice 与 token
slice。默认切点为填充后总 token 数除以微批数；对于 DBO 即一半。函数会同时返回：

- 原始范围，用于描述真实请求及其 token；
- 将最后一份扩展到 DP/CUDA Graph 填充长度的范围，用于固定 shape 的执行。

切点可以落在一个请求的中间，因此不能只按 request 列表二分。
[`split_attn_metadata`](../../vllm/v1/worker/ubatch_utils.py) 和其内部的
`_make_metadata_with_slice` 会为每一份重建 `CommonAttentionMetadata`：重算
`query_start_loc`，裁剪 `seq_lens`、block table 与 slot mapping，并在首/尾请求跨越
切点时校正 query/sequence 长度。这样每个 attention backend 看到的是自洽的局部 batch，
而输出仍可按微批编号拼回原 token 顺序。

## 5. `UBatchWrapper`、线程交接与 CUDA stream

模型初始化时，`GPUModelRunner` 在 `parallel_config.use_ubatching` 为真时用
[`UBatchWrapper`](../../vllm/v1/worker/gpu_ubatch_wrapper.py) 包装模型。wrapper 在有
full CUDA graph 时以 `CUDAGraphMode.FULL` 创建，否则以 `NONE` 创建；它不会改变未
切分 batch 的普通模型调用。

当 `ForwardContext.ubatch_slices` 存在时，wrapper 会：

1. 为两个 slice 各创建一份 `ForwardContext`，并切出 `input_ids`、`positions`、
   `inputs_embeds`、中间张量和 attention metadata；
2. 经 [`make_ubatch_contexts`](../../vllm/v1/worker/ubatching.py) 生成两个
   `UBatchContext`，二者共享一个通信 stream，使用原计算 stream，并各自拥有 CPU 与 GPU
   event；
3. 启动两个 Python 线程。线程进入 context 后先在 barrier 等待；主线程只唤醒第一个
   ubatch，后续交接由 context 完成；
4. 线程完成后按 ubatch id 排序，并用 `torch.cat(..., dim=0)` 合并输出。

`UBatchContext` 的 CPU event 构成一个环：当前线程在 `yield_()` 中 signal 下一个线程，
随后等待自己再次被唤醒。实现特意断言同一时刻只有一个 Python 线程运行，并在恢复后把
线程对应的 `ForwardContext` 与 CUDA current stream 重新设好。GPU 侧的
`yield_and_switch_from_compute_to_comm` 和反向函数会 record/wait 对应 event，确保同一
ubatch 的计算与通信依赖正确，同时允许另一 ubatch 的工作进入互补 stream。

`dbo_enabled()` 的含义值得注意：它检查当前线程是否已注册 `UBatchContext`，不是读取
全局配置。因此同一个进程中，非 DBO forward、CUDA Graph replay 或普通辅助代码调用
`dbo_*` helper 时会自然成为 no-op；只有实际的 ubatch 线程会执行 yield、换 stream 和
hook。

## 6. CUDA Graph 行为

DBO 的微批 CUDA Graph 由 `UBatchWrapper` 独立管理，而不是由普通的
`CUDAGraphWrapper` 管理。首次遇到某个 DBO 总 token shape 且 runtime mode 为 FULL 时，
wrapper 会先启动两个线程、确保两个 stream 的 CUDA context/BLAS handle 初始化完成，再
在计算 stream 中 capture 整个双微批执行。graph 以总 token 数为 key 缓存；之后同 shape
直接 `replay()`，不再需要 CPU 线程或 CPU event 同步。

微批执行不使用 piecewise CUDA Graph；在 eager/无 full graph 的情况下仍会通过两个线程
运行。若某个 shape 已缓存 DBO full graph，但某一步因 DP 协调回退为非 ubatch，代码会让
这一普通调用以 `CUDAGraphMode.NONE` 运行，避免把微批 graph 错当成普通 batch graph。
这也是使用 DBO 时观察到 CUDA Graph dispatch 与常规路径不同的原因。

## 7. MoE kernel 内如何形成重叠

[`FusedMoEModularKernel`](../../vllm/model_executor/layers/fused_moe/modular_kernel.py)
把 MoE 组织为 router、prepare/dispatch、fused experts、finalize/combine 四段。只有支持
异步 prepare/finalize 的 all-to-all 后端可以用于 DBO；不支持异步的实现会在 DBO 运行时
断言失败。

在 `_prepare` 与 `_finalize` 中，modular kernel 的共用协议是：先尝试执行已经登记的
receive hook；发起异步操作后，若后端返回 `(hook, receiver)`，DBO 会把 hook 登记到
**另一个** ubatch context，然后 `dbo_yield()`。被唤醒的下一个 ubatch 在合适位置运行该
hook；原 ubatch 重新取得执行权后由 `receiver()` 等待/取回通信结果。这让 CPU 不必在
同一 microbatch 的通信点空等，也把等待放到可被另一份计算覆盖的时间段。finalize 阶段还
会在 combine 返回前调用 shared experts，从而与 combine 通信重叠。

DeepEP 高吞吐实现
[`DeepEPHTPrepareAndFinalize`](../../vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py)
展示了更具体的 stream 编排：

1. dispatch 前在本 ubatch 的计算流 capture event；
2. `dbo_yield_and_switch_from_compute_to_comm()` 先让另一 ubatch 尽量提交计算，再在
   通信流启动 dispatch；
3. dispatch handle 按 `dbo_current_ubatch_id()` 保存到两个槽位，防止两线程相互覆盖；
4. 通过 `dbo_switch_to_compute_sync()` 回到计算流，并在通信结果依赖满足后执行本地
   experts；
5. finalize 的 combine 重复相同模式。它的异步 receiver 在 event 完成后复制结果，并再
   交接回计算流。

DeepEP 低延迟实现同样接入 `dbo_current_ubatch_id`、receive hook 与异步通信协议，但其
激活格式和专家 kernel 组合不同。支持的后端及其量化/激活格式组合应以
[MoE kernel features](moe_kernel_features.md) 的表格为准；“支持 DeepEP”不表示任意
模型量化和任意专家 kernel 都可组合。

为了防止 DBO 并发复用临时张量，多个组件都实现了双槽状态：DeepEP HT 的 dispatch
handle、`SharedExperts` 的输出缓存，以及 `WorkspaceManager` 的 workspace 都按当前
ubatch id 索引。workspace 扩容只替换请求该槽位的张量，避免另一线程仍持有旧 tensor view
时发生泄漏或悬挂引用。

## 8. 当前限制与兼容性边界

DBO 当前的边界应按下面的代码事实理解：

| 场景 | 当前行为 |
| --- | --- |
| CPU | 平台初始化会禁用 DBO。 |
| DP size = 1 | `coordinate_batch_across_dp` 直接返回不切分；实际 forward 不会进入 DBO。 |
| 非 DeepEP all-to-all | ubatching 配置校验拒绝，当前 DBO 不支持。 |
| Cascade attention | DBO/ubatching 配置时自动关闭。 |
| V2 model runner | 标记为不支持并回退到 V1；显式要求 V2 时会报配置错误。 |
| EAGLE speculative proposer / hidden-state extraction | 对 `should_ubatch` 有显式断言，报“not implemented”。 |
| Elastic EP | 执行路径明确报出 DBO 尚未支持。 |
| 小 batch 或 DP rank 负载不均 | 不一定触发；会安全地走单 batch。 |

因此不要把 DBO 视为所有 MoE 配置的通用加速选项。它优先针对 DP+EP、DeepEP、足够大且
跨 rank 较均衡的 batch。对低并发、短 decode、小 prefill 或通信本来不在关键路径上的工作
负载，切分、填充和线程调度的成本可能抵消收益。

## 9. 调优与排障建议

先保持默认阈值测量，再分别改变 decode 与 prefill 阈值；不要只看单请求延迟，也应看
TTFT、ITL、总体吞吐和 DP rank 的 token 分布。降低阈值会提高 DBO 覆盖率，但也更常支付
切分、双线程、metadata 重建和 padding 成本；提高阈值则只在更大的通信受限 batch 启用。
`VLLM_DBO_COMM_SMS` 也应在目标硬件上搜索：通信 SM 太少会拉长 all-to-all，太多会挤占
expert compute。

若预期 DBO 却没有性能变化，按以下顺序检查：确认日志未显示 CPU 禁用或 V2 回退；确认
`DP > 1`、EP、DeepEP backend 和 DeepEP 安装；确认实际 batch token 数跨过对应阈值；最后
检查所有 DP rank 是否都能切分且第二个 ubatch 非空。要分析时序，可使用 profiler 观察
compute/comm streams、DeepEP dispatch/combine 和两个 microbatch 的交错，而不能仅凭
`--enable-dbo` 是否出现在启动参数中判断。

## 10. 代码地图与测试

| 位置 | 责任 |
| --- | --- |
| [`vllm/config/parallel.py`](../../vllm/config/parallel.py) | DBO/ubatch 配置、阈值、固定双微批语义。 |
| [`vllm/config/vllm.py`](../../vllm/config/vllm.py) | DeepEP backend 与 cascade-attention 校验；V2 runner 不支持项。 |
| [`vllm/v1/worker/dp_utils.py`](../../vllm/v1/worker/dp_utils.py) | DP all-reduce、一致切分决策、统一填充。 |
| [`vllm/v1/worker/ubatch_utils.py`](../../vllm/v1/worker/ubatch_utils.py) | 阈值、token/request slice、attention metadata 切分。 |
| [`vllm/v1/worker/gpu_model_runner.py`](../../vllm/v1/worker/gpu_model_runner.py) | 选择执行模式、创建 slice、安装 wrapper。 |
| [`vllm/v1/worker/gpu_ubatch_wrapper.py`](../../vllm/v1/worker/gpu_ubatch_wrapper.py) | 两线程执行、full CUDA Graph capture/replay、SM 切分。 |
| [`vllm/v1/worker/ubatching.py`](../../vllm/v1/worker/ubatching.py) | `UBatchContext`、CPU/GPU event、stream 切换和 DBO helper。 |
| [`vllm/model_executor/layers/fused_moe/modular_kernel.py`](../../vllm/model_executor/layers/fused_moe/modular_kernel.py) | prepare/finalize 的 hook、yield 与 shared-expert 重叠协议。 |
| [`vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`](../../vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py) | DeepEP HT 下实际 dispatch/combine 的 DBO 编排。 |
| [`vllm/v1/worker/workspace.py`](../../vllm/v1/worker/workspace.py) | per-ubatch workspace，避免并发缓冲区复用。 |
| [`tests/v1/distributed/test_dbo.py`](../../tests/v1/distributed/test_dbo.py) | 两种 DeepEP backend、DP=2、EP 与 DBO 的端到端 GSM8K 正确性回归。 |

现有端到端测试使用 DeepSeek-V2-Lite、两个 DP rank，并把 decode/prefill 阈值分别固定为
16/256，以提高覆盖 DBO 的概率；它要求 DeepEP 和至少两张 GPU。该测试在 Blackwell 上
目前标记为 `xfail`，原因是已知的准确率不稳定。因此它验证的是受支持配置上的功能回归，
并不替代在具体硬件和负载上的性能 benchmark。
