# vLLM EPLB 代码与流程说明

EPLB 是 Expert Parallel Load Balancing 的缩写。它解决的问题很直接：MoE router 选出来的是逻辑 expert，但实际执行时每个 EP rank 只持有一部分物理 expert；当线上请求让某些 expert 长期过热时，EPLB 会统计负载，把热门逻辑 expert 复制到冗余物理槽位里，再把物理 expert 分布重新排到不同 rank 上，让 token 更均匀地落到各个 GPU。

这份文档按代码路径说明当前仓库里的 EPLB。核心路径是 `vllm/distributed/eplb/*`、`vllm/model_executor/layers/fused_moe/*` 和 `vllm/v1/worker/gpu_model_runner.py`。本分支还有 `VLLM_SC_EPLB` / `next_gate_lora` 相关实验路径，它不参与标准 EPLB 的重排闭环，只作为旁路预测/trace 逻辑单独说明。

## 配置入口

用户侧入口在 `vllm/config/parallel.py`。`EPLBConfig` 定义了 `window_size`、`step_interval`、`num_redundant_experts`、`log_balancedness`、`log_balancedness_interval`、`use_async`、`policy`、`communicator`。`ParallelConfig.enable_eplb` 是总开关。校验逻辑要求 EPLB 只能跑在 CUDA/ROCm 类设备上，必须启用 `enable_expert_parallel`，并且 TP 或 DP 至少有一个大于 1；如果没开 EPLB 却设置了 `num_redundant_experts`，会直接报错。

CLI 参数接在 `vllm/engine/arg_utils.py`，暴露 `--enable-eplb` 和 `--eplb-config`。用户文档已有一段在 `docs/serving/expert_parallel_deployment.md`，主要说明参数和部署示例。

`communicator=None` 时会自动选择：Elastic EP 强制 `pynccl`；普通 EPLB 优先 `nixl`，没有 NIXL 时退到 `torch_gloo`。这段选择在 `ParallelConfig` 初始化后期完成，原因是 async EPLB 下 `torch_nccl` 容易和主线程多 stream/高负载通信互相卡住。

`vllm/distributed/eplb/eplb_utils.py` 还有一个启动前环境修正：`override_envs_for_eplb()` 在 DP + EPLB + NCCL 类 communicator + DeepEP low latency 或 deep_gemm_mega_moe 时设置 `NCCL_MAX_CTAS=8`，避免 EPLB 权重交换和 MoE cooperative kernel 抢占 SM 导致死锁。调用点在 `vllm/v1/worker/gpu_worker.py` 的分布式初始化阶段。

## 模型和 MoE 层接入

EPLB 只接入实现了 `MixtureOfExperts` 接口的模型，接口在 `vllm/model_executor/models/interfaces.py`。模型需要暴露 MoE 层数、逻辑 expert 数、物理 expert 数、冗余 expert 数、`moe_layers` 和 `set_eplb_state()`。默认实现会遍历每个 MoE layer，收集 `layer.get_expert_weights()`，并把每层对应的负载视图和映射视图注册给 layer。

具体 MoE 层主要是 `vllm/model_executor/layers/fused_moe/layer.py` 里的 `FusedMoE`。构造时传入 `enable_eplb` 和 `num_redundant_experts` 后：

- `global_num_experts = num_experts + num_redundant_experts`，`logical_num_experts = num_experts`。
- 如果启用 EPLB，会创建一个空的 `EplbLayerState`，并要求物理 expert 能被 EP size 均分。
- `ExpertMapManager` 按包含冗余 expert 的全局 expert 数创建本 rank 的 expert map/routing table。
- router 创建时会拿到同一个 `EplbLayerState`，后续 forward 就能读映射、写负载。
- quant method 必须声明 `supports_eplb=True`。基类默认不支持；未量化路径支持；modular method 继承旧 quant method 的支持情况。

`FusedMoE.get_expert_weights()` 返回每层可搬运的 expert 权重视图，排除 shared experts、gate、routed transform、全局 activation scale 等非 expert 权重。EPLB 重排时搬的就是这些张量的 expert 维度切片。

## 初始化运行时状态

V1 GPU runner 里有两套写法：旧路径直接在 `vllm/v1/worker/gpu_model_runner.py` 持有 `self.eplb_state`，新一点的封装在 `vllm/v1/worker/gpu/eplb_utils.py` 的 `EPLBController`。两者做的事情一致：加载真实模型后，如果模型是 MoE 且 `enable_eplb=True`，创建 `EplbState` 并调用 `add_model()`。draft/speculator 模型如果也是 MoE，也会注册到同一个 `EplbState`，但 Elastic EP 不支持 draft MoE。

`EplbState.add_model()` 是初始化核心，在 `vllm/distributed/eplb/eplb_state.py`：

1. 校验多个已注册 MoE 模型的 EP 相关结构一致。
2. 建初始 `physical_to_logical_map`：前 `num_routed_experts` 个物理槽位一一映射到逻辑 expert，冗余槽位按 `i % num_routed_experts` 复制逻辑 expert。
3. 反推出 `logical_to_physical_map` 和 `logical_replica_count`。前者是稀疏表，`-1` 表示没有对应物理副本。
4. 为所有 MoE layer 复制同一套初始映射，并创建 `expert_load_pass` 和滑动窗口 `expert_load_window`。
5. 把这些 tensor 视图注册到模型和每个 layer。注意这些都是视图，后续 EPLB 改 map 后 layer 自动看到新映射。
6. 创建 `expert_buffer`，也就是单层权重搬运时的临时接收缓冲。
7. 创建 `EplbCommunicator`，保存为 `EplbModelState.communicator`。

`EplbModelState` 保存一个模型的 EPLB 运行态：映射表、负载统计、模型名、模型对象、所有 expert weights、临时 buffer、async 标志、communicator、async worker 和主线程之间的 `pending_result` 等。`EplbLayerState` 是每个 MoE layer 持有的轻量视图：当前层的 `expert_load_view`、`logical_to_physical_map`、`logical_replica_count` 和全局共享的 `should_record_tensor`。

`should_record_tensor` 是一个标量 bool tensor，所有 layer 共用。EPLB 不是每步都需要把负载写入滑动窗口：当 `step_interval > window_size` 时，前面那些一定会被覆盖的步可以不记录，从而少做一次 atomic 统计。

## forward 内的路由和负载统计

标准 forward 路径在 `vllm/model_executor/layers/fused_moe/router/base_router.py`。每个 router 子类先正常算出 `topk_weights, topk_ids`，此时 `topk_ids` 是逻辑 expert id。`BaseRouter.select_experts()` 在 capture/trace 之后调用 `_apply_eplb_mapping()`，如果 `eplb_state` 存在，就进入 `eplb_map_to_physical_and_record()`。

CUDA/ROCm 上这个函数是一个 Triton kernel，做两件事：

首先，用 `logical_replica_count` 看当前逻辑 expert 有几个物理副本。它用 token index 乘 Knuth hash multiplier 后对副本数取模，选择一个副本槽位，再通过 `logical_to_physical_map` 得到物理 expert id。这样热门逻辑 expert 的 token 会被分散到多个物理副本。

然后，如果 `should_record_tensor=True`，kernel 对 `expert_load_view[physical_id]` 做 `atomic_add(1)`。所以 EPLB 的负载指标本质是“本 forward pass 中每个物理 expert 处理的 token 计数”。非 CUDA 类平台这个函数直接返回原 `topk_ids`，实际 EPLB 配置也会在 `ParallelConfig` 里被限制住。

经过这一步，后面的 MoE kernel 看到的已经是物理 expert id。逻辑 id 只用于 router 输出、trace/capture 以及 EPLB 映射前的语义。

## 每步状态推进

runner 在每个真实 step 结束后调用 `eplb_step()`；dummy/profile 路径也会在必要时调用，避免 DP/EP rank 的 collective 不同步。主入口在 `vllm/v1/worker/gpu_model_runner.py`，封装版入口是 `EPLBController.step()`。

`EplbState.step()` 的流程是：

先处理 profile：`is_profile=True` 时直接调用 `rearrange(is_profile=True)`，只做通信 buffer 预留，不真的搬权重或提交映射。

如果是 dummy step，会清零 `expert_load_pass`，但仍推进 `expert_rearrangement_step`，这是为了让所有 EP rank 在同一时刻进入重排 collective。

如果打开 `log_balancedness`，会把每个模型的 `expert_load_pass` all-reduce 到 EP 组，按 rank 汇总 token 数，输出 `avg_tokens / max_tokens` 的 balancedness。这个日志本身有通信开销，所以默认关。

真实 step 且 `should_record_current_step=True` 时，把本 pass 的 `expert_load_pass` 拷到 `expert_load_window[expert_load_window_step]`，再清零 pass 计数，并推进滑动窗口指针。

如果 async 模式开启，主线程每步还会检查 async worker 是否已经把某一层的新权重搬到了 `expert_buffer`。所有 rank 都有 `pending_result` 后，主线程调用 `_move_to_workspace()`：把 buffer 写回真实 expert weight，提交这一层的新 map，然后记录 consumed event 让 async worker 继续下一层。

最后 `expert_rearrangement_step += 1`。达到 `step_interval` 后，如果 async worker 还没消费完上一轮重排，主线程只更新 `should_record_tensor` 并返回；否则重置计数并调用 `rearrange()`。

## 重排算法

`EplbState.rearrange()` 先把滑动窗口里的物理 expert 负载映射回逻辑 expert 负载。因为一个逻辑 expert 可能有多个物理副本，所以这里用当前 `physical_to_logical_map` 做 `scatter_add`，再对 window 维度求和，得到形状为 `[num_moe_layers, num_logical_experts]` 的全局逻辑负载。随后对 EP group 做 all-reduce，让每个 rank 都拿到同一份全局负载。

策略入口是 `vllm/distributed/eplb/policy/abstract.py` 的 `AbstractEplbPolicy.rebalance_experts()`。当前只有 `default`，实现在 `vllm/distributed/eplb/policy/default.py`，算法来自 DeepSeek EPLB：

`replicate_experts()` 根据逻辑 expert 负载决定冗余副本分配。它每次选择当前 `weight / replica_count` 最大的逻辑 expert 增加一个物理副本，目标是降低最热副本的负载。

`balanced_packing()` 把带权对象均匀装进若干 pack，每个 pack 的对象数相同，并尽量让 pack 总权重接近。

`rebalance_experts_hierarchical()` 先把 expert groups 均衡分到 node，再在 node 内复制 expert，最后把物理 expert 均衡分到 GPU。这样比纯全局均衡更尊重节点内/节点间网络层次。如果 group/node 条件不满足，会退化成 `num_groups=1,num_nodes=1` 的全局策略。

`preserve_intragpu_slots()` 是后处理：当 GPU 数和每 GPU slot 数没变时，同一个 GPU 内仍然存在的 expert 尽量留在原 slot，减少不必要的本地权重拷贝。

策略返回新的 `physical_to_logical_map`，形状是 `[num_moe_layers, num_physical_experts]`。

## 权重搬运和映射提交

同步模式下，`rearrange()` 拿到新 map 后直接调用 `vllm/distributed/eplb/rebalance_execute.py` 的 `rearrange_expert_weights_inplace()`。这个函数逐层处理，每层调用 `move_to_buffer()` 和 `move_from_buffer()`。

`move_to_buffer()` 对比 old/new map，分出几类情况：不变的 slot 不动；同 rank 内能找到的 expert 本地 copy 到 buffer；需要跨 rank 的 expert 通过 communicator 发/收。它会建立 send/recv 列表，调用 `communicator.execute()`，返回 `TransferMetadata`。

`move_from_buffer()` 把 buffer 中已收到的 expert 写回真实权重。如果同一个远端 expert 在本 rank 有多个目标副本，只接收一次，然后在本地复制到其他副本 slot。

搬完后 `_commit_eplb_maps()` / `_commit_eplb_maps_for_layer()` 更新 `physical_to_logical_map`，并用 `compute_logical_maps()` 重建 `logical_to_physical_map` 和 `logical_replica_count`。由于 layer 持有的是这些 tensor 的视图，提交后下一个 forward 立即按新映射路由。

communicator 抽象在 `vllm/distributed/eplb/eplb_communicator.py`。`torch_nccl` 使用 `torch.distributed.batch_isend_irecv`；`torch_gloo` 先 GPU/CPU staging 再走 gloo；`pynccl` 直接用 PyNCCL send/recv；`nixl` 是 receiver-initiated READ，会预注册所有 expert weights 和 buffer，通过远端地址元数据做 zero-copy RDMA read。

## 异步 EPLB

默认 `use_async=True`。异步路径把“算新 map + 跨 rank 搬权重”放到后台线程，主线程只在每步开头/结尾检查是否有一层已经准备好，然后短时间写回真实权重。

线程入口是 `vllm/distributed/eplb/async_worker.py:start_async_worker()`。它创建 CUDA stream，然后循环等待 `EplbState.rearrange_event`。`CpuGpuEvent` 同时包含 CPU `threading.Event` 和 CUDA event，用来避免 async 线程 wait 一个尚未 record 的 CUDA event 时直接穿透。

主线程 `rearrange()` 在 async 模式下不直接搬权重，而是给每个 `EplbModelState` 写入 `EplbStats` 快照并设置 `rebalanced=True`，然后 `rearrange_event.record()` 唤醒后台线程。

后台线程醒来后：

1. snapshot 当前 `physical_to_logical_map` 到 CPU。
2. 用 policy 计算新 map。
3. 每次只 transfer 一层，把结果写入共享 `expert_buffer`。
4. CUDA stream synchronize 后设置 `model_state.pending_result = AsyncEplbLayerResult(...)`。
5. 等待主线程消费 `expert_buffer` 并 record `consumed_event`，再处理下一层。

主线程在 `EplbState.step()` 里用 `_all_ranks_result_ready()` 确认所有 EP rank 都有 pending result，才调用 `_move_to_workspace()`。这个 all-reduce 是必要的：某个 rank 单独进入 `_move_to_workspace()` 会导致 collective 顺序错位。

## Elastic EP 旁路

Elastic EP 在 `vllm/distributed/elastic_ep/*` 里复用 EPLB 的 map 和权重搬运能力。扩缩容期间会临时 suppress EPLB，避免普通 step 的重排和扩缩容重排交错。新 worker 可以通过 `setup_eplb_from_mapping()` / `EplbState.from_mapping()` 从已有 expanded physical-to-logical map 初始化 EPLB state。

扩容/缩容时，`rank_mapping` 会传给 `rearrange()` 和 `rebalance_execute`，用于把旧 rank 的 expert slot 映射到新 EP group。scale down 时会先在仍存活的旧 group 上把 expert reshuffle 到保留 rank；scale up 时新 worker 先接收 expert mapping/weights，再切换 group 并做 EPLB reshuffle。

## 本分支的 SC-EPLB / next-gate LoRA

`vllm/envs.py` 定义了 `VLLM_SC_EPLB`。`vllm/v1/worker/gpu_worker.py` 在模型加载后调用 `maybe_attach_sc_eplb_next_gate_lora()`，源码在 `vllm/model_executor/layers/fused_moe/next_gate_lora.py`。

这条路径会扫描 `static_forward_context` 里的 `FusedMoE` 层，为相邻 MoE 层构建“当前层 hidden states -> 下一层 gate -> 可选 LoRA delta -> 下一层 router._compute_routing()` 的旁路 predictor，并挂到 `MoERunner` 上。`MoERunner._maybe_predict_next_gate()` 会在正常 MoE 执行前异步计算并缓存预测结果。

这不是标准 EPLB 的负载统计或权重重排输入。文件注释也写明：真实 inference 仍使用原始 router logits；预测 route 只缓存在 runner 上，不喂给 fused expert kernel。相关 trace 逻辑还出现在 `vllm/model_executor/layers/fused_moe/moe_trace.py`。

## 支持、限制和测试

模型层面的支持分散在各 MoE 模型文件中，常见模式是读取 `parallel_config.enable_eplb` 和 `parallel_config.eplb_config.num_redundant_experts`，构造 `FusedMoE(..., enable_eplb=..., num_redundant_experts=...)`，并在模型级 `set_eplb_state()` 中把状态传给每个 MoE layer。`MiniCPM` 明确对 EPLB 抛 `NotImplementedError`。

Spec decode 的 Medusa 路径在 `vllm/v1/spec_decode/medusa.py` 断言不支持 EPLB。draft MoE 可以注册 EPLB，但 Elastic EP + draft model 被禁止。

主要测试文件是：

- `tests/distributed/test_eplb_algo.py`：策略算法。
- `tests/distributed/test_eplb_execute.py`：权重搬运。
- `tests/distributed/test_eplb_events.py`：`CpuGpuEvent`。
- `tests/distributed/test_eplb_fused_moe_layer.py` 和 `test_eplb_fused_moe_layer_dep_nvfp4.py`：FusedMoE 集成。
- `tests/kernels/moe/test_routing.py`：逻辑 id 到物理 id 的映射和负载记录。
- `tests/kernels/moe/test_moe_layer.py`：MoE layer 在 EPLB 下的输出一致性和后端/量化组合。

## 一句话串起来

启动时，配置打开 EPLB，MoE 模型注册全局映射和负载 tensor；forward 时 router 先产出逻辑 expert id，再由 EPLB kernel 映射到物理副本并记录每个物理 expert 的 token 数；每个 engine step 后，`EplbState.step()` 把负载写入滑动窗口，达到 `step_interval` 后 all-reduce 全局负载，用 default policy 生成新物理布局，再通过 communicator 搬 expert 权重并提交映射。同步模式一次搬完，异步模式后台逐层搬，主线程每步消费一层结果。
