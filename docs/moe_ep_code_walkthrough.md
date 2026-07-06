# vLLM Expert Parallelism Code Walkthrough

本文梳理 vLLM 中 Expert Parallelism（EP）相关代码、关键函数和一次 MoE
forward 的执行流程。内容基于当前仓库代码，重点覆盖：

- EP 配置如何生成；
- EP 通信 group 和 all2all manager 如何初始化；
- `FusedMoE` 如何建立 expert map、router、quant method 和 runner；
- 一次 MoE forward 在 router、dispatch、expert compute、combine 间如何流转；
- 不同 all2all backend 的 prepare/finalize 分工；
- EPLB 和 Elastic EP 的主要入口。

## 1. 术语和总体模型

vLLM 的 MoE EP 可以理解为：每个 EP rank 完整持有一部分 expert 的权重，
token 根据 router 结果被 dispatch 到拥有目标 expert 的 rank 上计算，最后
再 combine 回原 token 顺序和原 rank。

主要概念：

- `TP`：Tensor Parallelism，普通 dense tensor 的切分维度。
- `DP`：Data Parallelism，多 engine / 多 batch shard。
- `PCP`：Prefill Context Parallelism，在 MoE 并行配置里类似 DP 参与展平。
- `EP`：Expert Parallelism，expert 权重按 expert 维度切分到 rank。
- `all2all_backend`：token dispatch/combine 的具体通信实现。
- `EPLB`：Expert Parallel Load Balancing，通过 redundant expert 和动态映射降低负载不均。
- `PrepareAndFinalize`：Modular MoE kernel 中负责 dispatch/combine 的组件。
- `FusedMoEExperts`：Modular MoE kernel 中负责本地 expert compute 的组件。

核心执行形态：

```text
hidden_states + router_logits
  -> router.select_experts()
  -> topk_weights + topk_ids
  -> prepare/dispatch
  -> local expert compute
  -> finalize/combine
  -> output
```

## 2. 关键代码目录

### EP 配置和 layer 初始化

- `vllm/model_executor/layers/fused_moe/config.py`
  - `FusedMoEParallelConfig`
  - `FusedMoEConfig`
- `vllm/model_executor/layers/fused_moe/layer.py`
  - `FusedMoE`
  - `FusedMoE.maybe_init_modular_kernel`
  - `FusedMoE.update_expert_map_info`
- `vllm/model_executor/layers/fused_moe/expert_map_manager.py`
  - `ExpertMapManager`
  - `determine_expert_map`
  - `determine_expert_placement_strategy`

### Router 和 runner

- `vllm/model_executor/layers/fused_moe/router/base_router.py`
  - `BaseRouter.select_experts`
  - EPLB logical-to-physical mapping
- `vllm/model_executor/layers/fused_moe/router/*`
  - `FusedTopKRouter`
  - `GroupedTopKRouter`
  - `FusedTopKBiasRouter`
  - `CustomRoutingRouter`
  - `ZeroExpertRouter`
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
  - `MoERunner._forward_impl`
  - `MoERunner._apply_quant_method`
  - sequence-parallel dispatch/combine hooks

### Modular kernel

- `vllm/model_executor/layers/fused_moe/modular_kernel.py`
  - `FusedMoEKernel`
  - `FusedMoEKernelModularImpl`
  - `FusedMoEPrepareAndFinalizeModular`
  - `FusedMoEExpertsModular`
  - `ExpertTokensMetadata`
- `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`
  - `FusedMoEModularMethod`
- `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`
  - `FusedMoEMethodBase`

### all2all backend

- `vllm/model_executor/layers/fused_moe/all2all_utils.py`
  - `maybe_roundup_layer_hidden_size`
  - `maybe_make_prepare_finalize`
- `vllm/model_executor/layers/fused_moe/prepare_finalize/*`
  - `naive_dp_ep.py`
  - `deepep_ht.py`
  - `deepep_ll.py`
  - `nixl_ep.py`
  - `flashinfer_nvlink_one_sided.py`
  - `flashinfer_nvlink_two_sided.py`
  - `mori.py`
  - `no_dp_ep.py`
  - `batched.py`
- `vllm/distributed/device_communicators/all2all.py`
  - `AgRsAll2AllManager`
  - `DeepEPHTAll2AllManager`
  - `DeepEPLLAll2AllManager`
  - `NixlEPAll2AllManager`
  - `FlashInferNVLinkTwoSidedManager`
  - `FlashInferNVLinkOneSidedManager`
  - `MoriAll2AllManager`

### distributed group

- `vllm/distributed/parallel_state.py`
  - `initialize_model_parallel`
  - `get_ep_group`
  - `GroupCoordinator.dispatch`
  - `GroupCoordinator.combine`
- `vllm/distributed/device_communicators/base_device_communicator.py`
  - `All2AllManagerBase`
  - `DeviceCommunicatorBase`
- `vllm/distributed/device_communicators/cuda_communicator.py`
  - 根据 `all2all_backend` 选择 all2all manager

### EPLB / Elastic EP

- `vllm/distributed/eplb/eplb_state.py`
  - `EplbState`
  - `EplbLayerState`
  - `EplbState.add_model`
  - `EplbState.step`
  - `EplbState.rearrange`
- `vllm/model_executor/layers/fused_moe/eep_reconfigure.py`
  - Elastic EP reconfigure helper
- `vllm/distributed/elastic_ep/elastic_execute.py`
  - `ElasticEPScalingExecutor`

## 3. EP 配置：`FusedMoEParallelConfig`

EP 的第一层入口是 `FusedMoEParallelConfig.make()`：

```python
FusedMoEParallelConfig.make(
    tp_size_=tp_size_,
    pcp_size_=pcp_size_,
    dp_size_=dp_size_,
    sp_size_=self.sp_size,
    vllm_parallel_config=vllm_config.parallel_config,
)
```

位置：`vllm/model_executor/layers/fused_moe/config.py`

### 3.1 是否启用 EP

核心判断：

```python
use_ep = (
    dp_size_ * pcp_size_ * tp_size_ > 1
    and vllm_parallel_config.enable_expert_parallel
)
```

也就是说：

- 必须有实际并行 rank 数大于 1；
- 必须设置 `enable_expert_parallel`；
- EP 会把原先的 TP/DP/PCP 组合展平到 expert parallel 维度。

### 3.2 非 EP 情况

如果 `use_ep=False`：

- `tp_size` 会通过 `flatten_tp_across_dp_and_pcp()` 展平；
- `ep_size=1`；
- `ep_rank=0`；
- `use_ep=False`；
- expert 不跨 EP rank 切分。

这一模式下如果有 DP 且使用新 modular interface，`allgather_reducescatter`
可以作为 naive dispatch/combine fallback。

### 3.3 EP 情况

如果 `use_ep=True`：

```python
ep_size = tp_size
ep_rank = tp_rank
return FusedMoEParallelConfig(
    tp_size=1,
    tp_rank=0,
    ep_size=ep_size,
    ep_rank=ep_rank,
    use_ep=True,
    ...
)
```

含义：

- dense tensor parallel 维度在 MoE 层内被“让位”给 EP；
- 每个 EP rank 持有完整的本地 expert 权重；
- `tp_size` 在 MoE 配置中变成 1；
- `ep_size` 等于展平后的原 `tp * dp * pcp` 规模；
- `ep_rank` 等于展平后的 rank。

## 4. EP group 和 all2all manager 初始化

### 4.1 `get_ep_group()`

位置：`vllm/distributed/parallel_state.py`

`get_ep_group()` 返回全局 `_EP`：

```python
def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, ...
    return _EP
```

`_EP` 是 `GroupCoordinator`，封装了：

- CPU process group；
- device process group；
- device communicator；
- all2all manager；
- rank/world size 信息；
- dispatch/combine 等通信方法。

### 4.2 `GroupCoordinator.dispatch/combine`

位置：`vllm/distributed/parallel_state.py`

`GroupCoordinator.dispatch()` 只是转发到 device communicator：

```python
return self.device_communicator.dispatch(...)
```

`GroupCoordinator.combine()` 同理：

```python
return self.device_communicator.combine(...)
```

对于 EP，真正的通信实现一般在：

```text
GroupCoordinator
  -> DeviceCommunicatorBase subclass
  -> all2all_manager
```

### 4.3 all2all manager 选择

位置：`vllm/distributed/device_communicators/cuda_communicator.py`

CUDA communicator 根据 `parallel_config.all2all_backend` 初始化：

- `"naive"` / `"allgather_reducescatter"` -> `AgRsAll2AllManager`
- `"deepep_high_throughput"` -> `DeepEPHTAll2AllManager`
- `"deepep_low_latency"` -> `DeepEPLLAll2AllManager`
- `"mori_high_throughput"` / `"mori_low_latency"` -> `MoriAll2AllManager`
- `"nixl_ep"` -> `NixlEPAll2AllManager`
- `"flashinfer_all2allv"` / `"flashinfer_nvlink_two_sided"` -> `FlashInferNVLinkTwoSidedManager`
- `"flashinfer_nvlink_one_sided"` -> `FlashInferNVLinkOneSidedManager`

`All2AllManagerBase` 定义通用接口：

- `get_handle(kwargs)`
- `dispatch_router_logits(...)`
- `dispatch(...)`
- `combine(...)`
- `set_num_sms(num_sms)`
- `max_sms_used()`
- `destroy()`

## 5. `FusedMoE` 初始化流程

位置：`vllm/model_executor/layers/fused_moe/layer.py`

`FusedMoE.__init__()` 是 EP layer 侧最重要的汇合点。

### 5.1 构建 parallel config

初始化阶段先计算：

```python
self.moe_parallel_config = FusedMoEParallelConfig.make(...)
```

随后设置：

- `self.global_num_experts = num_experts + num_redundant_experts`
- `self.logical_num_experts = num_experts`
- `self.layer_name = prefix`

如果启用 EPLB：

- 校验 expert 数可被 `ep_size` 整除；
- 创建 `EplbLayerState()`；
- redundant expert 只能在 EPLB 下使用。

### 5.2 ExpertMapManager

`FusedMoE` 创建：

```python
self.expert_map_manager = ExpertMapManager(...)
self.update_expert_map_info()
```

`ExpertMapManager` 负责：

- 决定 expert placement strategy；
- 生成 `expert_map`；
- 生成 ROCm AITER 需要的 `expert_mask`；
- 对 DeepEP-LL/NIXL 的 round-robin 生成 routing tables；
- 输出本 rank 的 `local_num_experts`。

### 5.3 Router

随后创建 router：

```python
self.router = create_fused_moe_router(...)
```

router 的职责是从 `router_logits` 得到：

- `topk_weights`
- `topk_ids`

如果 EPLB 启用，router 还会在 `BaseRouter.select_experts()` 中把 logical
expert id 映射到 physical expert id。

### 5.4 FusedMoEConfig

`FusedMoEConfig` 包含 MoE kernel 所需的完整静态信息：

- `num_experts`
- `experts_per_token`
- `hidden_dim`
- `intermediate_size_per_partition`
- `num_local_experts`
- `num_logical_experts`
- `moe_parallel_config`
- `routing_method`
- `moe_backend`
- `max_num_tokens`
- `activation`
- `in_dtype`

### 5.5 Quant method 和 weights

`FusedMoE` 根据 quant config 创建 `FusedMoEMethodBase` 子类：

```python
quant_method = self.quant_config.get_quant_method(self, prefix)
```

如果没有量化配置，则使用：

```python
UnquantizedFusedMoEMethod(self.moe_config)
```

然后调用：

```python
self.quant_method.create_weights(layer=self, **moe_quant_params)
```

注意这里传入的是 `num_experts=self.local_num_experts`，也就是本 rank 只为
本地 expert 创建权重。

### 5.6 MoERunner

最后创建：

```python
self.runner = MoERunner(
    layer_name=self.layer_name,
    moe_config=self.moe_config,
    router=self.router,
    quant_method=self.quant_method,
    ...
)
```

后续 `FusedMoE.forward()` 只做一件事：

```python
return self.runner.forward(hidden_states, router_logits, input_ids)
```

## 6. Expert map 和 placement

位置：`vllm/model_executor/layers/fused_moe/expert_map_manager.py`

### 6.1 `expert_map`

`expert_map` 是 global expert id 到 local expert id 的映射：

```text
expert_map[global_id] = local_id   if expert is local
expert_map[global_id] = -1         if expert is not local
```

非 EP 时可以是 `None`，表示所有 expert 都本地可见。

### 6.2 linear placement

linear placement 通常按连续 expert 段分配：

```text
rank 0: expert 0..k-1
rank 1: expert k..2k-1
...
```

这种 layout 对 DeepEP high-throughput 等后端较自然，因为 local expert 的
global id 可以通过 rank offset 还原。

### 6.3 round-robin placement

round-robin placement 按 `global_id % ep_size` 分配 owner：

```text
rank 0: expert 0, ep_size, 2*ep_size, ...
rank 1: expert 1, ep_size+1, ...
```

当前代码中 round-robin 对 all2all backend 有限制：

- DeepEP low-latency 支持；
- NIXL EP 支持；
- 其他 all2all backend 会 fallback 到 linear。

`ExpertMapManager._init_round_robin_expert_routing_tables()` 会生成：

- `global_to_physical`
- `physical_to_global`
- `local_to_global`

这些 routing tables 传给 DeepEP-LL / NIXL prepare-finalize，用于在 backend
需要 physical expert id 时做映射。

## 7. Router 流程

位置：`vllm/model_executor/layers/fused_moe/router/base_router.py`

`BaseRouter.select_experts()` 是公共模板方法：

```text
1. _validate_eplb_state()
2. _get_indices_type()
3. _compute_routing()
4. capture logical ids if needed
5. trace logical route if needed
6. _apply_eplb_mapping()
7. _convert_indices_dtype()
```

关键点：

- `_compute_routing()` 由具体 router 实现，例如 fused top-k、grouped top-k；
- capture/trace 记录的是 EPLB mapping 前的 logical id；
- EPLB 启用后，`topk_ids` 会从 logical expert id 转成 physical expert id；
- `indices_type_getter` 来自 quant method / modular kernel，因为不同 all2all
  backend 要求的 topk id dtype 可能不同。

## 8. 一次 MoE forward 的完整执行流程

### 8.1 顶层入口

```text
FusedMoE.forward()
  -> MoERunner.forward()
  -> torch.ops.vllm.moe_forward / native fallback
  -> MoERunner._forward_impl()
```

`MoERunner._forward_impl()` 的核心结构：

```text
1. layer.ensure_moe_quant_config_init()
2. sync shared experts stream
3. 如果 runner 内持有 gate，计算 router_logits
4. 进入 sequence parallel context
5. _maybe_dispatch()
6. _apply_quant_method()
7. _maybe_combine()
```

### 8.2 `_apply_quant_method()`

位置：`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

逻辑：

```text
1. 可能先运行 shared experts
2. 如果 quant method 是 monolithic：
     quant_method.apply_monolithic(...)
   否则：
     router.select_experts(...)
     quant_method.apply(...)
3. 可能运行 multi-stream overlapped shared experts
4. 返回 shared_output, fused_out
```

EP 重点在非 monolithic 路径：

```text
topk_weights, topk_ids = router.select_experts(...)
fused_out = quant_method.apply(..., topk_weights, topk_ids, ...)
```

### 8.3 `FusedMoEModularMethod.apply()`

位置：`vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`

这个类是旧 quant method 和新 modular kernel 之间的桥：

```python
return self.moe_kernel.apply(
    hidden_states=x,
    w1=layer.w13_weight,
    w2=layer.w2_weight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    activation=layer.activation,
    global_num_experts=layer.global_num_experts,
    expert_map=layer.expert_map,
    ...
)
```

### 8.4 `FusedMoEKernelModularImpl.apply()`

位置：`vllm/model_executor/layers/fused_moe/modular_kernel.py`

核心三段：

```text
1. _prepare()
2. _fused_experts()
3. _finalize()
```

#### `_prepare()`

调用 `prepare_finalize.prepare()` 或 `prepare_async()`。

职责：

- 可选输入量化；
- dispatch token；
- dispatch topk weights / topk ids；
- 返回本 rank 要计算的 activation；
- 返回本 rank 对应的 topk ids / weights；
- 返回 `ExpertTokensMetadata`。

`ExpertTokensMetadata` 包含：

- `expert_num_tokens`
- `expert_num_tokens_cpu`

它告诉 expert kernel 每个本地 expert 收到多少 token。

#### `_fused_experts()`

职责：

- 根据 `fused_experts.moe_problem_size()` 计算 M/N/K/top-k；
- 分配 workspace；
- 调用 `fused_experts.apply()` 执行本地 expert compute；
- 返回 `fused_out`。

如果某个 rank 没有收到 token，`M_full == 0` 时会返回空 tensor。

#### `_finalize()`

调用 `prepare_finalize.finalize()` 或 `finalize_async()`。

职责：

- 可选应用 topk weight；
- 可选 reduce top-k；
- combine 结果回原 rank / 原 token layout；
- 处理 shared expert overlap。

## 9. `maybe_make_prepare_finalize()` 和 backend 选择

位置：`vllm/model_executor/layers/fused_moe/all2all_utils.py`

`maybe_make_prepare_finalize()` 根据 `moe.moe_parallel_config` 和
`all2all_backend` 创建不同 `PrepareAndFinalize` 对象。

### 9.1 非 all2all kernel

如果：

```python
not moe.moe_parallel_config.use_all2all_kernels
```

且 `allow_new_interface=True`：

- DP 大于 1：返回 naive DP/EP prepare-finalize；
- 否则：返回 no-DP/no-EP prepare-finalize。

旧接口下可能返回 `None`，表示不创建 modular kernel wrapper。

### 9.2 DeepEP high-throughput

backend：

```text
deepep_high_throughput
```

对象：

```text
DeepEPHTPrepareAndFinalize
```

特点：

- activation format 是 `Standard`；
- 支持 async prepare/finalize；
- `output_is_reduced=True`，combine 内部完成跨 rank reduction；
- dispatch 前可能根据量化方式先量化；
- 使用 `buffer.get_dispatch_layout()` 计算 token 到 rank/expert 的布局；
- 使用 `buffer.dispatch()` 做 dispatch；
- combine 使用 `buffer.combine()`；
- HT combine 目前要求 BF16 输出。

### 9.3 DeepEP low-latency

backend：

```text
deepep_low_latency
```

对象：

```text
DeepEPLLPrepareAndFinalize
```

特点：

- activation format 是 `BatchedExperts`；
- topk indices dtype 是 `torch.int64`；
- 需要 `max_tokens_per_rank`；
- 支持 round-robin routing tables；
- hidden size 必须在 supported list 中，不足时由
  `maybe_roundup_layer_hidden_size()` 向上取支持值；
- dispatch 使用 `buffer.low_latency_dispatch()`；
- combine 使用 `buffer.low_latency_combine()`；
- 权重应用和 top-k reduction 发生在 combine kernel 中，因此 finalize 期望
  `TopKWeightAndReduceDelegate`。

### 9.4 allgather_reducescatter / naive

backend：

```text
naive
allgather_reducescatter
```

对象：

```text
AgRsAll2AllManager
MoEPrepareAndFinalizeNaiveDPEPModular
```

dispatch：

```text
all_gatherv(hidden_states, topk_weights, topk_ids)
```

combine：

```text
reduce_scatterv(output)
```

优点是简单、通用；缺点是通信量较大，不能真正只发给目标 expert rank。

### 9.5 NIXL EP

backend：

```text
nixl_ep
```

对象：

```text
NixlEPPrepareAndFinalize
NixlEPAll2AllManager
```

特点：

- 与 DeepEP-LL 类似，支持 max token per rank；
- 支持 routing tables；
- 支持 staged state / elastic reconfigure 场景；
- hidden size 可能需要 roundup。

### 9.6 FlashInfer NVLink

backend：

```text
flashinfer_nvlink_two_sided
flashinfer_all2allv
flashinfer_nvlink_one_sided
```

对象：

```text
FlashInferNVLinkTwoSidedPrepareAndFinalize
FlashInferNVLinkOneSidedPrepareAndFinalize
```

特点：

- 使用 FlashInfer 提供的 NVLink MoE all2all；
- one-sided backend 会初始化 workspace 和 dispatch payload layout；
- 对 quant dtype 有特定支持范围，例如 one-sided 支持 bf16、nvfp4、mxfp8。

### 9.7 Mori

backend：

```text
mori_high_throughput
mori_low_latency
```

对象：

```text
MoriPrepareAndFinalize
MoriAll2AllManager
```

当前 `FusedMoE` 中对 Mori 有额外限制：

- 需要 ROCm AITER fused MoE；
- 不支持 fused shared experts。

## 10. Prepare/finalize 接口契约

`FusedMoEPrepareAndFinalizeModular` 的核心接口：

```python
prepare(...) -> (
    a1q,
    a1q_scale,
    expert_tokens_meta,
    expert_topk_ids,
    expert_topk_weights,
)

finalize(
    output,
    fused_expert_output,
    topk_weights,
    topk_ids,
    apply_router_weight_on_input,
    weight_and_reduce_impl,
)
```

异步 backend 可以实现：

- `prepare_async()`
- `finalize_async()`
- `supports_async() -> True`

modular kernel wrapper 会处理：

- hook / receiver 解包；
- DBO microbatch 下的 recv hook 注册；
- shared expert overlap；
- prepare/finalize 的同步或异步路径。

## 11. Expert compute 接口契约

`FusedMoEExpertsModular.apply()` 接收：

- `output`
- `hidden_states`
- `w1`
- `w2`
- `topk_weights`
- `topk_ids`
- `activation`
- `global_num_experts`
- `expert_map`
- `a1q_scale`
- `a2_scale`
- `workspace13`
- `workspace2`
- `expert_tokens_meta`
- `apply_router_weight_on_input`

不同 expert backend 包括：

- Triton MoE；
- DeepGEMM；
- Cutlass；
- FlashInfer；
- TRT-LLM；
- ROCm AITER；
- Humming kernels；
- CPU/XPU fallback。

EP 对 expert compute 的主要输入影响：

- `w1/w2` 只包含本地 expert；
- `topk_ids` 可能仍是 global id，需要 `expert_map` 映射；
- `expert_tokens_meta` 给出每个本地 expert 的 token count；
- batched activation format 会把输入组织成 `[E, max_tokens, hidden]`。

## 12. Sequence Parallel / PCP 相关路径

`MoERunner._forward_impl()` 进入：

```python
with self._sequence_parallel_context():
    hidden_states, router_logits = self._maybe_dispatch(...)
    ...
    return self._maybe_combine(...)
```

这部分处理 sequence parallel 或 PCP 场景下的 token redistribution。

对 naive AG/RS backend：

- dispatch 可能用 `get_ep_group()` 或 `get_dp_group()`；
- `is_sequence_parallel=True` 时走 EP group；
- combine 也根据该 flag 选择 group。

## 13. EPLB 流程

EPLB 代码主要在：

- `vllm/distributed/eplb/eplb_state.py`
- `vllm/model_executor/layers/fused_moe/router/base_router.py`
- `vllm/model_executor/layers/fused_moe/layer.py`

### 13.1 初始化

`FusedMoE.__init__()` 中：

```python
if enable_eplb:
    self.eplb_state = EplbLayerState()
```

同时 `global_num_experts = num_experts + num_redundant_experts`。

`EplbState.add_model()` 会创建：

- `physical_to_logical_map`
- `logical_to_physical_map`
- `logical_replica_count`
- `expert_load_pass`
- `expert_load_window`
- expert weights buffer
- EPLB communicator

初始 physical-to-logical map 结构：

```text
[original routed experts, redundant experts]
```

redundant expert 会按 logical expert id 取模初始化。

### 13.2 router 阶段记录负载

`BaseRouter.select_experts()` 先生成 logical `topk_ids`。

如果 EPLB 启用：

```python
topk_ids = self._apply_eplb_mapping(topk_ids)
```

底层调用 `eplb_map_to_physical_and_record()`：

- 将 logical id 映射为 physical id；
- 根据 `record_enabled` 决定是否记录 expert load；
- 更新 `expert_load_view`。

### 13.3 step 和 rearrange

`EplbState.step()` 周期性执行：

- 同步当前 expert load；
- 写入 sliding window；
- 计算 balancedness 日志；
- 到达 `step_interval` 后调用 `rearrange()`；
- async 模式下可能等待后台 rearrange 完成。

balancedness 的思路：

```text
avg_tokens / max_tokens
```

越接近 1 表示越均衡。

## 14. Elastic EP / EEP

Elastic EP 相关入口：

- `vllm/distributed/elastic_ep/elastic_execute.py`
- `vllm/model_executor/layers/fused_moe/eep_reconfigure.py`
- `vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py`
- `vllm/distributed/device_communicators/all2all.py` 中的 `NixlEPAll2AllManager`

主要思想：

- EP group 规模可能动态变化；
- staged state 用于准备下一阶段的 all2all buffer / routing state；
- layer 需要更新 `FusedMoEParallelConfig` 和 expert map；
- NIXL EP backend 提供 staged commit 相关能力。

文档化理解上，可以把 Elastic EP 看作：

```text
旧 EP world
  -> 构造 staged EP state
  -> 更新 all2all manager handle
  -> 更新 expert map / routing tables
  -> commit staged state
  -> 新 EP world 生效
```

## 15. Backend 对比表

| backend | prepare/finalize | activation format | dispatch | combine | 备注 |
| --- | --- | --- | --- | --- | --- |
| `allgather_reducescatter` / `naive` | `MoEPrepareAndFinalizeNaiveDPEP*` | `Standard` | all-gather | reduce-scatter | 简单通用，通信量大 |
| `deepep_high_throughput` | `DeepEPHTPrepareAndFinalize` | `Standard` | DeepEP dispatch | DeepEP combine | HT，支持 async，combine BF16 |
| `deepep_low_latency` | `DeepEPLLPrepareAndFinalize` | `BatchedExperts` | `low_latency_dispatch` | `low_latency_combine` | LL，支持 routing tables |
| `nixl_ep` | `NixlEPPrepareAndFinalize` | 通常 batched | NIXL EP | NIXL EP | 支持 staged / elastic |
| `flashinfer_nvlink_two_sided` | `FlashInferNVLinkTwoSidedPrepareAndFinalize` | backend-specific | FlashInfer NVLink | FlashInfer NVLink | 两边参与 |
| `flashinfer_nvlink_one_sided` | `FlashInferNVLinkOneSidedPrepareAndFinalize` | backend-specific | FlashInfer one-sided | FlashInfer one-sided | 支持特定 quant dtype |
| `mori_high_throughput` / `mori_low_latency` | `MoriPrepareAndFinalize` | backend-specific | Mori | Mori | 当前依赖 ROCm AITER |

## 16. 常见代码阅读路径

### 16.1 想看 EP 是否开启

从这里开始：

```text
FusedMoE.__init__()
  -> FusedMoEParallelConfig.make()
```

重点看：

- `enable_expert_parallel`
- `tp_size_ * dp_size_ * pcp_size_`
- `use_ep`
- `ep_size`
- `ep_rank`

### 16.2 想看 expert 分配到哪些 rank

从这里开始：

```text
FusedMoE.__init__()
  -> ExpertMapManager(...)
  -> determine_expert_placement_strategy()
  -> determine_expert_map()
  -> update_expert_map_info()
```

重点看：

- `expert_placement_strategy`
- `expert_map`
- `expert_mask`
- `routing_tables`
- `local_num_experts`

### 16.3 想看一次 token 怎么 dispatch/combine

从这里开始：

```text
FusedMoE.forward()
  -> MoERunner._forward_impl()
  -> MoERunner._apply_quant_method()
  -> router.select_experts()
  -> quant_method.apply()
  -> FusedMoEKernel.apply()
  -> FusedMoEKernelModularImpl._prepare()
  -> FusedMoEKernelModularImpl._fused_experts()
  -> FusedMoEKernelModularImpl._finalize()
```

然后根据 `all2all_backend` 跳到对应 `prepare_finalize/*.py`。

### 16.4 想看 all2all backend 是谁创建的

从这里开始：

```text
cuda_communicator.py
  -> if self.use_all2all:
       self.all2all_manager = ...
```

再看：

```text
all2all_utils.maybe_make_prepare_finalize()
```

它决定 layer 的 `PrepareAndFinalize` 对象。

## 17. 调试建议

### 17.1 打印 EP 配置

`ExpertMapManager` 初始化时已经有 `logger.info_once`：

```text
[EP Rank x/y] Expert parallelism is enabled...
```

这能看到：

- EP rank；
- EP size；
- placement strategy；
- local/global expert 数；
- local -> global expert 映射。

### 17.2 看路由结果

可以使用已有 `moe_trace.py`：

```text
VLLM_MOE_TRACE_DIR=...
VLLM_MOE_TRACE_MAX_STEPS=...
```

它通过 router trace hook 捕获 EPLB mapping 前的 logical topk ids。

### 17.3 看 dispatch/combine 性能

如果需要关联 token 分布、dispatch/combine 延迟和本地 expert compute，可使用
当前新增的 `moe_profile.py` 实验 profiler：

```text
VLLM_MOE_PROFILE_DIR=/tmp/moe_profile
VLLM_MOE_PROFILE_MAX_RECORDS=1000
```

每条 JSONL 记录包含：

- `target_rank_assignments`
- `target_rank_unique_tokens`
- `dispatch_ms`
- `expert_compute_ms`
- `combine_ms`
- `local_expert_tokens`
- `received_tokens`

## 18. 端到端时序图

```text
FusedMoE.forward
  |
  v
MoERunner.forward
  |
  v
torch.ops.vllm.moe_forward / native call
  |
  v
MoERunner._forward_impl
  |
  +-- optional gate -> router_logits
  |
  +-- _maybe_dispatch for sequence parallel / PCP
  |
  v
MoERunner._apply_quant_method
  |
  +-- router.select_experts
  |     |
  |     +-- _compute_routing -> logical topk ids
  |     +-- optional trace/capture
  |     +-- optional EPLB logical -> physical mapping
  |     +-- dtype conversion
  |
  v
FusedMoEModularMethod.apply
  |
  v
FusedMoEKernel.apply
  |
  v
FusedMoEKernelModularImpl.apply
  |
  +-- _prepare
  |     |
  |     +-- quantize if needed
  |     +-- all2all dispatch
  |     +-- produce local activations
  |     +-- produce ExpertTokensMetadata
  |
  +-- _fused_experts
  |     |
  |     +-- allocate workspace
  |     +-- run local expert kernels
  |
  +-- _finalize
        |
        +-- apply weights/reduce if needed
        +-- all2all combine
        +-- write final output
```

## 19. 关键设计点总结

1. EP 配置不是独立于 TP/DP 的简单开关；vLLM 会把 TP/DP/PCP 展平成 EP
   rank 空间，并在 MoE 层内把 `tp_size` 置为 1。

2. `ExpertMapManager` 是理解 EP 权重切片和路由映射的关键。它决定每个 rank
   持有哪些 expert，以及特殊 backend 是否需要 routing tables。

3. `router.select_experts()` 输出的是后续 dispatch 的 token-to-expert 信息。
   EPLB 启用时，这里会把 logical expert id 映射到 physical expert id。

4. `PrepareAndFinalize` 是 EP 通信的抽象层。不同 all2all backend 的差异主要
   被封装在这里，而 expert compute 尽量复用相同的 `FusedMoEExperts` 接口。

5. `FusedMoEKernelModularImpl` 是主执行骨架，固定为 prepare -> expert compute
   -> finalize 三段。

6. DeepEP-HT 和 DeepEP-LL 的最大区别之一是 activation format：HT 走
   `Standard`，LL 走 `BatchedExperts`，并由 combine kernel 处理更多权重和
   reduce 工作。

7. naive AG/RS backend 不是真正按目标 expert rank 做稀疏 all2all，而是
   all-gather 后本地过滤/计算，再 reduce-scatter，适合作为通用 fallback。

8. EPLB 的核心是在 router 阶段记录负载、周期性重排 logical-to-physical
   mapping，并移动 expert 权重。

9. Elastic EP 在普通 EP 基础上增加 staged state 和 reconfigure 流程，当前和
   NIXL EP backend 关系最密切。

