# MoE EP 通信与计算负载实验

本实验使用
`benchmark_moe_ep_distribution.py` 绕过 router，直接构造
`topk_ids`，用于分别观察 EP 通信比例和 rank 间计算负载不均衡对延迟的影响。

实验使用 `locality_skew` 路由模式，其中：

- `local_share` 表示每个源 rank 留在本 rank 的 assignment 比例，远端通信
  比例约为 `1 - local_share`。
- `rank_skew` 表示远端 assignment 向 `hot_rank` 集中的程度。`0` 表示在
  其他 rank 间均匀分配，`1` 表示尽量集中到 `hot_rank`。

benchmark 始终在每个 measured stage 前执行 barrier，因此无需额外传入
`--stage-barrier`。

## 环境

所有 Python 命令都通过项目虚拟环境执行：

```bash
.venv/bin/python --version
```

绘图需要 matplotlib。如果当前虚拟环境中没有安装：

```bash
uv pip install matplotlib
```

## 单机冒烟实验

先运行一个较小的 3×3 sweep，检查 DeepEP 初始化、路由生成和结果输出是否正常：

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_distribution.py \
  --nproc-per-node 8 \
  --model-preset qwen3-30b-a3b \
  --backend deepep_low_latency \
  --pattern locality_skew \
  --local-shares 0,0.5,1 \
  --rank-skews 0,0.5,1 \
  --hot-rank 0 \
  --tokens 4096 \
  --warmup 3 \
  --iters 10 \
  --output-jsonl results/data/moe_ep_smoke.jsonl \
  --output-csv results/data/moe_ep_smoke.csv
```

其中 JSONL 保存每个 rank、每次迭代以及聚合后的详细记录；CSV 只保存经过
trim 后的 sweep summary，主要用于绘图。

## 完整二维实验

确认冒烟实验正常后，运行 `local_share × rank_skew` 的完整二维实验：

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_distribution.py \
  --nproc-per-node 8 \
  --model-preset qwen3-30b-a3b \
  --backend deepep_low_latency \
  --pattern locality_skew \
  --local-shares 0,0.25,0.5,0.75,1 \
  --rank-skews 0,0.25,0.5,0.75,1 \
  --hot-rank 0 \
  --tokens 4096 \
  --warmup 10 \
  --iters 100 \
  --trim-ratio 0.1 \
  --output-jsonl results/data/moe_ep_distribution.jsonl \
  --output-csv results/data/moe_ep_distribution.csv
```

如需测试 Qwen3-235B-A22B，将 preset 替换为：

```bash
--model-preset qwen3-235b-a22b
```

建议固定 GPU、backend、模型形状和 token 数量后再比较不同 sweep 点。正式实验可
重复运行三次，并为输出文件增加编号，以评估跨进程启动之间的波动。

## 分离通信和计算实验

只观察通信比例时，固定 `rank_skew=0`。此时所有 rank 收到的 assignment 数量
基本相同，计算负载保持平衡：

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_distribution.py \
  --nproc-per-node 8 \
  --model-preset qwen3-30b-a3b \
  --backend deepep_low_latency \
  --pattern locality_skew \
  --local-shares 0,0.25,0.5,0.75,1 \
  --rank-skew 0 \
  --warmup 10 \
  --iters 100 \
  --output-jsonl results/data/moe_ep_communication.jsonl \
  --output-csv results/data/moe_ep_communication.csv
```

只观察负载倾斜时，固定 `local_share`，使总远端通信比例保持不变：

```bash
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_distribution.py \
  --nproc-per-node 8 \
  --model-preset qwen3-30b-a3b \
  --backend deepep_low_latency \
  --pattern locality_skew \
  --local-share 0.5 \
  --rank-skews 0,0.25,0.5,0.75,1 \
  --hot-rank 0 \
  --warmup 10 \
  --iters 100 \
  --output-jsonl results/data/moe_ep_rank_skew.jsonl \
  --output-csv results/data/moe_ep_rank_skew.csv
```

## 多节点实验

每个节点执行相同命令，仅 `NODE_RANK` 不同：

```bash
.venv/bin/torchrun \
  --nnodes 2 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port 29500 \
  benchmarks/kernels/benchmark_moe_ep_distribution.py \
  --model-preset qwen3-30b-a3b \
  --backend deepep_low_latency \
  --pattern locality_skew \
  --local-shares 0,0.25,0.5,0.75,1 \
  --rank-skews 0,0.25,0.5,0.75,1 \
  --hot-rank 0 \
  --warmup 10 \
  --iters 100 \
  --output-jsonl results/data/moe_ep_distribution.jsonl \
  --output-csv results/data/moe_ep_distribution.csv
```

只有 global rank 0 写结果文件。多节点运行时应确保 rank 0 的输出目录存在且可写；
如果需要从其他节点访问结果，应将其放在共享文件系统中。

## 绘图

对完整二维实验生成汇总图：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
.venv/bin/python benchmarks/kernels/plot_moe_ep_distribution.py \
  --input-csv results/data/moe_ep_distribution.csv \
  --output results/plots/moe_ep_distribution.png \
  --skew-local-share 0.5
```

JSONL 和 CSV 数据保存在 `results/data/`，生成的 PNG 单独保存在
`results/plots/`。图片包含四张折线图：通信比例的 stage 耗时、固定
`local_share` 的负载倾斜 stage 耗时、所有 `local_share` 对应的
`rank_skew` stage-sum 曲线，以及所有 `rank_skew` 对应的 `local_share`
stage-sum 曲线。图中不绘制 end-to-end latency。

## 结果解读

通信实验中应先确认 `mean_token_imbalance` 接近 `1`，再观察
`mean_max_dispatch_ms`、`mean_max_combine_ms` 和
`mean_max_total_ms` 随 `mean_remote_share` 的变化。

负载倾斜实验中应先确认不同 `rank_skew` 下的 `mean_remote_share` 基本不变，
然后观察 `mean_token_imbalance`、`mean_max_compute_ms` 和 stage-sum latency
之间的关系。

图中的 `mean_max_total_ms` 是同一 rank 上 dispatch、compute 和 combine 耗时的
和，用于比较不同路由分布下的 stage 总成本；它不等于真实端到端延迟。CSV 和
JSONL 中仍保留 `mean_max_end_to_end_ms`，但绘图脚本不再展示该指标。
