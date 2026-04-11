---
title: Pipeline 同步
parent: 进阶主题
nav_order: 2
---

# Pipeline 同步原语
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CUTLASS 的高性能 GEMM Kernel 依赖 **软件流水线**（Software Pipelining）来重叠内存加载和计算。这需要精确的同步原语来协调生产者（数据加载）和消费者（MMA 计算）。

---

## CUDA 同步基础

### __syncthreads

最基本的 Threadblock 级同步屏障：

```cpp
__syncthreads();  // 所有线程到达后才继续
```

### cp.async（Ampere+）

异步 Global→Shared Memory 拷贝，不阻塞线程：

```cpp
// 发起异步拷贝
cp_async_fence();     // 标记一个 fence 点
// ... 发起更多拷贝 ...
cp_async_wait<N>();   // 等待直到最多 N 个 fence 未完成
```

### TMA（Hopper+）

Tensor Memory Accelerator，硬件自动执行多维数据传输：

```cpp
// TMA 使用 arrive/wait 模式
cute::copy(sm90_tma_load, tma_src, smem_dst);

// 使�� barrier 等待完成
pipeline.consumer_wait(stage);
// ... 使用数据 ...
pipeline.consumer_release(stage);
```

---

## Pipeline 抽象

CUTLASS 提供了 `PipelineAsync` 和 `PipelineTmaAsync` 等管线抽象，封装了生产者-消费者同步模式。

### 基本概念

```
Stage 0:  [Producer: Load]  →  [Consumer: Compute]
Stage 1:  [Producer: Load]  →  [Consumer: Compute]
Stage 2:  [Producer: Load]  →  [Consumer: Compute]
            ↓                      ↓
         循环使用               循环使用
```

- **Stage**：共享内存缓冲区的一个槽位
- **Producer**：负责将数据从 Global Memory 加载到 Shared Memory
- **Consumer**：负责从 Shared Memory 读取数据并执行 MMA

### PipelineAsync（Ampere）

基于 `cp.async` 的管线：

```cpp
using MainloopPipeline = cutlass::PipelineAsync<Stages>;

typename MainloopPipeline::Params pipeline_params;
pipeline_params.transaction_bytes = ... ;
MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params);

// 生产者
pipeline.producer_acquire(stage);
// ... 发起 cp.async 拷贝 ...
pipeline.producer_commit(stage);

// 消费者
pipeline.consumer_wait(stage);
// ... 执行 MMA 计算 ...
pipeline.consumer_release(stage);
```

### PipelineTmaAsync（Hopper）

基于 TMA 和 mbarrier 的管线：

```cpp
using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;

// 生产者（通常是一个 Warp 的线程 0）
pipeline.producer_acquire(stage);
// ... 发起 TMA 加载 ...
pipeline.producer_commit(stage, tma_transaction_bytes);

// 消费者（计算 Warp Group）
pipeline.consumer_wait(stage);
// ... 执行 WGMMA ...
pipeline.consumer_release(stage);
```

---

## Warp 特化（Hopper）

Hopper 架构支持 **Warp 特化**（Warp Specialization）：不同 Warp 执行不同角色。

### Producer-Consumer 模式

```
Threadblock (256 threads)
├── Producer Warps (Warp 0-3):   加载数据 (TMA)
└── Consumer Warps (Warp 4-7):   执行 MMA (WGMMA)
```

```cpp
if (warp_group_role == WarpGroupRole::Producer) {
    // 使用 TMA 加载数据到 Shared Memory
    for (int k = 0; k < K_tiles; ++k) {
        pipeline.producer_acquire(write_stage);
        tma_load(A_tma, smem_A(_, _, write_stage), ...);
        tma_load(B_tma, smem_B(_, _, write_stage), ...);
        pipeline.producer_commit(write_stage, transaction_bytes);
        ++write_stage;
    }
}
else {  // Consumer
    // 执行 WGMMA 计算
    for (int k = 0; k < K_tiles; ++k) {
        pipeline.consumer_wait(read_stage);
        warpgroup_arrive();
        gemm(tiled_mma, tCsA(_, _, read_stage), tCsB(_, _, read_stage), tCrC);
        warpgroup_commit_batch();
        pipeline.consumer_release(read_stage);
        ++read_stage;
    }
}
```

### Ping-Pong 模式

两个 Consumer Warp Group 交替执行：

```
Stage k:    Consumer Group A 计算  |  Consumer Group B 等待
Stage k+1:  Consumer Group A 等待  |  Consumer Group B 计算
```

这种模式通过重叠 MMA 的寄存器写回和下一轮的 Shared Memory 读取来进一步隐藏延迟。

---

## 同步原语总结

| 原语 | 架构 | 用途 |
|:-----|:-----|:-----|
| `__syncthreads()` | 所有 | Threadblock 全局同步 |
| `cp_async_fence/wait` | SM80+ | 异步拷贝同步 |
| `mbarrier` | SM90+ | 硬件屏障（TMA 使用） |
| `PipelineAsync` | SM80+ | cp.async ��线封装 |
| `PipelineTmaAsync` | SM90+ | TMA 管线封装 |
| `warpgroup_arrive/commit/wait` | SM90+ | Warp Group MMA 同步 |

---

## 下一步

- [Profiler](profiler) — 性能分析工具
- [Blackwell 支持](blackwell) — SM100 新特性
