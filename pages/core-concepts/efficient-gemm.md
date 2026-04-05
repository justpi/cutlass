---
title: 高效 GEMM 原理
parent: 核心概念
nav_order: 1
---

# 高效 GEMM 原理
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## GEMM 基础

矩阵乘法 GEMM（General Matrix Multiply）计算：

```
D = alpha * A * B + beta * C
```

其中 A 为 M×K 矩阵，B 为 K×N 矩阵，C 和 D 为 M×N 矩阵。

GEMM 的计算量为 `O(M * N * K)`，而数据量为 `O(M*K + K*N + M*N)`。当矩阵足够大时，计算密度（FLOPs/Byte）很高，使其成为 GPU 上的理想工作负载。

---

## 分层分块策略

CUTLASS 使用**分层分块**（Hierarchical Tiling）将 GEMM 映射到 GPU 的存储/计算层次上：

```
┌─────────────────────────────────────┐
│         Global Memory               │  ← 整个矩阵
│  ┌──────────────────────────┐       │
│  │    Threadblock Tile      │       │  ← 每个 Threadblock 处理一个 Tile
│  │  ┌───────────────┐       │       │
│  │  │  Warp Tile    │       │       │  ← 每个 Warp 处理子 Tile
│  │  │  ┌────────┐   │       │       │
│  │  │  │Thread  │   │       │       │  ← 每个 Thread 处理最小单元
│  │  │  │Tile    │   │       │       │
│  │  │  └────────┘   │       │       │
│  │  └───────────────┘       │       │
│  └──────────────────────────┘       │
└─────────────────────────────────────┘
```

### Threadblock 级分块

将输出矩阵 D 按 Threadblock Tile（如 128×128）划分，每个 Threadblock 负责计算一个输出 Tile。在 K 维度上，按 Tile_K（如 32）迭代：

1. 从 Global Memory 加载 A 的 128×32 子块和 B 的 32×128 子块到 Shared Memory
2. Threadblock 内的所有 Warp 从 Shared Memory 读取数据并计算
3. 沿 K 维度移动，重复步骤 1-2
4. 所有 K 迭代完成后，将结果写回 Global Memory

### Warp 级分块

在 Shared Memory 内，进一步将工作分配给各个 Warp。每个 Warp 负责一个更小的子 Tile（如 64×64），使用 Tensor Core 的 MMA 指令执行矩阵乘累加。

### Thread 级

每个线程持有累加器（Accumulator）寄存器，执行最细粒度的乘加运算。Tensor Core MMA 指令实际上是 Warp 级别的协作操作。

---

## 内存层次优化

### Global → Shared Memory

**挑战**：Global Memory 带宽有限（~2TB/s on A100），延迟高（~400 cycles）。

**策略**：
- **异步拷贝**（`cp.async`，Ampere+）：发起异步的 Global→Shared 拷贝，线程无需等待
- **TMA**（Tensor Memory Accelerator，Hopper+）：硬件自动执行多维数据搬运，线程完全不参与
- **向量化加载**：使用 128-bit 宽的加载指令，一次读取多个元素

### Shared → Register

**挑战**：从 Shared Memory 到寄存器的传输需要避免 Bank Conflict。

**策略**：
- **Swizzle**：对 Shared Memory 地址进行位变换，消除 Bank Conflict
- **LDMATRIX**（Turing+）：专用的 Shared→Register 加载指令，直接匹配 Tensor Core 的数据布局
- **WGMMA**（Hopper+）：直接从 Shared Memory 读取操作数，绕过显式的寄存器加载

---

## 软件流水线（Software Pipelining）

为了隐藏 Global Memory 的加载延迟，CUTLASS 使用软件流水线技术：

```
Stage 0:  [Load A₀,B₀]  [Compute  -  ]  [         -       ]
Stage 1:  [Load A₁,B₁]  [Compute A₀*B₀]  [         -       ]
Stage 2:  [Load A₂,B₂]  [Compute A₁*B₁]  [Store partial    ]
Stage 3:  [Load A₃,B₃]  [Compute A₂*B₂]  [Store partial    ]
...
```

通过多个 Stage（缓冲区），在计算当前 Tile 时同时加载下一个 Tile 的数据：

- **2-stage**：最简单，双缓冲
- **3+ stage**（Ampere+）：利用 `cp.async` 的多级流水线
- **Hopper**：TMA 支持更深的流水线，配合 `arrive/wait` 障碍同步

---

## Epilogue

GEMM 的主循环完成后，Epilogue 阶段负责将结果从累加器写回 Global Memory，同时可以融合额外的操作：

```
D = Epilogue(alpha * Accumulator + beta * C)
```

常见的 Epilogue 操作：
- **线性组合**：`D = alpha * acc + beta * C`
- **Bias 加法**：`D = acc + bias`
- **激活函数**：ReLU、GELU、SiLU 等
- **量化**：FP32 累加器 → FP8/INT8 输出
- **自定义融合**：任意可注入的逐元素操作

Epilogue 融合可以避免额外的 Kernel 启动和 Global Memory 往返，对性能至关重要。

---

## Tile 调度策略

CUTLASS 支持多种 Tile 到 Threadblock 的映射策略：

| 调度器 | 说明 |
|:--------|:-----|
| **Default** | 线性映射，每个 CTA 处理一个输出 Tile |
| **Stream-K** | 沿 K 维度将工作分散到所有 SM，改善负载均衡 |
| **Persistent** | CTA 常驻 SM，通过内部循环处理多个 Tile |
| **Cooperative** | 多个 CTA 协作处理同一个输出 Tile |
| **Ping-Pong** | 两组 Warp Group 交替执行加载和计算 |

---

## 性能要素总结

实现高效 GEMM 的关键因素：

1. **数据重用最大化**：通过分层 Tiling，在每层内存层次中最大化数据重用
2. **延迟隐藏**：通过软件流水线重叠计算和内存访问
3. **Tensor Core 利用率**：确保 MMA 指令持续发射，避免闲置
4. **Bank Conflict 消除**：通过 Swizzle 等技术避免 Shared Memory 访问冲突
5. **寄存器压力管理**：平衡 Tile 大小和寄存器使用量
6. **Epilogue 融合**：减少不必要的内存往返

---

## 下一步

- [GEMM API 3.x](gemm-api-3x) — 学习 CUTLASS 3.x 的 GEMM 编程模型
- [GEMM API 2.x](gemm-api-2x) — 了解经典的 CUTLASS 2.x API
- [CuTe 快速入门](../cute/quickstart) — 掌握 CUTLASS 3.x 的核心抽象
