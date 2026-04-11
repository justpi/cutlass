---
title: GEMM API 2.x
parent: 核心概念
nav_order: 3
---

# CUTLASS 2.x GEMM API
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CUTLASS 2.x GEMM API 紧密映射 GPU 的硬件层次结构：

```
Device（设备）
  └── Kernel
        └── Threadblock（线程块）
              └── Warp（线程束）
                    └── Thread（线程）
```

每一层都是 C++ 模板类，通过模板参数控制 Tile 大小、数据类型、布局等。

{: .note }
> 虽然 3.x 是推荐的新 API，但 2.x API 仍然可用且广泛使用。许多示例和已有项目基于 2.x。两套 API 可以共存于同一项目中。

---

## Device 层

`cutlass::gemm::device::Gemm` 是 2.x 中最常用的入口点：

```cpp
#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    float,                          // ElementA
    cutlass::layout::ColumnMajor,   // LayoutA
    float,                          // ElementB
    cutlass::layout::ColumnMajor,   // LayoutB
    float,                          // ElementC
    cutlass::layout::ColumnMajor,   // LayoutC
    float,                          // ElementAccumulator
    cutlass::arch::OpClassTensorOp, // 操作类型
    cutlass::arch::Sm80,            // 目标架构
    // Tile 形状 (M, N, K)
    cutlass::gemm::GemmShape<128, 128, 32>,
    // Warp 形状 (M, N, K)
    cutlass::gemm::GemmShape<64, 64, 32>,
    // MMA 指令形状
    cutlass::gemm::GemmShape<16, 8, 8>
>;

// 创建参数和运行
Gemm gemm_op;
Gemm::Arguments args(
    {M, N, K},         // 问题尺寸
    {ptr_A, lda},      // TensorRef A
    {ptr_B, ldb},      // TensorRef B
    {ptr_C, ldc},      // TensorRef C（源）
    {ptr_D, ldd},      // TensorRef D（目标）
    {alpha, beta}       // Epilogue 参数
);

cutlass::Status status = gemm_op(args);
```

### 模板参数一览

| 参数 | 说明 |
|:-----|:-----|
| `ElementA/B/C` | 各矩阵的数据类型 |
| `LayoutA/B/C` | 内存布局（RowMajor / ColumnMajor） |
| `ElementAccumulator` | 累加器类型（通常 FP32） |
| `OpClass` | `OpClassTensorOp`（Tensor Core）或 `OpClassSimt`（CUDA Core） |
| `ArchTag` | 目标架构（Sm70/75/80/86/89/90） |
| `ThreadblockShape` | Threadblock Tile 大小 |
| `WarpShape` | Warp Tile 大小 |
| `InstructionShape` | 单条 MMA 指令的形状 |

---

## Threadblock 层

Threadblock 层管理 Shared Memory 中的数据，并协调 Warp 之间的工作分配：

### 关键组件

- **MMA Pipelined / Multistage**：在 K 维度上迭代的主循环
- **Tile Iterator**：负责从 Global Memory 加载 Tile 到 Shared Memory
- **Shared Memory 布局**：定义 Shared Memory 中的数据排列方式

### Threadblock 级 MMA

```cpp
// 多级流水线 MMA（Ampere+）
using Mma = cutlass::gemm::threadblock::MmaMultistage<
    ThreadblockShape,
    IteratorA,      // A 的 Tile Iterator
    SmemLayoutA,    // A 在 Shared Memory 中的布局
    IteratorB,      // B 的 Tile Iterator
    SmemLayoutB,    // B 在 Shared Memory 中的布局
    ElementAccumulator,
    LayoutC,
    Policy,
    Stages          // 流水线级数
>;
```

---

## Warp 层

Warp 层直接与 Tensor Core MMA 指令交互：

```cpp
// Warp 级 MMA
using WarpMma = cutlass::gemm::warp::MmaTensorOp<
    WarpShape,           // Warp Tile 形状
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    Policy
>;
```

每个 Warp 从 Shared Memory 读取数据到寄存器，然后执行 `mma.sync` 或 `wgmma` 指令。

---

## Epilogue

2.x 的 Epilogue 组件：

```cpp
using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    ThreadblockShape,
    WarpMmaOperator,
    PartitionsK,
    OutputOp,             // 逐元素操作
    EpilogueOutputOp
>;
```

### 常用 OutputOp

| OutputOp | 功能 |
|:---------|:-----|
| `LinearCombination` | D = alpha * acc + beta * C |
| `LinearCombinationRelu` | D = ReLU(alpha * acc + beta * C) |
| `LinearCombinationBias` | D = alpha * acc + beta * C + bias |
| `LinearCombinationGELU` | D = GELU(alpha * acc + beta * C) |

---

## GemmShape 约束

选择 Tile 形状时需要注意的约束：

```
ThreadblockShape 必须是 WarpShape 的整数倍
WarpShape 必须是 InstructionShape 的整数倍
```

例如：
```
ThreadblockShape = <128, 128, 32>
WarpShape        = < 64,  64, 32>  → 每个 Threadblock 有 2×2=4 个 Warp
InstructionShape = < 16,   8,  8>  → 每个 Warp 执行 4×8=32 条 MMA 指令
```

---

## GemmUniversal

`device::GemmUniversal` 是 2.x 中功能更全面的接口，支持：

| 模式 | 说明 |
|:-----|:-----|
| `kGemm` | 标准单个 GEMM |
| `kBatched` | Batched GEMM（数组步长寻址） |
| `kArray` | Batched GEMM（指针数组寻址） |

```cpp
using GemmUniversal = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    ThreadblockSwizzle,
    Stages
>;
```

---

## 从 2.x 迁移到 3.x

CUTLASS 提供了向后兼容层，使 2.x 代码可以在 3.x 环境中编译运行：

1. 2.x 的 `#include` 路径保持不变
2. 可以逐步将组件替换为 3.x 版本
3. `cutlass::gemm::device::GemmUniversalAdapter` 可以包装 3.x Kernel 提供 2.x 类似的接口

{: .tip }
> 新项目建议直接使用 3.x API。如果需要支持 SM70/SM75，可以继续使用 2.x。

---

## 下一步

- [隐式 GEMM 卷积](convolution) — 将卷积表达为 GEMM
- [功能列表](functionality) — 查看各架构支持的操作
