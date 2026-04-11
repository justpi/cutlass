---
title: MMA 指令
parent: CuTe 教程
nav_order: 6
---

# CuTe MMA 指令抽象
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

GPU 的 Tensor Core 提供了 **MMA（Matrix Multiply-Accumulate）** 硬件指令，能够在一个时钟周期内完成小矩阵的乘累加运算。CuTe 通过 `MMA_Atom` 和 `TiledMMA` 对这些指令进行了优雅的抽象。

---

## 硬件 MMA 指令

不同架构的 MMA 指令：

| 架构 | 指令 | 粒度 | 典型形状 |
|:-----|:-----|:-----|:---------|
| **Volta (SM70)** | `mma.sync` | Warp (32 线程) | 8×8×4 FP16 |
| **Turing (SM75)** | `mma.sync` | Warp | 16×8×8 FP16, 8×8×16 INT8 |
| **Ampere (SM80)** | `mma.sync` | Warp | 16×8×16 FP16, 16×8×8 TF32 |
| **Hopper (SM90)** | `wgmma.mma_async` | Warp Group (128 线程) | 64×N×16 FP16 (N=8..256) |

每条 MMA 指令规定了：
- 参与的线程数和它们的角色
- 每个线程持有 A、B、C 矩阵的哪些元素
- 输入和输出的精度

---

## MMA_Atom

`MMA_Atom` 是 CuTe 对单条 MMA 硬件指令的抽象。它封装了：

1. **线程布局（Thread Layout）**：哪些线程参与这条指令
2. **值布局（Value Layout）**：每个线程持有哪些矩阵元素
3. **指令本身**：实际执行的 PTX 操作

### MMA_Traits

每个 MMA Atom 有对应的 `MMA_Traits`，描述其线程-值映射：

```cpp
// Ampere SM80: FP16 输入, FP32 累加器, TN 布局
using MMA_Atom_Arch = SM80_16x8x16_F32F16F16F32_TN;

// 查看其 Traits
using Traits = MMA_Traits<MMA_Atom_Arch>;
// Traits::Shape_MNK = Shape<_16, _8, _16>
// Traits::ThrID     = 线程 ID 布局（32 个线程如何分布）
// Traits::ALayout   = 每个线程持有 A 的哪些元素
// Traits::BLayout   = 每个线程持有 B 的哪些元素
// Traits::CLayout   = 每个线程持有 C 的哪些元素
```

### 常用 MMA Atom

**Ampere (SM80)**：

| Atom | 形状 M×N×K | A → 累加器 |
|:-----|:-----------|:-----------|
| `SM80_16x8x16_F32F16F16F32_TN` | 16×8×16 | FP16 → FP32 |
| `SM80_16x8x16_F32BF16BF16F32_TN` | 16×8×16 | BF16 → FP32 |
| `SM80_16x8x8_F32TF32TF32F32_TN` | 16×8×8 | TF32 → FP32 |
| `SM80_16x8x32_S32S8S8S32_TN` | 16×8×32 | INT8 → INT32 |

**Hopper (SM90)**：

| Atom | 形状 | 特点 |
|:-----|:-----|:-----|
| `SM90_64x8x16_F32F16F16F32_SS` | 64×8×16 | A,B 都在 Shared Memory |
| `SM90_64x8x16_F32F16F16F32_RS` | 64×8×16 | A 在 Register, B 在 Shared |
| `SM90_64x16x16_F32F16F16F32_SS` | 64×16×16 | 更大的 N 维度 |

{: .note }
> Hopper 的 WGMMA 指令后缀 `_SS` 表示 A、B 都从 Shared Memory 读取，`_RS` 表示 A 从 Register、B 从 Shared Memory 读取。

---

## TiledMMA

`TiledMMA` 通过对 MMA Atom 进行 Tiling（重复和排列），构建更大的矩阵运算：

```cpp
// 从 Atom 构建 TiledMMA
auto tiled_mma = make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},    // MMA Atom
    Layout<Shape<_2, _2, _1>>{}          // Atom 排列：M 方向 2 个, N 方向 2 个
);
// 结果：32×16×16 的 TiledMMA（使用 4 个 16×8×16 Atom）
```

### TiledMMA 的使用

```cpp
auto thr_mma = tiled_mma.get_slice(threadIdx.x);

// 分区 A, B, C Tensor
auto tCrA = thr_mma.partition_fragment_A(sA);  // A 在寄存器中的 Fragment
auto tCrB = thr_mma.partition_fragment_B(sB);  // B 在寄存器中的 Fragment
auto tCrC = thr_mma.partition_fragment_C(sC);  // C 的累加器 Fragment

// 执行 MMA
gemm(tiled_mma, tCrA, tCrB, tCrC);
```

### 理解 Partition

`partition_A/B/C` 根据 MMA 的线程-值映射，将 Tensor 分割为每个线程负责的部分：

```
原始 Tensor sA: (128, 32)
       ↓ partition_A with threadIdx.x
当前线程的 Fragment: (MMA_M, MMA_K, num_mma_k)
```

每个线程只看到自己负责的那部分数据，无需关心其他线程。

---

## Warp Group MMA（Hopper）

Hopper 的 WGMMA 指令以 **Warp Group**（128 线程）为单位执行，有特殊的工作方式：

### 关键区别

| 特性 | mma.sync (SM70-SM80) | wgmma.mma_async (SM90) |
|:-----|:---------------------|:-----------------------|
| 参与线程 | 32 (1 Warp) | 128 (4 Warps) |
| A 来源 | Register | Register 或 Shared Memory |
| B 来源 | Register | Shared Memory |
| 同步 | 隐式同步 | 异步，需要显式 fence |
| 累加器 | Register | Register |

### 使用 WGMMA

```cpp
// Hopper WGMMA Atom
using MMA = SM90_64x128x16_F32F16F16F32_SS;

auto tiled_mma = make_tiled_mma(MMA{});
// 一��� Warp Group (128 线程) 执行 64×128×16 的 MMA

// WGMMA 需要 warpgroup barrier
warpgroup_arrive();
gemm(tiled_mma, tCrA, tCrB, tCrC);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

---

## 选择合适的 MMA

### 决策流程

```
1. 确定目标架构 (SM70/75/80/90)
2. 确定数据类型 (FP16/BF16/TF32/INT8/FP8)
3. 选择对应的 MMA Atom
4. 根据 Tile 大小确定 TiledMMA 的排列方式
```

### 性能考虑

- **更大的 TiledMMA** → 更好的数据重用，但需要更多寄存器
- **M/N 维度均衡** → 通常比极端的长条形更高效
- **K 维度** → 由 MMA 指令固定，无法自由选择

---

## 下一步

- [GEMM 教程](gemm-tutorial) — 将所有概念组合，从零构建完整 GEMM
