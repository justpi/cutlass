---
title: Blackwell 架构
parent: 进阶主题
nav_order: 4
---

# Blackwell SM100 支持
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

NVIDIA Blackwell（SM100）架构引入了新一代的 Tensor Core 指令 `tcgen05.mma`，支持更多数据格式和更大的 MMA 操作。CUTLASS 4.x 提供了对 Blackwell 的全面支持。

---

## 新硬件特性

### tcgen05.mma 指令

Blackwell 的 MMA 指令相比 Hopper 的 WGMMA 有以下改进：

| 特性 | Hopper (WGMMA) | Blackwell (tcgen05.mma) |
|:-----|:----------------|:------------------------|
| 累加器位置 | Register File | Tensor Memory (TMem) |
| 输入源 | Shared Memory / Register | Shared Memory |
| MMA 形状 M | 64 | 128 或 256 |
| 1SM/2SM 模式 | 不支持 | 支持 |

### Tensor Memory (TMem)

Blackwell 引入了专用的 **Tensor Memory**，累加器存储在 TMem 中而非 Register File。这释放了寄存器资源，允许更大的 Tile。

### 1SM vs 2SM 模式

| 模式 | 说明 |
|:-----|:-----|
| **1SM** | 单个 SM 独立执行 MMA |
| **2SM** | 两个 SM 协作执行一个更大的 MMA |

2SM 模式使用 128×256 或 256×256 的超大 Tile，进一步提高数据重用率。

---

## 新数据格式

Blackwell 支持多种低精度和块缩放格式：

| 格式 | 位宽 | 说明 |
|:-----|:-----|:-----|
| **FP4 (E2M1)** | 4 bit | 极低精度浮点 |
| **MXFP8** | 8 bit | Microscaling FP8 |
| **MXFP6 (E3M2/E2M3)** | 6 bit | Microscaling FP6 |
| **MXFP4 (E2M1)** | 4 bit | Microscaling FP4 |
| **Block-Scaled** | 可变 | 分块共享缩放因子 |

### Block-Scaled GEMM

Block-Scaled GEMM 将矩阵分成小块，每块共享一个缩放因子（Scale Factor，SF）：

```
实际值 = 量化值 × 缩放因子
```

```cpp
// Block-Scaled GEMM 参数
using ElementA = cutlass::float_e4m3_t;  // FP8
using ElementB = cutlass::float_e4m3_t;  // FP8
using ElementSF = cutlass::float_e8m0_t; // Scale Factor 类型

// 缩放粒度：每 128 个元素共享一个 SF
constexpr int ScaleGranularityM = 128;
constexpr int ScaleGranularityN = 128;
```

---

## Cluster Launch Control (CLC)

Blackwell 引入了 **Cluster Launch Control** 机制，管理 GEMM Kernel 的三个阶段：

```
Prologue → Mainloop → Epilogue
```

### CLC 调度

CLC 允许不同 CTA 在不同阶段之间协调：
- **Prologue**：预加载数据
- **Mainloop**：执行 MMA 计算
- **Epilogue**：写回结果

### Multicast 加载

Blackwell 支持通过 Cluster 级别的 Multicast 将相同数据同时发送到多个 SM 的 Shared Memory，减少 Global Memory 带宽消耗。

---

## CUTLASS 中的 Blackwell 支持

### Collective Builder

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,                      // Blackwell
    cutlass::arch::OpClassBlockScaledTensorOp,  // Block-Scaled Tensor Op
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType
>::CollectiveOp;
```

### 调度策略

| 调度策略 | 说明 |
|:---------|:-----|
| `KernelScheduleSm100` | Blackwell 默认调度 |
| `KernelScheduleSm100Cooperative` | 多 CTA 协作 |
| `KernelScheduleSm100TmaWarpSpecialized1SmMma1x1` | 1SM MMA |
| `KernelScheduleSm100TmaWarpSpecialized2SmMma1x1` | 2SM MMA |

---

## 环境要求

| 要求 | 最低版本 |
|:-----|:---------|
| CUDA Toolkit | 12.8 |
| GPU | NVIDIA B200 / GB200 |
| `CUTLASS_NVCC_ARCHS` | `"100a"` |

```bash
cmake .. -DCUTLASS_NVCC_ARCHS="100a"
```

{: .note }
> 与 Hopper 的 `90a` 类似，Blackwell 的架构加速特性需要使用 `100a`（注意后缀 "a"）。

---

## 示例

CUTLASS 仓库中包含多个 Blackwell 示例：

| 示例 | 说明 |
|:-----|:-----|
| `examples/70_blackwell_gemm/` | 基础 Blackwell GEMM |
| `examples/71_blackwell_gemm_with_collective_builder/` | 使用 Collective Builder |
| Block-Scaled GEMM 示例 | FP4/FP8 + Scale Factor |

---

## 下一步

- [功能列表](../core-concepts/functionality) — 查看 Blackwell 支持的完整操作列表
- [Pipeline 同步](pipeline) — 理解同步原语
