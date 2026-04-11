---
title: GEMM API 3.x
parent: 核心概念
nav_order: 2
---

# CUTLASS 3.x GEMM API
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CUTLASS 3.x 基于 CuTe 重新设计了 GEMM API，具有更高的可组合性和可读性。核心思想是将 GEMM 分为 **CollectiveMainloop**（主循环）和 **CollectiveEpilogue**（Epilogue），并通过 **Kernel Layer** 将它们组合。

### 层次结构

```
Device Layer (cutlass::gemm::device::GemmUniversalAdapter)
  └── Kernel Layer (cutlass::gemm::kernel::GemmUniversal)
        ├── CollectiveMainloop  ─── 主循环（A*B 的迭代计算）
        └── CollectiveEpilogue  ─── Epilogue（结果后处理和写回）
```

---

## Device Layer

设备层是用户的主要入口点，负责：
- 参数验证
- Kernel 启动配置（Grid 大小、Shared Memory 大小等）
- 实际的 Kernel 启动

{% raw %}
```cpp
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

// 定义 Kernel 类型
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<_128, _128, _64>,       // Tile Shape (M, N, K)
    CollectiveMainloop,           // 主循环类型
    CollectiveEpilogue            // Epilogue 类型
>;

// 通过 Adapter 获得设备级接口
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// 创建参数
typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},                   // 问题尺寸
    {ptr_A, stride_A, ptr_B, stride_B},  // Mainloop 参数
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}  // Epilogue 参数
};

// 运行
Gemm gemm;
gemm.initialize(args);
gemm.run();
```
{% endraw %}

---

## Kernel Layer

Kernel 层是 CUDA `__global__` 函数的宿主，负责：
1. 从 Params 中恢复参数
2. 确定当前 Threadblock 的工作 Tile
3. 调用 CollectiveMainloop 执行主循环
4. 调用 CollectiveEpilogue 写回结果

### GemmUniversal

`cutlass::gemm::kernel::GemmUniversal` 是 3.x 的核心 Kernel 模板：

```cpp
template <
  class ProblemShape,         // 问题形状 (M, N, K [, L])
  class CollectiveMainloop,   // 主循环策略
  class CollectiveEpilogue,   // Epilogue 策略
  class TileScheduler = void  // Tile 调度器（可选）
>
class GemmUniversal;
```

---

## CollectiveMainloop

CollectiveMainloop 封装了 GEMM 主循环的所有逻辑：从 Global Memory 加载 A、B 的 Tile，存入 Shared Memory，然后执行 MMA 运算。

### 主要模板参数

```cpp
template <
  class ArchTag,             // 目标架构（如 Sm90）
  class OpClass,             // 操作类 (TensorOp, SimtOp)
  class ElementA,            // A 的数据类型
  class LayoutA,             // A 的布局
  class ElementB,            // B 的数据类型
  class LayoutB,             // B 的布局
  class ElementAccumulator,  // 累加器类型
  class TileShape_MNK,       // Tile 形状
  class ClusterShape_MNK,    // Cluster 形状（Hopper）
  class StageCountType,      // 流水线级数
  class KernelSchedule       // Kernel 调度策略
>
struct CollectiveMainloop;
```

### Kernel 调度策略

| 调度策略 | 说明 | 架构 |
|:---------|:-----|:-----|
| `KernelMultistage` | 多级流水线，经典 `cp.async` 加载 | SM80+ |
| `KernelTma` | 使用 TMA 加载数据 | SM90+ |
| `KernelTmaWarpSpecialized` | TMA + Warp 特化（Producer/Consumer） | SM90+ |
| `KernelTmaWarpSpecializedCooperative` | TMA + 多 CTA 协作 | SM90+ |
| `KernelTmaWarpSpecializedPingpong` | TMA + Ping-Pong 调度 | SM90+ |

---

## CollectiveEpilogue

CollectiveEpilogue 负责将累加器中的结果写回全局内存，同时可以融合额外的计算：

```cpp
template <
  class StrideC,             // C 矩阵的 Stride
  class StrideD,             // D 矩阵的 Stride
  class ThreadEpilogueOp,    // 逐元素操作
  class EpilogueSchedule     // Epilogue 调度策略
>
struct CollectiveEpilogue;
```

### Epilogue 融合示例

```cpp
// 线性组合：D = alpha * Acc + beta * C
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    float,    // 输出类型
    128 / cutlass::sizeof_bits<float>::value,  // 向量宽度
    float,    // 累加器类型
    float     // 计算类型
>;
```

---

## 完整示例

以 Hopper TMA GEMM 为例：

```cpp
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>

using namespace cute;

// 类型定义
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;

// 使用 Builder 自动选择最佳实现
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,           // 目标架构
    cutlass::arch::OpClassTensorOp, // Tensor Core 操作
    ElementA, cutlass::layout::RowMajor,    // A: FP16, 行主序
    8,                              // A 的对齐要求
    ElementB, cutlass::layout::ColumnMajor, // B: FP16, 列主序
    8,                              // B 的对齐要求
    ElementAccumulator,             // FP32 累加器
    Shape<_128, _128, _64>,         // Tile 形状
    Shape<_1, _1, _1>,             // Cluster 形状
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<...>::SharedStorage)
    >,
    cutlass::gemm::KernelTmaWarpSpecialized  // 调度策略
>::CollectiveOp;

// 组装 Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

{: .tip }
> `CollectiveBuilder` 是构建 CUTLASS 3.x GEMM 的推荐方式，它会根据架构、数据类型和调度策略自动选择最优实现。

---

## Stride 与 CuTe Layout

3.x API 使用 CuTe 的 Stride 来描述矩阵布局，而非 2.x 的 Layout tag：

```cpp
// 行主序 M×K 矩阵：stride = (K, 1)
auto stride_A = make_stride(K, Int<1>{});

// 列主序 K×N 矩阵：stride = (1, K)
auto stride_B = make_stride(Int<1>{}, K);
```

这比 2.x 的 `cutlass::layout::RowMajor` 更灵活，可以表示任意的步长模式。

---

## Batch GEMM

3.x 原生支持 Batched GEMM，通过在问题形状中添加 L 维度：

{% raw %}
```cpp
typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kBatch,
    {M, N, K, batch_count},         // (M, N, K, L)
    {ptr_A, stride_A, ptr_B, stride_B},
    {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}
};
```
{% endraw %}

---

## 下一步

- [GEMM API 2.x](gemm-api-2x) — 了解经典 API（仍广泛使用）
- [隐式 GEMM 卷积](convolution) — 基于 GEMM 实现卷积
- [CuTe Layout](../cute/layout) — 深入理解布局抽象
