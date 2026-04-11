---
title: 隐式 GEMM 卷积
parent: 核心概念
nav_order: 4
---

# 隐式 GEMM 卷积
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CUTLASS 通过 **隐式 GEMM**（Implicit GEMM）算法实现卷积操作。核心思想是将卷积的数据访问模式映射为矩阵乘法，从而复用 CUTLASS 高度优化的 GEMM 管线。

### 为什么使用 Implicit GEMM？

| 方法 | 说明 | 优缺点 |
|:-----|:-----|:-------|
| **直接卷积** | 按定义逐元素计算 | 简单但效率低 |
| **im2col + GEMM** | 显式展开为矩阵后调用 GEMM | 需要额外内存存储展开后的矩阵 |
| **Implicit GEMM** | 在 GEMM 框架内隐式计算卷积索引 | 无需额外内存，复用 GEMM 优化 |

Implicit GEMM 在加载数据时动态计算卷积的源地址，无需实际展开整个 im2col 矩阵。

---

## 卷积到 GEMM 的映射

### 前向传播（Fprop）

2D 卷积前向传播 `Output = Input * Filter`：

| GEMM 维度 | 卷积含义 |
|:-----------|:---------|
| **M** | N × P × Q（batch × 输出高 × 输出宽） |
| **N** | K（输出通道数） |
| **K** | C × R × S（输入通道 × 滤波器高 × 滤波器宽） |

其中：
- **A 矩阵** ← 输入张量（隐式 im2col 变换）
- **B 矩阵** ← 滤波器权重
- **C/D 矩阵** ← 输出张量

### 反向数据（Dgrad）

计算输入梯度 `dInput = dOutput * Filter^T`：

| GEMM 维度 | 卷积含义 |
|:-----------|:---------|
| **M** | N × H × W（batch × 输入高 × 输入宽） |
| **N** | C（输入通道数） |
| **K** | K × R × S（输出通道 × 滤波器高 × 滤波器宽） |

### 反向权重（Wgrad）

计算权重梯度 `dFilter = Input^T * dOutput`：

| GEMM 维度 | 卷积含义 |
|:-----------|:---------|
| **M** | K（输出通道数） |
| **N** | C × R × S（输入通道 × 滤波器高 × 滤波器宽） |
| **K** | N × P × Q（batch × 输出高 × 输出宽） |

---

## CUTLASS 2.x 卷积 API

```cpp
#include <cutlass/conv/device/implicit_gemm_convolution.h>

// 定义卷积操作
using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInput,                    // 输入类型
        LayoutInput,                     // NHWC
        ElementFilter,                   // 滤波器类型
        LayoutFilter,                    // KRSC
        ElementOutput,                   // 输出类型
        LayoutOutput,                    // NHWC
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages
    >
>;

// 创建参数
cutlass::conv::Conv2dProblemSize problem_size(
    N, H, W, C,    // 输入: batch, height, width, channels
    K, R, S,        // 滤波器: out_channels, height, width
    pad_h, pad_w,   // 填充
    stride_h, stride_w,  // 步长
    dilation_h, dilation_w  // 膨胀
);

Conv2dFprop conv_op;
conv_op(args);
```

---

## CUTLASS 3.x 卷积（Hopper+）

CUTLASS 3.5 引入了基于 CuTe 的 3.x 卷积 API，使用 TMA im2col 进行高效数据搬运：

### 特性

- **Rank-agnostic**：同一套代码支持 1D、2D、3D 卷积
- **TMA im2col**：Hopper 的 TMA 硬件直接支持 im2col 变换
- **CuTe 抽象**：与 3.x GEMM 相同的编程模型

### 支持的算法

| 算法 | 说明 | 架构 |
|:-----|:-----|:-----|
| Fprop | 前向传播 | SM90+ |
| Dgrad | 数据梯度 | SM90+ |
| Wgrad | 权重梯度 | SM90+ |

---

## 数据布局

卷积操作使用特定的张量布局：

| 张量 | 布局 | 维度含义 |
|:-----|:-----|:---------|
| Input（激活） | NHWC | Batch × Height × Width × Channel |
| Filter（权重） | KRSC | OutChannel × Height × Width × InChannel |
| Output | NHWC / NKPQ | Batch × OutChannel × OutHeight × OutWidth |

{: .note }
> CUTLASS 的卷积默认使用 **通道在最后**（channels-last）的布局，即 NHWC，而非 PyTorch 默认的 NCHW。通道在最后的布局更有利于 Tensor Core 的数据访问模式。

---

## 优化 Iterator

Implicit GEMM 的关键在于高效的地址计算。CUTLASS 提供了多种优化的 Tile Iterator：

### 默认 Iterator
对每个输出位置计算完整的 im2col 偏移，通用但有开销。

### 优化 Iterator
- **Few Channels**：针对通道数较少的情况优化
- **Fixed Channels**：通道数编译时已知时可进一步优化
- **Strided Dgrad**：针对步长 > 1 的 Dgrad 优化

### Analytic vs Precomputed

| 方式 | 说明 |
|:-----|:-----|
| **Analytic** | 运行时实时计算 im2col 偏移 |
| **Precomputed** | 预计算偏移表，存储在 Global Memory |

---

## Epilogue 融合

与 GEMM 一样，卷积也支持 Epilogue 融合：

```cpp
// 卷积 + Bias + ReLU 融合
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput, 8, ElementAccumulator, ElementCompute
>;
```

常见融合模式：
- Conv + Bias
- Conv + Bias + ReLU
- Conv + Bias + ReLU + MaxPool（更高级的融合）

---

## 下一步

- [功能列表](functionality) — 查看各架构支持的卷积配置
- [CUTLASS Profiler](../advanced/profiler) — 性能测试卷积操作
