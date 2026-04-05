---
title: 功能列表
parent: 核心概念
nav_order: 5
---

# 功能列表
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## GEMM 支持矩阵

### CUTLASS 3.x（推荐）

| 架构 | 数据类型 (A/B → Acc) | 操作类 |
|:-----|:---------------------|:-------|
| **SM90 (Hopper)** | FP16 → FP32, BF16 → FP32, FP8 (e4m3/e5m2) → FP32, TF32 → FP32, INT8 → INT32, FP64 → FP64 | TensorOp (WGMMA) |
| **SM100 (Blackwell)** | 上述所有 + FP4, MXFP8, MXFP6, MXFP4, Block-scaled 格式 | TensorOp (tcgen05.mma) |

### CUTLASS 2.x

| 架构 | 数据类型 (A/B → Acc) | 操作类 |
|:-----|:---------------------|:-------|
| **SM70 (Volta)** | FP16 → FP16/FP32 | TensorOp |
| **SM75 (Turing)** | FP16 → FP32, INT8 → INT32, INT4 → INT32, Binary | TensorOp |
| **SM80 (Ampere)** | FP16, BF16, TF32, FP64, INT8, INT4 | TensorOp |
| **SM89 (Ada)** | 上述 + FP8 (e4m3/e5m2) | TensorOp |
| **SM50-SM80** | FP32, FP64 | Simt |

---

## 卷积支持矩阵

### 2.x API

| 架构 | 算法 | 数据类型 | 布局 |
|:-----|:-----|:---------|:-----|
| SM80 | Fprop, Dgrad, Wgrad | FP16, BF16, TF32, INT8, INT4 | NHWC |
| SM75 | Fprop, Dgrad, Wgrad | FP16, INT8, INT4 | NHWC |
| SM70 | Fprop, Dgrad, Wgrad | FP16 | NHWC |

### 3.x API（SM90+）

| 维度 | 算法 | 数据类型 |
|:-----|:-----|:---------|
| 1D / 2D / 3D | Fprop | FP16, BF16, INT8, FP8 |
| 2D | Dgrad, Wgrad | FP16, BF16 |

---

## Epilogue 操作

| Epilogue | 说明 |
|:---------|:-----|
| `LinearCombination` | D = α·Acc + β·C |
| `LinearCombinationRelu` | D = ReLU(α·Acc + β·C) |
| `LinearCombinationBias` | D = α·Acc + β·C + Bias |
| `LinearCombinationGELU` | D = GELU(α·Acc + β·C) |
| `LinearCombinationSigmoid` | D = Sigmoid(α·Acc + β·C) |
| `LinearCombinationClamp` | D = Clamp(α·Acc + β·C, min, max) |
| EVT (Epilogue Visitor Tree) | 3.x 中的通用融合框架 |

---

## 特殊 GEMM 变体

| 变体 | 说明 | 架构 |
|:-----|:-----|:-----|
| **Grouped GEMM** | 单次 Kernel 执行多个不同尺寸的 GEMM | SM80+ |
| **Sparse GEMM** | 结构化稀疏（2:4）GEMM | SM80+ |
| **Stream-K GEMM** | 沿 K 维度分割以改善负载均衡 | SM80+ |
| **SYR2K / SYMM** | 对称矩阵运算 | SM80+ |
| **TRMM** | 三角矩阵乘法 | SM80+ |
| **Block-Scaled GEMM** | 分块缩放因子（FP4/MX 格式） | SM100 |

---

## 最低 CUDA Toolkit 要求

| 架构 | 最低 CUDA 版本 |
|:-----|:---------------|
| SM70 (Volta) | 11.4 |
| SM75 (Turing) | 11.4 |
| SM80 (Ampere) | 11.4 |
| SM89 (Ada) | 11.8 |
| SM90 (Hopper) | 11.8 |
| SM100 (Blackwell) | 12.8 |

---

## 下一步

- [CUTLASS Profiler](../advanced/profiler) — 测试和对比 Kernel 性能
- [Blackwell 支持](../advanced/blackwell) — SM100 新特性详解
