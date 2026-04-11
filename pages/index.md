---
title: 首页
layout: home
nav_order: 1
---

# CUTLASS 中文学习指南
{: .fs-9 }

NVIDIA CUTLASS — 高性能 CUDA C++ 矩阵运算模板库的系统学习文档
{: .fs-6 .fw-300 }

---

## 什么是 CUTLASS？

**CUTLASS**（CUDA Templates for Linear Algebra Subroutines and Solvers）是 NVIDIA 开源的 CUDA C++ 模板库，用于实现高性能的矩阵乘法（GEMM）及相关计算。它提供了与 cuBLAS 和 cuDNN 相当的性能，同时具备高度的灵活性和可组合性。

### 核心特性

| 特性 | 说明 |
|:-----|:-----|
| **分层抽象** | Device → Kernel → Collective → Warp → Thread 多层次分解 |
| **混合精度** | FP64、FP32、TF32、BF16、FP16、FP8、INT8、INT4、Binary |
| **Tensor Core** | 充分利用 Volta ~ Blackwell 架构的 Tensor Core 硬件 |
| **CuTe** | 3.0+ 引入的核心布局抽象库，大幅简化 Tensor 操作 |
| **Header-only** | 核心库仅含头文件，无需编译即可集成 |

### 支持的 GPU 架构

| 架构 | GPU 示例 | Compute Capability |
|:-----|:---------|:-------------------|
| Volta | V100, Titan V | SM 7.0 |
| Turing | RTX 2080, T4 | SM 7.5 |
| Ampere | A100, RTX 3090 | SM 8.0 / 8.6 |
| Ada | L40, RTX 4090 | SM 8.9 |
| Hopper | H100 | SM 9.0 / 9.0a |
| Blackwell | B200 | SM 10.0 / 10.0a |

---

## 文档结构

本学习指南分为四大部分：

### [入门篇](getting-started/overview)
从零开始了解 CUTLASS 的背景、编译环境搭建、代码结构和核心术语。

### [核心概念](core-concepts/efficient-gemm)
深入理解 GEMM 的分层实现原理、CUTLASS 3.x/2.x API 设计，以及卷积操作。

### [CuTe 教程](cute/quickstart)
学习 CUTLASS 3.0 引入的核心抽象库 CuTe，掌握 Layout、Tensor、算法和 MMA 指令。

### [进阶主题](advanced/programming-guidelines)
编程规范、性能分析工具、Pipeline 同步原语、Blackwell 架构支持等高级话题。

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# 配置构建（以 Ampere A100 为例）
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80"

# 编译并运行基础示例
make 00_basic_gemm -j
./examples/00_basic_gemm/00_basic_gemm
```

---

## 版本说明

本文档基于 **CUTLASS v4.4.2**（最新版本）整理，同时包含对 2.x 和 3.x API 的说明。

- [官方 CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- [官方 Doxygen API 文档](https://nvidia.github.io/cutlass)
