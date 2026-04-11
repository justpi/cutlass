---
title: CUTLASS 概述
parent: 入门篇
nav_order: 1
---

# CUTLASS 概述
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 什么是 CUTLASS

CUTLASS（**C**UDA **T**emplates for **L**inear **A**lgebra **S**ubroutines and **S**olvers）是 NVIDIA 推出的开源 CUDA C++ 模板抽象库，专门用于实现高性能的矩阵乘法（GEMM）及其相关计算。

CUTLASS 在每个层级和规模上实现了与 cuBLAS、cuDNN 相当的性能，同时将这些"运动部件"分解为可复用的模块化软件组件，通过 C++ 模板类进行抽象。

### 为什么需要 CUTLASS？

| | cuBLAS | CUTLASS |
|:--|:-------|:--------|
| **性能** | 极致优化 | 接近 cuBLAS |
| **灵活性** | 固定 API | 高度可定制 |
| **融合能力** | 有限 | 支持任意 Epilogue 融合 |
| **数据类型** | 预设类型 | 任意自定义类型 |
| **可理解性** | 闭源 | 开源 C++ 模板 |

cuBLAS 提供的是"黑盒"接口——你只能调用它预定义的功能。而 CUTLASS 将矩阵运算分解为可组合的构建块，让你可以：

- **自定义 Epilogue**：在 GEMM 输出阶段融合 bias、激活函数、归一化等操作
- **混合精度**：灵活组合输入/输出/累加器的精度
- **定制化 Kernel**：根据你的特定问题尺寸和硬件定制最优 Kernel

## 核心设计理念

### 分层分解（Hierarchical Decomposition）

CUTLASS 将 GEMM 操作按照 GPU 的并行层次进行分解：

```
Device Level（设备层）
  └── Kernel Level（Kernel 层）
        └── Collective Level（集合层）—— CUTLASS 3.x
              └── Warp Level（Warp 层）
                    └── Thread Level（线程层）
```

每一层负责不同粒度的计算和数据搬运：

| 层级 | 职责 | 数据存储 |
|:-----|:-----|:---------|
| **Device** | 启动 Kernel，管理 Host/Device 交互 | Host Memory |
| **Kernel** | Grid 级调度、Tile 分配 | Global Memory |
| **Collective** | Threadblock 间协调（3.x 新增） | Shared Memory |
| **Warp** | Warp 级 MMA 指令调度 | Register File |
| **Thread** | 单线程标量运算 | Register |

### CuTe —— 3.0 的核心抽象

CUTLASS 3.0 引入了 **CuTe**（CUDA Templates for Tensor Expressions），这是一套全新的 C++ CUDA 模板抽象，用于描述和操作线程与数据的层次化多维布局。

CuTe 的核心概念：
- **Layout**：描述多维坐标到一维索引的映射关系
- **Tensor**：将数据指针与 Layout 组合，表示多维数组
- **TiledMMA / TiledCopy**：描述线程组如何协作执行 MMA 或数据拷贝

CuTe 让程序员可以专注于算法的逻辑描述，而繁琐的索引计算由 CuTe 自动处理。

## CUTLASS 2.x vs 3.x

| 特性 | CUTLASS 2.x | CUTLASS 3.x |
|:-----|:------------|:------------|
| **核心抽象** | 自定义 Tile Iterator | CuTe Layout/Tensor |
| **层次模型** | Threadblock → Warp → Thread | Collective → 基于 CuTe 的统一模型 |
| **代码可读性** | 较复杂 | 大幅简化 |
| **可组合性** | 中等 | 高度可组合 |
| **Hopper 支持** | 有限 | 原生支持 TMA、WGMMA |
| **Blackwell 支持** | 不支持 | 原生支持 |

{: .note }
> CUTLASS 3.x 和 2.x 的代码可以在同一个项目中共存。新项目建议使用 3.x API。

## 混合精度支持

CUTLASS 支持广泛的数据类型组合：

| 数据类型 | 位宽 | 说明 |
|:---------|:-----|:-----|
| `double` | 64 bit | 双精度浮点 |
| `float` | 32 bit | 单精度浮点 |
| `tfloat32_t` | 32 bit | Tensor Float 32（Ampere+） |
| `bfloat16_t` | 16 bit | Brain Floating Point |
| `half_t` | 16 bit | IEEE 半精度浮点 |
| `float_e4m3_t` / `float_e5m2_t` | 8 bit | FP8 格式（Ada/Hopper+） |
| `int8_t` | 8 bit | 8 位整数 |
| `int4b_t` | 4 bit | 4 位整数 |
| `uint1b_t` | 1 bit | 二值（Binary） |

## 应用场景

CUTLASS 广泛应用于：

1. **深度学习框架**：PyTorch、TensorFlow 等框架的底层 GEMM 加速
2. **大模型推理**：LLM 推理中的矩阵运算优化
3. **自定义算子开发**：需要融合多步计算的自定义 CUDA Kernel
4. **科学计算**：高精度矩阵运算
5. **研究与教学**：理解 GPU 矩阵运算的实现原理

## 下一步

- [快速开始](quickstart) — 搭建编译环境，运行第一个 CUTLASS 示例
- [代码组织](code-organization) — 了解仓库目录结构
- [术语表](terminology) — 熟悉 CUTLASS 中的关键术语
