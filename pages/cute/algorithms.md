---
title: 算法
parent: CuTe 教程
nav_order: 5
---

# CuTe 算法
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CuTe 提供了一组作用于 Tensor 的算法原语。这些算法是泛型的，可以基于 Layout 和数据类型自动分派到最优的硬件指令。

---

## copy — 数据拷贝

`copy` 是最常用的算法，支持多种内存空间之间的数据搬运：

### 基本用法

```cpp
#include <cute/algorithm/copy.hpp>

// 自动选择最优拷贝方式
copy(src_tensor, dst_tensor);
```

CuTe 会根据源和目标的内存空间、数据类型和对齐方式自动选择最优的硬件指令。

### Copy Atom

`Copy_Atom` 封装了特定的硬件拷贝指令：

| Copy Atom | 说明 | 架构 |
|:-----------|:-----|:-----|
| `UniversalCopy<uint128_t>` | 128-bit 向量加载 | 所有 |
| `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>` | 异步 Global→Shared 拷贝 | SM80+ |
| `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` | 异步拷贝（绕过 L1） | SM80+ |
| `SM90_TMA_LOAD` | TMA 加载（Global→Shared） | SM90+ |
| `SM90_TMA_STORE` | TMA 存储（Shared→Global） | SM90+ |

### TiledCopy

将 Copy Atom 扩展为多线程协作拷贝：

```cpp
// 创建 TiledCopy：32×4 线程，每线程拷贝 1×8 个元素
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_32, _4>>{},    // 线程布局 (32 行 × 4 列线程)
    Layout<Shape<_1, _8>>{}      // 值布局 (每线程 1×8 个元素)
);

// 使用
auto thr_copy = tiled_copy.get_slice(threadIdx.x);
auto tSgS = thr_copy.partition_S(gmem_src);   // 源分区
auto tDsD = thr_copy.partition_D(smem_dst);   // 目标分区
copy(tiled_copy, tSgS, tDsD);
```

### 异步拷贝（cp.async）

Ampere+ 支持异步的 Global→Shared 拷贝：

```cpp
// 发起异步拷贝
copy(SM80_CP_ASYNC_CACHEALWAYS{}, tSgS, tDsD);

// 提交并等待
cp_async_fence();
cp_async_wait<0>();  // 等待所有完成
__syncthreads();
```

---

## gemm — 矩阵乘累加

CuTe 的 `gemm` 算法执行 `C += A * B`：

### 基本用法

```cpp
#include <cute/algorithm/gemm.hpp>

// 使用 TiledMMA 执行 GEMM
gemm(tiled_mma, tensor_A, tensor_B, tensor_C);
// C += A * B
```

### TiledMMA

`TiledMMA` 描述了一组线程如何协作执行 MMA：

```cpp
// 从 MMA Atom 构建 TiledMMA
using MMA = decltype(make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},   // MMA Atom（SM80 FP16→FP32）
    Layout<Shape<_2, _2, _1>>{},        // 扩展因子：2×2 = 4 个 Atom
));

TiledMMA mma;
auto thr_mma = mma.get_slice(threadIdx.x);
```

### 线程级 GEMM

每个线程在 K 维度上循环调用 MMA：

```cpp
// K 维度迭代
for (int k = 0; k < K_TILES; ++k) {
    gemm(mma, tCrA(_, _, k), tCrB(_, _, k), tCrC);
}
```

---

## axpby — 线性组合

```cpp
#include <cute/algorithm/axpby.hpp>

// Y = alpha * X + beta * Y
axpby(alpha, tensor_X, beta, tensor_Y);
```

---

## fill — 填充

```cpp
#include <cute/algorithm/fill.hpp>

// 将 Tensor 所有元素设为指定值
fill(tensor, 0.0f);

// 清零的快捷方式
clear(tensor);
```

---

## Predication（边界处理）

当矩阵维度不能被 Tile 大小整除时，需要 Predication：

```cpp
// 创建 Predicate Tensor
auto pred = make_tensor<bool>(shape(tCgC));

// 设置边界条件
for (int i = 0; i < size(pred); ++i) {
    auto coord = idx2crd(i, shape(tCgC));
    pred(i) = get<0>(coord) + blockIdx.x * BLK_M < M &&
              get<1>(coord) + blockIdx.y * BLK_N < N;
}

// 带 Predicate 的拷贝
copy_if(pred, src, dst);
```

---

## 算法选择策略

CuTe 的算法根据参数自动选择最优实现：

| 条件 | 选择 |
|:-----|:-----|
| src=gmem, dst=smem, SM80+ | `cp.async` |
| src=gmem, dst=smem, SM90+, TMA descriptor | TMA |
| src=smem, dst=reg, Tensor Core layout | `ldmatrix` |
| 通用场景 | 标量/向量加载+存储 |

---

## 下一步

- [MMA 指令](mma-atom) — 深入理解 Tensor Core MMA Atom
- [GEMM 教程](gemm-tutorial) — 从零构建完整 GEMM Kernel
