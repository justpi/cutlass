---
title: CuTe 快速入门
parent: CuTe 教程
nav_order: 1
---

# CuTe 快速入门
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 什么是 CuTe

**CuTe**（CUDA Templates for Tensor Expressions）是一套 C++ CUDA 模板抽象，用于定义和操作**层次化多维布局**（Hierarchically Multidimensional Layouts）的线程和数据。

CuTe 提供了两个核心概念：
1. **Layout** — 从逻辑坐标到物理索引的映射函数
2. **Tensor** — 数据指针 + Layout 的组合，表示多维数组

有了这些工具，开发者可以专注于算法的逻辑描述，而让 CuTe 负责复杂的索引计算。

### CuTe 的价值

在 CUTLASS 2.x 中，开发者需要手动管理大量的索引计算：计算线程 ID 到数据偏移的映射、Shared Memory 的 bank conflict 避免、Tensor Core 操作数的布局对齐等。

CuTe 将这些统一为 **Layout 代数**——通过数学化的 Layout 组合和变换，自动处理所有这些细节。

---

## Layout 基础

Layout 由 **Shape**（形状）和 **Stride**（步长）组成：

```cpp
#include <cute/layout.hpp>
using namespace cute;

// 创建一个 4×8 的列主序 Layout
auto layout = make_layout(
    make_shape(4, 8),       // Shape: (4, 8)
    make_stride(1, 4)       // Stride: (1, 4)
);

// 使用：layout(i, j) = i * 1 + j * 4
// layout(0, 0) = 0
// layout(1, 0) = 1
// layout(0, 1) = 4
// layout(3, 7) = 3 + 28 = 31
```

### 行主序 vs 列主序

```cpp
// 列主序 (4, 8):(1, 4) — 同一列元素连续
auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));

// 行主序 (4, 8):(8, 1) — 同一行元素连续
auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
```

### 层次化 Shape

CuTe 的独特之处在于 Shape 和 Stride 可以是**层次化**（嵌套）的：

```cpp
// 层次化 Shape: ((2, 4), 8)
auto layout = make_layout(
    make_shape(make_shape(2, 4), 8),
    make_stride(make_stride(1, 2), 8)
);
// 可以用 layout(make_coord(make_coord(i0, i1), j)) 索引
// 也可以展平后用 layout(k) 索引（0..63）
```

这种层次化结构非常适合描述 GPU 中的线程/数据层次：

```
((Warp内的线程, Warp数), 数据维��)
```

---

## Tensor 基础

Tensor 将数据和 Layout 组合在一起：

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// 从指针创建 Tensor
float* data_ptr = ...;
auto tensor = make_tensor(
    make_gmem_ptr(data_ptr),                    // 全局内存指针
    make_layout(make_shape(M, N), make_stride(N, 1))  // M×N 行主序
);

// 索引访问
tensor(i, j) = 1.0f;  // 等价于 data_ptr[i * N + j] = 1.0f

// 切片
auto row_0 = tensor(0, _);     // 第 0 行（返回一个新 Tensor）
auto col_3 = tensor(_, 3);     // 第 3 列
```

### 内存空间

CuTe 支持不同内存空间的指针：

```cpp
make_gmem_ptr(ptr)   // Global Memory
make_smem_ptr(ptr)   // Shared Memory
make_rmem_ptr(ptr)   // Register（寄存器）
```

---

## 核心操作

### Tiling（分块）

将 Tensor 按 Tile 大小分块：

```cpp
auto tensor = make_tensor(ptr, make_shape(128, 128));

// 将 128×128 的 Tensor 按 32×32 分块
// 结果形状: ((32, 32), (4, 4))
//            ^块内坐标  ^块间坐标
auto tiled = zipped_divide(tensor, make_shape(32, 32));
```

### Partitioning（分区）

将 Tensor 按线程分区，每个线程获得自己负责的部分：

```cpp
// thr_mma 是一个线程的 MMA 划分
auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
auto tA = thr_mma.partition_A(smem_tensor_A);  // 该线程负责的 A 部分
auto tB = thr_mma.partition_B(smem_tensor_B);  // 该线程负责的 B 部分
auto tC = thr_mma.partition_C(acc_tensor);      // 该线程的累加器部分
```

### Copy（拷贝）

```cpp
#include <cute/algorithm/copy.hpp>

// 自动匹配最优拷贝指令
copy(src_tensor, dst_tensor);

// 使用特定拷贝策略
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_32, _4>>{},   // 线程布局
    Layout<Shape<_1, _8>>{}     // 值布局（每线程拷贝的元素数）
);
```

### GEMM

```cpp
#include <cute/algorithm/gemm.hpp>

// MMA 运算
gemm(tiled_mma, tA, tB, tC);
```

---

## 一个完整示例

简单的 GEMM Kernel，使用 CuTe：

```cpp
template <class TA, class TB, class TC,
          class TiledMMA,
          class ALayout, class BLayout, class CLayout>
__global__ void gemm_kernel(
    TA const* A, TB const* B, TC* C,
    TiledMMA mma,
    ALayout layout_A, BLayout layout_B, CLayout layout_C) {

  using namespace cute;

  // 创建全局内存 Tensor
  auto mA = make_tensor(make_gmem_ptr(A), layout_A);  // (M, K)
  auto mB = make_tensor(make_gmem_ptr(B), layout_B);  // (N, K)
  auto mC = make_tensor(make_gmem_ptr(C), layout_C);  // (M, N)

  // 获取当前线程块的 Tile
  auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);
  auto gA = local_tile(mA, tile_shape_A, blk_coord, Step<_1, X, _1>{});
  auto gB = local_tile(mB, tile_shape_B, blk_coord, Step<X, _1, _1>{});
  auto gC = local_tile(mC, tile_shape_C, blk_coord, Step<_1, _1, X>{});

  // 获取当前线程的分区
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

  // 初始化累加器
  auto tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // 主循环：沿 K 维度迭代
  auto K_TILE_MAX = size<2>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    gemm(mma, tAgA(_, _, k_tile), tBgB(_, _, k_tile), tCrC);
  }

  // 写回结果
  copy(tCrC, tCgC);
}
```

---

## 学习路线

建议按以下顺序深入学习 CuTe：

1. **[Layout 详解](layout)** — 理解 Shape、Stride 和坐标映射
2. **[Layout 代数](layout-algebra)** — 学习 Layout 的组合、分割等运算
3. **[Tensor](tensor)** — 掌握 Tensor 的创建、切片和分区
4. **[算法](algorithms)** — 了解 copy、gemm 等核心算法
5. **[MMA 指令](mma-atom)** — 理解 Tensor Core MMA 的抽象
6. **[GEMM 教程](gemm-tutorial)** — 从零手写一个完整的 CuTe GEMM
