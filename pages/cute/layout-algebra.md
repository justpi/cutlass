---
title: Layout 代数
parent: CuTe 教程
nav_order: 3
---

# CuTe Layout 代数
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CuTe 提供了一套 **Layout 代数** — 一组用于组合和变换 Layout 的数学运算。这些运算是 CuTe 实现 Tiling、Partitioning 等高级功能的基础。

---

## 函数组合（Composition）

Layout 的**函数组合**将两个 Layout 串联：

```
(A ∘ B)(x) = A(B(x))
```

```cpp
auto a = make_layout(make_shape(4, 8), make_stride(1, 4));   // 4×8 列主序
auto b = make_layout(make_shape(2, 2), make_stride(1, 8));   // 选取特定元素

auto c = composition(a, b);
// c(i, j) = a(b(i, j))
```

**用途**：从一个大的 Layout 中选取子集。例如，给定一个 128×128 的矩阵 Layout，组合一个线程映射 Layout，得到每个线程应该访问的元素。

---

## 补集（Complement）

**补集**找到与给定 Layout"互补"的 Layout，使得两者合在一起恰好覆盖整个空间：

```cpp
auto a = make_layout(make_shape(4), make_stride(2));
// a 覆盖索引: 0, 2, 4, 6

auto b = complement(a, 16);
// b 覆盖"剩余"的索引: 1, 3, 5, 7, 8, 10, 12, 14
```

**用途**：已知线程布局（哪些线程访问哪些位置），计算出值布局（每个线程需要处理哪些值）。

---

## 逻辑乘积（Logical Product）

将两个 Layout 组合成一个更大的 Layout：

```cpp
auto a = make_layout(make_shape(2, 3));     // 2×3 基础 Layout
auto b = make_layout(make_shape(4, 5));     // 4×5 扩展因子

auto c = logical_product(a, b);
// 结果 Shape: ((2, 3), (4, 5))
// a 的每个位置被 b 复制/扩展
```

---

## 分块乘积（Blocked Product）

用于创建"分块"模式：

```cpp
auto atom = make_layout(make_shape(2, 4));       // 2×4 的原子块
auto tiler = make_layout(make_shape(3, 2));      // 3×2 的重复次数

auto blocked = blocked_product(atom, tiler);
// 结果: 6×8 的矩阵，内部按 2×4 块组织
```

**用途**：从 MMA Atom 构建 TiledMMA。一个 MMA Atom 描述了单条 MMA 指令的线程-值映射，通过 blocked product 将其扩展为更大的 Tile 操作。

---

## Raked Product

用于创建"交错"模式（而非分块）：

```cpp
auto atom = make_layout(make_shape(2, 4));
auto tiler = make_layout(make_shape(3, 2));

auto raked = raked_product(atom, tiler);
// 结果: 6×8 的矩阵，但原子块是交错排列而非连续排列
```

**区别**：
- `blocked_product`: 每个原子块占据一片连续区域
- `raked_product`: 每个原子块的元素均匀分布在整个区域

---

## Logical Divide（逻辑分割）

将 Layout 按 Tiler 分割为"内部"和"外部"两部分：

```cpp
auto layout = make_layout(make_shape(16, 32));
auto tiler = make_shape(4, 8);

auto [inner, outer] = logical_divide(layout, tiler);
// inner: 4×8 — 每个 Tile 内部的坐标
// outer: 4×4 — Tile 的网格坐标
```

**用途**：将矩阵分成多个 Tile，inner 描述 Tile 内的元素，outer 描述 Tile 的位置。

---

## Zipped Divide

类似 logical_divide，但将结果组织为 `(Tile, RestM, RestN)` 的形式：

```cpp
auto layout = make_layout(make_shape(128, 128));
auto result = zipped_divide(layout, make_shape(32, 32));
// 结果形状: ((32, 32), (4, 4))
//            ^Tile 内    ^Tile 间
```

---

## 实际应用示例

### 构建线程到数据的映射

```cpp
// 目标：32 个线程各负责读取 4 个 FP16 元素（128-bit 加载）
// 矩阵是 16×8 的列主序

auto data_layout = make_layout(make_shape(16, 8), make_stride(1, 16));  // 数据布局
auto thread_layout = make_layout(make_shape(16, 2));  // 线程排列：16行×2列
auto value_layout = make_layout(make_shape(1, 4));    // 每线程读 1×4 个值

// 通过 blocked_product 组合
// 最终：16×2 个线程，每个读 1×4 个元素，覆盖 16×8 的矩阵
```

### Tiling 一个 GEMM

```cpp
// M=128, N=128, K=64 的问题
auto layout_A = make_layout(make_shape(128, 64));    // A: 128×64
auto layout_B = make_layout(make_shape(128, 64));    // B: 128×64

// 按 Threadblock Tile 分块
auto tiler_A = make_shape(64, 32);   // 每次处理 64×32
auto tiler_B = make_shape(64, 32);   // 每次处理 64×32

auto [tile_A, rest_A] = logical_divide(layout_A, tiler_A);
// tile_A: 64×32, rest_A: 2×2 (两个 M 块, 两个 K 块)
```

---

## 总结

| 运算 | 功能 | 典型用途 |
|:-----|:-----|:---------|
| `composition` | 函数组合 A(B(x)) | 从大 Layout 中选取子集 |
| `complement` | 互补 Layout | 计算值 Layout |
| `logical_product` | 逻辑扩展 | 从 Atom 构建 Tiled 操作 |
| `blocked_product` | 分块扩展 | 构建 TiledMMA/TiledCopy |
| `raked_product` | 交错扩展 | 交错的数据分配 |
| `logical_divide` | 分块分割 | Tiling 操作 |
| `zipped_divide` | 打包分割 | Tile 迭代 |

---

## 下一步

- [Tensor](tensor) — 将 Layout 与数据结合
- [算法](algorithms) — 了解 copy、gemm 等操作
