---
title: Layout 详解
parent: CuTe 教程
nav_order: 2
---

# CuTe Layout 详解
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Layout 的定义

Layout 是 CuTe 中最核心的概念。一个 Layout 定义了从**逻辑坐标空间**到**一维索引空间**的映射函数：

```
Layout: 坐标 → 索引
```

Layout 由两部分组成：
- **Shape**：定义坐标空间的维度大小
- **Stride**：定义每个维度的步长

```cpp
Layout<Shape, Stride>
```

给定坐标 `(i, j, ...)`，索引计算为：

```
index = i * stride_0 + j * stride_1 + ...
```

---

## 创建 Layout

### 基本创建

```cpp
using namespace cute;

// 方式1：明确指定 Shape 和 Stride
auto layout_a = make_layout(make_shape(4, 8), make_stride(1, 4));

// 方式2：使用编译时常量
auto layout_b = make_layout(make_shape(Int<4>{}, Int<8>{}),
                            make_stride(Int<1>{}, Int<4>{}));

// 方式3：只指定 Shape（默认列主序，即按列紧凑）
auto layout_c = make_layout(make_shape(4, 8));
// 等价于 (4, 8):(1, 4) — 列主序
```

### 编译时 vs 运行时

CuTe 区分编译时已知和运行时才知道的值：

```cpp
// 编译时常量：使用 Int<N>{} 或 _N
auto static_layout = make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _4{}));

// 运行时值：使用普通 int
auto dynamic_layout = make_layout(make_shape(M, N), make_stride(1, M));

// 混合
auto mixed_layout = make_layout(make_shape(_128{}, N), make_stride(_1{}, _128{}));
```

编译时已知的值使得编译器可以进行更多优化（常量折叠、循环展开等）。

---

## 索引操作

### 多维索引

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));

// 二维索引
int idx = layout(2, 3);  // = 2*1 + 3*4 = 14

// 等价的坐标对象
int idx = layout(make_coord(2, 3));  // = 14
```

### 一维索引（展平）

Layout 也接受一维索引，会自动展平：

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));

layout(0)  = 0   // (0,0) → 0*1 + 0*4 = 0
layout(1)  = 1   // (1,0) → 1*1 + 0*4 = 1
layout(4)  = 4   // (0,1) → 0*1 + 1*4 = 4
layout(31) = 31  // (3,7) → 3*1 + 7*4 = 31
```

一维到多维的映射规则：按照 **Shape 的列主序** 展开。即先遍历第一个维度，再遍历第二个。

---

## 常见 Layout 模式

### 列主序（Column Major）

```cpp
// 4×8 列主序：同一列的元素在内存中连续
auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
// 内存排列: [col0: 0,1,2,3] [col1: 4,5,6,7] ... [col7: 28,29,30,31]
```

### 行主序（Row Major）

```cpp
// 4×8 行主序：同一行的元素在内存中连续
auto row_major = make_layout(make_shape(4, 8), make_stride(8, 1));
// 内存排列: [row0: 0,1,...,7] [row1: 8,9,...,15] ...
```

### Stride 为 0（广播）

```cpp
// Stride 为 0 意味着该维度不影响索引 → 广播
auto broadcast = make_layout(make_shape(4, 8), make_stride(1, 0));
// layout(i, j) = i  对任意 j，结果相同
```

---

## 层次化 Layout

CuTe 允许 Shape 和 Stride 本身也是元组，形成嵌套/层次化结构：

```cpp
// 层次化 Shape: ((2, 4), 8)
// 层次化 Stride: ((1, 2), 8)
auto layout = make_layout(
    make_shape(make_shape(2, 4), 8),
    make_stride(make_stride(1, 2), 8)
);

// 索引方式1：完全层次化
layout(make_coord(make_coord(1, 2), 3));  // = 1*1 + 2*2 + 3*8 = 29

// 索引方式2：展平为一维
layout(13);  // = 29（展平后的第 13 个逻辑元素）
```

### 层次化的实际意义

层次化 Layout 用于表达 GPU 的并行层次：

```cpp
// 线程布局：((线程内的值, Warp 内的线程), Warp 数)
// 数据布局：((元素块, 线程块内的偏移), 块数)
```

例如，描述 32 个线程各持有 4 个元素的 Layout：

```cpp
// (线程数, 每线程元素数) = (32, 4)
auto thread_layout = make_layout(make_shape(32, 4));
// thread_layout(tid, vid) 给出该线程的第 vid 个元素的索引
```

---

## Layout 的大小和范围

| 函数 | 说明 |
|:-----|:-----|
| `size(layout)` | 逻辑元素总数（Shape 各维度的乘积） |
| `size<I>(layout)` | 第 I 个维度的大小 |
| `cosize(layout)` | 物理范围（最大索引 + 1） |
| `rank(layout)` | 维度数（顶层） |
| `depth(layout)` | 嵌套深度 |

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(2, 8));
size(layout);    // = 32（4 × 8）
cosize(layout);  // = 64（(4-1)*2 + (8-1)*8 + 1 = 6+56+1 = 63... 实际为 max_index+1）
rank(layout);    // = 2
```

---

## Stride 的含义

Stride 不仅决定了元素间距，还暗含了数据访问模式：

| Stride 模式 | 含义 |
|:-------------|:-----|
| `(1, M)` | 列主序，列内连续 |
| `(N, 1)` | 行主序，行内连续 |
| `(0, 1)` | 第一维广播 |
| `(2, 2*M)` | 交错访问（stride=2 意味着跳一个元素） |

### 紧凑 Layout（Compact Layout）

当 Layout 的所有逻辑元素恰好占据一段连续内存时，称为紧凑 Layout：

```cpp
// 紧凑：(4, 8):(1, 4) — 32 个元素占据 [0, 31]
// 非紧凑：(4, 8):(2, 8) — 32 个元素分散在 [0, 62] 中
```

---

## 打印 Layout

CuTe 提供了方便的打印功能：

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));
print_layout(layout);
```

输出类似：
```
       0    1    2    3    4    5    6    7
    +----+----+----+----+----+----+----+----+
 0  |  0 |  4 |  8 | 12 | 16 | 20 | 24 | 28 |
    +----+----+----+----+----+----+----+----+
 1  |  1 |  5 |  9 | 13 | 17 | 21 | 25 | 29 |
    +----+----+----+----+----+----+----+----+
 2  |  2 |  6 | 10 | 14 | 18 | 22 | 26 | 30 |
    +----+----+----+----+----+----+----+----+
 3  |  3 |  7 | 11 | 15 | 19 | 23 | 27 | 31 |
    +----+----+----+----+----+----+----+----+
```

这对调试非常有用！

---

## 下一步

- [Layout 代数](layout-algebra) — 学习 Layout 的组合、补集、乘积等运算
- [Tensor](tensor) — 将 Layout 与数据组合
