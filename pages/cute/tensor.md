---
title: Tensor
parent: CuTe 教程
nav_order: 4
---

# CuTe Tensor
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 什么是 Tensor

CuTe 的 `Tensor` 是一个轻量级容器，将**数据**（指针或值数组）与 **Layout** 组合，形成可通过多维坐标索引的视图。

```
Tensor = Engine（数据） + Layout（索引映射）
```

Tensor 本身不拥有数据，而是一个"视图"（View），类似于 C++20 的 `std::mdspan`。

---

## 创建 Tensor

### 从指针创建

```cpp
using namespace cute;

// 全局内存 Tensor
float* gmem_ptr = ...;
auto gmem_tensor = make_tensor(
    make_gmem_ptr(gmem_ptr),
    make_layout(make_shape(M, N), make_stride(N, 1))  // M×N 行主序
);

// 共享内存 Tensor
__shared__ float smem_data[128 * 32];
auto smem_tensor = make_tensor(
    make_smem_ptr(smem_data),
    make_layout(make_shape(128, 32))  // 128×32 列主序
);
```

### 从寄存器创建

```cpp
// 创建寄存器中的 Tensor（用于累加器等）
auto reg_tensor = make_tensor<float>(make_shape(8, 4));
// 实际分配在寄存器中（编译器会将小数组分配到寄存器）

// 或使用 make_fragment
auto frag = make_fragment_like(some_other_tensor);
```

### 从已有 Tensor 派生

```cpp
// 重新解释 Layout
auto tensor_b = make_tensor(tensor_a.data(), new_layout);

// 重新解释数据类型
auto tensor_half = recast<half_t>(tensor_float);
```

---

## 索引和切片

### 多维索引

```cpp
auto tensor = make_tensor(ptr, make_layout(make_shape(4, 8, 2)));

// 直接索引
float val = tensor(2, 3, 1);

// 坐标索引
float val = tensor(make_coord(2, 3, 1));
```

### 切片

使用 `_`（全选）进行切片：

```cpp
auto tensor = make_tensor(ptr, make_shape(M, N, K));

// 切片操作，返回新的 Tensor 视图
auto slice_k0 = tensor(_, _, 0);     // M×N 切片（K=0）
auto row_2 = tensor(2, _, _);        // N×K 切片（M=2）
auto col_3 = tensor(_, 3, _);        // M×K 切片（N=3）
auto elem = tensor(1, 2, 3);         // 单个元素
```

### 层次化索引

对于层次化 Layout 的 Tensor：

```cpp
auto tensor = make_tensor(ptr,
    make_layout(make_shape(make_shape(2, 4), 8)));

// 层次化索引
tensor(make_coord(1, 2), 3);  // 内层 (1, 2)，外层 3

// 展平索引
tensor(5, 3);  // 等价（5 = 1 + 2*2）
```

---

## 分区操作

### local_tile

从全局 Tensor 中提取当前 Threadblock 的 Tile：

```cpp
auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K));

// 提取 Threadblock (bx, by) 的 Tile
auto gA = local_tile(
    mA,                      // 源 Tensor
    make_shape(BLK_M, BLK_K), // Tile 大小
    make_coord(bx, _)         // Tile 坐标（M 维度选 bx，K 维度全选）
);
// gA 形状: (BLK_M, BLK_K, num_K_tiles)
```

### local_partition

按线程 ID 对 Tensor 进行分区：

```cpp
auto smem_tensor = make_tensor(smem_ptr, make_shape(128, 32));

// 将 128×32 的 Tensor 按 32 个线程分区
auto thr_tensor = local_partition(
    smem_tensor,
    Layout<Shape<_32, _1>>{},   // 线程布局
    threadIdx.x                  // 当前线程 ID
);
// 每个线程获得 4×32 的子 Tensor
```

### TiledMMA 分区

```cpp
TiledMMA mma = ...;
auto thr_mma = mma.get_slice(threadIdx.x);

// 分区：获取当前线程负责的部分
auto tAgA = thr_mma.partition_A(gA);   // A 的分区
auto tBgB = thr_mma.partition_B(gB);   // B 的分区
auto tCgC = thr_mma.partition_C(gC);   // C 的分区

// 创建累加器 Fragment
auto tCrC = thr_mma.make_fragment_C(tCgC);
clear(tCrC);
```

---

## 常用操作

### 元素操作

```cpp
// 清零
clear(tensor);

// 填充
fill(tensor, 1.0f);

// 逐元素操作
cute::transform(src, dst, [](float x) { return x * 2.0f; });
```

### 拷贝

```cpp
// 基本拷贝
copy(src_tensor, dst_tensor);

// 带 Atom 的拷贝（使用特定硬件指令）
copy(copy_atom, src_tensor, dst_tensor);
```

### 打印

```cpp
// 打印 Tensor 的元数据
print(tensor);        // 打印 Layout 信息
print_tensor(tensor); // 打印所有元素值（小 Tensor）
```

---

## Tensor 的生命周期

### 不同内存空间

| 内存空间 | 创建方式 | 生命周期 | 带宽 |
|:---------|:---------|:---------|:-----|
| Global | `make_gmem_ptr` | 手动管理 | ~2 TB/s |
| Shared | `make_smem_ptr` | Kernel 生命周期 | ~19 TB/s |
| Register | `make_tensor<T>(shape)` | 线程生命周期 | 极高 |

### 典型数据流

```
Global Memory (gmem_tensor)
    ↓ cp.async / TMA
Shared Memory (smem_tensor)
    ↓ ldmatrix / wgmma
Register File  (reg_tensor / fragment)
    ↓ MMA 计算
Register File  (accumulator)
    ↓ store
Global Memory  (output)
```

---

## 实际示例：Shared Memory Tile

```cpp
__global__ void kernel(...) {
  using namespace cute;

  // 在 Shared Memory 中分配
  __shared__ half_t smem_A[128 * 32];
  __shared__ half_t smem_B[32 * 128];

  // 创建 Shared Memory Tensor
  auto sA = make_tensor(make_smem_ptr(smem_A),
                         make_shape(Int<128>{}, Int<32>{}));
  auto sB = make_tensor(make_smem_ptr(smem_B),
                         make_shape(Int<32>{}, Int<128>{}));

  // 创建 Global Memory Tensor 的当前 Tile
  auto gA = local_tile(gmem_A, make_shape(_128{}, _32{}),
                        make_coord(blockIdx.x, _));
  auto gB = local_tile(gmem_B, make_shape(_32{}, _128{}),
                        make_coord(_, blockIdx.y));

  // 分区并拷贝
  auto tAgA = thr_copy.partition_S(gA);  // 源分区
  auto tAsA = thr_copy.partition_D(sA);  // 目标分区

  // Global → Shared 拷贝
  copy(copy_atom, tAgA(_, _, 0), tAsA);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
}
```

---

## 下一步

- [算法](algorithms) — 了解 copy、gemm 等核心算法
- [MMA 指令](mma-atom) — 理解 Tensor Core MMA 的抽象
