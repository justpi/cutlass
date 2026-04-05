---
title: GEMM 教程
parent: CuTe 教程
nav_order: 7
---

# CuTe GEMM 教程
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

本教程从零开始，逐步构建一个高性能的 GEMM Kernel，计算 `D = A × B`。我们将经历以下阶段：

1. **朴素版本** — 每线程计算一个元素
2. **Shared Memory 版本** — 使用 Tiling 和 Shared Memory
3. **向量化版本** — 优化内存加载
4. **流水线版本** — 使用 Software Pipelining 隐藏延迟

---

## 第一步：朴素 GEMM

最简单的实现：每个线程从 Global Memory 读取 A 和 B 的完整行/列，计算输出的一个元素。

```cpp
template <class TA, class TB, class TC>
__global__ void gemm_naive(
    int M, int N, int K,
    TA const* A, int ldA,   // M×K, 列主序
    TB const* B, int ldB,   // K×N, 列主序
    TC* C, int ldC) {       // M×N, 列主序

  int ix = blockIdx.x * blockDim.x + threadIdx.x;  // M 维度
  int iy = blockIdx.y * blockDim.y + threadIdx.y;  // N 维度

  if (ix < M && iy < N) {
    TC sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[ix + k * ldA] * B[k + iy * ldB];
    }
    C[ix + iy * ldC] = sum;
  }
}
```

**问题**：
- 每个元素重复从 Global Memory 读取 A 和 B
- 没有数据重用
- 内存带宽严重受限

---

## 第二步：Shared Memory Tiling

用 CuTe 实现分块 GEMM：

```cpp
template <class TA, class TB, class TC,
          int BLK_M, int BLK_N, int BLK_K>
__global__ void gemm_shared(
    int M, int N, int K,
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC* C, int ldC) {

  using namespace cute;

  // 创建全局 Tensor
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(1, ldA));
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(1, ldB));
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(1, ldC));

  // 当前 Threadblock 的 Tile
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                        make_coord(blockIdx.x, _));  // (BLK_M, BLK_K, k)
  auto gB = local_tile(mB, make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                        make_coord(blockIdx.y, _));  // (BLK_N, BLK_K, k)
  auto gC = local_tile(mC, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),
                        make_coord(blockIdx.x, blockIdx.y));  // (BLK_M, BLK_N)

  // 分配 Shared Memory
  __shared__ TA smemA[BLK_M * BLK_K];
  __shared__ TB smemB[BLK_N * BLK_K];
  auto sA = make_tensor(make_smem_ptr(smemA),
                         make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
  auto sB = make_tensor(make_smem_ptr(smemB),
                         make_shape(Int<BLK_N>{}, Int<BLK_K>{}));

  // 线程分区（用于加载）
  auto tAgA = local_partition(gA, Layout<Shape<_32, _4>>{}, threadIdx.x);
  auto tAsA = local_partition(sA, Layout<Shape<_32, _4>>{}, threadIdx.x);
  auto tBgB = local_partition(gB, Layout<Shape<_32, _4>>{}, threadIdx.x);
  auto tBsB = local_partition(sB, Layout<Shape<_32, _4>>{}, threadIdx.x);

  // 线程分区（用于计算）
  auto tCgC = local_partition(gC, Layout<Shape<_16, _8>>{}, threadIdx.x);
  auto tCrC = make_tensor_like(tCgC);
  clear(tCrC);

  // 主循环：沿 K 方向迭代
  int num_k_tiles = size<2>(gA);
  for (int k = 0; k < num_k_tiles; ++k) {
    // Global → Shared
    copy(tAgA(_, _, k), tAsA);
    copy(tBgB(_, _, k), tBsB);
    __syncthreads();

    // Shared Memory 中的 GEMM
    gemm(sA, sB, tCrC);  // tCrC += sA * sB
    __syncthreads();
  }

  // 写回 Global Memory
  copy(tCrC, tCgC);
}
```

**改进**：数据在 Shared Memory 中被 Threadblock 内所有线程复用。

---

## 第三步：使用 TiledMMA 和 TiledCopy

引入硬件 MMA 指令和优化的拷贝操作：

```cpp
template <class TA, class TB, class TC,
          class TiledMMA, class CopyA, class CopyB,
          class SmemLayoutA, class SmemLayoutB>
__global__ void gemm_tiled(
    int M, int N, int K,
    TA const* A, TB const* B, TC* C,
    TiledMMA mma, CopyA copy_a, CopyB copy_b,
    SmemLayoutA sA_layout, SmemLayoutB sB_layout) {

  using namespace cute;
  // ... 创建 Tensor（同上）

  // 共享内存
  extern __shared__ char smem[];
  auto sA = make_tensor(make_smem_ptr((TA*)smem), sA_layout);
  auto sB = make_tensor(make_smem_ptr((TB*)(smem + cosize(sA_layout) * sizeof(TA))),
                         sB_layout);

  // TiledCopy 分区
  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  auto tAgA = thr_copy_a.partition_S(gA);  // 全局源
  auto tAsA = thr_copy_a.partition_D(sA);  // 共享目标

  // TiledMMA 分区
  auto thr_mma = mma.get_slice(threadIdx.x);
  auto tCsA = thr_mma.partition_A(sA);
  auto tCsB = thr_mma.partition_B(sB);
  auto tCrC = thr_mma.partition_fragment_C(gC);
  clear(tCrC);

  // 主循环
  for (int k = 0; k < num_k_tiles; ++k) {
    // 使用 TiledCopy 加载（可能使用 cp.async）
    copy(copy_a, tAgA(_, _, k), tAsA);
    copy(copy_b, tBgB(_, _, k), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // 使用 TiledMMA 计算（Tensor Core）
    gemm(mma, tCsA, tCsB, tCrC);
    __syncthreads();
  }

  // Epilogue: 写回
  copy(tCrC, tCgC);
}
```

---

## 第四步：Software Pipelining

使用多个 Stage 重叠加载和计算：

```cpp
// 多 Stage 共享内存
constexpr int STAGES = 3;
auto sA = make_tensor(smem_ptr_A, make_shape(BLK_M, BLK_K, Int<STAGES>{}));
auto sB = make_tensor(smem_ptr_B, make_shape(BLK_N, BLK_K, Int<STAGES>{}));

// 预填充前 STAGES-1 个 Stage
for (int s = 0; s < STAGES - 1; ++s) {
    copy(copy_a, tAgA(_, _, s), tAsA(_, _, s));
    copy(copy_b, tBgB(_, _, s), tBsB(_, _, s));
    cp_async_fence();
}

// 主循环
int smem_pipe_read = 0;
int smem_pipe_write = STAGES - 1;

for (int k = 0; k < num_k_tiles; ++k) {
    // 等待当前 Stage 数据就绪
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // 计算当前 Stage
    int read_stage = smem_pipe_read;
    gemm(mma, tCsA(_, _, read_stage), tCsB(_, _, read_stage), tCrC);

    // 异步加载下一个 Stage
    if (k + STAGES - 1 < num_k_tiles) {
        copy(copy_a, tAgA(_, _, k + STAGES - 1), tAsA(_, _, smem_pipe_write));
        copy(copy_b, tBgB(_, _, k + STAGES - 1), tBsB(_, _, smem_pipe_write));
        cp_async_fence();
    }

    // 推进 Stage 指针
    smem_pipe_read = (smem_pipe_read + 1) % STAGES;
    smem_pipe_write = (smem_pipe_write + 1) % STAGES;
}
```

---

## 性能对比总结

| 版本 | 主要优化 | 数据重用层级 |
|:-----|:---------|:-------------|
| 朴素 | 无 | 无 |
| Shared Memory | Threadblock Tiling | Shared Memory |
| TiledMMA | Tensor Core + 向量化加载 | Register + Shared |
| 流水线 | 重叠加载与计算 | 全层级 + 延迟隐藏 |

{: .tip }
> 实际的 CUTLASS Kernel 在此基础上还有更多优化：Swizzle 消除 bank conflict、Epilogue 融合、Persistent Kernel 等。建议参考 `examples/` 中的示例代码。

---

## 参考示例

| 示例 | 说明 |
|:-----|:-----|
| `examples/cute/tutorial/sgemm_1.cu` | CuTe GEMM 基础 |
| `examples/cute/tutorial/sgemm_2.cu` | 带 Shared Memory |
| `examples/cute/tutorial/sgemm_nt_1.cu` | 优化版本 |
| `examples/48_hopper_warp_specialized_gemm/` | Hopper 完整 GEMM |
