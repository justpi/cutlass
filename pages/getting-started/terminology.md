---
title: 术语表
parent: 入门篇
nav_order: 4
---

# CUTLASS 术语表
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 基础概念

### GEMM
**General Matrix Multiply** — 通用矩阵乘法，计算 `D = alpha * A * B + beta * C`，是 CUTLASS 的核心运算。

### Tensor Core
NVIDIA GPU 中的专用矩阵运算硬件单元，支持 `D = A * B + C` 的混合精度矩阵乘累加运算。从 Volta 架构（SM70）开始引入。

### Epilogue
GEMM 计算完成后的输出处理阶段。可以融合 bias 加法、激活函数（ReLU、GELU 等）、缩放等操作，避免额外的 Kernel 启动和全局内存访问。

### Mainloop
GEMM 的主循环部分，负责在 K 维度上迭代执行矩阵乘累加操作。这是 GEMM Kernel 中计算最密集的部分。

---

## CuTe 术语

### Layout
CuTe 的核心抽象，由 **Shape**（形状）和 **Stride**（步长）组成，定义了从多维逻辑坐标到一维物理索引的映射。

```cpp
// 一个 4x8 的列主序 Layout
auto layout = make_layout(make_shape(4, 8), make_stride(1, 4));
// layout(i, j) = i * 1 + j * 4
```

### Shape
描述每个维度的大小。CuTe 的 Shape 可以是层次化的：

```cpp
auto shape = make_shape(4, make_shape(2, 3));  // 层次化 Shape
```

### Stride
描述每个维度上相邻元素之间的内存偏移量。

### Tensor
将数据指针（或迭代器）与 Layout 组合，形成可索引的多维数组视图。

### Tile
用于对 Layout 或 Tensor 进行分块操作的大小描述符。

### TiledMMA
描述一组线程如何协作执行矩阵乘累加（MMA）操作。由 MMA Atom（单次硬件 MMA 指令的线程-值映射）通过 Tiling 构建而成。

### TiledCopy
描述一组线程如何协作执行数据拷贝操作。由 Copy Atom（单次硬件拷贝指令的映射）通过 Tiling 构建。

### MMA Atom
对单条硬件 MMA 指令的抽象封装，包含该指令所需的线程分布和数据分布信息。

### Copy Atom
对单次数据拷贝操作的抽象封装（如 `cp.async`、TMA 等），描述了参与线程和数据元素的映射关系。

---

## 硬件术语

### SM（Streaming Multiprocessor）
GPU 的基本计算单元。不同架构的 SM 编号：
- SM70 = Volta
- SM75 = Turing
- SM80 = Ampere
- SM89 = Ada
- SM90 = Hopper
- SM100 = Blackwell

### Warp
32 个线程组成的执行单元，是 GPU 调度的基本粒度。

### Warp Group
4 个 Warp（128 个线程）组成的组，Hopper 架构中 WGMMA 指令以 Warp Group 为单位执行。

### Cluster
Hopper 引入的概念，多个 Threadblock 组成的协作组，可以通过分布式共享内存（DSMEM）直接通信。

### TMA（Tensor Memory Accelerator）
Hopper 引入的硬件单元，支持高效的多维数据传输（Global Memory ↔ Shared Memory），无需线程参与地址计算。

### WGMMA（Warp Group Matrix Multiply-Accumulate）
Hopper 引入的矩阵乘累加指令，以 Warp Group（128 线程）为粒度执行，支持直接从共享内存读取操作数。

---

## CUTLASS 架构术语

### Device Level
设备级别，负责 Host-Device 交互、参数传递和 Kernel 启动。对应 `cutlass::gemm::device::` 命名空间。

### Kernel Level
Kernel 级别，负责 Grid 中的 Tile 分配和 Threadblock 调度。

### Collective Level（3.x）
集合级别，CUTLASS 3.x 引入。封装了 Threadblock 内的主循环（Mainloop）和 Epilogue 的协作逻辑。

### Threadblock Level（2.x）
线程块级别，CUTLASS 2.x 的主要抽象层。管理 Shared Memory 中的数据分块和 Warp 间协作。

---

## 计算模式

### Tile 分块
将大矩阵分割为小的 Tile（分块），每个 Threadblock 处理一个或多个 Tile。典型的 Tile 大小如 128×128、256×128 等。

### Software Pipelining（软件流水线）
在主循环中重叠数据加载和计算，用多个缓冲区（Stage）隐藏内存延迟。

### Persistent Kernel（持久化 Kernel）
Kernel 启动后常驻 SM，通过内部调度器动态获取新的工作 Tile，避免重复的 Kernel 启动开销。

### Cooperative / Ping-Pong 调度
多种 Kernel 调度策略：
- **Cooperative**：多个 Threadblock 协作处理同一个 Tile
- **Ping-Pong**：交替使用两组 Warp Group 实现计算和数据加载的重叠

---

## 数据布局

### Row Major（行主序）
矩阵按行存储，同一行的元素在内存中连续。

### Column Major（列主序）
矩阵按列存储，同一列的元素在内存中连续。

### Interleaved（交错布局）
多个矩阵元素按特定模式交错存储，常用于量化场景。

---

## 下一步

- [高效 GEMM 原理](../core-concepts/efficient-gemm) — 理解 GEMM 在 CUDA 上的高效实现
- [CuTe 快速入门](../cute/quickstart) — 深入学习 CuTe 的核心概念
