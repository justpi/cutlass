---
title: Profiler 工具
parent: 进阶主题
nav_order: 3
---

# CUTLASS Profiler
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 概述

CUTLASS Profiler 是一个命令行驱动的性能分析工具，用于：
- 枚举所有可用的 CUTLASS Kernel
- 对特定问题尺寸进行性能测试
- 与 cuBLAS 进行性能对比
- 验证计算结果的正确性

---

## 编译

```bash
cd build
make cutlass_profiler -j
```

编译后的可执行文件位于 `./tools/profiler/cutlass_profiler`。

---

## 基本用法

### 枚举可用 Kernel

```bash
# 列出所有 GEMM Kernel
./tools/profiler/cutlass_profiler --operation=gemm --enumerate

# 按数据类型过滤
./tools/profiler/cutlass_profiler --operation=gemm --enumerate \
  --A=f16 --B=f16 --C=f32

# 按架构过滤
./tools/profiler/cutlass_profiler --operation=gemm --enumerate \
  --op_class=tensorop --arch=sm80
```

### 运行性能测试

```bash
# 单个问题尺寸
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=4096 --n=4096 --k=4096

# 多个问题尺寸
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=1024,2048,4096 --n=1024,2048,4096 --k=1024

# 范围扫描
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=1024:8192:1024 --n=1024:8192:1024 --k=4096
```

### 指定数据类型

```bash
# FP16 输入，FP32 累加器
./tools/profiler/cutlass_profiler --operation=gemm \
  --A=f16:column --B=f16:row --C=f32 \
  --accumulator-type=f32 \
  --m=4096 --n=4096 --k=4096
```

---

## 卷积性能测试

```bash
# 2D ���积前向传播
./tools/profiler/cutlass_profiler --operation=conv2d_fprop \
  --n=1 --h=224 --w=224 --c=3 \
  --k=64 --r=7 --s=7 \
  --pad_h=3 --pad_w=3 \
  --stride_h=2 --stride_w=2

# Dgrad
./tools/profiler/cutlass_profiler --operation=conv2d_dgrad \
  --n=1 --h=56 --w=56 --c=64 \
  --k=64 --r=3 --s=3
```

---

## 输出格式

默认输出包含：

| 字段 | 说明 |
|:-----|:-----|
| **Operation** | Kernel 名称 |
| **Status** | 运行状态（Success/Failed） |
| **Verification** | 正确性验证结果 |
| **Runtime (ms)** | 执行时间 |
| **GFLOPs** | 算力利用率 |
| **GB/s** | 带宽利用率 |

### CSV 输出

```bash
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=4096 --n=4096 --k=4096 \
  --output=results.csv
```

---

## 与 cuBLAS 对比

```bash
# 同时运行 CUTLASS 和 cuBLAS
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=4096 --n=4096 --k=4096 \
  --providers=cutlass,cublas
```

---

## 常用选项

| 选项 | 说明 | 示例 |
|:-----|:-----|:-----|
| `--operation` | 操作类型 | `gemm`, `conv2d_fprop`, `conv2d_dgrad`, `conv2d_wgrad` |
| `--A/--B/--C` | 数据类型和布局 | `f16:column`, `f32:row` |
| `--op_class` | 操作类 | `tensorop`, `simt` |
| `--arch` | 目标架构 | `sm80`, `sm90` |
| `--warmup-iterations` | 预热次数 | 默认 10 |
| `--profiling-iterations` | 测试次数 | 默认 100 |
| `--verification-enabled` | 启用正确性验证 | `true`/`false` |
| `--providers` | 对比提供者 | `cutlass`, `cublas` |
| `--output` | 输出文件 | `results.csv` |
| `--append` | 追加到输出文件 | |
| `--sort` | 结果排序字段 | `gflops` |

---

## Kernel 过滤

```bash
# 按名称过滤
./tools/profiler/cutlass_profiler --operation=gemm \
  --kernels=cutlass_tensorop_f16_s16816gemm_*

# 排除特定 Kernel
./tools/profiler/cutlass_profiler --operation=gemm \
  --excluded-kernels=*simt*
```

---

## 下一步

- [Blackwell 支持](blackwell) — SM100 架构新特性
- [编程规范](programming-guidelines) — CUTLASS 代码规范
