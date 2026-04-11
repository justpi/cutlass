---
title: 快速开始
parent: 入门篇
nav_order: 2
---

# 快速开始
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 环境要求

### 必需工具

| 工具 | 最低版本 | 推荐版本 |
|:-----|:---------|:---------|
| CMake | 3.19 | 最新 |
| CUDA Toolkit | 11.4 | 12.4+ |
| C++ 编译器 | C++17 | GCC 9+, Clang 14+ |
| Python（可选） | 3.8 | 3.10+ |

### 支持的操作系统

| 操作系统 | 编译器 |
|:---------|:-------|
| Ubuntu 18.04+ | GCC 7.5+, Clang 10+ |
| Windows 10 | Visual Studio 2019+ |

{: .warning }
> GCC 8.5.0 存在关于折叠表达式和运算符重载的已知问题，建议使用 GCC 7.5 或 GCC 9+。

---

## 编译 CUTLASS

### 1. 克隆仓库

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
```

### 2. 设置 CUDA 编译器

```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
```

### 3. CMake 配置

```bash
mkdir build && cd build

# 基本配置（以 Ampere SM80 为例）
cmake .. -DCUTLASS_NVCC_ARCHS="80"
```

#### 常用 CMake 选项

| 选项 | 说明 | 示例 |
|:-----|:-----|:-----|
| `CUTLASS_NVCC_ARCHS` | 目标 GPU 架构 | `"80"`, `"80;90a"` |
| `CUTLASS_ENABLE_TESTS` | 启用测试目标 | `ON` / `OFF` |
| `CUTLASS_ENABLE_EXAMPLES` | 启用示例目标 | `ON` / `OFF` |
| `CUTLASS_ENABLE_CUBLAS` | 启用 cuBLAS 对比验证 | `ON` / `OFF` |
| `CMAKE_BUILD_TYPE` | 构建类型 | `Release` / `Debug` |

#### 多架构编译

```bash
# 同时编译 Ampere 和 Hopper
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"
```

{: .note }
> Hopper 的架构加速特性需要使用 `90a`（注意后缀 "a"），而非 `90`。如果使用了 SM90a 特性但编译目标设为 SM90，Kernel 运行时会报错。

#### 不同平台的配置

**使用 Clang 作为 Host 编译器**：
```bash
cmake .. \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCUTLASS_NVCC_ARCHS="80"
```

**Windows + Visual Studio**：
```bash
cmake .. -G "Visual Studio 16 2019" -A x64 -DCUTLASS_NVCC_ARCHS="80"
```

### 4. 编译

```bash
# 编译某个示例
make 00_basic_gemm -j

# 编译全部单元测试
make test_unit -j

# 编译性能分析工具
make cutlass_profiler -j
```

---

## 运行示例

### 基础 GEMM 示例

```bash
# 编译
cd build
make 00_basic_gemm -j

# 运行
./examples/00_basic_gemm/00_basic_gemm
```

该示例演示了一个简单的单精度 GEMM（C = alpha * A * B + beta * C）。

### Tensor Core 示例

```bash
# Ampere Tensor Core GEMM（需要 SM80+）
make 14_ampere_tf32_tensorop_gemm -j
./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm
```

---

## 运行单元测试

```bash
cd build

# 编译并运行全部测试
make test_unit -j
ctest

# 运行特定测试
make test_unit_gemm_device -j
./test/unit/gemm/device/test_unit_gemm_device
```

测试使用 Google Test 框架，支持过滤：
```bash
./test/unit/gemm/device/test_unit_gemm_device --gtest_filter="*SM80*"
```

---

## 使用 CUTLASS Profiler

```bash
make cutlass_profiler -j

# 列出可用的 GEMM Kernel
./tools/profiler/cutlass_profiler --operation=gemm --enumerate

# 运行特定尺寸的 GEMM
./tools/profiler/cutlass_profiler --operation=gemm \
  --m=4096 --n=4096 --k=4096
```

---

## 作为 Header-only 库集成

CUTLASS 核心库是纯头文件的，无需编译即可集成到你的项目中：

### CMake 集成

```cmake
# 方式1：add_subdirectory
add_subdirectory(path/to/cutlass)
target_link_libraries(my_target PRIVATE cutlass)

# 方式2：仅包含头文件
target_include_directories(my_target PRIVATE path/to/cutlass/include)
```

### 在 CUDA 代码中使用

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// 定义 GEMM 类型
using Gemm = cutlass::gemm::device::Gemm<
  float,                           // A 矩阵元素类型
  cutlass::layout::ColumnMajor,    // A 矩阵布局
  float,                           // B 矩阵元素类型
  cutlass::layout::ColumnMajor,    // B 矩阵布局
  float,                           // C 矩阵元素类型
  cutlass::layout::ColumnMajor     // C 矩阵布局
>;

// 创建并运行 GEMM
Gemm gemm_op;
cutlass::Status status = gemm_op(args);
```

---

## Python 接口

CUTLASS 也提供 Python 绑定：

```bash
# 安装
pip install nvidia-cutlass

# 或从源码安装
cd cutlass/python
pip install -e .
```

```python
import cutlass

# 创建 GEMM 操作
plan = cutlass.op.Gemm(
    element_A=cutlass.DataType.f16,
    element_B=cutlass.DataType.f16,
    element_C=cutlass.DataType.f16,
    layout_A=cutlass.LayoutType.ColumnMajor,
    layout_B=cutlass.LayoutType.ColumnMajor,
)

plan.run(A, B, C, D)
```

---

## 常见问题

### 编译时间过长

CUTLASS 大量使用 C++ 模板，编译时间可能很长。建议：
- 通过 `CUTLASS_NVCC_ARCHS` 仅编译需要的架构
- 使用 `make -j` 并行编译
- 只编译需要的目标，避免 `make all`

### Kernel 运行时错误

检查以下几点：
1. 编译的架构是否与运行的 GPU 匹配
2. Hopper 特性是否使用了 `90a` 而非 `90`
3. CUDA Toolkit 版本是否满足最低要求

---

## 下一步

- [代码组织](code-organization) — 了解仓库目录结构
- [术语表](terminology) — 熟悉核心术语
- [高效 GEMM 原理](../core-concepts/efficient-gemm) — 理解 GEMM 的分层优化
