---
title: 代码组织
parent: 入门篇
nav_order: 3
---

# 代码组织
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 顶层目录结构

```
cutlass/
├── include/                    # 核心头文件库（Header-only）
│   ├── cutlass/               #   CUTLASS 主库
│   └── cute/                  #   CuTe 布局/张量库
├── examples/                   # 示例程序（66+ 个）
├── test/                       # 测试套件
│   ├── unit/                  #   Google Test 单元测试
│   └── python/                #   Python 测试
├── tools/                      # 工具
│   ├── profiler/              #   性能分析器
│   ├── library/               #   Kernel 库生成
│   └── util/                  #   通用工具
├── python/                     # Python 绑定
│   ├── cutlass/               #   Python CUTLASS 包
│   ├── cutlass_library/       #   库生成脚本
│   └── pycute/                #   PyCuTe
├── media/                      # 文档和图片
│   ├── docs/                  #   Markdown 文档
│   └── images/                #   图片资源
├── docs/                       # Doxygen 生成的 API 文档
├── cmake/                      # CMake 工具模块
├── CMakeLists.txt              # 主构建配置
└── CUDA.cmake                  # CUDA 编译配置
```

---

## include/ — 核心库

这是 CUTLASS 最核心的部分，完全由头文件组成。

### include/cutlass/

```
cutlass/
├── arch/                # 硬件架构抽象
│   ├── mma_sm70.h      #   Volta MMA 指令
│   ├── mma_sm75.h      #   Turing MMA 指令
│   ├── mma_sm80.h      #   Ampere MMA 指令
│   └── mma_sm90.h      #   Hopper WGMMA 指令
├── gemm/                # GEMM 组件
│   ├── device/         #   设备级 GEMM（入口点）
│   ├── kernel/         #   Kernel 级实现
│   ├── collective/     #   集合级操作（3.x）
│   ├── threadblock/    #   线程块级操作（2.x）
│   ├── warp/           #   Warp 级操作
│   └── thread/         #   线程级操作
├── conv/                # 卷积（基于 Implicit GEMM）
│   ├── device/         #   设备级卷积
│   ├── kernel/         #   Kernel 级实现
│   └── threadblock/    #   线程块级操作
├── epilogue/            # Epilogue（输出后处理）
│   ├── collective/     #   集合级 Epilogue（3.x）
│   ├── threadblock/    #   线程块级 Epilogue（2.x）
│   ├── warp/           #   Warp 级操作
│   └── thread/         #   线程级操作
├── layout/              # 内存布局描述符
├── reduction/           # 归约操作
├── transform/           # 数据变换
├── pipeline/            # 执行流水线抽象
├── platform/            # CUDA 标准库组件
├── numeric_types.h      # 数值类型定义
├── array.h              # Array 容器
├── coord.h              # 坐标类型
└── cutlass.h            # 主入口头文件
```

**命名空间对应关系**：目录结构直接映射到 C++ 命名空间。例如 `include/cutlass/gemm/device/gemm.h` 中的类位于 `cutlass::gemm::device` 命名空间。

### include/cute/

CuTe 是 CUTLASS 3.0 引入的核心抽象库：

```
cute/
├── layout.hpp           # Layout 核心类型
├── tensor.hpp           # Tensor 核心类型
├── algorithm/           # 算法（copy, gemm, reduce 等）
├── arch/                # PTX 指令封装
│   ├── copy_sm80.hpp   #   Ampere 异步拷贝
│   ├── copy_sm90_tma.hpp #  Hopper TMA
│   └── mma_sm90.hpp    #   Hopper WGMMA
├── atom/                # 原子操作
│   ├── mma_atom.hpp    #   MMA 原子
│   └── copy_atom.hpp   #   Copy 原子
├── container/           # 容器类型（Tuple 等）
├── numeric/             # 数值类型
├── int_tuple.hpp        # 整数元组
├── stride.hpp           # Stride 类型
└── pointer.hpp          # 指针封装
```

---

## examples/ — 示例程序

按编号组织，涵盖从基础到高级的各种用法：

| 范围 | 主题 |
|:-----|:-----|
| 00-05 | 基础 GEMM、工具使用 |
| 07-11 | Volta/Turing Tensor Core GEMM |
| 12-15 | Epilogue 融合（Bias、ReLU 等） |
| 14-29 | Ampere 特性（TF32、异步拷贝等） |
| 35-56 | CUTLASS 3.x API、CuTe 教程 |
| 57+ | 高级主题（FP8、稀疏、Grouped GEMM 等） |

每个示例包含独立的 `CMakeLists.txt` 和详细注释。

---

## test/ — 测试套件

```
test/
├── unit/                    # 单元测试（Google Test）
│   ├── gemm/               #   GEMM 测试
│   │   ├── device/         #     设备级
│   │   ├── warp/           #     Warp 级
│   │   └── thread/         #     线程级
│   ├── conv/               #   卷积测试
│   ├── epilogue/           #   Epilogue 测试
│   ├── cute/               #   CuTe 测试
│   ├── layout/             #   布局测试
│   └── core/               #   核心类型测试
└── python/                  # Python 测试
    ├── cutlass/            #   CUTLASS Python 接口测试
    └── pycute/             #   PyCuTe 测试
```

---

## tools/ — 工具

### Profiler

`tools/profiler/` 是 CUTLASS 的性能分析工具，可以：
- 枚举所有可用 Kernel
- 对指定问题尺寸进行性能测试
- 对比 cuBLAS 性能
- 输出 CSV 格式的性能数据

### Library

`tools/library/` 包含 Kernel 实例化和注册的基础设施，用于生成预编译的 Kernel 库。

---

## python/ — Python 生态

```
python/
├── cutlass/                 # 高级 Python API
│   ├── op/                 #   GEMM、Conv 等操作符
│   ├── backend/            #   CUDA 代码生成后端
│   ├── emit/               #   代码发射器
│   └── epilogue/           #   Epilogue 配置
├── cutlass_library/         # 库生成和管理
│   ├── manifest.py         #   Kernel 清单
│   └── generator.py        #   代码生成器
└── pycute/                  # PyCuTe：Python 中的 CuTe
    ├── layout.py           #   Layout 操作
    └── tensor.py           #   Tensor 操作
```

---

## 关键文件

| 文件 | 说明 |
|:-----|:-----|
| `include/cutlass/cutlass.h` | 主入口头文件，包含版本信息和基础类型 |
| `include/cutlass/gemm/device/gemm.h` | 设备级 GEMM 模板（2.x） |
| `include/cutlass/gemm/device/gemm_universal_adapter.h` | 通用 GEMM 适配器（3.x） |
| `include/cute/layout.hpp` | CuTe Layout 核心定义 |
| `include/cute/tensor.hpp` | CuTe Tensor 核心定义 |
| `CMakeLists.txt` | 主构建系统配置（~1000 行） |
| `CUDA.cmake` | CUDA 编译器检测和配置 |

---

## 下一步

- [术语表](terminology) — 了解 CUTLASS 中的关键术语
- [高效 GEMM](../core-concepts/efficient-gemm) — 理解 GEMM 的分层实现
