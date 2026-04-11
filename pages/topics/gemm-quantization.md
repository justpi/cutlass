---
title: GEMM 量化精度全解
parent: 实战专题
nav_order: 1
---

# GEMM 量化精度全解
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 为什么需要量化 GEMM

随着大语言模型规模膨胀，**算力**和**显存带宽**成为主要瓶颈。降低数据精度有三大收益：

| 收益维度 | 说明 |
|:---------|:-----|
| **算力 ×2~×16** | Tensor Core 对低精度有更高 throughput（FP8 是 FP16 的 2 倍，FP4 是 FP8 的 2 倍） |
| **带宽减半~减八** | 模型权重、KV cache 体积成比例缩小 |
| **显存容量** | 同样的 GPU 能放下更大的模型 |

代价是**精度损失**和**实现复杂度**：每种低精度都需要量化方案（per-tensor / per-channel / per-block）和反量化逻辑。CUTLASS 的优势是：把这些细节都封装在了 Collective Mainloop / Epilogue 里，让你能用统一的 API 配置。

---

## 数据类型总览

CUTLASS 支持的精度按"位宽"组织：

| 位宽 | 类型 | C++ 类型 | 架构 | 典型用途 |
|:-----|:-----|:---------|:-----|:---------|
| 64 | FP64 | `double` | SM70+ | 科学计算 |
| 32 | FP32 | `float` | SM50+ | 训练（基线） |
| 32 | TF32 | `cutlass::tfloat32_t` | SM80+ | 训练加速 |
| 16 | FP16 | `cutlass::half_t` | SM70+ | 训练 / 推理主力 |
| 16 | BF16 | `cutlass::bfloat16_t` | SM80+ | 训练 / 推理 |
| 8 | FP8 E4M3 | `cutlass::float_e4m3_t` | SM89/90+ | 推理（前向激活） |
| 8 | FP8 E5M2 | `cutlass::float_e5m2_t` | SM89/90+ | 训练反向梯度 |
| 8 | INT8 | `int8_t` | SM75+ | 量化推理 |
| 6 | FP6 E3M2 / E2M3 | `cutlass::float_e3m2_t` 等 | SM100 | 块缩放 |
| 4 | FP4 E2M1 | `cutlass::float_e2m1_t` | SM100 | 极低精度推理 |
| 4 | INT4 | `cutlass::int4b_t` | SM75+ | INT4 权重量化 |
| 1 | Binary | `cutlass::uint1b_t` | SM75+ | 1-bit 网络 |

每种精度组合都对应不同的 Tensor Core 指令，CUTLASS 通过模板自动选择。

---

## 精度配置的核心：四元组 (A, B, Acc, D)

CUTLASS GEMM 的精度由**四个独立的类型**决定：

```cpp
ElementA          // 输入 A 矩阵的类型
ElementB          // 输入 B 矩阵的类型
ElementAccumulator // 累加器类型（中间寄存器）
ElementD          // 输出 D 矩阵的类型
```

**关键原则**：累加器精度通常**高于**输入精度，避免累加溢出。例如 FP16×FP16 累加到 FP32，INT8×INT8 累加到 INT32。

| 输入 A/B | 推荐累加器 | 输出 D | 说明 |
|:---------|:-----------|:-------|:-----|
| FP16 | FP32 | FP16 / FP32 | 训练标配 |
| BF16 | FP32 | BF16 / FP32 | LLM 训练标配 |
| TF32 | FP32 | FP32 | 替代 FP32 训练 |
| FP8 (E4M3) | FP32 | BF16 / FP16 | 推理 |
| INT8 | INT32 | INT8 | 量化推理 |
| FP4 | FP32 | BF16 | 极低精度推理 |

---

## FP16 / BF16 GEMM（基线）

最常用的配置。Hopper 上使用 `KernelTmaWarpSpecialized` 调度策略。

```cpp
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

using namespace cute;

// 1. 类型定义
using ElementA           = cutlass::half_t;       // FP16 输入
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = float;                  // FP32 累加
using ElementD           = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// 2. Tile / Cluster 形状
using TileShape    = Shape<_128, _128, _64>;     // M, N, K
using ClusterShape = Shape<_2, _1, _1>;           // Hopper Cluster

// 3. 主循环（Mainloop）
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,                          // 8 = 128-bit / sizeof(half)
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(int)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

// 4. Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 8,
    ElementD, LayoutC, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

// 5. 组装 Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

把 `cutlass::half_t` 全部换成 `cutlass::bfloat16_t` 即得到 BF16 版本。

{: .tip }
> **对齐参数**：`CollectiveBuilder` 第二个 int 是 `Alignment`（向量化宽度），单位是元素。FP16 用 8 表示一次加载 8 个 half (128-bit)，FP8 应该用 16，FP4 用 32。

---

## TF32 GEMM

TF32 是 Ampere 引入的特殊格式（19 位）：使用 FP32 的指数范围 + FP16 的尾数精度，专供 Tensor Core 使用。**输入和累加都是 FP32**，由硬件内部转换为 TF32 进行运算。

```cpp
using ElementA           = cutlass::tfloat32_t;   // 内部使用 TF32
using ElementB           = cutlass::tfloat32_t;
using ElementAccumulator = float;                  // FP32 累加
using ElementC           = float;
using ElementD           = float;

// 其他配置同 FP16
```

**实战技巧——3xTF32 算法**：通过 3 次 TF32 MMA 模拟出接近 FP32 的精度，速度比纯 FP32 快得多。参考示例 `examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/`。

---

## FP8 GEMM（Hopper / Ada）

FP8 是 LLM 推理的主流格式。CUTLASS 同时支持两种 FP8 编码：

| 类型 | 指数位 | 尾数位 | 范围 | 用途 |
|:-----|:-------|:-------|:-----|:-----|
| `float_e4m3_t` | 4 | 3 | ±448 | 前向激活、权重 |
| `float_e5m2_t` | 5 | 2 | ±57344 | 反向梯度（动态范围更大） |

### 基础 FP8 GEMM

```cpp
using ElementA           = cutlass::float_e4m3_t;  // FP8 E4M3
using ElementB           = cutlass::float_e4m3_t;
using ElementAccumulator = float;                   // 必须 FP32 累加
using ElementC           = cutlass::bfloat16_t;     // 输出反量化为 BF16
using ElementD           = cutlass::bfloat16_t;

using TileShape    = Shape<_128, _128, _128>;       // FP8 K 维度可以更大
using ClusterShape = Shape<_2, _1, _1>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,                          // FP8 对齐 = 16
    ElementB, LayoutB, 16,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(int)>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;
```

参考完整示例：`examples/54_hopper_fp8_warp_specialized_gemm/`

### Per-Tensor / Per-Channel 缩放

FP8 的动态范围有限，必须配合**缩放因子**使用。最简单的是 **per-tensor scaling**：

```
quantized = clip(real / scale, FP8_MAX) → FP8
real      = quantized × scale            → BF16/FP32
```

CUTLASS 支持在 Epilogue 里自动应用 scale：

```cpp
// 缩放参数
struct ScaleArguments {
    ElementAccumulator scale_a;   // A 的反量化系数
    ElementAccumulator scale_b;   // B 的反量化系数
    ElementAccumulator scale_d;   // D 的量化系数（如果输出也是 FP8）
};

// 在 Epilogue 中应用：
// D = scale_d * (scale_a * scale_b * Acc + beta * C)
```

### Per-Block / Blockwise Scaling

更精细的方案：把 A 和 B 划分成小块（如 1×128 或 128×128），每块用一个独立的 scale。这能显著提升精度，避免 outlier 主导。

```cpp
// Hopper Blockwise FP8 GEMM
// 参考: examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    1, 128, 128                                   // ScaleM, ScaleN, ScaleK
>;
```

| 缩放粒度 | 精度 | 实现复杂度 | 推荐场景 |
|:---------|:-----|:-----------|:---------|
| Per-tensor | 低 | 极低 | 推理快速验证 |
| Per-channel | 中 | 低 | 量化推理 |
| Per-block (1×128) | 高 | 中 | 高质量量化推理 |
| Per-block (128×128) | 极高 | 中 | DeepSeek-V3 等 |

### FP8 累加器选择

Hopper FP8 GEMM 支持两种累加：
- **FP32 累加**（默认）：精度高，性能稍低
- **FP16 累加**：性能更高，但有精度损失风险

```cpp
// 启用 FP16 累加（用于注意力分数等容许低精度的场景）
using ElementAccumulator = cutlass::half_t;
```

---

## INT8 GEMM

INT8 推理是较成熟的方案，CUTLASS 从 SM75 (Turing) 开始支持。

```cpp
using ElementA           = int8_t;
using ElementB           = int8_t;
using ElementAccumulator = int32_t;                 // INT32 累加器
using ElementC           = int8_t;
using ElementD           = int8_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;       // INT8 推荐 K-major

// Ampere SM80
using TileShape   = Shape<_128, _128, _64>;
using OpClass     = cutlass::arch::OpClassTensorOp;
```

INT8 的反量化在 Epilogue 完成：

```
int_acc = ΣA*B (int32)
real    = scale × int_acc + bias
quant_d = round(real / scale_d) → int8
```

CUTLASS 提供了组合这些操作的 EpilogueOp：

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    int8_t,                                         // 输出
    8,                                              // 向量宽度
    int32_t,                                        // 累加器
    float                                           // 缩放计算精度
>;
```

---

## INT4 / 混合精度 GEMM

INT4 通常用于**权重量化**：A 是激活（FP16/BF16），B 是 INT4 权重。CUTLASS 提供混合精度 GEMM 来处理这种情况：

```cpp
// 权重 INT4 + 激活 FP16
using ElementA           = cutlass::half_t;         // 激活
using ElementB           = cutlass::int4b_t;        // INT4 权重
using ElementScale       = cutlass::half_t;         // 反量化 scale
using ElementAccumulator = float;
using ElementD           = cutlass::half_t;

// 注意：B 的对齐要求是 64（每次加载 256-bit = 64 个 int4）
```

### Mixed-Dtype 数据流

```
       Global Memory                    Shared Memory                Register
A (FP16) ──── TMA ──────────────────► smem_A (FP16) ────────────► reg_A (FP16)
                                                                      │
B (INT4) ──── TMA ──────────────────► smem_B (INT4)                   │
                                            │                         │
                                            ▼ Convert + Scale         │
                                       smem_B' (FP16) ────────────► reg_B (FP16)
                                                                      │
                                                                      ▼
                                                                  WGMMA → reg_C (FP32)
```

CUTLASS 通过 **mainloop transform** 在加载阶段把 INT4 反量化为 FP16，然后用普通的 FP16 MMA 计算。参考示例：

| 示例 | 组合 |
|:-----|:-----|
| `examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu` | INT4 × BF16 |
| `examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu` | INT4 × FP8 |
| `examples/86_blackwell_mixed_dtype_gemm/` | Blackwell 混合精度 |

### Group-wise 反量化

INT4 通常每 64 或 128 个元素共享一组 (scale, zero-point):

```cpp
// 反量化公式
fp16_value = scale[group_id] * (int4_value - zero_point[group_id])
```

CUTLASS 通过 `MainloopWithDequant` 风格的策略支持这种 group-wise 反量化，scale 和 zero-point 作为额外的 tensor 输入。

---

## FP4 GEMM（Blackwell）

Blackwell 的 `tcgen05.mma` 引入了对 FP4 的原生支持。CUTLASS 提供两种 FP4 变体：

### NVFP4

NVIDIA 私有的 FP4 格式：每 16 个 FP4 共享一个 FP8 (E4M3) 缩放因子。

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ElementD = cutlass::bfloat16_t;

// Blackwell SM100 配置
using ArchTag    = cutlass::arch::Sm100;
using OpClass    = cutlass::arch::OpClassBlockScaledTensorOp;
```

参考示例：`examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu`

### MXFP4

OCP 标准的 MX 格式：每 32 个元素共享一个 UE8M0（8 位无符号指数）缩放因子。

```cpp
using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
```

| 格式 | 元素位宽 | Block 大小 | Scale 类型 | 标准化 |
|:-----|:---------|:-----------|:-----------|:-------|
| **NVFP4** | 4 | 16 | FP8 (E4M3) | NVIDIA |
| **MXFP4** | 4 | 32 | UE8M0 | OCP |
| **MXFP6** | 6 | 32 | UE8M0 | OCP |
| **MXFP8** | 8 | 32 | UE8M0 | OCP |

### Block-Scaled 数据布局

FP4/MX GEMM 需要两个张量：
1. **数据张量**：FP4 元素紧凑存储
2. **Scale Factor 张量**：每 block 一个 scale

```cpp
typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
        ptr_A,         stride_A,           // FP4 数据
        ptr_B,         stride_B,
        ptr_SFA,       stride_SFA,         // A 的 scale factors
        ptr_SFB,       stride_SFB          // B 的 scale factors
    },
    {
        {1.0f, 0.0f},                       // alpha, beta
        ptr_C, stride_C,
        ptr_D, stride_D
    }
};
```

---

## 完整精度对比表

| 配置 | 算力（H100 TFLOPS） | 显存压缩 | 精度损失 | 典型场景 |
|:-----|:--------------------|:---------|:---------|:---------|
| FP32 | 67 | 1× | 0 | 训练基线 |
| TF32 | 495 | 1× | 极低 | 训练加速 |
| FP16/BF16 | 989 | 2× | 几乎无 | 训练 / 推理 |
| FP8 (E4M3) | 1979 | 4× | 小 | 推理主力 |
| INT8 | 1979 | 4× | 中 | 量化推理 |
| FP4 (B200) | 9000+ | 8× | 较大 | 极端推理 |
| INT4 (权重) | ~FP16 | 4× | 中 | 权重量化推理 |

---

## 选型建议

```
开始选型
   │
   ├─ 训练？
   │     ├─ 大规模 LLM → BF16 + FP8 混合
   │     └─ 一般任务 → FP16 / BF16
   │
   └─ 推理？
         ├─ 追求极致吞吐 → FP4 (B200) / FP8 (H100)
         ├─ 追求精度 → BF16
         ├─ 显存受限（权重大）→ INT4 权重量化
         └─ 边缘部署 → INT8
```

---

## 常见陷阱

1. **累加器精度不够**：FP8 累加到 FP16 容易溢出，建议默认 FP32。
2. **对齐参数错误**：FP8 的 Alignment 应该是 16（128-bit / 8-bit），不是 8。
3. **Scale 张量布局**：Block-scaled GEMM 的 scale 布局有特殊要求，必须用 CUTLASS 提供的 helper 计算。
4. **硬件支持不匹配**：INT4 需要 SM75+，FP8 需要 SM89/90+，FP4 需要 SM100。
5. **混合 dtype 的 K 维度**：mixed-dtype GEMM 的 K 维度通常需要是 group size 的倍数。

---

## 参考示例索引

| 精度 | 架构 | 示例路径 |
|:-----|:-----|:---------|
| FP16/BF16 | Hopper | `examples/48_hopper_warp_specialized_gemm/` |
| FP16/BF16 | Hopper Builder | `examples/49_hopper_gemm_with_collective_builder/` |
| FP8 | Hopper | `examples/54_hopper_fp8_warp_specialized_gemm/` |
| FP8 | Ada | `examples/58_ada_fp8_gemm/` |
| FP8 Blockwise | Hopper | `examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/` |
| INT4 + FP16 | Hopper | `examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu` |
| INT4 + FP8 | Hopper | `examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu` |
| FP4 NVFP4 | Blackwell | `examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu` |
| MXFP8 + BF16 | Blackwell | `examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu` |
| Blockwise FP8 | Blackwell | `examples/81_blackwell_gemm_blockwise/` |
| FP4 GEMV | Blackwell | `examples/91_fp4_gemv/` |
| Mixed dtype | Blackwell | `examples/86_blackwell_mixed_dtype_gemm/` |

---

## 下一步

- [FlashAttention 实战](flash-attention) — 在 Attention Kernel 中应用混合精度
- [Blackwell 架构](../advanced/blackwell) — SM100 narrow precision 详解
- [GEMM API 3.x](../core-concepts/gemm-api-3x) — 配置 GEMM 的基础 API
