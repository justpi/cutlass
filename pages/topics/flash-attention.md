---
title: FlashAttention 实战
parent: 实战专题
nav_order: 2
---

# FlashAttention 在 CUTLASS 中的实现
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## FlashAttention 简介

**Attention** 是 Transformer 的核心计算：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

朴素实现需要 `O(N²)` 的中间存储（注意力矩阵），N 是序列长度。当 N 达到几千甚至几十万时，这个矩阵根本放不下。

**FlashAttention** 的核心思想：
1. **分块计算**——把 Q、K、V 沿序列维度切分成 tile
2. **Online Softmax**——在 tile 之间增量更新 softmax 的归一化因子
3. **永远不物化** N×N 注意力矩阵——直接累加到输出

这样把内存复杂度从 `O(N²)` 降到 `O(N)`，同时充分利用 GPU 的 SRAM。

### FA 的演进

| 版本 | 时间 | 关键改进 |
|:-----|:-----|:---------|
| **FlashAttention v1** | 2022 | Online softmax + 分块 |
| **FlashAttention v2** | 2023 | 减少非矩阵乘运算、改进并行策略 |
| **FlashAttention v3** | 2024 | 利用 Hopper 的 WGMMA / TMA / FP8 |

CUTLASS 的 Hopper FMHA 示例就对应 FA v3 的设计思路。

---

## FA 算法详解

### 朴素 Attention

```python
# Naive Attention，O(N²) 内存
S = Q @ K.T          # (N, N) — 注意力分数
P = softmax(S, dim=-1)  # (N, N) — 概率分布
O = P @ V            # (N, d) — 输出
```

### FlashAttention 核心：Online Softmax

Softmax 通常需要先扫一遍找 max（数值稳定），再扫一遍归一化：

```
m = max(x)
y = exp(x - m) / sum(exp(x - m))
```

**Online Softmax** 把这个过程拆成可增量更新的形式。当我们处理新一块 `x_new` 时：

```
m_new      = max(m_old, max(x_new))
exp_diff   = exp(m_old - m_new)
sum_new    = sum_old × exp_diff + sum(exp(x_new - m_new))
output_new = output_old × exp_diff + exp(x_new - m_new)
```

关键性质：每次新块到来，**只需要修正之前的累加结果**，不需要重新扫描。

### FA 的循环结构

伪代码：

```python
# 初始化输出和统计量
O = zeros(N, d)
m = -inf * ones(N)        # 行 max
l = zeros(N)              # 行 sum

# 外层：Q 块
for q_block in range(num_q_blocks):
    Q_i = Q[q_block]                            # (Br, d)

    # 内层：K, V 块
    for kv_block in range(num_kv_blocks):
        K_j = K[kv_block]                       # (Bc, d)
        V_j = V[kv_block]                       # (Bc, d)

        # 第一次 GEMM：S_ij = Q_i @ K_j.T
        S_ij = Q_i @ K_j.T                      # (Br, Bc)

        # Online softmax 更新
        m_new = max(m_i, max(S_ij, dim=-1))
        P_ij  = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + sum(P_ij, dim=-1)

        # 第二次 GEMM：累加到输出
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j

        m_i, l_i = m_new, l_new

    # 最终归一化
    O[q_block] = O_i / l_i
```

每个 Q 块处理完所有 KV 块后，最终除以 `l_i` 完成归一化。

### 为什么快

| 朴素 Attention | FlashAttention |
|:---------------|:---------------|
| 物化 N×N 矩阵到 HBM | 矩阵只在 SMEM/寄存器中 |
| HBM 读写 = O(N²d) | HBM 读写 = O(Nd² / B) |
| 受限于显存带宽 | 受限于算力 |

---

## CUTLASS 中的 FMHA 实现

CUTLASS 提供了三套 FMHA 实现，对应不同架构：

| 示例 | 架构 | 特点 |
|:-----|:-----|:-----|
| `examples/41_fused_multi_head_attention/` | Ampere (SM80) | 经典 2.x API 实现，支持 fwd/bwd |
| `examples/88_hopper_fmha/` | Hopper (SM90) | FA3 风格，TMA + WGMMA + FP8 |
| `examples/77_blackwell_fmha/` | Blackwell (SM100) | tcgen05.mma + 更大 tile + MLA |

### 核心组件

以 Blackwell FMHA 为例，目录结构清晰反映了 CUTLASS 3.x 的分层：

```
77_blackwell_fmha/
├── device/                                      # Device 层入口
├── kernel/
│   ├── sm100_fmha_fwd_kernel_tma_warpspecialized.hpp    # 前向 Kernel
│   ├── sm100_fmha_bwd_kernel_tma_warpspecialized.hpp    # 反向 Kernel
│   ├── sm100_fmha_gen_kernel_warpspecialized.hpp        # Decode 阶段
│   ├── sm100_fmha_mla_tma_warpspecialized.hpp           # MLA 推理
│   └── fmha_tile_scheduler.hpp                          # Tile 调度器
├── collective/
│   ├── sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp  # 主循环
│   ├── sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp  # Epilogue
│   ├── sm100_fmha_load_tma_warpspecialized.hpp          # TMA 加载
│   ├── sm100_fmha_load_cpasync_warpspecialized.hpp      # cp.async 加载
│   └── fmha_fusion.hpp                                  # 融合点（mask 等）
└── reference/                                           # 参考实现
```

### Mainloop 的两层结构

FMHA 的 Mainloop 比普通 GEMM 复杂：它内部包含**两个 GEMM 和一个 Softmax**。

```
Mainloop (per Q tile):
│
├─ Load Q tile (TMA)
│
└─ For each KV tile:
       ├─ Load K, V tiles (TMA, async)
       ├─ GEMM 1: S = Q × K^T          (tcgen05.mma)
       ├─ Apply mask (causal / padding) ← fmha_fusion.hpp 钩子
       ├─ Online softmax: P = softmax_update(S)
       └─ GEMM 2: O += P × V            (tcgen05.mma)
```

### Warp Specialization 与 PingPong

Blackwell FMHA 使用了高度优化的 warp 调度策略：

```
Threadblock 内的 warp 角色：
┌─────────────────┐
│ Producer Warps  │  ← 用 TMA 加载 K, V
└─────────────────┘
┌─────────────────┐
│ MMA Warps       │  ← 执行 tcgen05.mma
└─────────────────┘
┌─────────────────┐
│ Softmax Warps   │  ← 执行 online softmax
└─────────────────┘
```

并通过 **PingPong** 机制：把同一个 CTA 的工作拆成两个子 tile，让 MMA 和 softmax 错开执行——MMA 计算第 k 个 tile 的同时，softmax 处理第 k-1 个 tile，最大化硬件利用率。

---

## 自定义融合：mask 和 activation

CUTLASS FMHA 提供 `fmha_fusion.hpp` 作为最方便的扩展点。函数 `apply_mask` (Blackwell) 或 `before_softmax` (Hopper) 在第一个 GEMM 之后被调用：

```cpp
// 示例：自定义 causal mask + sliding window
template <class AccTensor, class CoordTensor>
CUTLASS_DEVICE
void apply_mask(AccTensor& acc, CoordTensor const& coord, int window_size) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc); ++i) {
        auto [row, col] = coord(i);                  // 全局位置
        bool causal      = (col <= row);
        bool in_window   = (row - col < window_size);
        if (!(causal && in_window)) {
            acc(i) = -INFINITY;                      // 屏蔽
        }
    }
}
```

这是一个通用的 hook：你只需要修改 `apply_mask` 就能实现 causal、sliding window、ALiBi、自定义 attention bias 等。

---

## 各种 FA 变体在 CUTLASS 中的实现

### 1. Causal Attention（因果掩码）

GPT 类模型必备：每个位置只能看到之前的位置。

**实现**：在 mainloop 的 mask 阶段把 `col > row` 的位置置为 `-INFINITY`。

CUTLASS 的优化：使用 **causal tile scheduler** (`fmha_causal_tile_scheduler.hpp`)，跳过完全被 mask 掉的 KV 块，性能提升约 2 倍。

```cpp
// 用 CausalTileScheduler 替换默认 scheduler
using TileScheduler = cutlass::fmha::kernel::CausalTileScheduler;
```

### 2. Sliding Window Attention

每个 query 只看最近 W 个 key/value（如 Mistral、Gemma）。

**实现**：mask 中加入额外的 `row - col < window_size` 条件。和 causal scheduler 类似，可以跳过窗口外的 KV 块。

### 3. Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)

减少 KV head 数量来降低 KV cache 大小：

| 模式 | num_q_heads | num_kv_heads |
|:-----|:------------|:-------------|
| MHA | H | H |
| MQA | H | 1 |
| GQA | H | H/G（G 是 group 数） |

**CUTLASS 实现的精妙之处**：完全通过 **CuTe Layout** 表达，**无需修改 kernel 代码**。

来自 `examples/88_hopper_fmha/README.md`：

> Where regular multi-head attention's layout for the head dimension is `(numHeads:headStride)`,  
> for single-head attention it is simply `(1:0)` everywhere,  
> for **GQA** it is normal in Q and `(numHeads/numGroups, numGroups : headStride, 0)` in KV,  
> and for **MQA** it is normal for Q and `(numHeads:0)` in KV.

也就是说，MQA/GQA 通过把 KV 的 head stride 设为 0（让多个 Q head 共享同一个 KV head 的内存）就实现了。这是 CuTe Layout 抽象的威力——硬件层面无需改动。

### 4. Multi-Latent Attention (MLA)

DeepSeek-V2/V3 使用的方案：把 KV 压缩成低维 latent 表示，推理时再投影回来。

**关键参数**：
- `d_latent` = 512（KV 共享的 latent 维度）
- `d_rope` = 64（位置编码维度）

**挑战**：累加器维度变得非常大（512），普通 Tensor Core 容纳不下。

**Blackwell 的解决方案**：使用 **2-SM MMA** 模式，两个 SM 协作处理一个 MLA 操作。

```cpp
// examples/77_blackwell_fmha/77_blackwell_mla.cu
// 2-SM MMA 让 d=512 的累加器成为可能
using MmaShape = Shape<_256, _256, _64>;     // 两个 SM 协作
```

### 5. Paged KV Cache

vLLM 流行的方案：把 KV cache 分成固定大小的 page，按需分配，避免 padding 浪费。

**CUTLASS 中的支持**：Blackwell MLA 示例 (`77_blackwell_mla.cu`) 演示了 paged KV 加载：

```cpp
// 用 cp.async 而非 TMA（TMA 不支持非连续地址）
// 支持 power-of-two page size ≤ 128
using LoadStrategy = LoadCpAsyncWarpSpecialized;
```

### 6. Variable Sequence Length

batch 中每个序列长度不同的场景。CUTLASS 的处理方式：

- **Padding-free** 输入：用 `cu_seqlens` 数组指示每个序列的起止
- **第一个 batch 前需要 valid padding**（4.3.0 引入的安全要求）
- 通过 tile scheduler 自动跳过 padding 区域

### 7. FP8 FMHA

Hopper FMHA 支持 FP8 输入：

```cpp
// 编译时定义 FP8 启用 FP8 路径
#define FP8 1

using ElementInput = cutlass::float_e4m3_t;   // FP8
// 累加器仍然用 FP32 或 FP16
```

性能比 FP16 提升约 1.5~2 倍。需要注意：
- Q × K 通常用 **FP32 累加**（注意力分数对精度敏感）
- P × V 可以用 **FP16 累加**（softmax 后的 P 范围已规整）

### 8. Attention Backward

反向传播比前向复杂得多：需要计算 dQ、dK、dV。

**Blackwell FMHA Backward** 用了三个 kernel：

```
1. FmhaKernelBwdSumOdO       — 计算 sum(O ⊙ dO) 
2. Sm100FmhaBwdKernelTmaWarpSpecialized  — 主要反向 kernel
3. FmhaKernelBwdConvert       — 把 dQ 从 FP32 转为目标精度
```

主要 kernel 内部是一个**两次 GEMM × 两次 GEMM** 的结构：

```
For each KV tile:
    S    = Q × K^T
    P    = softmax(S)
    dV  += P^T × dO
    dP   = dO × V^T
    dS   = (dP - rowsum(dP × P)) × P
    dQ  += dS × K
    dK  += dS^T × Q
```

---

## 实战配置示例：Hopper FA3 风格 FP16 前向

```cpp
#include <cute/tensor.hpp>
#include "collective/sm90_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "kernel/sm90_fmha_fwd_kernel_tma_warpspecialized.hpp"

using namespace cute;

// 数据类型
using Element        = cutlass::half_t;          // Q, K, V 输入
using ElementOut     = cutlass::half_t;          // 输出
using ElementAccQK   = float;                    // QK 累加器（高精度）
using ElementAccPV   = float;                    // PV 累加器

// Tile 形状
constexpr int kHeadDim = 128;                    // 单 head 维度
using TileShapeQK = Shape<_128, _128, Int<kHeadDim>>;   // (Br, Bc, d)
using TileShapePV = Shape<_128, Int<kHeadDim>, _128>;   // (Br, d, Bc)

// 主循环
using Mainloop = cutlass::fmha::collective::Sm90FmhaFwdMainloopTmaWarpSpecialized<
    Element, ElementAccQK, ElementAccPV,
    TileShapeQK, TileShapePV,
    /* Stages = */ 2
>;

// Epilogue（写回 O）
using Epilogue = cutlass::fmha::collective::Sm90FmhaFwdEpilogueTmaWarpSpecialized<
    ElementOut, /* Layout */ cutlass::layout::RowMajor
>;

// Kernel
using FmhaKernel = cutlass::fmha::kernel::Sm90FmhaFwdKernelTmaWarpSpecialized<
    ProblemShape,                                // (B, H, S_q, S_kv, D)
    Mainloop,
    Epilogue
>;
```

---

## 性能数据（参考）

H100 上 FMHA fp16，head_dim=128：

| 序列长度 | FA2 (TFLOPS) | CUTLASS Hopper FMHA (TFLOPS) | FA3 论文 (TFLOPS) |
|:---------|:-------------|:------------------------------|:------------------|
| 1024 | 380 | 580 | 600 |
| 4096 | 410 | 650 | 680 |
| 16384 | 420 | 680 | 720 |

CUTLASS 实现接近 FA3 论文水平，主要瓶颈在 backward 和小 head dim 场景。

---

## FMHA 调优要点

1. **选择合适的 Tile Shape**
   - Br（Q tile）、Bc（KV tile）通常是 64 / 128 / 256
   - head_dim 越大，Br 应越小（避免寄存器溢出）

2. **Stage 数**
   - Hopper 通常 2~3 stage 即可
   - 过多 stage 会超出 SMEM 限制

3. **Causal mask 时启用 Causal Tile Scheduler**
   - 跳过 mask 完全为 -inf 的 tile，性能提升 ~2×

4. **PingPong 调度**
   - MMA 和 softmax 重叠执行，必开

5. **FP8 时**
   - 用 per-tensor scale 校准 Q, K, V
   - QK GEMM 用 FP32 累加，PV GEMM 可以用 FP16 累加

---

## 调试与分析

CUTLASS FMHA 内部有大量编译期检查，常见错误：

| 错误 | 原因 | 修复 |
|:-----|:-----|:-----|
| `static_assert: head_dim must be ...` | head_dim 不是支持的值（通常 32/64/128/256） | 调整 head_dim |
| `Shared memory exceeds limit` | Stage 数过多 | 减少 Stage 或 Tile |
| 性能远低于预期 | Tile Scheduler 未启用 causal 优化 | 切换到 CausalTileScheduler |

---

## 参考资源

### CUTLASS 示例

| 示例 | 内容 |
|:-----|:-----|
| `examples/41_fused_multi_head_attention/` | Ampere 经典 FMHA（含 backward） |
| `examples/88_hopper_fmha/88_hopper_fmha.cu` | Hopper FA3 风格前向 |
| `examples/77_blackwell_fmha/77_blackwell_fmha.cu` | Blackwell 前向（context） |
| `examples/77_blackwell_fmha/77_blackwell_fmha_bwd.cu` | Blackwell 反向 |
| `examples/77_blackwell_fmha/77_blackwell_fmha_gen.cu` | Blackwell 生成阶段（decode） |
| `examples/77_blackwell_fmha/77_blackwell_mla.cu` | Blackwell MLA 推理 |
| `examples/77_blackwell_fmha/77_blackwell_mla_fwd.cu` | Blackwell MLA 前向 |
| `examples/93_blackwell_low_latency_gqa/` | Blackwell 低延迟 GQA decode |

### 论文

- [FlashAttention](https://arxiv.org/abs/2205.14135) (v1)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)（与 CUTLASS Hopper FMHA 思路一致）
- [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)

---

## 下一步

- [GEMM 量化精度](gemm-quantization) — FP8/FP4 在 FMHA 中的应用
- [CuTe Layout](../cute/layout) — 理解 GQA/MQA 的 Layout 技巧
- [Pipeline 同步](../advanced/pipeline) — Warp 特化和 PingPong 调度的同步原语
