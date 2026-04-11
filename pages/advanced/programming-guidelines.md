---
title: 编程规范
parent: 进阶主题
nav_order: 1
---

# CUTLASS 编程规范
{: .no_toc }

<details open markdown="block">
  <summary>目录</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 代码格式

{: .warning }
> **禁止使用自动格式化工具**（如 `clang-format`）处理 CUTLASS 代码。CUTLASS 的格式化规则与常见工具的默认行为不同。

### 缩进与行宽

- **2 个空格**缩进，不使用 Tab
- **每行最多 100 个字符**

### 函数声明

函数参数使用**双缩进**（4 个空格），右括号和左花括号在同一行：

```cpp
void possibly_an_unusually_long_function_name(
    std::uint32_t foo,
    std::uint32_t const* bar,
    TypeA a,
    TypeB b) {
  // 函数体使用 2 个空格缩进
}
```

函数调用在左括号后换行，参数使用**单缩进**（2 个空格）：

```cpp
detail::very_long_function_object_name<TemplateArgument>{}(
  params.long_parameter_name, some_operator.another_long_function_name());
```

### 控制流

```cpp
// 总是使用花括号
if (condition) {
  // ...
}
else {                   // else 在新一行
  // ...
}

for (int k = 0; k < num_iters; ++k) {
  // ...
}
```

规则：
- `if`/`for`/`while` 后加空格
- 总是使用花括号，即使只有一行
- `else` 另起一行（不使用 `} else {`）

---

## East Const

CUTLASS 使用 **East const** 风格——`const`/`constexpr` 放在类型**之后**：

```cpp
// ✅ 正确（East const）
float const x = 1.0f;
float const& ref = x;
float const* ptr = &x;
float const* const cptr = &x;
float constexpr kValue = 42.0f;

// ❌ 错误（West const）
const float x = 1.0f;
const float& ref = x;
const float* ptr = &x;
constexpr float kValue = 42.0f;
```

**规则**：`const` 修饰它**左边**的类型。

---

## 指针和引用对齐

使用**左对齐**——`*` 和 `&` 紧贴类型：

```cpp
// ✅ 正确
int const& var;
int const* var;

// ❌ 错误
int const &var;
int const *var;
```

---

## 设计模式

### Params 模式

具有非平凡构造函数的类应定义内部 `struct Params`，存放 Grid 不变的状态：

```cpp
class MyOperator {
public:
  struct Params {
    float const* ptr;
    int stride;

    Params() = default;

    Params(float const* ptr_, int stride_)
      : ptr(ptr_), stride(stride_) { }
  };

  // 从 Params 构造
  CUTLASS_DEVICE
  MyOperator(Params const& params)
    : ptr_(params.ptr), stride_(params.stride) { }

private:
  float const* ptr_;
  int stride_;
};
```

**好处**：
- Params 在 Host 端构造，通过 Constant Memory 传递
- 避免每个线程重复计算初始状态

### SharedStorage 模式

需要 Shared Memory 的类定义 `struct SharedStorage`：

```cpp
class MyMma {
public:
  struct SharedStorage {
    cute::array_aligned<half_t, 128 * 32> smem_A;
    cute::array_aligned<half_t, 32 * 128> smem_B;
  };

  // 非重叠生命周期可以使用 union 节省空间
  union SharedStorage {
    typename Mainloop::SharedStorage mainloop;
    typename Epilogue::SharedStorage epilogue;
  };
};
```

### 循环展开

所有编译时已知迭代次数的循环都应标注 `CUTLASS_PRAGMA_UNROLL`：

```cpp
int const kN = 8;
Array<float, kN> x;

CUTLASS_PRAGMA_UNROLL
for (int i = 0; i < kN; ++i) {
  x[i] = float(i);
}
```

---

## 命名规范

### 避免"fast"或"optimized"

```cpp
// ❌ 不推荐
void compute_fast(...);
void optimized_gemm(...);

// ✅ 推��：描述算法或特征
void compute_on_device(...);
void gemm_multistage(...);
```

### 避免无约束的模板函数使用常见名称

由于 ADL（Argument-Dependent Lookup），在 `cutlass::` 命名空间中定义无约束的模板函数（如 `swap`、`min`）会导致外部代码的编译错误���

```cpp
// ❌ 不要这样做
namespace cutlass {
template<class T>
void swap(T& a, T& b) { ... }  // 会与 std::swap 冲突
}

// ✅ 使用约束或不常见的名称
namespace cutlass {
template<class T>
  requires CutlassType<T>
void swap(T& a, T& b) { ... }
}
```

---

## C++ 惯用法

### 标准 C++ 优先

CUTLASS 是 C++ 项目（CUDA C++ 是 C++ 方言），尽量使用标准 C++ 惯用法：

- 遵循 [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)
- 参考 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Host 代码使用标准 C++
- Device 代码使用尽可能接近标准 C++ 的 CUDA C++

### 平台可移植性

- 避免使用编译器特定扩展（除非性能绝对需要）
- 偏差应该限制在最小范围内
- 性能要求需要有测量数据支持

---

## 下一步

- [Pipeline 同步](pipeline) — 了解流水线同步原语
- [Profiler](profiler) — 使用性能分析工具
