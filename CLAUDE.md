# CLAUDE.md - CUTLASS Repository Guide

## Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers) is NVIDIA's open-source, header-only C++ template library for high-performance GEMM and related computations on NVIDIA GPUs. Version 3.5.0, BSD-3-Clause licensed.

Key concepts:
- Hierarchical decomposition of GEMM (device → kernel → threadblock/collective → warp → thread)
- CuTe (CUTLASS 3.x core): C++ CUDA template abstractions for multidimensional layouts of threads and data
- Mixed-precision support: FP16, BF16, TF32, FP32, FP64, INT4, INT8, binary
- Targets Volta (SM70), Turing (SM75), Ampere (SM80/86), Ada (SM89), Hopper (SM90/90a)

## Repository Structure

```
include/cutlass/       # Core header-only library (primary namespace: cutlass::)
include/cute/          # CuTe layout/tensor library (namespace: cute::)
examples/              # 66+ example programs (00_basic_gemm through 66+)
test/unit/             # Google Test unit tests, mirroring include/ hierarchy
test/python/           # Python test suite
python/cutlass/        # Python bindings and code emission
python/cutlass_library/# Library generation scripts
python/pycute/         # PyCute Python bindings
tools/                 # Profiler, utilities, library generation
media/docs/            # Markdown documentation
docs/                  # Doxygen-generated HTML API docs
cmake/                 # CMake utility modules
```

### Key include/ hierarchy

```
cutlass/
  arch/          # Hardware-level abstractions (SM-specific)
  gemm/          # GEMM: device/, kernel/, threadblock/, warp/, thread/, collective/
  conv/          # Convolution via implicit GEMM
  epilogue/      # Output transformations (bias, activation, etc.)
  layout/        # Memory layout descriptors
  reduction/     # Reduction operations
  transform/     # Data transformations
  pipeline/      # Execution pipeline abstractions
  platform/      # CUDA standard library components
cute/
  algorithm/     # Core operations (copy, gemm, reduce)
  arch/          # PTX wrapper structs
  atom/          # MMA atoms and copy atoms
```

## Building

```bash
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80"       # specify target arch(s)
make test_unit -j                          # build and run unit tests
make cutlass_profiler -j                   # build the profiler
```

Requirements:
- CMake >= 3.19
- C++17 compiler (GCC >= 7.3, Clang >= 7.0, MSVC 2019+)
- CUDA Toolkit >= 11.4 (12.4 recommended)
- For Hopper features: use `CUTLASS_NVCC_ARCHS="90a"` (note the "a")

Key CMake options:
- `CUTLASS_NVCC_ARCHS` - target GPU architectures (e.g., "80;90a")
- `CUTLASS_ENABLE_TESTS` - enable test targets
- `CUTLASS_ENABLE_EXAMPLES` - enable example targets

## Testing

- Framework: Google Test (gtest)
- Test location: `test/unit/` organized by component (gemm/, conv/, cute/, epilogue/, etc.)
- Python tests: `test/python/` (PyCute, GEMM, Conv2D, EVT, PyTorch integration)
- Build all tests: `make test_unit -j` from build directory
- Individual test targets follow naming pattern: `test_unit_<component>`

## Coding Conventions

**These are critical - CUTLASS has strict style rules. Do NOT use auto-formatters.**

### Formatting
- **2-space indentation**, no tabs
- **Max 100 characters** per line
- **No automatic code formatting** (no clang-format)
- Double indent for function parameter declarations
- Always use braces with `if`, `for`, `while`
- Space after control keywords: `if (condition) {`
- `else` on new line after closing brace of `if`

### East const
CUTLASS uses "East const" - `const`/`constexpr` goes **after** the type:
```cpp
float const x = 1.0f;          // NOT: const float x
float const& ref = x;          // NOT: const float& ref
float const* ptr = &x;         // NOT: const float* ptr
float constexpr kVal = 42.0f;  // NOT: constexpr float kVal
```

### Left alignment for pointers/references
```cpp
int const& var;   // correct
int const* var;   // correct
int const &var;   // WRONG
int const *var;   // WRONG
```

### Naming
- File extensions: `.h` (headers), `.cu` (CUDA sources)
- Headers mirror namespace hierarchy: `include/cutlass/gemm/device/gemm.h`
- Primary namespace: `cutlass::`, sub-namespaces: `cutlass::gemm::`, `cutlass::conv::`, etc.
- CuTe namespace: `cute::`

### Design Patterns
- **Params pattern**: Classes with nontrivial constructors define an inner `struct Params` for grid-invariant state passed to kernels via constant memory
- **SharedStorage pattern**: Classes needing shared memory define `struct SharedStorage`; use `union` for non-overlapping lifetimes
- **Loop unrolling**: Use `CUTLASS_PRAGMA_UNROLL` on all loops expected to be unrolled
- **Templates**: Extensive use of C++ templates for compile-time specialization across data types, layouts, and architectures
- Avoid naming functions "fast" or "optimized" - describe the algorithm instead
- Avoid unconstrained template functions with common names (ADL issues)

## Python Bindings

```
python/cutlass/        # High-level Python API for GEMM, convolution, grouped GEMM
python/cutlass_library/# Library generation and profiler integration
python/pycute/         # PyCute layout algebra in Python
```

Dependencies: cuda-python >= 11.8.0, numpy, scipy, networkx, pydot, treelib.
Python >= 3.8 required.

## Key Documentation

- `media/docs/quickstart.md` - Build and run guide
- `media/docs/gemm_api_3x.md` - CUTLASS 3.x GEMM API (current)
- `media/docs/gemm_api.md` - CUTLASS 2.x GEMM API (legacy)
- `media/docs/programming_guidelines.md` - Full coding style and design patterns
- `media/docs/code_organization.md` - Detailed directory structure
- `media/docs/cute/00_quickstart.md` - CuTe documentation

## Important Notes for AI Assistants

1. **Header-only library**: The core library requires no compilation to use - just include the headers. Tests and examples need CMake builds.
2. **Template-heavy codebase**: Most logic is in headers via C++ templates. Compile times can be very long.
3. **Two API generations**: CUTLASS 2.x (threadblock/warp/thread hierarchy) and 3.x (CuTe-based collective operations) coexist. New code should prefer 3.x patterns.
4. **Architecture-specific code**: Many components are specialized per GPU architecture (SM70, SM75, SM80, SM89, SM90a). Check which arch you're targeting.
5. **No clang-format**: Do NOT auto-format code. Follow the manual style guide strictly.
6. **East const is mandatory**: Always write `Type const` not `const Type`.
7. **Performance over simplicity**: When there's a tradeoff, CUTLASS chooses performance. Expect complex template metaprogramming.
