# Refal-5λ + libtorch Integration Guide

## Overview

This guide documents how to integrate libtorch (PyTorch's C++ backend) with Refal-5λ, a functional pattern-matching language. This allows you to write high-level symbolic computation in Refal while leveraging libtorch for numerical/tensor operations.

## Table of Contents

1. [Background](#background)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [The Wrapper Script](#the-wrapper-script)
5. [Compilation Command](#compilation-command)
6. [Why This Approach](#why-this-approach)
7. [Native Code API Reference](#native-code-api-reference)
8. [Example: TorchWrapper Module](#example-torchwrapper-module)
9. [Example: Pure Refal Main Program](#example-pure-refal-main-program)
10. [Troubleshooting](#troubleshooting)

---

## Background

### What is Refal-5λ?

Refal-5λ is a functional programming language focused on symbolic computation and pattern matching. It compiles to C++ and has excellent interoperability with native code through "native insertions" (`%%` blocks).

### Why integrate with libtorch?

- Refal excels at symbolic manipulation, parsing, and pattern matching
- libtorch excels at numerical computation, neural networks, and tensor operations
- Combining them allows symbolic AI approaches with numerical backends

---

## Prerequisites

### 1. Refal-5λ Compiler

Clone and build from: https://github.com/bmstu-iu9/refal-5-lambda

```bash
git clone https://github.com/bmstu-iu9/refal-5-lambda.git
cd refal-5-lambda
./bootstrap.sh
```

### 2. libtorch

Download from https://pytorch.org/ (select LibTorch, C++, your platform)

```bash
# Example for Linux CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
```

### 3. Environment Variables

```bash
export LIBTORCH=$HOME/path/to/libtorch
export REFAL_HOME=$HOME/path/to/refal-5-lambda
export REFAL_RT=$REFAL_HOME/lib/scratch-rt
```

---

## Directory Structure

```
refal-5-lambda/
├── lib/
│   ├── scratch-rt/           # Runtime C++ sources
│   │   ├── refalrts.h        # Main runtime header
│   │   ├── refalrts.cpp
│   │   ├── refalrts-vm.cpp
│   │   ├── refalrts-vm-api.cpp
│   │   ├── refalrts-functions.cpp
│   │   ├── refalrts-dynamic.cpp
│   │   ├── platform-Linux/
│   │   ├── platform-POSIX/
│   │   ├── debug-stubs/
│   │   └── exe/
│   └── src/
│       └── Library.ref       # Standard library (must be compiled with your program)
├── torch-g++.sh              # Wrapper script (you create this)
├── TorchWrapper.ref          # Your libtorch bindings
└── main.ref                  # Your pure Refal program
```

---

## The Wrapper Script

Create `torch-g++.sh` in your refal-5-lambda directory:

```bash
#!/bin/bash
# torch-g++.sh - Wrapper script for compiling Refal with libtorch

# === CONFIGURATION - Adjust these paths ===
LIBTORCH=$HOME/path/to/libtorch
REFAL_RT=$HOME/path/to/refal-5-lambda/lib/scratch-rt

# === COMPILATION ===
g++ -std=c++17 \
    -I${LIBTORCH}/include \
    -I${LIBTORCH}/include/torch/csrc/api/include \
    -I${REFAL_RT} \
    -I${REFAL_RT}/debug-stubs \
    "$@" \
    ${REFAL_RT}/refalrts.cpp \
    ${REFAL_RT}/refalrts-vm.cpp \
    ${REFAL_RT}/refalrts-vm-api.cpp \
    ${REFAL_RT}/refalrts-functions.cpp \
    ${REFAL_RT}/refalrts-dynamic.cpp \
    ${REFAL_RT}/platform-Linux/refalrts-platform-specific.cpp \
    ${REFAL_RT}/platform-POSIX/refalrts-platform-POSIX.cpp \
    ${REFAL_RT}/debug-stubs/refalrts-profiler.cpp \
    ${REFAL_RT}/debug-stubs/refalrts-diagnostic-initializer.cpp \
    ${REFAL_RT}/exe/refalrts-main.cpp \
    -L${LIBTORCH}/lib \
    -Wl,-rpath,${LIBTORCH}/lib \
    -ltorch -ltorch_cpu -lc10 -ldl
```

Make it executable:
```bash
chmod +x torch-g++.sh
```

### What the wrapper script does:

1. **Includes libtorch headers** (`-I${LIBTORCH}/include ...`)
2. **Includes Refal runtime headers** (`-I${REFAL_RT} ...`)
3. **Passes through rlc's arguments** (`"$@"`)
4. **Compiles all Refal runtime C++ files** (the `${REFAL_RT}/*.cpp` files)
5. **Links libtorch libraries** (`-ltorch -ltorch_cpu -lc10`)
6. **Sets runtime library path** (`-Wl,-rpath,...`)

---

## Compilation Command

```bash
# Clean previous builds
rm -f *.cpp *.rasl

# Compile
rlc -x -Od --prefix= \
    -c "$HOME/path/to/refal-5-lambda/torch-g++.sh -Wall -g -o" \
    $HOME/path/to/refal-5-lambda/lib/src/Library.ref \
    TorchWrapper.ref \
    main.ref \
    -o my_program

# Run
./my_program
```

### Explanation of flags:

| Flag | Purpose |
|------|---------|
| `-x` | Build executable (not library) |
| `-Od` | Direct C++ code generation (required for native insertions) |
| `--prefix=` | Empty prefix - bypasses pre-compiled runtime, allows native code |
| `-c "..."` | Custom C++ compiler command (our wrapper script) |
| `Library.ref` | Standard library - MUST be included for runtime symbols |
| `TorchWrapper.ref` | Your native bindings module |
| `main.ref` | Your main program |
| `-o my_program` | Output executable name |

---

## Why This Approach

### Why `--prefix=` (empty prefix)?

Refal-5λ normally uses "prefix files" - pre-compiled binary blobs containing the runtime. However, when you have native C++ insertions (`%%` blocks), the compiler detects this and refuses to use prefix mode because:

1. Native code needs to be compiled fresh with your C++ compiler
2. The cookies (module identifiers) must match between all compiled units
3. Pre-compiled prefixes have fixed cookies that won't match

Using `--prefix=` forces "from-source" compilation of everything.

### Why include Library.ref?

The standard library provides essential functions (`Prout`, `Add`, `Type`, etc.). When compiling without a prefix, these aren't available unless you compile Library.ref alongside your code.

**Critical**: All `.ref` files compiled together share the same "cookies" (unique identifiers). This is how the runtime links functions across modules. Compiling Library.ref separately would give it different cookies, causing "unresolved external" errors at runtime.

### Why use a wrapper script?

The `rlc` compiler invokes the C++ compiler with specific arguments. We need to:
1. Add libtorch include paths and libraries
2. Add all Refal runtime source files
3. Ensure libraries come AFTER object files (linker requirement)

A wrapper script intercepts the compilation and adds our requirements.

### Why `-Od` flag?

`-Od` means "direct code generation" - it compiles Refal functions directly to C++ functions instead of generating interpreted bytecode (RASL). This is required when you have native `%%` insertions because those insertions become part of the generated C++ code.

---

## Native Code API Reference

### Function Signature

Native Refal functions receive these parameters:
```cpp
refalrts::FnResult func_YourFunction(
    refalrts::VM *vm,        // Virtual machine context
    refalrts::Iter arg_begin, // Start of argument expression
    refalrts::Iter arg_end    // End of argument expression
)
```

### Key Data Types

```cpp
// Node tags - check what type of data a node contains
refalrts::cDataNumber    // Integer number
refalrts::cDataChar      // Character
refalrts::cDataFunction  // Function reference
refalrts::cDataIdentifier // Identifier

// Node structure
struct Node {
    DataTag tag;
    union {
        char char_info;
        RefalNumber number_info;
        RefalFunction *function_info;
        RefalIdentifier ident_info;
        // ...
    };
};
```

### Essential Functions

#### Extracting Arguments

```cpp
// Get content between call brackets
refalrts::Iter content_b = 0, content_e = 0;
refalrts::call_left(content_b, content_e, arg_begin, arg_end);

// Move through the expression
refalrts::move_left(content_b, content_e);   // Advance forward
refalrts::move_right(content_b, content_e);  // Advance backward

// Check if empty
refalrts::empty_seq(content_b, content_e);

// Extract s-variable (single symbol)
refalrts::Iter svar = 0;
refalrts::svar_left(svar, content_b, content_e);

// Extract t-variable (term - symbol or bracketed expression)
refalrts::Iter tvar = 0;
refalrts::tvar_left(tvar, content_b, content_e);
```

#### Checking Types

```cpp
// Check node type by examining tag
if (content_b->tag == refalrts::cDataNumber) {
    int value = content_b->number_info;
}

if (content_b->tag == refalrts::cDataChar) {
    char ch = content_b->char_info;
}
```

#### Building Results

```cpp
// Modify existing node to be a number
refalrts::reinit_number(arg_begin, 42);

// Allocate new nodes
refalrts::Iter new_node;
refalrts::alloc_number(vm, new_node, 42);
refalrts::alloc_char(vm, new_node, 'A');

// Clean up unused nodes
refalrts::splice_to_freelist(vm, start, end);
```

#### Return Values

```cpp
return refalrts::cSuccess;              // Function succeeded
return refalrts::cRecognitionImpossible; // Pattern match failed
```

---

## Example: TorchWrapper Module

```refal
* TorchWrapper.ref - libtorch bindings for Refal

%%
#include <torch/torch.h>
#include <iostream>
#include <unordered_map>

namespace {
    std::unordered_map<int, torch::Tensor> g_tensors;
    int g_next_id = 1;
}
%%

*============================================================================
* <TorchRand s.Rows s.Cols> == s.TensorID
* Creates a random tensor of given dimensions
*============================================================================
$ENTRY TorchRand {
%%
    refalrts::Iter content_b = 0, content_e = 0;
    refalrts::call_left(content_b, content_e, arg_begin, arg_end);
    
    // Get first number (rows)
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int rows = content_b->number_info;
    refalrts::move_left(content_b, content_e);
    
    // Get second number (cols)
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int cols = content_b->number_info;
    
    // Create tensor
    torch::Tensor t = torch::rand({rows, cols});
    int id = g_next_id++;
    g_tensors[id] = t;
    
    // Return ID
    refalrts::reinit_number(arg_begin, id);
    refalrts::splice_to_freelist(vm, arg_begin->next, arg_end);
    return refalrts::cSuccess;
%%
}

*============================================================================
* <TorchPrint s.TensorID> == empty
* Prints a tensor to stdout
*============================================================================
$ENTRY TorchPrint {
%%
    refalrts::Iter content_b = 0, content_e = 0;
    refalrts::call_left(content_b, content_e, arg_begin, arg_end);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id = content_b->number_info;
    
    if (g_tensors.find(id) != g_tensors.end()) {
        std::cout << g_tensors[id] << std::endl;
    } else {
        std::cout << "Tensor " << id << " not found!" << std::endl;
    }
    
    refalrts::splice_to_freelist(vm, arg_begin, arg_end);
    return refalrts::cSuccess;
%%
}

*============================================================================
* <TorchAdd s.TensorA s.TensorB> == s.TensorResult
* Element-wise addition
*============================================================================
$ENTRY TorchAdd {
%%
    refalrts::Iter content_b = 0, content_e = 0;
    refalrts::call_left(content_b, content_e, arg_begin, arg_end);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id_a = content_b->number_info;
    refalrts::move_left(content_b, content_e);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id_b = content_b->number_info;
    
    torch::Tensor result = g_tensors[id_a] + g_tensors[id_b];
    int result_id = g_next_id++;
    g_tensors[result_id] = result;
    
    refalrts::reinit_number(arg_begin, result_id);
    refalrts::splice_to_freelist(vm, arg_begin->next, arg_end);
    return refalrts::cSuccess;
%%
}

*============================================================================
* <TorchMatMul s.TensorA s.TensorB> == s.TensorResult
* Matrix multiplication
*============================================================================
$ENTRY TorchMatMul {
%%
    refalrts::Iter content_b = 0, content_e = 0;
    refalrts::call_left(content_b, content_e, arg_begin, arg_end);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id_a = content_b->number_info;
    refalrts::move_left(content_b, content_e);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id_b = content_b->number_info;
    
    torch::Tensor result = torch::matmul(g_tensors[id_a], g_tensors[id_b]);
    int result_id = g_next_id++;
    g_tensors[result_id] = result;
    
    refalrts::reinit_number(arg_begin, result_id);
    refalrts::splice_to_freelist(vm, arg_begin->next, arg_end);
    return refalrts::cSuccess;
%%
}

*============================================================================
* <TorchFree s.TensorID> == empty
* Free a tensor from memory
*============================================================================
$ENTRY TorchFree {
%%
    refalrts::Iter content_b = 0, content_e = 0;
    refalrts::call_left(content_b, content_e, arg_begin, arg_end);
    
    if (content_b->tag != refalrts::cDataNumber) {
        return refalrts::cRecognitionImpossible;
    }
    int id = content_b->number_info;
    g_tensors.erase(id);
    
    refalrts::splice_to_freelist(vm, arg_begin, arg_end);
    return refalrts::cSuccess;
%%
}
```

---

## Example: Pure Refal Main Program

```refal
* main.ref - Pure Refal program using TorchWrapper

$EXTERN TorchRand, TorchPrint, TorchAdd, TorchMatMul, TorchFree;

*============================================================================
* Entry point
*============================================================================
$ENTRY Go {
  = <Main>;
}

Main {
  = <Prout 'Creating two random 3x3 matrices...'>
    <TestMatrixOps>;
}

TestMatrixOps {
  = <DoMatrixOps <TorchRand 3 3> <TorchRand 3 3>>;
}

DoMatrixOps {
  s.A s.B
    = <Prout 'Matrix A:'>
      <TorchPrint s.A>
      <Prout 'Matrix B:'>
      <TorchPrint s.B>
      <Prout 'A + B:'>
      <PrintAndFree <TorchAdd s.A s.B>>
      <Prout 'A * B (matmul):'>
      <PrintAndFree <TorchMatMul s.A s.B>>
      <TorchFree s.A>
      <TorchFree s.B>;
}

PrintAndFree {
  s.Tensor
    = <TorchPrint s.Tensor>
      <TorchFree s.Tensor>;
}
```

### Key Points:

1. **`$EXTERN`** imports functions from other modules
2. **`$ENTRY`** exports functions (makes them visible to other modules)
3. Tensor handles are just integers (`s.TensorID`) - opaque to Refal
4. All tensor operations happen in C++, Refal just orchestrates

---

## Troubleshooting

### Error: "unexpected native file Generated X.cpp while compilation runs with prefix"

**Cause**: You're using a prefix but have native `%%` insertions.

**Solution**: Use `--prefix=` (empty) to disable prefix mode.

### Error: "unresolved external: Add#0:0, Prout#0:0, ..."

**Cause**: Standard library not linked.

**Solution**: Include `Library.ref` in your compilation:
```bash
rlc ... Library.ref YourProgram.ref ...
```

### Error: "multiple definition of cookie_ns_..."

**Cause**: A module is being compiled twice (once by rlc, once in wrapper script).

**Solution**: Don't include pre-compiled `.cpp` files in the wrapper script if rlc is generating them.

### Error: "undefined reference to refalrts::..."

**Cause**: Runtime C++ files not being compiled.

**Solution**: Ensure your wrapper script includes all runtime `.cpp` files from `lib/scratch-rt/`.

### Error: "INTERNAL ERROR: can't find signature in executable"

**Cause**: Missing runtime initialization when manually compiling with g++.

**Solution**: Use `rlc` instead of raw g++ - it handles signature embedding.

### Runtime: "INTERNAL ERROR: unresolved external: ..."

**Cause**: Cookies don't match between modules.

**Solution**: Compile all `.ref` files together in a single `rlc` invocation.

---

## Quick Reference: Build Commands

```bash
# Set environment
export LIBTORCH=$HOME/path/to/libtorch
export REFAL_HOME=$HOME/path/to/refal-5-lambda

# Clean
rm -f *.cpp *.rasl

# Build
rlc -x -Od --prefix= \
    -c "$REFAL_HOME/torch-g++.sh -Wall -g -o" \
    $REFAL_HOME/lib/src/Library.ref \
    TorchWrapper.ref \
    main.ref \
    -o my_program

# Run
./my_program
```

---

## License

This guide is provided as-is for educational purposes. Refal-5λ is licensed under BSD-2-Clause. libtorch/PyTorch has its own license terms.

---

## References

- Refal-5λ: https://github.com/bmstu-iu9/refal-5-lambda
- Refal-5λ Documentation: https://bmstu-iu9.github.io/refal-5-lambda/
- libtorch: https://pytorch.org/cppdocs/
- PyTorch C++ API: https://pytorch.org/tutorials/advanced/cpp_frontend.html
