# RefTorch - libtorch Bindings for Refal-5λ

RefTorch provides tensor computation capabilities to Refal-5λ by wrapping libtorch (PyTorch's C++ backend).

## Features

- **Tensor Creation**: zeros, ones, random, from lists, identity matrices, ranges
- **Tensor Info**: shape, dimensions, element count, printing
- **Math Operations**: add, subtract, multiply, divide, matrix multiply
- **Unary Operations**: neg, abs, sqrt, exp, log, pow, sigmoid, tanh
- **Comparisons**: eq, gt, lt, ge, le

## Quick Start

### Prerequisites

1. **Refal-5λ** - https://github.com/bmstu-iu9/refal-5-lambda
2. **libtorch** - https://pytorch.org/ (download LibTorch C++)

### Setup

```bash
# Set environment variables
export LIBTORCH=/path/to/libtorch
export REFAL_HOME=/path/to/refal-5-lambda

# Make build script executable
chmod +x build.sh torch-g++.sh
```

### Build and Run Example

```bash
./build.sh examples/01_basic_tensors program
./program
```

## Project Structure

```
refal-torch/
├── build.sh              # Build script
├── torch-g++.sh          # Compiler wrapper
├── src/
│   ├── core/
│   │   ├── TensorCore.ref    # Creation, memory
│   │   └── TensorInfo.ref    # Shape, print
│   └── ops/
│       └── TensorMath.ref    # Math operations
├── include/
│   └── RefTorch.refi         # Header with $EXTERN declarations
└── examples/
    └── 01_basic_tensors.ref  # Basic usage example
```

## Usage

### In Your Refal Program

```refal
* Import the functions you need
$EXTERN TZeros, TRand, TAdd, TPrint, TFree;

$ENTRY Go {
  = <Main>;
}

Main {
  = <DoComputation <TRand 3 3> <TRand 3 3>>;
}

DoComputation {
  s.A s.B
    = <Prout 'A + B ='>
      <TPrint <TAdd s.A s.B>>
      <TFree s.A>
      <TFree s.B>;
}
```

### Building Your Program

```bash
./build.sh path/to/your_program output_name
./output_name
```

## API Reference

### Tensor Creation

| Function | Description | Example |
|----------|-------------|---------|
| `<TZeros e.Shape>` | Tensor of zeros | `<TZeros 3 4>` → 3×4 zeros |
| `<TOnes e.Shape>` | Tensor of ones | `<TOnes 2 2>` → 2×2 ones |
| `<TRand e.Shape>` | Uniform random [0,1) | `<TRand 3 3>` |
| `<TRandn e.Shape>` | Normal distribution | `<TRandn 2 2>` |
| `<TFull s.Val e.Shape>` | Filled with value | `<TFull 5 2 3>` |
| `<TEye s.N>` | Identity matrix | `<TEye 4>` |
| `<TArange ...>` | Range of values | `<TArange 0 10 2>` |
| `<TFromList (e.Shape) e.Values>` | From list | `<TFromList (2 2) 1 2 3 4>` |

### Tensor Info

| Function | Description |
|----------|-------------|
| `<TPrint s.T>` | Print tensor to stdout |
| `<TShape s.T>` | Get shape as list |
| `<TDim s.T>` | Number of dimensions |
| `<TNumel s.T>` | Total element count |

### Math Operations

| Function | Description |
|----------|-------------|
| `<TAdd s.A s.B>` | A + B |
| `<TSub s.A s.B>` | A - B |
| `<TMul s.A s.B>` | A * B (element-wise) |
| `<TDiv s.A s.B>` | A / B |
| `<TMatMul s.A s.B>` | Matrix multiply A @ B |
| `<TMulScalar s.T s.Val>` | T * (Val/1000) |

### Memory Management

| Function | Description |
|----------|-------------|
| `<TFree s.T>` | Free a tensor |
| `<TFreeAll>` | Free all tensors |
| `<TClone s.T>` | Deep copy |

## Fixed-Point Convention

Since Refal only has integers, floating-point values use a fixed-point convention:

- **Scalars passed to functions**: Multiply by 1000 (e.g., 1.5 → 1500)
- **Values returned from `TToList`, `TItem`**: Divided by 1000 to get actual value

Example:
```refal
* Multiply tensor by 2.5
<TMulScalar s.T 2500>

* Get scalar value (returns value * 1000)
<TItem s.ScalarTensor>  /* returns e.g. 3141 for π ≈ 3.141 */
```

## License

MIT License
