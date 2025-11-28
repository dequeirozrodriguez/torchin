![Logo](logo.png)

# RefTorch - PyTorch/libtorch Bindings for Refal-5λ

A comprehensive tensor computation library bringing PyTorch's power to the Refal-5λ functional programming language.

Torchin is a fond nickname for the lib in honor of Valentin Turchin.
## Overview

RefTorch provides **250+ functions** for tensor operations, neural networks, and deep learning, seamlessly integrated with Refal-5λ's pattern-matching paradigm.

## Project Structure

```
refal-torch/
├── build.sh                    # Build script
├── torch-g++.sh               # Compiler wrapper
├── include/
│   └── RefTorch.refi          # Header with all $EXTERN declarations
├── src/
│   ├── core/
│   │   ├── TensorCore.ref     # Tensor creation and memory management
│   │   └── TensorInfo.ref     # Shape queries, printing
│   ├── ops/
│   │   ├── TensorMath.ref     # Math operations (+, -, *, /, matmul, etc.)
│   │   ├── TensorManip.ref    # Reshape, transpose, concat, slice
│   │   └── TensorReduce.ref   # Sum, mean, max, min, sorting
│   ├── nn/
│   │   ├── TensorNN.ref       # Layers, activations, pooling, conv
│   │   ├── TensorLoss.ref     # Loss functions (MSE, CrossEntropy, etc.)
│   │   └── TensorOptim.ref    # Optimizers (SGD, Adam), gradients
│   └── util/
│       ├── TensorIO.ref       # Save/load tensors
│       └── TensorUtil.ref     # Seeds, device info, debugging
└── examples/
    ├── 01_basic_tensors.ref   # Basic tensor operations
    ├── 02_manip_reduce.ref    # Manipulation and reduction
    ├── 03_neural_network.ref  # Neural network training
    └── 04_utilities.ref       # I/O, seeds, debugging
```

## Quick Start

### Prerequisites

1. Refal-5λ compiler (`rlc`) installed
2. libtorch (CPU version) downloaded

### Setup

```bash
# Edit paths in build.sh and torch-g++.sh
export LIBTORCH=/path/to/libtorch
export REFAL_HOME=/path/to/refal-5-lambda

chmod +x build.sh torch-g++.sh
```

### Building

```bash
./build.sh examples/01_basic_tensors program
./program
```

### Usage

```refal
* Import functions you need
$EXTERN TRand, TAdd, TPrint, TFree;

$ENTRY Go {
  = <DoComputation <TRand 3 3> <TRand 3 3>>;
}

DoComputation {
  s.A s.B = <TPrint <TAdd s.A s.B>>
            <TFree s.A> <TFree s.B>;
}
```

## Module Reference

### Core (TensorCore, TensorInfo) - ~25 functions

| Function | Description |
|----------|-------------|
| `<TZeros e.Shape>` | Create tensor of zeros |
| `<TOnes e.Shape>` | Create tensor of ones |
| `<TRand e.Shape>` | Uniform random [0,1) |
| `<TRandn e.Shape>` | Normal random N(0,1) |
| `<TFull e.Shape s.Val>` | Fill with value (÷1000) |
| `<TEye s.N>` | Identity matrix |
| `<TArange s.N>` | Range [0, N) |
| `<TFromList (e.Shape) e.Values>` | Create from list |
| `<TFree s.T>` | Free tensor |
| `<TFreeAll>` | Free all tensors |
| `<TPrint s.T>` | Print tensor |
| `<TShape s.T>` | Get shape as list |
| `<TDim s.T>` | Number of dimensions |
| `<TNumel s.T>` | Total elements |

### Math (TensorMath) - ~35 functions

| Function | Description |
|----------|-------------|
| `<TAdd s.A s.B>` | Element-wise add |
| `<TSub s.A s.B>` | Element-wise subtract |
| `<TMul s.A s.B>` | Element-wise multiply |
| `<TDiv s.A s.B>` | Element-wise divide |
| `<TMatMul s.A s.B>` | Matrix multiplication |
| `<TDot s.A s.B>` | Dot product |
| `<TNeg s.T>` | Negate |
| `<TAbs s.T>` | Absolute value |
| `<TSqrt s.T>` | Square root |
| `<TExp s.T>` | Exponential |
| `<TLog s.T>` | Natural log |
| `<TPow s.T s.P>` | Power (p÷1000) |
| `<TSin/TCos/TTan s.T>` | Trigonometric |
| `<TSigmoid/TTanh s.T>` | Activations |

### Manipulation (TensorManip) - ~30 functions

| Function | Description |
|----------|-------------|
| `<TReshape s.T e.Shape>` | Reshape tensor |
| `<TView s.T e.Shape>` | View with -1 inference |
| `<TFlatten s.T>` | Flatten to 1D |
| `<TTranspose s.T s.D0 s.D1>` | Swap dimensions |
| `<TT s.T>` | 2D transpose |
| `<TCat s.Dim e.Tensors>` | Concatenate |
| `<TStack s.Dim e.Tensors>` | Stack (new dim) |
| `<TSlice s.T s.Dim s.Start s.End>` | Slice |
| `<TSelect s.T s.Dim s.Idx>` | Select index |
| `<TSplit s.T s.Size s.Dim>` | Split into chunks |
| `<TFlip s.T e.Dims>` | Flip along dims |
| `<TRepeat s.T e.Reps>` | Repeat tensor |

### Reduction (TensorReduce) - ~40 functions

| Function | Description |
|----------|-------------|
| `<TSum s.T>` | Sum all elements |
| `<TSumDim s.T s.Dim>` | Sum along dim |
| `<TMean s.T>` | Mean |
| `<TProd s.T>` | Product |
| `<TMax s.T>` / `<TMin s.T>` | Max/Min |
| `<TArgmax s.T>` | Index of max |
| `<TStd s.T>` / `<TVar s.T>` | Std/Variance |
| `<TNorm s.T>` | Frobenius norm |
| `<TSort s.T s.Dim>` | Sort |
| `<TTopK s.T s.K s.Dim>` | Top-K values |
| `<TCumsum s.T s.Dim>` | Cumulative sum |

### Neural Networks (TensorNN) - ~45 functions

| Function | Description |
|----------|-------------|
| `<TRelu s.T>` | ReLU activation |
| `<TGelu s.T>` | GELU activation |
| `<TSoftmax s.T s.Dim>` | Softmax |
| `<TLinear s.X s.W s.B>` | Linear layer |
| `<TInitLinear s.In s.Out>` | Init weights |
| `<TDropout s.T s.P>` | Dropout (p÷1000) |
| `<TLayerNorm s.T e.Shape>` | Layer norm |
| `<TBatchNorm1d ...>` | Batch norm |
| `<TMaxPool2d s.T ...>` | Max pooling |
| `<TConv2d s.T s.W s.B ...>` | 2D convolution |
| `<TEmbedding s.Idx s.Table>` | Embedding lookup |

### Loss Functions (TensorLoss) - ~20 functions

| Function | Description |
|----------|-------------|
| `<TMSE s.Pred s.Target>` | Mean squared error |
| `<TMAE s.Pred s.Target>` | Mean absolute error |
| `<TCrossEntropy s.Logits s.Targets>` | Cross-entropy |
| `<TBCEWithLogits s.Logits s.Targets>` | Binary CE |
| `<TNLL s.LogProbs s.Targets>` | Negative log-likelihood |
| `<TKLDiv s.P s.Q>` | KL divergence |
| `<TTripletMargin s.A s.P s.N s.M>` | Triplet loss |

### Optimizers (TensorOptim) - ~20 functions

| Function | Description |
|----------|-------------|
| `<TRequiresGrad s.T s.Bool>` | Enable gradients |
| `<TBackward s.Loss>` | Backpropagation |
| `<TGrad s.T>` | Get gradient |
| `<TZeroGrad s.T>` | Zero gradient |
| `<TUpdateSGD s.T s.LR>` | SGD update |
| `<TUpdateAdam s.T s.LR ...>` | Adam update |
| `<TClipGradNorm e.Ts s.Max>` | Gradient clipping |
| `<TLRCosine ...>` | Cosine LR schedule |

### I/O (TensorIO) - ~12 functions

| Function | Description |
|----------|-------------|
| `<TSave s.T e.File>` | Save PyTorch format |
| `<TLoad e.File>` | Load PyTorch format |
| `<TSaveText s.T e.File>` | Save as text |
| `<TSaveCSV s.T e.File>` | Save as CSV |
| `<TSaveBinary s.T e.File>` | Save binary |
| `<TFileExists e.File>` | Check file exists |

### Utilities (TensorUtil) - ~25 functions

| Function | Description |
|----------|-------------|
| `<TSetSeed s.Seed>` | Set random seed |
| `<TCudaAvailable>` | Check CUDA |
| `<TToFloat32 s.T>` | Convert dtype |
| `<TEqual s.A s.B>` | Exact equality |
| `<TAllClose s.A s.B ...>` | Approx equality |
| `<TCheckNaN s.T>` | Check for NaN |
| `<TSummary s.T>` | Statistics summary |
| `<TVersion>` | LibTorch version |

## Fixed-Point Convention

Since Refal only has integers, floating-point values use fixed-point:
- **Scalars**: multiply by 1000 (e.g., 1.5 → 1500)
- **Learning rates**: typically 1-100 (÷1000 = 0.001-0.1)
- **Tolerances**: may use 1e-6 scale (÷1000000)

Example:
```refal
<TMulScalar s.T 2500>  /* Multiply by 2.5 */
<TUpdateSGD s.W 10>    /* LR = 0.01 */
```

## Training Example

```refal
$EXTERN TRandn, TZeros, TRand, TLinear, TInitLinear, TMSE;
$EXTERN TRequiresGrad, TBackward, TUpdateAdam, TZeroGrad, TFree;

$ENTRY Go { = <Train>; }

Train {
  = <DoTrain 
      <TRequiresGrad <TRandn 1 4> 1>  /* Weight */
      <TRequiresGrad <TZeros 1> 1>    /* Bias */
      <TRand 10 4>                     /* X */
      <TRand 10 1>>;                   /* Y */
}

DoTrain {
  s.W s.B s.X s.Y = <TrainLoop 1 100 s.W s.B s.X s.Y>;
}

TrainLoop {
  s.I 100 s.W s.B s.X s.Y = <Done s.W s.B s.X s.Y>;
  s.I s.Max s.W s.B s.X s.Y
    = <TZeroGrad s.W> <TZeroGrad s.B>
      <DoStep s.I s.Max s.W s.B s.X s.Y 
              <TLinear s.X s.W s.B>>;
}

DoStep {
  s.I s.Max s.W s.B s.X s.Y s.Pred
    = <DoLoss s.I s.Max s.W s.B s.X s.Y s.Pred
              <TMSE s.Pred s.Y>>;
}

DoLoss {
  s.I s.Max s.W s.B s.X s.Y s.Pred s.Loss
    = <TBackward s.Loss>
      <TUpdateAdam s.W 10>  /* LR = 0.01 */
      <TUpdateAdam s.B 10>
      <TFree s.Pred> <TFree s.Loss>
      <TrainLoop <Add s.I 1> s.Max s.W s.B s.X s.Y>;
}

Done {
  s.W s.B s.X s.Y = <Prout 'Training complete!'> 
                    <TFree s.W> <TFree s.B> <TFree s.X> <TFree s.Y>;
}
```

## License

MIT License
