refal-torch/
├── README.md
├── build.sh                    # Build script
├── torch-g++.sh               # Compiler wrapper
│
├── src/
│   ├── core/                  # Core tensor operations
│   │   ├── TensorCore.ref     # Creation, memory management
│   │   └── TensorInfo.ref     # Shape, dtype, device queries
│   │
│   ├── ops/                   # Operations
│   │   ├── TensorMath.ref     # Basic math (+, -, *, /, pow, sqrt, etc.)
│   │   ├── TensorReduce.ref   # Reductions (sum, mean, max, min, etc.)
│   │   ├── TensorManip.ref    # Reshape, transpose, concat, slice
│   │   └── TensorCompare.ref  # Comparison ops (eq, gt, lt, etc.)
│   │
│   ├── nn/                    # Neural network building blocks
│   │   ├── NNLinear.ref       # Linear/Dense layers
│   │   ├── NNActivation.ref   # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── NNLoss.ref         # MSE, CrossEntropy, etc.
│   │   └── NNOptim.ref        # SGD, Adam optimizers
│   │
│   └── util/                  # Utilities
│       ├── TensorIO.ref       # Save/Load tensors, print utilities
│       └── TensorRandom.ref   # Random number generation
│
├── include/
│   └── RefTorch.refi          # Header file with all $EXTERN declarations
│
├── examples/
│   ├── 01_basic_tensors.ref   # Basic tensor creation and ops
│   ├── 02_matrix_ops.ref      # Matrix multiplication, transpose
│   ├── 03_simple_nn.ref       # Simple neural network
│   └── 04_xor_problem.ref     # Train XOR classifier
│
└── test/
    └── test_all.ref           # Test suite
