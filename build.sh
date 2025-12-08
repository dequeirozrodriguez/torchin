#!/bin/bash
# build.sh - Build script for refal-torch project

set -e

# === CONFIGURATION ===
LIBTORCH="${LIBTORCH:-$HOME/qwen3_perl/qwen3_refal/torch/libtorch}"
REFAL_HOME="${REFAL_HOME:-$HOME/qwen3_perl/refal-5-lambda}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Refal-Torch Build ===${NC}"

TARGET="${1:-examples/01_basic_tensors}"
OUTPUT="${2:-program}"

SOURCES=(
    "$REFAL_HOME/lib/src/Library.ref"
    "$PROJECT_DIR/src/core/TensorCore.ref"
    "$PROJECT_DIR/src/core/TensorInfo.ref"
    "$PROJECT_DIR/src/ops/TensorMath.ref"
    "$PROJECT_DIR/src/ops/TensorManip.ref"
    "$PROJECT_DIR/src/ops/TensorReduce.ref"
    "$PROJECT_DIR/src/nn/TensorNN.ref"
    "$PROJECT_DIR/src/nn/TensorLoss.ref"
    "$PROJECT_DIR/src/nn/TensorOptim.ref"
    "$PROJECT_DIR/src/util/TensorIO.ref"
    "$PROJECT_DIR/src/util/TensorUtil.ref"

    # --- QWEN3 MODULES ---
    "$PROJECT_DIR/src/models/QwenConfig.ref"
    "$PROJECT_DIR/src/models/WeightLoader.ref"
    "$PROJECT_DIR/src/models/QwenMLP.ref"
    "$PROJECT_DIR/src/models/QwenAttention.ref"
    "$PROJECT_DIR/src/models/QwenBlock.ref"
    "$PROJECT_DIR/src/models/QwenModel.ref"
    "$PROJECT_DIR/src/models/QwenGenerate.ref"
)

if [ -f "$PROJECT_DIR/${TARGET}.ref" ]; then
    SOURCES+=("$PROJECT_DIR/${TARGET}.ref")
elif [ -f "$TARGET" ]; then
    SOURCES+=("$TARGET")
else
    echo -e "${RED}Error: Target not found: $TARGET${NC}"
    exit 1
fi

if [[ "$TARGET" == *"test/"* ]] || [[ "$TARGET" == *"test_"* ]]; then
    echo "Adding test framework..."
    # Insert test_framework before the last file (the target)
    SOURCES=("${SOURCES[@]:0:${#SOURCES[@]}-1}" "$PROJECT_DIR/test/test_framework.ref" "${SOURCES[@]: -1}")
fi
# ---------------------------------

# Header Fix
echo -e "${YELLOW}Preparing Header...${NC}"
if [ -f "$PROJECT_DIR/include/RefTorch.refi" ]; then
    cp "$PROJECT_DIR/include/RefTorch.refi" "$PROJECT_DIR/RefTorch.refi"
fi

# vars for wrapper
export LIBTORCH
export REFAL_HOME

echo -e "${YELLOW}Compiling...${NC}"
rlc -x -Od --prefix= \
    -c "$PROJECT_DIR/torch-g++.sh -Wall -g -o" \
    "${SOURCES[@]}" \
    -o "$PROJECT_DIR/$OUTPUT"

# cleanup
rm -f "$PROJECT_DIR/RefTorch.refi"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful: $PROJECT_DIR/$OUTPUT${NC}"
else
    echo -e "${RED}Build failed${NC}"
    exit 1
fi
