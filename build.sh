#!/bin/bash
# build.sh - Build script for refal-torch project

set -e

# === CONFIGURATION ===
LIBTORCH="${LIBTORCH:-$HOME/qwen3_perl/qwen3_refal/torch/libtorch}"
REFAL_HOME="${REFAL_HOME:-$HOME/qwen3_perl/refal-5-lambda}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Refal-Torch Build ===${NC}"
echo "LIBTORCH: $LIBTORCH"
echo "REFAL_HOME: $REFAL_HOME"
echo "PROJECT_DIR: $PROJECT_DIR"
echo ""

# Checks
if [ ! -d "$LIBTORCH" ]; then
    echo -e "${RED}Error: LIBTORCH directory not found: $LIBTORCH${NC}"
    exit 1
fi
if [ ! -d "$REFAL_HOME" ]; then
    echo -e "${RED}Error: REFAL_HOME directory not found: $REFAL_HOME${NC}"
    exit 1
fi

export LIBTORCH
export REFAL_HOME

# Clean previous builds (garbage .cpp files in root)
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -f "$PROJECT_DIR"/*.cpp "$PROJECT_DIR"/*.rasl "$PROJECT_DIR"/*.o
rm -f RefTorch.refi  # Clean up potential leftover link

# Target setup
TARGET="${1:-examples/01_basic_tensors}"
OUTPUT="${2:-01_basic_tensors}" 

# Define Sources
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
)

# Add Target File
if [ -f "$PROJECT_DIR/${TARGET}.ref" ]; then
    SOURCES+=("$PROJECT_DIR/${TARGET}.ref")
elif [ -f "$TARGET" ]; then
    SOURCES+=("$TARGET")
else
    echo -e "${RED}Error: Target not found: $TARGET${NC}"
    exit 1
fi

echo -e "${YELLOW}Preparing Header...${NC}"
# FIX: Copy header to root so rlc finds it in Current Working Directory
if [ -f "$PROJECT_DIR/include/RefTorch.refi" ]; then
    cp "$PROJECT_DIR/include/RefTorch.refi" "$PROJECT_DIR/RefTorch.refi"
else
    echo -e "${RED}Error: include/RefTorch.refi not found!${NC}"
    exit 1
fi

echo -e "${YELLOW}Compiling...${NC}"
# rlc command:
# -x  : Build executable
# -Od : Debug mode (faster compile, easier debugging)
# -c  : Specifies the C++ compiler wrapper command
rlc -x -Od --prefix= \
    -c "$PROJECT_DIR/torch-g++.sh -Wall -g -o" \
    "${SOURCES[@]}" \
    -o "$PROJECT_DIR/$OUTPUT"

# Cleanup header
rm -f "$PROJECT_DIR/RefTorch.refi"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful: $PROJECT_DIR/$OUTPUT${NC}"
    echo "Run with: ./$OUTPUT"
else
    echo -e "${RED}Build failed${NC}"
    exit 1
fi
