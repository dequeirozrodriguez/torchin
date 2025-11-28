#!/bin/bash
# build.sh - Build script for refal-torch project

set -e

# === CONFIGURATION - Adjust these paths ===
LIBTORCH="${LIBTORCH:-$HOME/qwen3_perl/qwen3_refal/torch/libtorch}"
REFAL_HOME="${REFAL_HOME:-$HOME/qwen3_perl/refal-5-lambda}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Refal-Torch Build ===${NC}"
echo "LIBTORCH: $LIBTORCH"
echo "REFAL_HOME: $REFAL_HOME"
echo "PROJECT_DIR: $PROJECT_DIR"
echo ""

# Check prerequisites
if [ ! -d "$LIBTORCH" ]; then
    echo -e "${RED}Error: LIBTORCH directory not found: $LIBTORCH${NC}"
    echo "Set LIBTORCH environment variable to your libtorch installation"
    exit 1
fi

if [ ! -d "$REFAL_HOME" ]; then
    echo -e "${RED}Error: REFAL_HOME directory not found: $REFAL_HOME${NC}"
    echo "Set REFAL_HOME environment variable to your refal-5-lambda installation"
    exit 1
fi

# Export for wrapper script
export LIBTORCH
export REFAL_HOME

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -f "$PROJECT_DIR"/*.cpp "$PROJECT_DIR"/*.rasl
rm -f "$PROJECT_DIR"/src/**/*.cpp "$PROJECT_DIR"/src/**/*.rasl 2>/dev/null || true

# Determine what to build
TARGET="${1:-examples/01_basic_tensors}"
OUTPUT="${2:-program}"

echo -e "${YELLOW}Building: $TARGET${NC}"

# Collect all source files
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

# Add target file
if [ -f "$PROJECT_DIR/${TARGET}.ref" ]; then
    SOURCES+=("$PROJECT_DIR/${TARGET}.ref")
elif [ -f "$TARGET" ]; then
    SOURCES+=("$TARGET")
else
    echo -e "${RED}Error: Target not found: $TARGET${NC}"
    exit 1
fi

echo "Sources:"
for src in "${SOURCES[@]}"; do
    echo "  - $src"
done
echo ""

# Build
echo -e "${YELLOW}Compiling...${NC}"
rlc -x -Od --prefix= \
    -c "$PROJECT_DIR/torch-g++.sh -Wall -g -o" \
    "${SOURCES[@]}" \
    -o "$PROJECT_DIR/$OUTPUT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful: $PROJECT_DIR/$OUTPUT${NC}"
    echo ""
    echo "Run with: ./$OUTPUT"
else
    echo -e "${RED}Build failed${NC}"
    exit 1
fi
