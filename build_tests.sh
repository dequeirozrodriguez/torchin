#!/bin/bash
# build_tests.sh - Build and run RefTorch tests
#
# Usage:
#   ./build_tests.sh           # Run all tests
#   ./build_tests.sh core      # Run only core tests
#   ./build_tests.sh qwen      # Run all Qwen tests
#   ./build_tests.sh qwen_mlp  # Run specific test

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

TEST_SUITE="${1:-all}"

run_test() {
    local name=$1
    echo ""
    echo "========================================"
    echo "Running: test_$name"
    echo "========================================"
    
    if [ ! -f "test/test_$name.ref" ]; then
        echo "Error: test/test_$name.ref not found!"
        return 1
    fi
    
    ./build.sh "test/test_$name" "test_$name" 2>&1
    
    if [ -f "test_$name" ]; then
        ./test_$name
        rm -f "test_$name"
    else
        echo "ERROR: Failed to build test_$name"
        return 1
    fi
}

case "$TEST_SUITE" in
    all)
        echo "Running ALL RefTorch tests..."
        echo ""
        echo "=== General Tests ==="
        run_test "core"
        run_test "math"
        run_test "manip"
        run_test "reduce"
        run_test "nn"
        run_test "util"
        run_test "llm"
        
        echo ""
        echo "=== Qwen3 Tests ==="
        run_test "qwen_config"
        run_test "qwen_mlp"
        run_test "qwen_attention"
        run_test "qwen_block"
        run_test "qwen_model"
        run_test "qwen_generate"
        run_test "qwen_weight_loader"
        run_test "qwen_integration"
        
        echo ""
        echo "========================================"
        echo "All test suites completed!"
        echo "========================================"
        ;;
    
    general)
        echo "Running General RefTorch tests..."
        run_test "core"
        run_test "math"
        run_test "manip"
        run_test "reduce"
        run_test "nn"
        run_test "util"
        run_test "llm"
        ;;
    
    qwen)
        echo "Running Qwen3 test suite..."
        run_test "qwen_config"
        run_test "qwen_mlp"
        run_test "qwen_attention"
        run_test "qwen_block"
        run_test "qwen_model"
        run_test "qwen_generate"
        run_test "qwen_weight_loader"
        run_test "qwen_integration"
        ;;
    
    *)
        # Run single test by name
        run_test "$TEST_SUITE"
        ;;
esac
