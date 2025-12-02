#!/bin/bash
# build_tests.sh - Build and run RefTorch tests
#
# Usage:
#   ./build_tests.sh           # Run all tests
#   ./build_tests.sh core      # Run only core tests
#   ./build_tests.sh math      # Run only math tests
#   ./build_tests.sh manip     # Run only manipulation tests
#   ./build_tests.sh reduce    # Run only reduction tests
#   ./build_tests.sh nn        # Run only neural network tests
#   ./build_tests.sh util      # Run only utility tests

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Determine which tests to run
TEST_SUITE="${1:-all}"

run_test() {
    local name=$1
    echo ""
    echo "========================================"
    echo "Running: $name"
    echo "========================================"
    
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
        run_test "core"
        run_test "math"
        run_test "manip"
        run_test "reduce"
        run_test "nn"
        run_test "util"
        echo ""
        echo "========================================"
        echo "All test suites completed!"
        echo "========================================"
        ;;
    core|math|manip|reduce|nn|util)
        run_test "$TEST_SUITE"
        ;;
    *)
        echo "Unknown test suite: $TEST_SUITE"
        echo "Available: all, core, math, manip, reduce, nn, util"
        exit 1
        ;;
esac
