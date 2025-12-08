#!/bin/bash
# setup_test_symlinks.sh - Create symlinks for organized test subdirectories
#
# This allows tests to be organized in subdirectories while still being
# visible to the Refal compiler which expects flat paths.
#
# Structure:
#   test/
#     test_framework.ref          (stays in root - shared)
#     test_core.ref → general/test_core.ref (symlink)
#     test_qwen_config.ref → qwen/test_qwen_config.ref (symlink)
#     general/
#       test_core.ref (actual file)
#     qwen/
#       test_qwen_config.ref (actual file)
#
# Usage:
#   1. Move your existing general tests to test/general/
#   2. Run this script to create symlinks

set -e

cd "$(dirname "$0")/test"

echo "Creating test symlinks..."
echo ""

# General tests
if [ -d "general" ]; then
    echo "=== General Tests ==="
    for f in general/test_*.ref; do
        if [ -f "$f" ]; then
            name=$(basename "$f")
            # Remove existing file/symlink if present
            rm -f "$name"
            ln -sf "$f" "$name"
            echo "  $name -> $f"
        fi
    done
    echo ""
fi

# Qwen tests
if [ -d "qwen" ]; then
    echo "=== Qwen Tests ==="
    for f in qwen/test_*.ref; do
        if [ -f "$f" ]; then
            name=$(basename "$f")
            # Remove existing file/symlink if present
            rm -f "$name"
            ln -sf "$f" "$name"
            echo "  $name -> $f"
        fi
    done
    echo ""
fi

echo "Done! Symlinks created."
echo ""
echo "You can now run:"
echo "  ./build_tests.sh general    # Run general tests"
echo "  ./build_tests.sh qwen       # Run Qwen tests"
echo "  ./build_tests.sh all        # Run all tests"
