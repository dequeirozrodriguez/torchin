#!/bin/bash
# torch-g++.sh - Wrapper script for compiling Refal with libtorch

# Use environment variables or defaults
LIBTORCH="${LIBTORCH:-$HOME/qwen3_perl/qwen3_refal/torch/libtorch}"
REFAL_HOME="${REFAL_HOME:-$HOME/qwen3_perl/refal-5-lambda}"
REFAL_RT="$REFAL_HOME/lib/scratch-rt"

# Compile with all necessary includes and libraries
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
