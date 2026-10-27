# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions: https://llvm.org/LICENSE.txt
# RUN: %mojo-build-no-debug-no-assert %s --target-triple aarch64-unknown-opencl --target-cpu generic --emit asm -o %t.cl
# RUN: FileCheck %s --input-file=%t.cl
# CHECK: __kernel void mojo_kernel_{{[0-9]+}}(
# CHECK-DAG: exp(
# CHECK-DAG: exp2(
# CHECK-DAG: log(
# CHECK-DAG: log2(
# CHECK-DAG: sqrt(
# CHECK-DAG: sincos(

from std.memory import Pointer
from std.math import exp, exp2, log, log2, sqrt, rsqrt, recip, sin, cos


@export("scalar_math")
def scalar_math(
    x: Pointer[Float32, MutAnyOrigin], y: Pointer[Float32, MutAnyOrigin]
) abi("C"):
    """Exercise standard-library math dispatch, including exp's fallback elaboration.
    """
    var v = x[unsafe_offset=0]
    y[unsafe_offset=0] = exp(v)
    y[unsafe_offset=1] = exp2(v)
    y[unsafe_offset=2] = log(v)
    y[unsafe_offset=3] = log2(v)
    y[unsafe_offset=4] = sqrt(v)
    y[unsafe_offset=5] = rsqrt(v)
    y[unsafe_offset=6] = recip(v)
    y[unsafe_offset=7] = sin(v)
    y[unsafe_offset=8] = cos(v)
